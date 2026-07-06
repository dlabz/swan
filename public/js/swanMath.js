import { WGSL } from './wgsl_shaders.js';
import { SwanStorage } from './swan.js';

export class SwanMathEngine {
    constructor(ui) {
        this.ui = ui;
        this.device = null;
        this.DIM = 1536;
        this.VOCAB_SIZE = 151936;
        this.LAYERS = 28;
        this.pipelines = {};
        this.weights = {}; // Cache for all weights
    }

    async init() {
        // Prefer the discrete/high-performance GPU (ATI 5500M on MBP with switchable graphics).
        // Falls back to any available adapter if high-performance is unavailable.
        let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) {
            adapter = await navigator.gpu.requestAdapter();
        }
        if (!adapter) throw new Error("No WebGPU adapter found.");

        // Log adapter info for debugging
        try {
            const info = await adapter.requestAdapterInfo();
            console.log(`[GPU] Adapter: ${info.vendor} / ${info.architecture} / ${info.device}`);
        } catch(e) { /* requestAdapterInfo may not be available everywhere */ }
        
        console.log(`[GPU] maxStorageBufferBindingSize: ${(adapter.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB`);
        console.log(`[GPU] maxBufferSize: ${(adapter.limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB`);
        
        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize
            }
        });

        this.useCpuFallback = adapter.limits.maxStorageBufferBindingSize < 233400000;

        // Add uncaught error logging
        this.device.addEventListener('uncaughterror', (e) => {
            this.ui.log("WebGPU Error: " + e.error.message, 'error');
        });

        await this.compilePipelines();
        await this.allocateBuffers();
        this.ui.log("WebGPU Engine compiled and buffers allocated.");
    }

    async compilePipelines() {
        const createPipeline = (wgsl, entryPoint) => {
            const module = this.device.createShaderModule({ code: wgsl });
            return this.device.createComputePipeline({
                layout: 'auto',
                compute: { module, entryPoint }
            });
        };

        this.pipelines.embedLookup = createPipeline(WGSL.EmbedLookup, 'main');
        this.pipelines.rmsnorm = createPipeline(WGSL.RMSNorm, 'main');
        this.pipelines.matmul = createPipeline(WGSL.MatMul, 'main');
        this.pipelines.add = createPipeline(WGSL.Add, 'main');
        this.pipelines.rope = createPipeline(WGSL.RoPE, 'main');
        this.pipelines.write_cache = createPipeline(WGSL.WriteCache, 'main');
        this.pipelines.attention = createPipeline(WGSL.Attention, 'main');
        this.pipelines.silu_mul = createPipeline(WGSL.SiLU_Mul, 'main');
    }

    async createBuffer(size, usage, mappedAtCreation = false) {
        return this.device.createBuffer({ size, usage, mappedAtCreation });
    }

    async uploadBuffer(data, usage) {
        if (!(data instanceof Uint8Array) && data instanceof ArrayBuffer) {
            data = new Uint8Array(data);
        } else if (!(data instanceof Uint8Array)) {
            data = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        }
        const buffer = await this.createBuffer(data.byteLength, usage, true);
        new Uint8Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }

    async loadLayerWeight(filename) {
        const file = await SwanStorage.getModelFile(filename, '/model/i7/' + filename);
        const arrayBuffer = await file.arrayBuffer();
        const scale = new Float32Array(arrayBuffer, 0, 1)[0];
        const weightData = new Uint8Array(arrayBuffer, 4);
        const gpuBuffer = await this.uploadBuffer(weightData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        return { buffer: gpuBuffer, scale, arrayBuffer };
    }

    /**
     * CPU-side embedding lookup. Avoids binding the 222MB embed matrix as a GPU storage buffer.
     * Dequantizes int8 -> float32 on CPU and writes directly into hiddenState GPU buffer.
     */
    async cpuEmbedLookup(tokenIds) {
        if (!Array.isArray(tokenIds)) tokenIds = [tokenIds];
        const w = this.weights['model.embed_tokens.weight'];
        const scale = w.scale;
        const i8Data = new Int8Array(w.arrayBuffer, 4); // Skip the 4-byte scale header
        const f32 = new Float32Array(tokenIds.length * this.DIM);
        for (let t = 0; t < tokenIds.length; t++) {
            const tokenId = tokenIds[t];
            const offset = tokenId * this.DIM;
            for (let i = 0; i < this.DIM; i++) {
                f32[t * this.DIM + i] = i8Data[offset + i] * scale;
            }
        }
        // Upload float32 token embedding directly to hiddenState
        this.device.queue.writeBuffer(this.hiddenState, 0, f32);
    }

    /**
     * CPU-side LM head (weight-tied with embed_tokens).
     * Computes logits = normOutput (f32) · embed_tokens^T (int8) and returns argmax.
     * This avoids the 222MB storage buffer binding limit issue on integrated GPUs.
     */
    async cpuLmHead(seqLen = 1) {
        // Read the last row of normOutput from GPU
        const readNorm = this.device.createBuffer({
            size: this.DIM * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const enc = this.device.createCommandEncoder();
        const offset = (seqLen - 1) * this.DIM * 4;
        enc.copyBufferToBuffer(this.normOutput, offset, readNorm, 0, this.DIM * 4);
        this.device.queue.submit([enc.finish()]);
        await readNorm.mapAsync(GPUMapMode.READ);
        const normData = new Float32Array(readNorm.getMappedRange().slice(0));
        readNorm.unmap();
        readNorm.destroy();

        // Dot product with each vocab row (int8 weights)
        const w = this.weights['model.embed_tokens.weight'];
        const scale = w.scale;
        const i8Data = new Int8Array(w.arrayBuffer, 4);
        const VOCAB = this.VOCAB_SIZE;
        const DIM = this.DIM;

        let maxVal = -Infinity;
        let maxIdx = 0;

        // Log top-5 for first two positions (reuse pos tracking via a counter)
        const doLog = this._logPos !== undefined && this._logPos <= 1;
        const logits = doLog ? new Float32Array(VOCAB) : null;

        for (let v = 0; v < VOCAB; v++) {
            const rowOffset = v * DIM;
            let sum = 0;
            for (let i = 0; i < DIM; i++) {
                sum += normData[i] * i8Data[rowOffset + i];
            }
            sum *= scale;
            if (doLog) logits[v] = sum;
            if (sum > maxVal) { maxVal = sum; maxIdx = v; }
        }

        if (doLog) {
            const indexed = Array.from(logits).map((v, i) => [v, i]);
            indexed.sort((a, b) => b[0] - a[0]);
            const top5 = indexed.slice(0, 5).map(([v, i]) => `${i}(${v.toFixed(2)})`).join(', ');
            console.log(`[pos=${this._logPos}] Top-5 logits (CPU): ${top5}`);
        }

        return maxIdx;
    }

    async loadAllWeights(onProgress) {
        this.ui.log("Pre-loading model weights into VRAM. This will take a moment...");
        
        const load = async (name) => {
            if (!this.weights[name]) {
                this.weights[name] = await this.loadLayerWeight(name + '.bin');
            }
        };

        await load('model.embed_tokens.weight');
        await load('model.norm.weight');

        const layerPrefixes = [
            'input_layernorm.weight',
            'post_attention_layernorm.weight',
            'self_attn.q_proj.weight', 'self_attn.q_proj.bias',
            'self_attn.k_proj.weight', 'self_attn.k_proj.bias',
            'self_attn.v_proj.weight', 'self_attn.v_proj.bias',
            'self_attn.o_proj.weight',
            'mlp.gate_proj.weight',
            'mlp.up_proj.weight',
            'mlp.down_proj.weight'
        ];

        // Load layer by layer
        for (let l = 0; l < this.LAYERS; l++) {
            if (onProgress) onProgress(l, this.LAYERS);
            const promises = layerPrefixes.map(p => load(`model.layers.${l}.${p}`));
            await Promise.all(promises);
        }
        
        // Dummy bias buffer for projections that don't have biases
        this.dummyBias = await this.uploadBuffer(new Uint32Array([0]), GPUBufferUsage.STORAGE);

        this.embedScaleBuf = null;
        this.wNormFinalScaleBuf = null;
        this.wNormScaleBuf = [];
        this.wNorm2ScaleBuf = [];

        this.ui.log("All weights loaded successfully.", "system");
    }

    async allocateBuffers() {
        const BUF_STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        this.maxPrefillLen = 512;
        this.tokensInputBuffer = await this.createBuffer(this.maxPrefillLen * 4, BUF_STORAGE);

        this.hiddenState = await this.createBuffer(this.maxPrefillLen * this.DIM * 4, BUF_STORAGE);
        this.residual = await this.createBuffer(this.maxPrefillLen * this.DIM * 4, BUF_STORAGE);
        this.normOutput = await this.createBuffer(this.maxPrefillLen * this.DIM * 4, BUF_STORAGE);
        
        this.qBuffer = await this.createBuffer(this.maxPrefillLen * 1536 * 4, BUF_STORAGE);
        this.kBuffer = await this.createBuffer(this.maxPrefillLen * 256 * 4, BUF_STORAGE);
        this.vBuffer = await this.createBuffer(this.maxPrefillLen * 256 * 4, BUF_STORAGE);
        this.attnOutput = await this.createBuffer(this.maxPrefillLen * 1536 * 4, BUF_STORAGE);
        
        this.gateBuffer = await this.createBuffer(this.maxPrefillLen * 8960 * 4, BUF_STORAGE);
        this.upBuffer = await this.createBuffer(this.maxPrefillLen * 8960 * 4, BUF_STORAGE);
        this.downBuffer = await this.createBuffer(this.maxPrefillLen * 1536 * 4, BUF_STORAGE);
        
        this.logitsBuffer = await this.createBuffer(this.VOCAB_SIZE * 4, BUF_STORAGE);
        this.readbackBuffer = await this.createBuffer(this.VOCAB_SIZE * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        
        // KV Cache: maxSeqLen * kv_heads(2) * head_dim(128) * 4 bytes
        this.kCaches = [];
        this.vCaches = [];
        const maxSeqLen = 512;
        const cacheSize = maxSeqLen * 2 * 128 * 4;
        for (let i = 0; i < this.LAYERS; i++) {
            this.kCaches.push(await this.createBuffer(cacheSize, BUF_STORAGE));
            this.vCaches.push(await this.createBuffer(cacheSize, BUF_STORAGE));
        }

        // Cache systems
        this.bgCache = {};
        this.bufCache = {};

        this.embedScaleBuf = null;
        this.wNormFinalScaleBuf = null;
        this.wNormScaleBuf = [];
        this.wNorm2ScaleBuf = [];

        // Shared mutable config buffers (updated once per forwardSequence)
        this.ropeConfigBuf = await this.createBuffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.writeCacheConfigBuf = await this.createBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.attnConfigBuf = await this.createBuffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.lastRowBuf = await this.createBuffer(this.DIM * 4, BUF_STORAGE);
    }

    getRMSNormConfigBuf(seqLen) {
        const key = `rmsnormConfig_${seqLen}`;
        if (this.bufCache[key]) return this.bufCache[key];
        
        const data = new ArrayBuffer(16);
        new Uint32Array(data)[0] = this.DIM;
        new Float32Array(data)[1] = 1e-6;
        new Uint32Array(data)[2] = seqLen;
        
        const buf = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(buf, 0, new Uint8Array(data));
        this.bufCache[key] = buf;
        return buf;
    }
    
    getAddConfigBuf(seqLen) {
        const key = `addConfig_${seqLen}`;
        if (this.bufCache[key]) return this.bufCache[key];
        const buf = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(buf, 0, new Uint32Array([this.DIM, seqLen, 0, 0]));
        this.bufCache[key] = buf;
        return buf;
    }
    
    getSiLUConfigBuf(seqLen) {
        const key = `siluConfig_${seqLen}`;
        if (this.bufCache[key]) return this.bufCache[key];
        const buf = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(buf, 0, new Uint32Array([8960, seqLen, 0, 0]));
        this.bufCache[key] = buf;
        return buf;
    }
    
    getEmbedConfigBuf(seqLen) {
        const key = `embedConfig_${seqLen}`;
        if (this.bufCache[key]) return this.bufCache[key];
        const buf = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(buf, 0, new Uint32Array([this.DIM, seqLen, 0, 0]));
        this.bufCache[key] = buf;
        return buf;
    }

    getMatMulBindGroup(name, input, weightObj, biasObj, output, inDim, outDim, seqLen = 1) {
        const cacheKey = `${name}_${seqLen}`;
        if (this.bgCache[cacheKey]) {
            return this.bgCache[cacheKey];
        }

        const configBuf = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(configBuf, 0, new Uint32Array([inDim, outDim, seqLen, 0]));
        
        const scalesBuf = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(scalesBuf, 0, new Float32Array([weightObj.scale, biasObj ? biasObj.scale : 0.0]));
        
        const hasBiasBuf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(hasBiasBuf, 0, new Uint32Array([biasObj ? 1 : 0]));

        const bg = this.device.createBindGroup({
            layout: this.pipelines.matmul.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: input } },
                { binding: 1, resource: { buffer: weightObj.buffer } },
                { binding: 2, resource: { buffer: biasObj ? biasObj.buffer : this.dummyBias } },
                { binding: 3, resource: { buffer: output } },
                { binding: 4, resource: { buffer: configBuf } },
                { binding: 5, resource: { buffer: scalesBuf } },
                { binding: 6, resource: { buffer: hasBiasBuf } }
            ]
        });

        this.bgCache[cacheKey] = bg;
        return bg;
    }

    async forwardSequence(tokenIds, startPos) {
        if (!Array.isArray(tokenIds)) tokenIds = [tokenIds];
        const seqLen = tokenIds.length;
        
        // Update shared mutable configs
        const ropeConfigData = new ArrayBuffer(32);
        new Uint32Array(ropeConfigData)[0] = 1536;      
        new Uint32Array(ropeConfigData)[1] = 128;       
        new Uint32Array(ropeConfigData)[2] = startPos;       
        new Float32Array(ropeConfigData)[3] = 1000000.0; 
        new Uint32Array(ropeConfigData)[4] = seqLen; 
        this.device.queue.writeBuffer(this.ropeConfigBuf, 0, new Uint8Array(ropeConfigData));

        const writeCacheConfigData = new Uint32Array([256, startPos, seqLen, this.maxPrefillLen]);
        this.device.queue.writeBuffer(this.writeCacheConfigBuf, 0, writeCacheConfigData);

        const attnConfigData = new Uint32Array([12, 2, 128, startPos, seqLen, 0, 0, 0]);
        this.device.queue.writeBuffer(this.attnConfigBuf, 0, attnConfigData);

        const addConfig = this.getAddConfigBuf(seqLen);
        const siluConfig = this.getSiLUConfigBuf(seqLen);
        const rmsnormConfig = this.getRMSNormConfigBuf(seqLen);

        this.device.pushErrorScope('validation');
        this.device.pushErrorScope('out-of-memory');

        const encoder = this.device.createCommandEncoder();

        // 1. Embed Lookup
        if (this.useCpuFallback) {
            await this.cpuEmbedLookup(tokenIds);
            encoder.copyBufferToBuffer(this.hiddenState, 0, this.residual, 0, seqLen * this.DIM * 4);
        } else {
            this.device.queue.writeBuffer(this.tokensInputBuffer, 0, new Uint32Array(tokenIds));
            
            const embedConfig = this.getEmbedConfigBuf(seqLen);
            const embedKey = `embed_${seqLen}`;
            if (!this.bgCache[embedKey]) {
                if (!this.embedScaleBuf) {
                    this.embedScaleBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                    this.device.queue.writeBuffer(this.embedScaleBuf, 0, new Float32Array([this.weights['model.embed_tokens.weight'].scale]));
                }
                this.bgCache[embedKey] = this.device.createBindGroup({
                    layout: this.pipelines.embedLookup.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.weights['model.embed_tokens.weight'].buffer } },
                        { binding: 1, resource: { buffer: this.hiddenState } },
                        { binding: 2, resource: { buffer: this.tokensInputBuffer } },
                        { binding: 3, resource: { buffer: embedConfig } },
                        { binding: 4, resource: { buffer: this.embedScaleBuf } }
                    ]
                });
            }
            const bgEmbed = this.bgCache[embedKey];

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.embedLookup);
            pass.setBindGroup(0, bgEmbed);
            pass.dispatchWorkgroups(Math.ceil(this.DIM / 64), seqLen);
            pass.end();
            encoder.copyBufferToBuffer(this.hiddenState, 0, this.residual, 0, seqLen * this.DIM * 4);
        }

        let pass;

        for (let l = 0; l < this.LAYERS; l++) {
            encoder.copyBufferToBuffer(this.hiddenState, 0, this.residual, 0, seqLen * this.DIM * 4);

            const norm1Key = `norm1_${l}_${seqLen}`;
            if (!this.bgCache[norm1Key]) {
                if (!this.wNormScaleBuf[l]) {
                    this.wNormScaleBuf[l] = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                    this.device.queue.writeBuffer(this.wNormScaleBuf[l], 0, new Float32Array([this.weights[`model.layers.${l}.input_layernorm.weight`].scale]));
                }
                this.bgCache[norm1Key] = this.device.createBindGroup({
                    layout: this.pipelines.rmsnorm.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.hiddenState } },
                        { binding: 1, resource: { buffer: this.weights[`model.layers.${l}.input_layernorm.weight`].buffer } },
                        { binding: 2, resource: { buffer: this.normOutput } },
                        { binding: 3, resource: { buffer: rmsnormConfig } },
                        { binding: 4, resource: { buffer: this.wNormScaleBuf[l] } }
                    ]
                });
            }
            const bgNorm1 = this.bgCache[norm1Key];

            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.rmsnorm);
            pass.setBindGroup(0, bgNorm1);
            pass.dispatchWorkgroups(Math.ceil(this.DIM / 256), seqLen);
            pass.end();

            const bgQ = this.getMatMulBindGroup(
                `q_${l}`,
                this.normOutput,
                this.weights[`model.layers.${l}.self_attn.q_proj.weight`],
                this.weights[`model.layers.${l}.self_attn.q_proj.bias`],
                this.qBuffer, 1536, 1536, seqLen
            );
            const bgK = this.getMatMulBindGroup(
                `k_${l}`,
                this.normOutput,
                this.weights[`model.layers.${l}.self_attn.k_proj.weight`],
                this.weights[`model.layers.${l}.self_attn.k_proj.bias`],
                this.kBuffer, 1536, 256, seqLen
            );
            const bgV = this.getMatMulBindGroup(
                `v_${l}`,
                this.normOutput,
                this.weights[`model.layers.${l}.self_attn.v_proj.weight`],
                this.weights[`model.layers.${l}.self_attn.v_proj.bias`],
                this.vBuffer, 1536, 256, seqLen
            );
            
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgQ);
            pass.dispatchWorkgroups(Math.ceil(1536 / 64), 1);
            pass.setBindGroup(0, bgK);
            pass.dispatchWorkgroups(Math.ceil(256 / 64), 1);
            pass.setBindGroup(0, bgV);
            pass.dispatchWorkgroups(Math.ceil(256 / 64), 1);
            pass.end();

            const ropeKey = `rope_${l}`;
            if (!this.bgCache[ropeKey]) {
                this.bgCache[ropeKey] = this.device.createBindGroup({
                    layout: this.pipelines.rope.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.qBuffer } },
                        { binding: 1, resource: { buffer: this.kBuffer } },
                        { binding: 2, resource: { buffer: this.ropeConfigBuf } }
                    ]
                });
            }
            const bgRope = this.bgCache[ropeKey];

            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.rope);
            pass.setBindGroup(0, bgRope);
            pass.dispatchWorkgroups(Math.ceil(1536 / 128), seqLen);
            pass.end();
            
            // WRITE CACHE
            const writeKKey = `writeK_${l}`;
            if (!this.bgCache[writeKKey]) {
                this.bgCache[writeKKey] = this.device.createBindGroup({
                    layout: this.pipelines.write_cache.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.kBuffer } },
                        { binding: 1, resource: { buffer: this.kCaches[l] } },
                        { binding: 2, resource: { buffer: this.writeCacheConfigBuf } }
                    ]
                });
            }
            const bgWriteK = this.bgCache[writeKKey];

            const writeVKey = `writeV_${l}`;
            if (!this.bgCache[writeVKey]) {
                this.bgCache[writeVKey] = this.device.createBindGroup({
                    layout: this.pipelines.write_cache.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.vBuffer } },
                        { binding: 1, resource: { buffer: this.vCaches[l] } },
                        { binding: 2, resource: { buffer: this.writeCacheConfigBuf } }
                    ]
                });
            }
            const bgWriteV = this.bgCache[writeVKey];

            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.write_cache);
            pass.setBindGroup(0, bgWriteK);
            pass.dispatchWorkgroups(Math.ceil(256 / 64), seqLen);
            pass.setBindGroup(0, bgWriteV);
            pass.dispatchWorkgroups(Math.ceil(256 / 64), seqLen);
            pass.end();

            // ATTENTION
            const attnKey = `attn_${l}`;
            if (!this.bgCache[attnKey]) {
                this.bgCache[attnKey] = this.device.createBindGroup({
                    layout: this.pipelines.attention.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.qBuffer } },
                        { binding: 1, resource: { buffer: this.kCaches[l] } },
                        { binding: 2, resource: { buffer: this.vCaches[l] } },
                        { binding: 3, resource: { buffer: this.attnOutput } },
                        { binding: 4, resource: { buffer: this.attnConfigBuf } }
                    ]
                });
            }
            const bgAttn = this.bgCache[attnKey];

            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.attention);
            pass.setBindGroup(0, bgAttn);
            pass.dispatchWorkgroups(12, seqLen); 
            pass.end();

            const bgO = this.getMatMulBindGroup(
                `o_${l}`,
                this.attnOutput,
                this.weights[`model.layers.${l}.self_attn.o_proj.weight`],
                null,
                this.downBuffer, 1536, 1536, seqLen
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgO);
            pass.dispatchWorkgroups(Math.ceil(1536 / 64), 1);
            pass.end();

            const add1Key = `add1_${l}_${seqLen}`;
            if (!this.bgCache[add1Key]) {
                this.bgCache[add1Key] = this.device.createBindGroup({
                    layout: this.pipelines.add.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.residual } },
                        { binding: 1, resource: { buffer: this.downBuffer } },
                        { binding: 2, resource: { buffer: addConfig } }
                    ]
                });
            }
            const bgAdd1 = this.bgCache[add1Key];

            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.add);
            pass.setBindGroup(0, bgAdd1);
            pass.dispatchWorkgroups(Math.ceil((this.DIM * seqLen) / 256));
            pass.end();
            encoder.copyBufferToBuffer(this.residual, 0, this.hiddenState, 0, seqLen * this.DIM * 4);

            const norm2Key = `norm2_${l}_${seqLen}`;
            if (!this.bgCache[norm2Key]) {
                if (!this.wNorm2ScaleBuf[l]) {
                    this.wNorm2ScaleBuf[l] = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                    this.device.queue.writeBuffer(this.wNorm2ScaleBuf[l], 0, new Float32Array([this.weights[`model.layers.${l}.post_attention_layernorm.weight`].scale]));
                }
                this.bgCache[norm2Key] = this.device.createBindGroup({
                    layout: this.pipelines.rmsnorm.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.hiddenState } },
                        { binding: 1, resource: { buffer: this.weights[`model.layers.${l}.post_attention_layernorm.weight`].buffer } },
                        { binding: 2, resource: { buffer: this.normOutput } },
                        { binding: 3, resource: { buffer: rmsnormConfig } },
                        { binding: 4, resource: { buffer: this.wNorm2ScaleBuf[l] } }
                    ]
                });
            }
            const bgNorm2 = this.bgCache[norm2Key];

            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.rmsnorm);
            pass.setBindGroup(0, bgNorm2);
            pass.dispatchWorkgroups(Math.ceil(this.DIM / 256), seqLen);
            pass.end();

            const bgGate = this.getMatMulBindGroup(
                `gate_${l}`,
                this.normOutput,
                this.weights[`model.layers.${l}.mlp.gate_proj.weight`],
                null,
                this.gateBuffer, 1536, 8960, seqLen
            );
            const bgUp = this.getMatMulBindGroup(
                `up_${l}`,
                this.normOutput,
                this.weights[`model.layers.${l}.mlp.up_proj.weight`],
                null,
                this.upBuffer, 1536, 8960, seqLen
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgGate);
            pass.dispatchWorkgroups(Math.ceil(8960 / 64), 1);
            pass.setBindGroup(0, bgUp);
            pass.dispatchWorkgroups(Math.ceil(8960 / 64), 1);
            pass.end();

            const siluKey = `silu_${l}_${seqLen}`;
            if (!this.bgCache[siluKey]) {
                this.bgCache[siluKey] = this.device.createBindGroup({
                    layout: this.pipelines.silu_mul.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.gateBuffer } },
                        { binding: 1, resource: { buffer: this.upBuffer } },
                        { binding: 2, resource: { buffer: siluConfig } }
                    ]
                });
            }
            const bgSilu = this.bgCache[siluKey];

            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.silu_mul);
            pass.setBindGroup(0, bgSilu);
            pass.dispatchWorkgroups(Math.ceil((8960 * seqLen) / 256));
            pass.end();

            const bgDown = this.getMatMulBindGroup(
                `down_${l}`,
                this.gateBuffer,
                this.weights[`model.layers.${l}.mlp.down_proj.weight`],
                null,
                this.downBuffer, 8960, 1536, seqLen
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgDown);
            pass.dispatchWorkgroups(Math.ceil(1536 / 64), 1);
            pass.end();

            const add2Key = `add2_${l}_${seqLen}`;
            if (!this.bgCache[add2Key]) {
                this.bgCache[add2Key] = this.device.createBindGroup({
                    layout: this.pipelines.add.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.residual } },
                        { binding: 1, resource: { buffer: this.downBuffer } },
                        { binding: 2, resource: { buffer: addConfig } }
                    ]
                });
            }
            const bgAdd2 = this.bgCache[add2Key];

            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.add);
            pass.setBindGroup(0, bgAdd2);
            pass.dispatchWorkgroups(Math.ceil((this.DIM * seqLen) / 256));
            pass.end();
            encoder.copyBufferToBuffer(this.residual, 0, this.hiddenState, 0, seqLen * this.DIM * 4);
        }

        const normFinalKey = `normFinal_${seqLen}`;
        if (!this.bgCache[normFinalKey]) {
            if (!this.wNormFinalScaleBuf) {
                this.wNormFinalScaleBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                this.device.queue.writeBuffer(this.wNormFinalScaleBuf, 0, new Float32Array([this.weights['model.norm.weight'].scale]));
            }
            this.bgCache[normFinalKey] = this.device.createBindGroup({
                layout: this.pipelines.rmsnorm.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.hiddenState } },
                    { binding: 1, resource: { buffer: this.weights['model.norm.weight'].buffer } },
                    { binding: 2, resource: { buffer: this.normOutput } },
                    { binding: 3, resource: { buffer: rmsnormConfig } },
                    { binding: 4, resource: { buffer: this.wNormFinalScaleBuf } }
                ]
            });
        }
        const bgNormFinal = this.bgCache[normFinalKey];

        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.rmsnorm);
        pass.setBindGroup(0, bgNormFinal);
        pass.dispatchWorkgroups(Math.ceil(this.DIM / 256), seqLen);
        pass.end();

        if (this.useCpuFallback) {
            this.device.queue.submit([encoder.finish()]);
            const res = await this.cpuLmHead(seqLen);
            
            const oomError = await this.device.popErrorScope();
            const valError = await this.device.popErrorScope();
            if (oomError) console.error('[GPU OOM]', oomError.message);
            if (valError) console.error('[GPU Validation]', valError.message);
            
            return res;
        } else {
            encoder.copyBufferToBuffer(this.normOutput, (seqLen - 1) * this.DIM * 4, this.lastRowBuf, 0, this.DIM * 4);

            const bgHead = this.getMatMulBindGroup(
                `head`,
                this.lastRowBuf,
                this.weights['model.embed_tokens.weight'],
                null,
                this.logitsBuffer, 1536, this.VOCAB_SIZE, 1
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgHead);
            pass.dispatchWorkgroups(Math.ceil(this.VOCAB_SIZE / 64), 1);
            pass.end();

            encoder.copyBufferToBuffer(this.logitsBuffer, 0, this.readbackBuffer, 0, this.VOCAB_SIZE * 4);
            this.device.queue.submit([encoder.finish()]);

            const oomError = await this.device.popErrorScope();
            const valError = await this.device.popErrorScope();
            if (oomError) console.error('[GPU OOM]', oomError.message);
            if (valError) console.error('[GPU Validation]', valError.message);

            await this.readbackBuffer.mapAsync(GPUMapMode.READ);
            const logits = new Float32Array(this.readbackBuffer.getMappedRange().slice(0));
            this.readbackBuffer.unmap();

            let maxVal = -Infinity;
            let maxIdx = 0;
            for (let i = 0; i < this.VOCAB_SIZE; i++) {
                if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
            }
            return maxIdx;
        }
    }
}
