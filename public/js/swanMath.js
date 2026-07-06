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
        
        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize
            }
        });

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
        this.pipelines.attention = createPipeline(WGSL.Attention, 'main');
        this.pipelines.silu_mul = createPipeline(WGSL.SiLU_Mul, 'main');
    }

    async createBuffer(size, usage, mappedAtCreation = false) {
        return this.device.createBuffer({ size, usage, mappedAtCreation });
    }

    async uploadBuffer(data, usage) {
        const buffer = await this.createBuffer(data.byteLength, usage, true);
        new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        buffer.unmap();
        return buffer;
    }

    async loadLayerWeight(filename) {
        const file = await SwanStorage.getModelFile(filename, '/model/i7/' + filename);
        const arrayBuffer = await file.arrayBuffer();
        const scale = new Float32Array(arrayBuffer, 0, 1)[0];
        const weightData = new Uint8Array(arrayBuffer, 4);
        const gpuBuffer = await this.uploadBuffer(weightData, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        return { buffer: gpuBuffer, scale };
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
        this.ui.log("All weights loaded successfully.", "system");
    }

    async allocateBuffers() {
        const BUF_STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        this.hiddenState = await this.createBuffer(this.DIM * 4, BUF_STORAGE);
        this.residual = await this.createBuffer(this.DIM * 4, BUF_STORAGE);
        this.normOutput = await this.createBuffer(this.DIM * 4, BUF_STORAGE);
        
        this.qBuffer = await this.createBuffer(1536 * 4, BUF_STORAGE);
        this.kBuffer = await this.createBuffer(256 * 4, BUF_STORAGE);
        this.vBuffer = await this.createBuffer(256 * 4, BUF_STORAGE);
        this.attnOutput = await this.createBuffer(1536 * 4, BUF_STORAGE);
        
        this.gateBuffer = await this.createBuffer(8960 * 4, BUF_STORAGE);
        this.upBuffer = await this.createBuffer(8960 * 4, BUF_STORAGE);
        this.downBuffer = await this.createBuffer(1536 * 4, BUF_STORAGE);
        
        this.logitsBuffer = await this.createBuffer(this.VOCAB_SIZE * 4, BUF_STORAGE);
        this.readbackBuffer = await this.createBuffer(this.VOCAB_SIZE * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        
        // KV Cache: maxSeqLen * kv_heads(2) * head_dim(128) * 4 bytes
        this.kCaches = [];
        this.vCaches = [];
        const maxSeqLen = 512;
        const cacheSize = maxSeqLen * 2 * 128 * 4; // 524288 bytes per layer
        for (let i = 0; i < this.LAYERS; i++) {
            this.kCaches.push(await this.createBuffer(cacheSize, BUF_STORAGE));
            this.vCaches.push(await this.createBuffer(cacheSize, BUF_STORAGE));
        }

        // Shared uniforms
        this.addConfig = await this.uploadBuffer(new Uint32Array([this.DIM]), GPUBufferUsage.UNIFORM);
        this.siluConfig = await this.uploadBuffer(new Uint32Array([8960]), GPUBufferUsage.UNIFORM);
        const rmsnormConfigData = new ArrayBuffer(8);
        new Uint32Array(rmsnormConfigData)[0] = this.DIM;
        new Float32Array(rmsnormConfigData)[1] = 1e-6;
        this.rmsnormConfig = await this.uploadBuffer(rmsnormConfigData, GPUBufferUsage.UNIFORM);
    }

    getMatMulBindGroup(input, weightObj, biasObj, output, inDim, outDim) {
        const configBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(configBuf, 0, new Uint32Array([inDim, outDim]));
        
        const scalesBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(scalesBuf, 0, new Float32Array([weightObj.scale, biasObj ? biasObj.scale : 0.0]));
        
        const hasBiasBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(hasBiasBuf, 0, new Uint32Array([biasObj ? 1 : 0]));

        return this.device.createBindGroup({
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
    }

    async forward(token, pos) {
        // Capture any GPU validation/OOM errors that occur during this forward pass
        this.device.pushErrorScope('validation');
        this.device.pushErrorScope('out-of-memory');

        const encoder = this.device.createCommandEncoder();
        // Track transient per-step uniform buffers for cleanup after submit
        const transientBuffers = [];
        const tempBuf = (data, usage) => {
            const buf = this.device.createBuffer({
                size: data.byteLength,
                usage,
                mappedAtCreation: true
            });
            new Uint8Array(buf.getMappedRange()).set(
                new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            );
            buf.unmap();
            transientBuffers.push(buf);
            return buf;
        };
        
        // 1. Embed Lookup
        const embedConfig = tempBuf(new Uint32Array([this.DIM, token]), GPUBufferUsage.UNIFORM);
        const embedScale = tempBuf(
            new Float32Array([this.weights['model.embed_tokens.weight'].scale]),
            GPUBufferUsage.UNIFORM
        );
        
        const bgEmbed = this.device.createBindGroup({
            layout: this.pipelines.embedLookup.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.weights['model.embed_tokens.weight'].buffer } },
                { binding: 1, resource: { buffer: this.hiddenState } },
                { binding: 2, resource: { buffer: embedConfig } },
                { binding: 3, resource: { buffer: embedScale } }
            ]
        });

        let pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.embedLookup);
        pass.setBindGroup(0, bgEmbed);
        pass.dispatchWorkgroups(Math.ceil(this.DIM / 64));
        pass.end();

        // 2. Transformer Layers
        for (let l = 0; l < this.LAYERS; l++) {
            // Save hidden state as residual at the start of this layer
            encoder.copyBufferToBuffer(this.hiddenState, 0, this.residual, 0, this.DIM * 4);

            // a. Pre-Attention RMSNorm: hiddenState -> normOutput
            const wNormScale = tempBuf(
                new Float32Array([this.weights[`model.layers.${l}.input_layernorm.weight`].scale]),
                GPUBufferUsage.UNIFORM
            );
            const bgNorm1 = this.device.createBindGroup({
                layout: this.pipelines.rmsnorm.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.hiddenState } },
                    { binding: 1, resource: { buffer: this.weights[`model.layers.${l}.input_layernorm.weight`].buffer } },
                    { binding: 2, resource: { buffer: this.normOutput } },
                    { binding: 3, resource: { buffer: this.rmsnormConfig } },
                    { binding: 4, resource: { buffer: wNormScale } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.rmsnorm);
            pass.setBindGroup(0, bgNorm1);
            pass.dispatchWorkgroups(Math.ceil(this.DIM / 256));
            pass.end();

            // b. QKV Projections: normOutput -> qBuffer/kBuffer/vBuffer
            const bgQ = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.self_attn.q_proj.weight`],
                this.weights[`model.layers.${l}.self_attn.q_proj.bias`],
                this.qBuffer, 1536, 1536
            );
            const bgK = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.self_attn.k_proj.weight`],
                this.weights[`model.layers.${l}.self_attn.k_proj.bias`],
                this.kBuffer, 1536, 256
            );
            const bgV = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.self_attn.v_proj.weight`],
                this.weights[`model.layers.${l}.self_attn.v_proj.bias`],
                this.vBuffer, 1536, 256
            );
            
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgQ);
            pass.dispatchWorkgroups(Math.ceil(1536 / 64));
            pass.setBindGroup(0, bgK);
            pass.dispatchWorkgroups(Math.ceil(256 / 64));
            pass.setBindGroup(0, bgV);
            pass.dispatchWorkgroups(Math.ceil(256 / 64));
            pass.end();

            // c. RoPE: rotate qBuffer (all heads) and kBuffer (kv_heads only)
            const ropeConfigData = new ArrayBuffer(16);
            new Uint32Array(ropeConfigData)[0] = 1536;      // Q total dim
            new Uint32Array(ropeConfigData)[1] = 128;       // head_dim
            new Uint32Array(ropeConfigData)[2] = pos;       // sequence position
            new Float32Array(ropeConfigData)[3] = 1000000.0; // RoPE base theta
            const ropeConfig = tempBuf(ropeConfigData, GPUBufferUsage.UNIFORM);
            
            const bgRope = this.device.createBindGroup({
                layout: this.pipelines.rope.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.qBuffer } },
                    { binding: 1, resource: { buffer: this.kBuffer } },
                    { binding: 2, resource: { buffer: ropeConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.rope);
            pass.setBindGroup(0, bgRope);
            pass.dispatchWorkgroups(Math.ceil(1536 / 128));
            pass.end();

            // d. Attention: (qBuffer + kCaches + vCaches) -> attnOutput
            // Config: [num_heads=12, kv_heads=2, head_dim=128, pos]
            const attnConfig = tempBuf(new Uint32Array([12, 2, 128, pos]), GPUBufferUsage.UNIFORM);
            const bgAttn = this.device.createBindGroup({
                layout: this.pipelines.attention.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.qBuffer } },
                    { binding: 1, resource: { buffer: this.kCaches[l] } },
                    { binding: 2, resource: { buffer: this.vCaches[l] } },
                    { binding: 3, resource: { buffer: this.attnOutput } },
                    { binding: 4, resource: { buffer: this.kBuffer } },
                    { binding: 5, resource: { buffer: this.vBuffer } },
                    { binding: 6, resource: { buffer: attnConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.attention);
            pass.setBindGroup(0, bgAttn);
            pass.dispatchWorkgroups(12); // one workgroup per Q head
            pass.end();

            // e. O Projection: attnOutput -> downBuffer
            const bgO = this.getMatMulBindGroup(
                this.attnOutput,
                this.weights[`model.layers.${l}.self_attn.o_proj.weight`],
                null,
                this.downBuffer, 1536, 1536
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgO);
            pass.dispatchWorkgroups(Math.ceil(1536 / 64));
            pass.end();

            // f. Residual Add: residual += downBuffer
            const bgAdd1 = this.device.createBindGroup({
                layout: this.pipelines.add.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.residual } },
                    { binding: 1, resource: { buffer: this.downBuffer } },
                    { binding: 2, resource: { buffer: this.addConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.add);
            pass.setBindGroup(0, bgAdd1);
            pass.dispatchWorkgroups(Math.ceil(this.DIM / 256));
            pass.end();
            // residual now has post-attention hidden state; sync to hiddenState
            encoder.copyBufferToBuffer(this.residual, 0, this.hiddenState, 0, this.DIM * 4);

            // g. Post-Attention RMSNorm: hiddenState -> normOutput
            const wNorm2Scale = tempBuf(
                new Float32Array([this.weights[`model.layers.${l}.post_attention_layernorm.weight`].scale]),
                GPUBufferUsage.UNIFORM
            );
            const bgNorm2 = this.device.createBindGroup({
                layout: this.pipelines.rmsnorm.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.hiddenState } },
                    { binding: 1, resource: { buffer: this.weights[`model.layers.${l}.post_attention_layernorm.weight`].buffer } },
                    { binding: 2, resource: { buffer: this.normOutput } },
                    { binding: 3, resource: { buffer: this.rmsnormConfig } },
                    { binding: 4, resource: { buffer: wNorm2Scale } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.rmsnorm);
            pass.setBindGroup(0, bgNorm2);
            pass.dispatchWorkgroups(Math.ceil(this.DIM / 256));
            pass.end();

            // h. MLP Gate & Up: normOutput -> gateBuffer / upBuffer
            const bgGate = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.mlp.gate_proj.weight`],
                null,
                this.gateBuffer, 1536, 8960
            );
            const bgUp = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.mlp.up_proj.weight`],
                null,
                this.upBuffer, 1536, 8960
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgGate);
            pass.dispatchWorkgroups(Math.ceil(8960 / 64));
            pass.setBindGroup(0, bgUp);
            pass.dispatchWorkgroups(Math.ceil(8960 / 64));
            pass.end();

            // i. SiLU Mul: gateBuffer = SiLU(gateBuffer) * upBuffer
            const bgSilu = this.device.createBindGroup({
                layout: this.pipelines.silu_mul.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.gateBuffer } },
                    { binding: 1, resource: { buffer: this.upBuffer } },
                    { binding: 2, resource: { buffer: this.siluConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.silu_mul);
            pass.setBindGroup(0, bgSilu);
            pass.dispatchWorkgroups(Math.ceil(8960 / 256));
            pass.end();

            // j. MLP Down: gateBuffer -> downBuffer
            const bgDown = this.getMatMulBindGroup(
                this.gateBuffer,
                this.weights[`model.layers.${l}.mlp.down_proj.weight`],
                null,
                this.downBuffer, 8960, 1536
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgDown);
            pass.dispatchWorkgroups(Math.ceil(1536 / 64));
            pass.end();

            // k. Residual Add: residual += downBuffer -> hiddenState
            const bgAdd2 = this.device.createBindGroup({
                layout: this.pipelines.add.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.residual } },
                    { binding: 1, resource: { buffer: this.downBuffer } },
                    { binding: 2, resource: { buffer: this.addConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.add);
            pass.setBindGroup(0, bgAdd2);
            pass.dispatchWorkgroups(Math.ceil(this.DIM / 256));
            pass.end();
            encoder.copyBufferToBuffer(this.residual, 0, this.hiddenState, 0, this.DIM * 4);
        }

        // 3. Final RMSNorm: hiddenState -> normOutput
        const wNormFinalScale = tempBuf(
            new Float32Array([this.weights['model.norm.weight'].scale]),
            GPUBufferUsage.UNIFORM
        );
        const bgNormFinal = this.device.createBindGroup({
            layout: this.pipelines.rmsnorm.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.hiddenState } },
                { binding: 1, resource: { buffer: this.weights['model.norm.weight'].buffer } },
                { binding: 2, resource: { buffer: this.normOutput } },
                { binding: 3, resource: { buffer: this.rmsnormConfig } },
                { binding: 4, resource: { buffer: wNormFinalScale } }
            ]
        });
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.rmsnorm);
        pass.setBindGroup(0, bgNormFinal);
        pass.dispatchWorkgroups(Math.ceil(this.DIM / 256));
        pass.end();

        // 4. LM Head (weight-tied to embed_tokens: shape [VOCAB_SIZE, DIM])
        const bgHead = this.getMatMulBindGroup(
            this.normOutput,
            this.weights['model.embed_tokens.weight'],
            null,
            this.logitsBuffer, 1536, this.VOCAB_SIZE
        );
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.matmul);
        pass.setBindGroup(0, bgHead);
        pass.dispatchWorkgroups(Math.ceil(this.VOCAB_SIZE / 64));
        pass.end();

        // 5. Copy logits to CPU-readable buffer
        encoder.copyBufferToBuffer(this.logitsBuffer, 0, this.readbackBuffer, 0, this.VOCAB_SIZE * 4);
        
        this.device.queue.submit([encoder.finish()]);

        // Pop GPU error scopes — errors are reported here after submit
        const oomError = await this.device.popErrorScope();
        const valError = await this.device.popErrorScope();
        if (oomError) console.error('[GPU OOM]', oomError.message);
        if (valError) console.error('[GPU Validation]', valError.message);

        // Free all transient per-step buffers now that GPU commands are submitted
        for (const buf of transientBuffers) buf.destroy();

        // 6. Argmax on CPU
        await this.readbackBuffer.mapAsync(GPUMapMode.READ);
        const logits = new Float32Array(this.readbackBuffer.getMappedRange());

        // Log top-5 logits for diagnosis on the first two steps
        if (pos <= 1) {
            const indexed = Array.from(logits).map((v, i) => [v, i]);
            indexed.sort((a, b) => b[0] - a[0]);
            const top5 = indexed.slice(0, 5).map(([v, i]) => `${i}(${v.toFixed(2)})`).join(', ');
            console.log(`[pos=${pos}] Top-5 logits: ${top5}`);
        }

        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let i = 0; i < this.VOCAB_SIZE; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        this.readbackBuffer.unmap();
        
        return maxIdx;
    }
}
