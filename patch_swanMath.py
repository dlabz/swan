import re

with open('public/js/swanMath.js', 'r') as f:
    code = f.read()

# 1. Compile write_cache pipeline
code = code.replace(
    "this.pipelines.rope = createPipeline(WGSL.RoPE, 'main');",
    "this.pipelines.rope = createPipeline(WGSL.RoPE, 'main');\n        this.pipelines.write_cache = createPipeline(WGSL.WriteCache, 'main');"
)

# 2. cpuEmbedLookup array support
old_cpu_embed = """    async cpuEmbedLookup(tokenId) {
        const w = this.weights['model.embed_tokens.weight'];
        const scale = w.scale;
        const i8Data = new Int8Array(w.arrayBuffer, 4); // Skip the 4-byte scale header
        const offset = tokenId * this.DIM;
        const f32 = new Float32Array(this.DIM);
        for (let i = 0; i < this.DIM; i++) {
            f32[i] = i8Data[offset + i] * scale;
        }
        // Upload float32 token embedding directly to hiddenState
        this.device.queue.writeBuffer(this.hiddenState, 0, f32);
    }"""
new_cpu_embed = """    async cpuEmbedLookup(tokenIds) {
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
    }"""
code = code.replace(old_cpu_embed, new_cpu_embed)

# 3. cpuLmHead last token extraction
# Find readNorm copyBufferToBuffer
old_lm_head = """    async cpuLmHead() {
        // Read normOutput from GPU
        const readNorm = this.device.createBuffer({
            size: this.DIM * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.normOutput, 0, readNorm, 0, this.DIM * 4);"""
new_lm_head = """    async cpuLmHead(seqLen = 1) {
        // Read the last row of normOutput from GPU
        const readNorm = this.device.createBuffer({
            size: this.DIM * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const enc = this.device.createCommandEncoder();
        const offset = (seqLen - 1) * this.DIM * 4;
        enc.copyBufferToBuffer(this.normOutput, offset, readNorm, 0, this.DIM * 4);"""
code = code.replace(old_lm_head, new_lm_head)

# 4. allocateBuffers update
old_alloc = """        const BUF_STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
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
        
        this.logitsBuffer = await this.createBuffer(this.VOCAB_SIZE * 4, BUF_STORAGE);"""

new_alloc = """        const BUF_STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
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
        
        this.logitsBuffer = await this.createBuffer(this.VOCAB_SIZE * 4, BUF_STORAGE);"""
code = code.replace(old_alloc, new_alloc)

# Remove the shared uniforms from allocateBuffers, we will create them dynamically in forwardSequence
# because they depend on seqLen
old_shared_uniforms = """        // Shared uniforms
        this.addConfig = await this.uploadBuffer(new Uint32Array([this.DIM]), GPUBufferUsage.UNIFORM);
        this.siluConfig = await this.uploadBuffer(new Uint32Array([8960]), GPUBufferUsage.UNIFORM);
        const rmsnormConfigData = new ArrayBuffer(8);
        new Uint32Array(rmsnormConfigData)[0] = this.DIM;
        new Float32Array(rmsnormConfigData)[1] = 1e-6;
        this.rmsnormConfig = await this.uploadBuffer(new Uint8Array(rmsnormConfigData), GPUBufferUsage.UNIFORM);"""
code = code.replace(old_shared_uniforms, "")


# 5. getMatMulBindGroup
old_matmul_bg = """    getMatMulBindGroup(input, weightObj, biasObj, output, inDim, outDim) {
        const configBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(configBuf, 0, new Uint32Array([inDim, outDim]));"""
new_matmul_bg = """    getMatMulBindGroup(input, weightObj, biasObj, output, inDim, outDim, seqLen = 1) {
        const configBuf = this.device.createBuffer({ size: 12, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(configBuf, 0, new Uint32Array([inDim, outDim, seqLen]));"""
code = code.replace(old_matmul_bg, new_matmul_bg)

# 6. Replace forward with forwardSequence
# It's better to replace the entire function.
# I'll just write the entire function using string replace.

import sys
with open('public/js/swanMath.js', 'w') as f:
    f.write(code)
print("Applied initial transformations.")
