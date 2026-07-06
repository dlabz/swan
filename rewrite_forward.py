import re

with open('public/js/swanMath.js', 'r') as f:
    lines = f.readlines()

forward_start = -1
for i, line in enumerate(lines):
    if "async forward(token, pos) {" in line:
        forward_start = i
        break

if forward_start != -1:
    lines = lines[:forward_start]

forward_code = """
    async forwardSequence(tokenIds, startPos) {
        if (!Array.isArray(tokenIds)) tokenIds = [tokenIds];
        const seqLen = tokenIds.length;
        
        this.device.pushErrorScope('validation');
        this.device.pushErrorScope('out-of-memory');

        const encoder = this.device.createCommandEncoder();
        const transientBuffers = [];
        const tempBuf = (data, usage) => {
            if (!(data instanceof Uint8Array) && data instanceof ArrayBuffer) {
                data = new Uint8Array(data);
            } else if (!(data instanceof Uint8Array)) {
                data = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            }
            const buf = this.device.createBuffer({
                size: data.byteLength,
                usage,
                mappedAtCreation: true
            });
            new Uint8Array(buf.getMappedRange()).set(data);
            buf.unmap();
            transientBuffers.push(buf);
            return buf;
        };
        
        // Dynamic shared uniforms based on seqLen
        const addConfig = tempBuf(new Uint32Array([this.DIM, seqLen]), GPUBufferUsage.UNIFORM);
        const siluConfig = tempBuf(new Uint32Array([8960, seqLen]), GPUBufferUsage.UNIFORM);
        const rmsnormConfigData = new ArrayBuffer(12);
        new Uint32Array(rmsnormConfigData)[0] = this.DIM;
        new Float32Array(rmsnormConfigData)[1] = 1e-6;
        new Uint32Array(rmsnormConfigData)[2] = seqLen;
        const rmsnormConfig = tempBuf(new Uint8Array(rmsnormConfigData), GPUBufferUsage.UNIFORM);

        // 1. Embed Lookup
        if (this.useCpuFallback) {
            await this.cpuEmbedLookup(tokenIds);
            encoder.copyBufferToBuffer(this.hiddenState, 0, this.residual, 0, seqLen * this.DIM * 4);
        } else {
            // Upload tokenIds array
            this.device.queue.writeBuffer(this.tokensInputBuffer, 0, new Uint32Array(tokenIds));
            
            const embedConfig = tempBuf(new Uint32Array([this.DIM, seqLen]), GPUBufferUsage.UNIFORM);
            const embedScale = tempBuf(new Float32Array([this.weights['model.embed_tokens.weight'].scale]), GPUBufferUsage.UNIFORM);
            const bgEmbed = this.device.createBindGroup({
                layout: this.pipelines.embedLookup.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.weights['model.embed_tokens.weight'].buffer } },
                    { binding: 1, resource: { buffer: this.hiddenState } },
                    { binding: 2, resource: { buffer: this.tokensInputBuffer } },
                    { binding: 3, resource: { buffer: embedConfig } },
                    { binding: 4, resource: { buffer: embedScale } }
                ]
            });
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
                    { binding: 3, resource: { buffer: rmsnormConfig } },
                    { binding: 4, resource: { buffer: wNormScale } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.rmsnorm);
            pass.setBindGroup(0, bgNorm1);
            pass.dispatchWorkgroups(Math.ceil(this.DIM / 256), seqLen);
            pass.end();

            const bgQ = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.self_attn.q_proj.weight`],
                this.weights[`model.layers.${l}.self_attn.q_proj.bias`],
                this.qBuffer, 1536, 1536, seqLen
            );
            const bgK = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.self_attn.k_proj.weight`],
                this.weights[`model.layers.${l}.self_attn.k_proj.bias`],
                this.kBuffer, 1536, 256, seqLen
            );
            const bgV = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.self_attn.v_proj.weight`],
                this.weights[`model.layers.${l}.self_attn.v_proj.bias`],
                this.vBuffer, 1536, 256, seqLen
            );
            
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgQ);
            pass.dispatchWorkgroups(Math.ceil(1536 / 64), seqLen);
            pass.setBindGroup(0, bgK);
            pass.dispatchWorkgroups(Math.ceil(256 / 64), seqLen);
            pass.setBindGroup(0, bgV);
            pass.dispatchWorkgroups(Math.ceil(256 / 64), seqLen);
            pass.end();

            const ropeConfigData = new ArrayBuffer(20);
            new Uint32Array(ropeConfigData)[0] = 1536;      
            new Uint32Array(ropeConfigData)[1] = 128;       
            new Uint32Array(ropeConfigData)[2] = startPos;       
            new Float32Array(ropeConfigData)[3] = 1000000.0; 
            new Uint32Array(ropeConfigData)[4] = seqLen; 
            const ropeConfig = tempBuf(new Uint8Array(ropeConfigData), GPUBufferUsage.UNIFORM);
            
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
            pass.dispatchWorkgroups(Math.ceil(1536 / 128), seqLen);
            pass.end();
            
            // WRITE CACHE
            const writeCacheConfig = tempBuf(new Uint32Array([256, startPos, seqLen, this.maxPrefillLen]), GPUBufferUsage.UNIFORM);
            const bgWriteK = this.device.createBindGroup({
                layout: this.pipelines.write_cache.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.kBuffer } },
                    { binding: 1, resource: { buffer: this.kCaches[l] } },
                    { binding: 2, resource: { buffer: writeCacheConfig } }
                ]
            });
            const bgWriteV = this.device.createBindGroup({
                layout: this.pipelines.write_cache.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.vBuffer } },
                    { binding: 1, resource: { buffer: this.vCaches[l] } },
                    { binding: 2, resource: { buffer: writeCacheConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.write_cache);
            pass.setBindGroup(0, bgWriteK);
            pass.dispatchWorkgroups(Math.ceil(256 / 64), seqLen);
            pass.setBindGroup(0, bgWriteV);
            pass.dispatchWorkgroups(Math.ceil(256 / 64), seqLen);
            pass.end();

            // ATTENTION
            const attnConfig = tempBuf(new Uint32Array([12, 2, 128, startPos, seqLen]), GPUBufferUsage.UNIFORM);
            const bgAttn = this.device.createBindGroup({
                layout: this.pipelines.attention.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.qBuffer } },
                    { binding: 1, resource: { buffer: this.kCaches[l] } },
                    { binding: 2, resource: { buffer: this.vCaches[l] } },
                    { binding: 3, resource: { buffer: this.attnOutput } },
                    { binding: 4, resource: { buffer: attnConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.attention);
            pass.setBindGroup(0, bgAttn);
            pass.dispatchWorkgroups(12, seqLen); 
            pass.end();

            const bgO = this.getMatMulBindGroup(
                this.attnOutput,
                this.weights[`model.layers.${l}.self_attn.o_proj.weight`],
                null,
                this.downBuffer, 1536, 1536, seqLen
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgO);
            pass.dispatchWorkgroups(Math.ceil(1536 / 64), seqLen);
            pass.end();

            const bgAdd1 = this.device.createBindGroup({
                layout: this.pipelines.add.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.residual } },
                    { binding: 1, resource: { buffer: this.downBuffer } },
                    { binding: 2, resource: { buffer: addConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.add);
            pass.setBindGroup(0, bgAdd1);
            pass.dispatchWorkgroups(Math.ceil((this.DIM * seqLen) / 256));
            pass.end();
            encoder.copyBufferToBuffer(this.residual, 0, this.hiddenState, 0, seqLen * this.DIM * 4);

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
                    { binding: 3, resource: { buffer: rmsnormConfig } },
                    { binding: 4, resource: { buffer: wNorm2Scale } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.rmsnorm);
            pass.setBindGroup(0, bgNorm2);
            pass.dispatchWorkgroups(Math.ceil(this.DIM / 256), seqLen);
            pass.end();

            const bgGate = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.mlp.gate_proj.weight`],
                null,
                this.gateBuffer, 1536, 8960, seqLen
            );
            const bgUp = this.getMatMulBindGroup(
                this.normOutput,
                this.weights[`model.layers.${l}.mlp.up_proj.weight`],
                null,
                this.upBuffer, 1536, 8960, seqLen
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgGate);
            pass.dispatchWorkgroups(Math.ceil(8960 / 64), seqLen);
            pass.setBindGroup(0, bgUp);
            pass.dispatchWorkgroups(Math.ceil(8960 / 64), seqLen);
            pass.end();

            const bgSilu = this.device.createBindGroup({
                layout: this.pipelines.silu_mul.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.gateBuffer } },
                    { binding: 1, resource: { buffer: this.upBuffer } },
                    { binding: 2, resource: { buffer: siluConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.silu_mul);
            pass.setBindGroup(0, bgSilu);
            pass.dispatchWorkgroups(Math.ceil((8960 * seqLen) / 256));
            pass.end();

            const bgDown = this.getMatMulBindGroup(
                this.gateBuffer,
                this.weights[`model.layers.${l}.mlp.down_proj.weight`],
                null,
                this.downBuffer, 8960, 1536, seqLen
            );
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.matmul);
            pass.setBindGroup(0, bgDown);
            pass.dispatchWorkgroups(Math.ceil(1536 / 64), seqLen);
            pass.end();

            const bgAdd2 = this.device.createBindGroup({
                layout: this.pipelines.add.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.residual } },
                    { binding: 1, resource: { buffer: this.downBuffer } },
                    { binding: 2, resource: { buffer: addConfig } }
                ]
            });
            pass = encoder.beginComputePass();
            pass.setPipeline(this.pipelines.add);
            pass.setBindGroup(0, bgAdd2);
            pass.dispatchWorkgroups(Math.ceil((this.DIM * seqLen) / 256));
            pass.end();
            encoder.copyBufferToBuffer(this.residual, 0, this.hiddenState, 0, seqLen * this.DIM * 4);
        }

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
                { binding: 3, resource: { buffer: rmsnormConfig } },
                { binding: 4, resource: { buffer: wNormFinalScale } }
            ]
        });
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.rmsnorm);
        pass.setBindGroup(0, bgNormFinal);
        pass.dispatchWorkgroups(Math.ceil(this.DIM / 256), seqLen);
        pass.end();

        if (this.useCpuFallback) {
            this.device.queue.submit([encoder.finish()]);
            const res = await this.cpuLmHead(seqLen);
            
            for (const buf of transientBuffers) { buf.destroy(); }
            transientBuffers.length = 0;
            
            const oomError = await this.device.popErrorScope();
            const valError = await this.device.popErrorScope();
            if (oomError) console.error('[GPU OOM]', oomError.message);
            if (valError) console.error('[GPU Validation]', valError.message);
            
            return res;
        } else {
            // Need a view over just the LAST row of normOutput to project to logits
            // We can just use normOutput, but we will adjust getMatMulBindGroup to take an offset?
            // Actually, we can copy the last row of normOutput into a 1xDIM buffer first.
            const lastRowBuf = tempBuf(new Float32Array(this.DIM), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            encoder.copyBufferToBuffer(this.normOutput, (seqLen - 1) * this.DIM * 4, lastRowBuf, 0, this.DIM * 4);

            const bgHead = this.getMatMulBindGroup(
                lastRowBuf,
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
            
            for (const buf of transientBuffers) { buf.destroy(); }
            transientBuffers.length = 0;

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
"""

with open('public/js/swanMath.js', 'w') as f:
    f.writelines(lines)
    f.write(forward_code)

print("Rewrote forwardSequence successfully.")
