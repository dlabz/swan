import re

with open('public/js/swanMath.js', 'r') as f:
    code = f.read()

old_matmul = """    getMatMulBindGroup(input, weightObj, biasObj, output, inDim, outDim, seqLen = 1) {
        const configBuf = this.device.createBuffer({ size: 12, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(configBuf, 0, new Uint32Array([inDim, outDim, seqLen]));
        
        const scalesBuf = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(scalesBuf, 0, new Float32Array([weightObj.scale, biasObj ? biasObj.scale : 0.0]));
        
        const hasBiasBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(hasBiasBuf, 0, new Uint32Array([biasObj ? 1 : 0]));"""

new_matmul = """    getMatMulBindGroup(input, weightObj, biasObj, output, inDim, outDim, transientBuffers, seqLen = 1) {
        const tempBuf = (data) => {
            const buf = this.device.createBuffer({
                size: data.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(buf, 0, data);
            if (transientBuffers) transientBuffers.push(buf);
            return buf;
        };
        const configBuf = tempBuf(new Uint32Array([inDim, outDim, seqLen]));
        const scalesBuf = tempBuf(new Float32Array([weightObj.scale, biasObj ? biasObj.scale : 0.0]));
        const hasBiasBuf = tempBuf(new Uint32Array([biasObj ? 1 : 0]));"""

code = code.replace(old_matmul, new_matmul)

# Now find all calls to getMatMulBindGroup and add transientBuffers
code = code.replace(
    "this.qBuffer, 1536, 1536, seqLen",
    "this.qBuffer, 1536, 1536, transientBuffers, seqLen"
).replace(
    "this.kBuffer, 1536, 256, seqLen",
    "this.kBuffer, 1536, 256, transientBuffers, seqLen"
).replace(
    "this.vBuffer, 1536, 256, seqLen",
    "this.vBuffer, 1536, 256, transientBuffers, seqLen"
).replace(
    "this.downBuffer, 1536, 1536, seqLen",
    "this.downBuffer, 1536, 1536, transientBuffers, seqLen"
).replace(
    "this.gateBuffer, 1536, 8960, seqLen",
    "this.gateBuffer, 1536, 8960, transientBuffers, seqLen"
).replace(
    "this.upBuffer, 1536, 8960, seqLen",
    "this.upBuffer, 1536, 8960, transientBuffers, seqLen"
).replace(
    "this.downBuffer, 8960, 1536, seqLen",
    "this.downBuffer, 8960, 1536, transientBuffers, seqLen"
).replace(
    "this.logitsBuffer, 1536, this.VOCAB_SIZE, 1",
    "this.logitsBuffer, 1536, this.VOCAB_SIZE, transientBuffers, 1"
)

with open('public/js/swanMath.js', 'w') as f:
    f.write(code)

print("Patched MatMul leak")
