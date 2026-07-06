import re

with open('public/js/swanMath.js', 'r') as f:
    code = f.read()

old_tempBuf = """        const tempBuf = (data, usage) => {
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
        };"""

new_tempBuf = """        const tempBuf = (data, usage) => {
            if (!(data instanceof Uint8Array) && data instanceof ArrayBuffer) {
                data = new Uint8Array(data);
            } else if (!(data instanceof Uint8Array)) {
                data = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            }
            const buf = this.device.createBuffer({
                size: data.byteLength,
                usage: usage | GPUBufferUsage.COPY_DST,
            });
            this.device.queue.writeBuffer(buf, 0, data);
            transientBuffers.push(buf);
            return buf;
        };"""

code = code.replace(old_tempBuf, new_tempBuf)

with open('public/js/swanMath.js', 'w') as f:
    f.write(code)

print("Patched tempBuf")
