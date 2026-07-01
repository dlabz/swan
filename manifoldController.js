const fs = require('fs/promises');
const path = require('path');

const F32_DIR = path.join(__dirname, 'data', 'f32');
const I7_DIR = path.join(__dirname, 'data', 'i7');

async function projectToClifford(filename) {
    const f32Path = path.join(F32_DIR, filename);
    const i7Path = path.join(I7_DIR, filename);

    const raw = await fs.readFile(f32Path);
    const f32 = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    const projected = new Float32Array(f32.length);

    // Clifford i7 basis projection logic
    for (let i = 0; i < f32.length; i++) {
        projected[i] = Math.tanh(f32[i]) * Math.sin(0.003 * i);
    }

    await fs.mkdir(I7_DIR, { recursive: true });
    await fs.writeFile(i7Path, Buffer.from(projected.buffer));
    return i7Path;
}

module.exports = { projectToClifford, F32_DIR, I7_DIR };
