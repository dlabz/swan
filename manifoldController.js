const fs = require('fs/promises');
const path = require('path');
const { hf_hub_download } = require('@huggingface/hub');

const F32_DIR = path.join(__dirname, 'data', 'f32');
const I7_DIR = path.join(__dirname, 'data', 'i7');

async function fetchAndProject(filename, repoId = "dlabz/swan") {
    // 1. Download from HF Hub
    const localF32Path = await hf_hub_download({
        repo: repoId,
        path: filename,
        cacheDir: F32_DIR
    });

    // 2. Projection Logic
    const raw = await fs.readFile(localF32Path);
    const f32 = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    const projected = new Float32Array(f32.length);

    for (let i = 0; i < f32.length; i++) {
        projected[i] = Math.tanh(f32[i]) * Math.sin(0.003 * i);
    }

    const targetPath = path.join(I7_DIR, filename);
    await fs.mkdir(I7_DIR, { recursive: true });
    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    
    return targetPath;
}

module.exports = { fetchAndProject, I7_DIR };
