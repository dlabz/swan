const fs = require('fs/promises');
const path = require('path');
const { hfHubDownload } = require('@huggingface/hub');

// Ensure this is an absolute path
const I7_DIR = path.resolve(__dirname, 'data', 'i7');

async function fetchAndProject(filename = "model.safetensors", repoId = "Qwen/Qwen2.5-Coder-1.5B-Instruct") {
    // Ensure dir exists before trying to download
    await fs.mkdir(I7_DIR, { recursive: true });

    const localFilePath = await hfHubDownload({
        repoId: repoId,
        filename: filename,
    });

    const raw = await fs.readFile(localFilePath);
    const view = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    
    const projected = new Float32Array(view.length);
    for (let i = 0; i < view.length; i++) {
        projected[i] = Math.tanh(view[i]) * Math.sin(0.003 * i);
    }

    const targetPath = path.join(I7_DIR, `projected_${filename}`);
    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    
    return targetPath;
}

module.exports = { fetchAndProject, I7_DIR };