const fs = require('fs/promises');
const path = require('path');
const { downloadFile } = require('@huggingface/hub');

const I7_DIR = path.resolve(__dirname, 'data', 'i7');

async function fetchAndProject(filename = "model.safetensors", repoId = "Qwen/Qwen2.5-Coder-1.5B-Instruct") {
    await fs.mkdir(I7_DIR, { recursive: true });
    const targetPath = path.join(I7_DIR, `projected_${filename}`);

    console.log(`Downloading ${filename} from ${repoId}...`);

    // 1. Correct API usage for @huggingface/hub
    const fileBlob = await downloadFile({
        repo: { type: 'model', name: repoId },
        path: filename,
    });

    if (!fileBlob) throw new Error("Could not download file from Hugging Face.");

    // 2. Memory-safe processing
    const buffer = Buffer.from(await fileBlob.arrayBuffer());
    const view = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    
    const projected = new Float32Array(view.length);
    for (let i = 0; i < view.length; i++) {
        projected[i] = Math.tanh(view[i]) * Math.sin(0.003 * i);
    }

    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    return targetPath;
}

module.exports = { fetchAndProject, I7_DIR };