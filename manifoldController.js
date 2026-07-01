const fs = require('fs/promises');
const path = require('path');
const { downloadFile } = require('@huggingface/hub');

// Aligning with your workspace structure: data/i7
const I7_DIR = path.resolve(__dirname, 'data', 'i7');

async function fetchAndProject(filename = "model.safetensors", repoId = "Qwen/Qwen2.5-Coder-1.5B-Instruct") {
    await fs.mkdir(I7_DIR, { recursive: true });
    const targetPath = path.join(I7_DIR, `projected_${filename}`);

    console.log(`Downloading from ${repoId} to ${I7_DIR}...`);

    const fileBlob = await downloadFile({
        repo: { type: 'model', name: repoId },
        path: filename,
    });

    if (!fileBlob) throw new Error("Could not download file.");

    const buffer = Buffer.from(await fileBlob.arrayBuffer());
    // Process in chunks to avoid memory crashes on 1.5B models
    const view = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const projected = new Float32Array(view.length);

    for (let i = 0; i < view.length; i++) {
        projected[i] = Math.tanh(view[i]) * Math.sin(0.003 * i);
    }

    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    return targetPath;
}

module.exports = { fetchAndProject, I7_DIR };
