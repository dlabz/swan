const fs = require('fs/promises');
const path = require('path');

const I7_DIR = path.resolve(__dirname, 'data', 'i7');

async function fetchAndProject(filename = "model.safetensors", repoId = "Qwen/Qwen2.5-Coder-1.5B-Instruct") {
    await fs.mkdir(I7_DIR, { recursive: true });
    const targetPath = path.join(I7_DIR, `projected_${filename}`);

    // Official Hugging Face Resolve URL pattern
    const url = `https://huggingface.co/${repoId}/resolve/main/${filename}`;
    console.log(`Fetching from: ${url}`);

    const response = await fetch(url);
    if (!response.ok) throw new Error(`Download failed: ${response.statusText}`);

    const arrayBuffer = await response.arrayBuffer();
    const view = new Float32Array(arrayBuffer, 0, arrayBuffer.byteLength / 4);
    
    const projected = new Float32Array(view.length);
    for (let i = 0; i < view.length; i++) {
        projected[i] = Math.tanh(view[i]) * Math.sin(0.003 * i);
    }

    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    console.log("Projection complete and saved.");
    return targetPath;
}

module.exports = { fetchAndProject, I7_DIR };