const fs = require('fs/promises');
const path = require('path');
// Import specific function to ensure API compatibility
const { hfHubDownload } = require('@huggingface/hub');

const I7_DIR = path.join(__dirname, 'data', 'i7');

async function fetchAndProject(filename = "model.safetensors", repoId = "Qwen/Qwen2.5-Coder-1.5B-Instruct") {
    console.log(`Downloading ${filename} from ${repoId}...`);

    // Download the specific weight file from the Qwen repo
    const localFilePath = await hfHubDownload({
        repoId: repoId,
        filename: filename,
    });

    // Perform projection on the safetensors file
    // Note: For actual model tensors, you'd normally use a library like 
    // @huggingface/transformers or torch. This performs a simulated projection.
    const raw = await fs.readFile(localFilePath);
    const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    const view = new Float32Array(buffer);
    
    const projected = new Float32Array(view.length);
    for (let i = 0; i < view.length; i++) {
        projected[i] = Math.tanh(view[i]) * Math.sin(0.003 * i);
    }

    const targetPath = path.join(I7_DIR, `projected_${filename}`);
    await fs.mkdir(I7_DIR, { recursive: true });
    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    
    return targetPath;
}

module.exports = { fetchAndProject };