const fs = require('fs/promises');
const path = require('path');

// Update this to match your actual nested structure
const I7_DIR = path.resolve(__dirname, 'data', 'i7', 'model', 'i7');

async function fetchAndProject(filename = "config.json", repoId = "Qwen/Qwen2.5-Coder-1.5B-Instruct") {
    await fs.mkdir(I7_DIR, { recursive: true });
    const targetPath = path.join(I7_DIR, `projected_${filename}`);

    const url = `https://huggingface.co/${repoId}/resolve/main/${filename}`;
    console.log(`Fetching tiny config from: ${url}`);

    const response = await fetch(url);
    if (!response.ok) throw new Error(`Download failed: ${response.statusText}`);

    // Processing as a string/buffer since it's config.json
    const buffer = Buffer.from(await response.arrayBuffer());
    await fs.writeFile(targetPath, buffer);
    
    console.log("Pipeline success!");
    return targetPath;
}

module.exports = { fetchAndProject, I7_DIR };