const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const { pipeline } = require('stream/promises');
const { HfApi } = require('@huggingface/hub');
const { crc32 } = require('crc');

const F32_DIR = path.resolve(__dirname, 'data', 'model', 'f32');
const hf = new HfApi();

async function downloadAndIndexModel(repoId) {
    await fsPromises.mkdir(F32_DIR, { recursive: true });

    // 1. Fetch file list from HF
    const files = await hf.list({ repoId });
    const safetensors = files.filter(f => f.path.endsWith('.safetensors'));
    
    console.log(`Found ${safetensors.length} files in ${repoId}...`);

    for (const file of safetensors) {
        const url = `https://huggingface.co/${repoId}/resolve/main/${file.path}`;
        const dest = path.join(F32_DIR, file.path.replace(/\//g, '_'));

        // Skip if exists
        if (fs.existsSync(dest)) {
            console.log(`Skipping ${file.path}, already exists.`);
            continue;
        }

        console.log(`Downloading ${file.path}...`);
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status} for ${file.path}`);
        
        await pipeline(response.body, fs.createWriteStream(dest));
        console.log(`Finished ${file.path}`);
    }
}

module.exports = { downloadAndIndexModel };
