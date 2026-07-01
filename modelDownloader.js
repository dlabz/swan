const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const { pipeline } = require('stream/promises');
const { list } = require('@huggingface/hub'); // Correct import
const { crc32 } = require('crc');

const F32_DIR = path.resolve(__dirname, 'data', 'model', 'f32');

async function downloadAndIndexModel(repoId) {
    await fsPromises.mkdir(F32_DIR, { recursive: true });

    // 1. Fetch file list (using the named export 'list')
    console.log(`Querying Hugging Face for ${repoId}...`);
    const files = await list({ repo: repoId });
    
    // Filter for model files only
    const modelFiles = files.filter(f => f.path.endsWith('.safetensors'));
    
    console.log(`Found ${modelFiles.length} files. Starting download...`);

    for (const file of modelFiles) {
        const url = `https://huggingface.co/${repoId}/resolve/main/${file.path}`;
        const dest = path.join(F32_DIR, file.path.replace(/\//g, '_'));

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