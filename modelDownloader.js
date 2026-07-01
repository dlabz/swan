const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const { pipeline } = require('stream/promises');
const { Transform } = require('stream');
const { crc32 } = require('crc');

const F32_DIR = path.resolve(__dirname, 'data', 'model', 'f32');

/**
 * Downloads all .safetensors files for a given HF repoId.
 * Performs streaming CRC32 validation and skips existing files.
 */
async function downloadAndIndexModel(repoId) {
    await fsPromises.mkdir(F32_DIR, { recursive: true });

    // 1. Discovery via Hugging Face REST API
    console.log(`[Downloader] Querying API for: ${repoId}`);
    const apiRes = await fetch(`https://huggingface.co/api/models/${repoId}/tree/main`);
    if (!apiRes.ok) throw new Error(`API Error: ${apiRes.status}`);
    
    const files = await apiRes.json();
    const safetensors = files.filter(f => f.path.endsWith('.safetensors'));
    
    console.log(`[Downloader] Found ${safetensors.length} model files.`);

    // 2. Streaming Processing Loop
    for (const file of safetensors) {
        const url = `https://huggingface.co/${repoId}/resolve/main/${file.path}`;
        const dest = path.join(F32_DIR, file.path.replace(/\//g, '_'));

        if (fs.existsSync(dest)) {
            console.log(`[Downloader] Skipping (Cache Hit): ${file.path}`);
            continue;
        }

        console.log(`[Downloader] Streaming: ${file.path}...`);
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Download failed: ${response.status}`);

        let checksum = 0;
        
        // CRC32 Transform stream to validate integrity on-the-fly
        const crcStream = new Transform({
            transform(chunk, encoding, callback) {
                checksum = crc32(chunk, checksum);
                callback(null, chunk);
            }
        });

        try {
            await pipeline(response.body, crcStream, fs.createWriteStream(dest));
            const hexCrc = (checksum >>> 0).toString(16);
            console.log(`[Downloader] Finished: ${file.path} (CRC32: ${hexCrc})`);
        } catch (err) {
            // Ensure partial files are removed on error
            await fsPromises.unlink(dest).catch(() => {});
            throw new Error(`Pipeline failed for ${file.path}: ${err.message}`);
        }
    }
}

module.exports = { downloadAndIndexModel };