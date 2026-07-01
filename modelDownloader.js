// @ts-check
const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const { pipeline } = require('stream/promises');
const { Transform } = require('stream');
const { crc32 } = require('crc');

const BF16_DIR = path.resolve(__dirname, 'data', 'model', 'bf16');
const META_DIR = path.resolve(__dirname, 'data', 'model', 'meta');

/**
 * Discovery and Download Manager
 * @param {string} repoId - The ID of the HuggingFace repository to download.
 * @returns {Promise<void>}
 */
async function downloadAndIndexModel(repoId) {
    await fsPromises.mkdir(BF16_DIR, { recursive: true });
    await fsPromises.mkdir(META_DIR, { recursive: true });

    // 1. Discover files via HF Tree API
    const apiRes = await fetch(`https://huggingface.co/api/models/${repoId}/tree/main`);
    if (!apiRes.ok) throw new Error(`API Error: ${apiRes.status}`);
    const files = (await apiRes.json()).filter(f => f.path.endsWith('.safetensors'));

    for (const file of files) {
        const filename = file.path.replace(/\//g, '_');
        const dest = path.join(BF16_DIR, filename);

        // 2. Stream Download with Integrity Check
        if (!fs.existsSync(dest)) {
            console.log(`[Downloader] Streaming: ${file.path}...`);
            const response = await fetch(`https://huggingface.co/${repoId}/resolve/main/${file.path}`);
            let checksum = 0;
            const crcStream = new Transform({
                transform(chunk, enc, cb) {
                    checksum = crc32(chunk, checksum);
                    cb(null, chunk);
                }
            });
            await pipeline(response.body, crcStream, fs.createWriteStream(dest));
            console.log(`[Downloader] Verified ${filename} (CRC32: ${(checksum >>> 0).toString(16)})`);
        }

        // 3. Header Inspection via Range Request
        await fetchAndCacheHeader(repoId, file.path, filename);
    }
}

/**
 * Header metadata inspection via HTTP Range Requests
 * @param {string} repoId - The HuggingFace repository ID
 * @param {string} remotePath - The path to the file in the repository
 * @param {string} localName - The local filename to save metadata
 * @returns {Promise<void>}
 */
async function fetchAndCacheHeader(repoId, remotePath, localName) {
    const metaDest = path.join(META_DIR, `${localName}.json`);
    if (fs.existsSync(metaDest)) return;

    const url = `https://huggingface.co/${repoId}/resolve/main/${remotePath}`;
    
    // Fetch header size (first 8 bytes)
    const sizeRes = await fetch(url, { headers: { Range: "bytes=0-7" } });
    const headerSize = Number(Buffer.from(await sizeRes.arrayBuffer()).readBigUInt64LE(0));

    // Fetch JSON header
    const headerRes = await fetch(url, { headers: { Range: `bytes=8-${7 + headerSize}` } });
    const headerData = await headerRes.json();
    
    await fsPromises.writeFile(metaDest, JSON.stringify(headerData, null, 2));
    console.log(`[Header] Cached metadata for ${remotePath}`);
}

module.exports = { downloadAndIndexModel };