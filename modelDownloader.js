const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const { pipeline } = require('stream/promises');
const { crc32 } = require('crc'); // You may need 'npm install crc'

const F32_DIR = path.resolve(__dirname, 'data', 'model', 'f32');
const META_DIR = path.resolve(__dirname, 'data', 'model', 'meta');

async function downloadAndIndexModel(url, modelName, expectedCrc) {
    const dest = path.join(F32_DIR, `${modelName}.raw`);
    
    // 1. Cache Check
    try {
        const stats = await fsPromises.stat(dest);
        if (stats.size > 0) {
            console.log(`Cache hit for ${modelName}, verifying integrity...`);
            return { rawPath: dest, cached: true };
        }
    } catch {}

    // 2. Download
    console.log(`Downloading ${modelName} from ${url}...`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    // Stream and compute CRC32 simultaneously
    const writeStream = fs.createWriteStream(dest);
    let checksum = 0;
    
    // Custom transform stream for CRC32 calculation
    const crcStream = new TransformStream({
        transform(chunk, controller) {
            checksum = crc32(chunk, checksum);
            controller.enqueue(chunk);
        }
    });

    await pipeline(response.body, crcStream, writeStream);

    // 3. Verify
    const hexCrc = (checksum >>> 0).toString(16);
    console.log(`Download complete. CRC32: ${hexCrc}`);
    
    if (expectedCrc && hexCrc !== expectedCrc) {
        await fsPromises.unlink(dest);
        throw new Error(`Integrity mismatch! Expected ${expectedCrc}, got ${hexCrc}`);
    }

    return { rawPath: dest, cached: false };
}

module.exports = { downloadAndIndexModel };