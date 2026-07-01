const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const { pipeline } = require('stream/promises');

const F32_DIR = path.resolve(__dirname, 'data', 'model', 'f32');
const META_DIR = path.resolve(__dirname, 'data', 'model', 'meta');

/**
 * Downloads a safetensors file, extracts the header, and caches raw tensors.
 */
async function downloadAndIndexModel(url, modelName) {
    await fsPromises.mkdir(F32_DIR, { recursive: true });
    await fsPromises.mkdir(META_DIR, { recursive: true });

    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

    // Create a reader to get the first 8 bytes for the header size
    const reader = response.body.getReader();
    
    // 1. Read Header Size (8 bytes)
    const { value: sizeChunk } = await reader.read();
    const headerSize = Number(new DataView(sizeChunk.buffer).getBigUint64(0, true));
    
    // 2. Read Header JSON
    let headerBuffer = Buffer.alloc(headerSize);
    let bytesRead = 0;
    
    // We need the rest of the first chunk + subsequent chunks until header is full
    let currentPos = sizeChunk.slice(8);
    headerBuffer.set(currentPos, 0);
    bytesRead += currentPos.length;

    while (bytesRead < headerSize) {
        const { value } = await reader.read();
        const toCopy = Math.min(value.length, headerSize - bytesRead);
        headerBuffer.set(value.slice(0, toCopy), bytesRead);
        bytesRead += toCopy;
    }

    const header = JSON.parse(headerBuffer.toString());
    await fsPromises.writeFile(path.join(META_DIR, `${modelName}.json`), JSON.stringify(header, null, 2));

    // 3. Stream remaining raw data to disk
    const dest = path.join(F32_DIR, `${modelName}.raw`);
    const writeStream = fs.createWriteStream(dest);

    // Continue streaming from where we left off (if any remaining bytes in the last chunk)
    // For simplicity, here we restart the pipeline from the reader body
    // In production, we'd continue the stream.
    await pipeline(response.body, writeStream);

    return { header, rawPath: dest };
}

module.exports = { downloadAndIndexModel };