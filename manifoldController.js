const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const { encodeBF16To5Cell8Bit } = require('./manifoldMath');

const I7_DIR = path.resolve(__dirname, 'data/model/i7');
const META_DIR = path.resolve(__dirname, 'data/model/meta');

/**
 * Projects monolith safetensors into granular 5-Cell 8-bit manifold shards.
 * Uses 2-pass memory-bounded streaming to calculate exact tensor scaling.
 * @returns {Promise<void>}
 */
async function projectAllLayers() {
    await fsPromises.mkdir(I7_DIR, { recursive: true });
    
    // Get list of cached metadata files
    const metaFiles = await fsPromises.readdir(META_DIR);

    for (const metaFile of metaFiles) {
        const tensorName = metaFile.replace('.json', '');
        const safetensorPath = path.resolve(__dirname, 'data/model/bf16', metaFile.replace('.json', ''));
        const targetPath = path.join(I7_DIR, `${tensorName}.bin`);

        // Skip if already projected
        if (fs.existsSync(targetPath)) continue;

        try {
            const metadata = JSON.parse(await fsPromises.readFile(path.join(META_DIR, metaFile), 'utf8'));
            const fileHandle = await fsPromises.open(safetensorPath, 'r');
            
            // 1. Read the 8-byte header size
            const headerSizeBuffer = Buffer.alloc(8);
            await fileHandle.read(headerSizeBuffer, 0, 8, 0);
            const headerSize = Number(headerSizeBuffer.readBigUInt64LE(0));
            
            // 2. The data section starts EXACTLY at 8 bytes (the size field) + header size
            const dataOffsetBase = 8 + headerSize;

            // 3. Iterate tensors defined in this file
            for (const [key, info] of Object.entries(metadata)) {
                if (key === '__metadata__') continue;

                const dataStart = dataOffsetBase + info.data_offsets[0];
                const byteLength = info.data_offsets[1] - info.data_offsets[0];

                const binPath = path.join(I7_DIR, `${key}.bin`);

                // 4. Stream in aligned chunks of 6MB (divisible by 6 bytes for BF16 alignment)
                const CHUNK_SIZE = 6 * 1024 * 1024;
                
                // Pass 1: Scan chunk-by-chunk to find the absolute max value of the tensor
                let maxVal = 0;
                let bytesRead = 0;
                while (bytesRead < byteLength) {
                    const currentChunkSize = Math.min(CHUNK_SIZE, byteLength - bytesRead);
                    const chunkBuffer = Buffer.alloc(currentChunkSize);
                    await fileHandle.read(chunkBuffer, 0, currentChunkSize, dataStart + bytesRead);
                    
                    const numElements = currentChunkSize >> 1;
                    const view = new DataView(chunkBuffer.buffer, chunkBuffer.byteOffset, chunkBuffer.byteLength);
                    
                    // Small local bitcast buffers
                    const bitBuffer = new ArrayBuffer(4);
                    const bitU32 = new Uint32Array(bitBuffer);
                    const bitF32 = new Float32Array(bitBuffer);
                    
                    for (let i = 0; i < numElements; i++) {
                        const bf16Bits = view.getUint16(i * 2, true);
                        bitU32[0] = bf16Bits << 16;
                        const val = Math.abs(bitF32[0]);
                        if (val > maxVal) maxVal = val;
                    }
                    bytesRead += currentChunkSize;
                }

                // Pass 2: Stream chunk-by-chunk, scale, and write to disk
                const writeStream = fs.createWriteStream(binPath);
                
                // Write the Float32 scale factor in the first 4 bytes
                const scaleBuf = Buffer.alloc(4);
                scaleBuf.writeFloatLE(maxVal / 127.0, 0);
                writeStream.write(scaleBuf);

                bytesRead = 0;
                while (bytesRead < byteLength) {
                    const currentChunkSize = Math.min(CHUNK_SIZE, byteLength - bytesRead);
                    const chunkBuffer = Buffer.alloc(currentChunkSize);
                    await fileHandle.read(chunkBuffer, 0, currentChunkSize, dataStart + bytesRead);
                    
                    const rgbaBuffer = encodeBF16To5Cell8Bit(chunkBuffer, maxVal);
                    writeStream.write(rgbaBuffer);
                    
                    bytesRead += currentChunkSize;
                }

                writeStream.end();
                await new Promise((resolve) => writeStream.on('finish', resolve));
                
                // Free memory hints for large layers
                if (global.gc) global.gc();
            }

            await fileHandle.close();
            console.log(`[Manifold] Projected shards for ${tensorName}`);
        } catch (e) {
            const err = /** @type {any} */ (e);
            console.error(`[Manifold] Failed projection for ${tensorName}: ${err.message || String(err)}`);
        }
    }
}

module.exports = { projectAllLayers, I7_DIR };