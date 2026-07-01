const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');

const I7_DIR = path.resolve(__dirname, 'data/model/i7');
const META_DIR = path.resolve(__dirname, 'data/model/meta');

/**
 * Projects monolith safetensors into granular I7 manifold shards.
 * Uses strict offset arithmetic to prevent BFloat16/Float32 alignment drift.
 */
async function projectAllLayers() {
    await fsPromises.mkdir(I7_DIR, { recursive: true });
    
    // Get list of cached metadata files
    const metaFiles = await fsPromises.readdir(META_DIR);

    for (const metaFile of metaFiles) {
        const tensorName = metaFile.replace('.json', '');
        const safetensorPath = path.resolve(__dirname, 'data/model/f32', metaFile.replace('.json', ''));
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

                // 4. Extract exact slice
                const buffer = Buffer.alloc(byteLength);
                await fileHandle.read(buffer, 0, byteLength, dataStart);
                
                // Write to individual shard
                await fsPromises.writeFile(path.join(I7_DIR, `${key}.bin`), buffer);
            }

            await fileHandle.close();
            console.log(`[Manifold] Projected shards for ${tensorName}`);
        } catch (e) {
            console.error(`[Manifold] Failed projection for ${tensorName}: ${e.message}`);
        }
    }
}

module.exports = { projectAllLayers, I7_DIR };