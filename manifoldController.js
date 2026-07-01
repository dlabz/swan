const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');

const F32_DIR = path.resolve(__dirname, 'data', 'model', 'f32');
const META_DIR = path.resolve(__dirname, 'data', 'model', 'meta');
const I7_DIR = path.resolve(__dirname, 'data', 'model', 'i7');

/**
 * Normalizes Hugging Face tensor names to your frontend's expected format.
 * E.g., "lm_head.weight" -> "lm_head.bin"
 */
function normalizeName(tensorName) {
    if (tensorName === 'lm_head.weight') return 'lm_head.bin';
    if (tensorName === 'model.embed_tokens.weight') return 'embed_tokens.bin';
    return `${tensorName}.bin`;
}

/**
 * Projects all tensors dynamically using the JSON metadata index.
 * Uses random-access file reading to maintain low memory usage.
 */
async function projectAllLayers() {
    await fsPromises.mkdir(I7_DIR, { recursive: true });

    // 1. Get all cached metadata files
    const metaFiles = (await fsPromises.readdir(META_DIR)).filter(f => f.endsWith('.json'));

    if (metaFiles.length === 0) {
        throw new Error("No metadata found. Please run the downloader first.");
    }

    for (const metaFile of metaFiles) {
        const safetensorFilename = metaFile.replace('.json', '');
        const safetensorPath = path.join(F32_DIR, safetensorFilename);
        const metaPath = path.join(META_DIR, metaFile);

        if (!fs.existsSync(safetensorPath)) {
            console.warn(`[Projector] Missing raw file for metadata: ${safetensorFilename}`);
            continue;
        }

        console.log(`\n[Projector] Opening ${safetensorFilename} for projection...`);

        // 2. Open file for random access
        const fileHandle = await fsPromises.open(safetensorPath, 'r');
        
        // Read header size to calculate absolute offsets
        const sizeBuffer = Buffer.alloc(8);
        await fileHandle.read(sizeBuffer, 0, 8, 0);
        const headerSize = Number(sizeBuffer.readBigUInt64LE(0));
        const dataOffsetBase = 8 + headerSize;

        // 3. Parse Metadata
        const metadata = JSON.parse(await fsPromises.readFile(metaPath, 'utf8'));
        const tensors = Object.keys(metadata).filter(k => k !== '__metadata__');

        console.log(`[Projector] Found ${tensors.length} tensors. Processing...`);

        // 4. Iterate and Project Tensor-by-Tensor
        for (const tensorName of tensors) {
            const tensorInfo = metadata[tensorName];
            const start = dataOffsetBase + tensorInfo.data_offsets[0];
            const end = dataOffsetBase + tensorInfo.data_offsets[1];
            const byteLength = end - start;

            const targetName = normalizeName(tensorName);
            const targetPath = path.join(I7_DIR, `projected_${targetName}`);

            // Skip if already projected
            if (fs.existsSync(targetPath)) {
                console.log(`[Projector] Skipping (Already projected): ${targetName}`);
                continue;
            }

            console.log(`[Projector] Processing: ${targetName} (${(byteLength / 1024 / 1024).toFixed(2)} MB)`);

            // Read exact byte chunk from disk
            const buffer = Buffer.alloc(byteLength);
            await fileHandle.read(buffer, 0, byteLength, start);

            // Safely align Float32 processing
            const validLength = byteLength - (byteLength % 4);
            const floats = new Float32Array(buffer.buffer, buffer.byteOffset, validLength / 4);
            const projected = new Float32Array(floats.length);

            // 5. Apply Mathematical Manifold Projection
            for (let i = 0; i < floats.length; i++) {
                projected[i] = Math.tanh(floats[i]) * Math.sin(0.003 * i);
            }

            // 6. Write to i7 storage
            await fsPromises.writeFile(targetPath, Buffer.from(projected.buffer));
        }

        await fileHandle.close();
    }

    console.log(`\n[Projector] All tensors successfully projected to ${I7_DIR}`);
}

module.exports = { projectAllLayers, I7_DIR };