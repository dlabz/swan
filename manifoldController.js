const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');

const F32_DIR = path.resolve(__dirname, 'data', 'model', 'f32');
const I7_DIR = path.resolve(__dirname, 'data', 'model', 'i7');
const META_DIR = path.resolve(__dirname, 'data', 'model', 'meta');

function normalizeName(tensorName) {
    if (tensorName === 'lm_head.weight') return 'lm_head.bin';
    if (tensorName === 'model.embed_tokens.weight') return 'embed_tokens.bin';
    return `${tensorName}.bin`;
}

function formatSize(bytes) {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

async function projectAllLayers() {
    await fsPromises.mkdir(I7_DIR, { recursive: true });
    const metaFiles = (await fsPromises.readdir(META_DIR)).filter(f => f.endsWith('.json'));

    if (metaFiles.length === 0) throw new Error("No metadata found.");

    for (const metaFile of metaFiles) {
        const safetensorPath = path.join(F32_DIR, metaFile.replace('.json', ''));
        if (!fs.existsSync(safetensorPath)) continue;

        console.log(`\n[Extractor] Opening ${metaFile.replace('.json', '')} for extraction...`);
        const fileHandle = await fsPromises.open(safetensorPath, 'r');
        
        const sizeBuffer = Buffer.alloc(8);
        await fileHandle.read(sizeBuffer, 0, 8, 0);
        const dataOffsetBase = 8 + Number(sizeBuffer.readBigUInt64LE(0));

        const metadata = JSON.parse(await fsPromises.readFile(path.join(META_DIR, metaFile), 'utf8'));
        const tensors = Object.keys(metadata).filter(k => k !== '__metadata__');

        for (const tensorName of tensors) {
            const info = metadata[tensorName];
            const byteLength = info.data_offsets[1] - info.data_offsets[0];
            const targetPath = path.join(I7_DIR, `projected_${normalizeName(tensorName)}`);

            if (fs.existsSync(targetPath)) {
                console.log(`[Extractor] Skipped: ${normalizeName(tensorName)}`);
                continue;
            }

            console.log(`[Extractor] Processing: ${normalizeName(tensorName)} (${formatSize(byteLength)})`);

            // NO DUMMY MATH: We stream the raw bfloat16 bytes directly to disk
            const CHUNK_SIZE = 16 * 1024 * 1024; 
            const writeStream = fs.createWriteStream(targetPath);
            let bytesProcessed = 0;

            while (bytesProcessed < byteLength) {
                const currentChunkSize = Math.min(CHUNK_SIZE, byteLength - bytesProcessed);
                const buffer = Buffer.alloc(currentChunkSize);
                
                await fileHandle.read(buffer, 0, currentChunkSize, dataOffsetBase + info.data_offsets[0] + bytesProcessed);
                writeStream.write(buffer);
                
                bytesProcessed += currentChunkSize;
            }

            await new Promise(res => writeStream.end(res));
        }
        await fileHandle.close();
    }
    console.log(`\n[Extractor] All weights successfully extracted to pure BF16 format.`);
}

module.exports = { projectAllLayers, I7_DIR };