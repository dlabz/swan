
const fs = require('fs/promises');
const path = require('path');

const I7_DIR = path.resolve(__dirname, 'data', 'model', 'i7');

// ... [Manifest and other project functions remain the same] ...

/**
 * Stabilized Softmax Projection
 * Applies exponentiation and normalization to convert raw logits to probabilities
 * on the manifold.
 */
async function projectSoftmax(filename, rows, cols) {
    const rawPath = path.resolve(I7_DIR, filename);
    const targetPath = path.resolve(I7_DIR, `projected_${filename}`);
    
    console.log(`Processing Softmax Projection: ${filename}`);
    
    const buffer = await fs.readFile(rawPath);
    const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const projected = new Float32Array(data.length);

    // Process row by row (vocab distribution)
    for (let r = 0; r < rows; r++) {
        let rowMax = -Infinity;
        const rowOffset = r * cols;
        
        // 1. Find Max (for stability)
        for (let c = 0; c < cols; c++) rowMax = Math.max(rowMax, data[rowOffset + c]);
        
        // 2. Exponentiate and Sum
        let sum = 0;
        for (let c = 0; c < cols; c++) {
            const exp = Math.exp(data[rowOffset + c] - rowMax);
            projected[rowOffset + c] = exp;
            sum += exp;
        }
        
        // 3. Normalize
        for (let c = 0; c < cols; c++) {
            projected[rowOffset + c] /= sum;
        }
    }

    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    return targetPath;
}

module.exports = { 
    projectWeights, 
    projectLmHead, 
    projectEmbedding, 
    projectSoftmax, // Exported for the manifest
    processAllLayers,
    I7_DIR 
};

