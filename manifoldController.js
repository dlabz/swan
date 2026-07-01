const fs = require('fs/promises');
const path = require('path');

const I7_DIR = path.resolve(__dirname, 'data', 'model', 'i7');

async function downloadAndProject(filename, rows, cols, isLmHead = false) {
    if (isLmHead) {
        return await projectLmHead(filename, rows, cols);
    } else {
        return await projectEmbedding(filename, rows, cols);
    }
}

async function projectLmHead(filename, rows, cols) {
    const rawPath = path.resolve(I7_DIR, filename);
    const targetPath = path.resolve(I7_DIR, `projected_${filename}`);
    
    console.log(`Processing LM-Head (Transposed): ${filename}`);
    
    const buffer = await fs.readFile(rawPath);
    const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const projected = new Float32Array(data.length);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const originalIndex = r * cols + c;
            const targetIndex = c * rows + r; // Transpose
            
            const val = data[originalIndex];
            projected[targetIndex] = Math.tanh(val) * Math.sin(0.003 * targetIndex);
        }
    }

    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    return targetPath;
}

async function projectEmbedding(filename, rows, cols) {
    const rawPath = path.resolve(I7_DIR, filename);
    const targetPath = path.resolve(I7_DIR, `projected_${filename}`);
    
    console.log(`Processing Embedding (Standard): ${filename}`);
    
    const buffer = await fs.readFile(rawPath);
    const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const projected = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
        projected[i] = Math.tanh(data[i]) * Math.sin(0.003 * i);
    }

    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    return targetPath;
}

module.exports = { downloadAndProject, projectLmHead, projectEmbedding, I7_DIR };