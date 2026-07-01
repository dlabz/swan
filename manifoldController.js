const fs = require('fs/promises');
const path = require('path');

const I7_DIR = path.resolve(__dirname, 'data', 'model', 'i7');

// Qwen2.5-1.5B Architecture: 28 layers, hidden_size 1536
const H = 1536;
const I = 8960; // Intermediate size

const MANIFEST = [
    { name: 'model.embed_tokens.bin', rows: 151936, cols: H, type: 'embed' },
    ...Array.from({ length: 28 }, (_, i) => [
        { name: `model.layers.${i}.self_attn.q_proj.bin`, rows: H, cols: H, type: 'weight' },
        { name: `model.layers.${i}.self_attn.k_proj.bin`, rows: H, cols: H, type: 'weight' },
        { name: `model.layers.${i}.self_attn.v_proj.bin`, rows: H, cols: H, type: 'weight' },
        { name: `model.layers.${i}.self_attn.o_proj.bin`, rows: H, cols: H, type: 'weight' },
        { name: `model.layers.${i}.mlp.gate_proj.bin`, rows: I, cols: H, type: 'weight' },
        { name: `model.layers.${i}.mlp.up_proj.bin`, rows: I, cols: H, type: 'weight' },
        { name: `model.layers.${i}.mlp.down_proj.bin`, rows: H, cols: I, type: 'weight' }
    ]).flat(),
    { name: 'lm_head.bin', rows: 32000, cols: H, type: 'lm_head' }
];

async function processAllLayers() {
    for (const layer of MANIFEST) {
        try {
            console.log(`Processing ${layer.name}...`);
            if (layer.type === 'embed') await projectEmbedding(layer.name, layer.rows, layer.cols);
            else if (layer.type === 'lm_head') await projectLmHead(layer.name, layer.rows, layer.cols);
            else await projectWeights(layer.name, layer.rows, layer.cols);
        } catch (err) {
            console.error(`Error processing ${layer.name}:`, err.message);
        }
    }
}

async function projectLmHead(filename, rows, cols) {
    const rawPath = path.resolve(I7_DIR, filename);
    const targetPath = path.resolve(I7_DIR, `projected_${filename}`);
    const buffer = await fs.readFile(rawPath);
    const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const projected = new Float32Array(data.length);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const target = c * rows + r;
            projected[target] = Math.tanh(data[idx]) * Math.sin(0.003 * target);
        }
    }
    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    return targetPath;
}

async function projectEmbedding(filename, rows, cols) {
    const rawPath = path.resolve(I7_DIR, filename);
    const targetPath = path.resolve(I7_DIR, `projected_${filename}`);
    const buffer = await fs.readFile(rawPath);
    const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const projected = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
        projected[i] = Math.tanh(data[i]) * Math.sin(0.003 * i);
    }
    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    return targetPath;
}

async function projectWeights(filename, rows, cols) {
    const rawPath = path.resolve(I7_DIR, filename);
    const targetPath = path.resolve(I7_DIR, `projected_${filename}`);
    const buffer = await fs.readFile(rawPath);
    const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const projected = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
        projected[i] = Math.tanh(data[i]) * Math.sin(0.003 * i);
    }
    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    return targetPath;
}

async function projectSoftmax(filename, rows, cols) {
    const rawPath = path.resolve(I7_DIR, filename);
    const targetPath = path.resolve(I7_DIR, `projected_${filename}`);
    const buffer = await fs.readFile(rawPath);
    const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    const projected = new Float32Array(data.length);
    for (let r = 0; r < rows; r++) {
        let rowMax = -Infinity;
        const rowOffset = r * cols;
        for (let c = 0; c < cols; c++) rowMax = Math.max(rowMax, data[rowOffset + c]);
        let sum = 0;
        for (let c = 0; c < cols; c++) {
            const exp = Math.exp(data[rowOffset + c] - rowMax);
            projected[rowOffset + c] = exp;
            sum += exp;
        }
        for (let c = 0; c < cols; c++) projected[rowOffset + c] /= sum;
    }
    await fs.writeFile(targetPath, Buffer.from(projected.buffer));
    return targetPath;
}

module.exports = { 
    projectWeights, 
    projectLmHead, 
    projectEmbedding, 
    projectSoftmax,
    processAllLayers,
    I7_DIR 
};
