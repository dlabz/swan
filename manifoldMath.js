// @ts-check

/**
 * Manifold Math operations for BFloat16 <-> I7 conversion.
 */

const LUT_SIZE = 256;

// Precomputed Math Shortcuts Lookup Table
const I7_TO_FLOAT32_LUT = new Float32Array(LUT_SIZE);
for (let i = 0; i < LUT_SIZE; i++) {
    // Cast index (0-255) to signed 8-bit (-128 to 127)
    I7_TO_FLOAT32_LUT[i] = (i << 24 >> 24);
}

/**
 * Optimized BF16 to Float32 conversion.
 */
function bf16ToFloat32Array(bf16Buffer) {
    const numElements = bf16Buffer.length >> 1;
    const f32Array = new Float32Array(numElements);
    const view = new DataView(bf16Buffer.buffer, bf16Buffer.byteOffset, bf16Buffer.byteLength);

    for (let i = 0; i < numElements; i++) {
        // Shift left by 16 to move 16-bit BF16 to upper 16 bits of F32
        f32Array[i] = view.getUint16(i * 2, true) << 16;
    }
    return f32Array;
}

/**
 * Projects a Float32Array into the I7 Manifold format.
 * Structure: 4-byte Float32 LE scale factor + N-byte Int8 data.
 * @param {Float32Array} f32Array 
 * @returns {Buffer}
 */
function quantizeToI7(f32Array) {
    let maxAbs = 0;
    for (let i = 0; i < f32Array.length; i++) {
        const absVal = Math.abs(f32Array[i]);
        if (absVal > maxAbs) maxAbs = absVal;
    }

    const scale = maxAbs === 0 ? 1 : maxAbs / 127;
    const invScale = maxAbs === 0 ? 1 : 127 / maxAbs;

    const i7Buffer = Buffer.alloc(4 + f32Array.length);
    i7Buffer.writeFloatLE(scale, 0);

    for (let i = 0; i < f32Array.length; i++) {
        let quantized = Math.round(f32Array[i] * invScale);
        // Clamp to [-127, 127]
        if (quantized > 127) quantized = 127;
        else if (quantized < -127) quantized = -127;
        
        i7Buffer.writeInt8(quantized, 4 + i);
    }
    return i7Buffer;
}

/**
 * Optimized Dequantization.
 */
function dequantizeFromI7(i7Buffer) {
    const scale = i7Buffer.readFloatLE(0);
    const numElements = i7Buffer.length - 4;
    
    const i8View = new Int8Array(i7Buffer.buffer, i7Buffer.byteOffset + 4, numElements);
    const f32Array = new Float32Array(numElements);
    
    for (let i = 0; i < numElements; i++) {
        // Offset by 128 to use as LUT index (0-255)
        f32Array[i] = I7_TO_FLOAT32_LUT[i8View[i] + 128] * scale;
    }
    return f32Array;
}

module.exports = {
    I7_TO_FLOAT32_LUT,
    bf16ToFloat32Array,
    quantizeToI7,
    dequantizeFromI7
};