// @ts-check

/**
 * Manifold Math operations for BFloat16 <-> I7 conversion.
 */

const LUT_SIZE = 256;

// Precomputed Math Shortcuts Lookup Table
const I7_TO_FLOAT32_LUT = new Float32Array(LUT_SIZE);
for (let i = 0; i < LUT_SIZE; i++) {
    // Map index (0-255) back to signed 8-bit (-128 to 127)
    // Note: 128 maps to 0, 127 maps to -1, 0 maps to -128
    const val = (i <= 127) ? (i - 128) : (i - 128); 
    // Actually, simply using (i << 24 >> 24) is a faster way to cast 
    // an unsigned byte back to a signed 8-bit integer in JS
    I7_TO_FLOAT32_LUT[i] = (i << 24 >> 24);
}

/**
 * Optimized BF16 to Float32 conversion.
 */
function bf16ToFloat32Array(bf16Buffer) {
    const numElements = bf16Buffer.length >> 1;
    const f32Array = new Float32Array(numElements);
    // Use a single DataView for the whole buffer to improve performance
    const view = new DataView(bf16Buffer.buffer, bf16Buffer.byteOffset, bf16Buffer.byteLength);

    for (let i = 0; i < numElements; i++) {
        // Shift left by 16 to move 16-bit BF16 to upper 16 bits of F32
        f32Array[i] = view.getUint16(i * 2, true) << 16;
    }
    return f32Array;
}

/**
 * Optimized Dequantization.
 */
function dequantizeFromI7(i7Buffer) {
    const scale = i7Buffer.readFloatLE(0);
    const numElements = i7Buffer.length - 4;
    
    // Create a direct view on the buffer starting after the header
    const i8View = new Int8Array(i7Buffer.buffer, i7Buffer.byteOffset + 4, numElements);
    const f32Array = new Float32Array(numElements);
    
    for (let i = 0; i < numElements; i++) {
        // Using the LUT with direct Int8 index. 
        // JS engine optimizes lookups on TypedArrays heavily.
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