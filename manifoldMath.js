// @ts-check

/**
 * Manifold Math operations for BFloat16 <-> I7 conversion.
 */

const LUT_SIZE = 256;

// Precomputed Math Shortcuts Lookup Table
const I7_TO_FLOAT32_LUT = new Float32Array(LUT_SIZE);
for (let i = 0; i < LUT_SIZE; i++) {
    // Cast index (0-255) to signed 8-bit (-128 to 127) using offset subtraction
    I7_TO_FLOAT32_LUT[i] = i - 128;
}

/**
 * Converts a Little-Endian BFloat16 buffer into a Float32Array.
 * @param {Buffer} bf16Buffer
 * @returns {Float32Array}
 */
function bf16ToFloat32Array(bf16Buffer) {
    const numElements = bf16Buffer.length >> 1;
    const f32Array = new Float32Array(numElements);
    const view = new DataView(bf16Buffer.buffer, bf16Buffer.byteOffset, bf16Buffer.byteLength);
    const u32 = new Uint32Array(1);
    const f32 = new Float32Array(u32.buffer);
    
    for (let i = 0; i < numElements; i++) {
        u32[0] = view.getUint16(i * 2, true) << 16;
        f32Array[i] = f32[0];
    }
    return f32Array;
}

/**
 * Projects a Float32Array into the I7 Manifold format.
 * Uses symmetric quantization to Int8 (1 sign, 7 magnitude).
 * Returns a buffer containing:
 * - 4 bytes: Float32 Scale factor (Little-Endian)
 * - N bytes: I7 quantized data
 * @param {Float32Array} f32Array
 * @returns {Buffer}
 */
function quantizeToI7(f32Array) {
    let maxVal = 0;
    for (let i = 0; i < f32Array.length; i++) {
        const val = Math.abs(f32Array[i]);
        if (val > maxVal) maxVal = val;
    }
    
    const scale = maxVal / 127.0;
    const safeScale = scale || 1.0;
    
    const buffer = Buffer.alloc(4 + f32Array.length);
    buffer.writeFloatLE(scale, 0);
    
    for (let i = 0; i < f32Array.length; i++) {
        let scaled = Math.round(f32Array[i] / safeScale);
        if (Number.isNaN(scaled)) scaled = 0;
        const byteVal = Math.max(-127, Math.min(127, scaled));
        buffer.writeInt8(byteVal, 4 + i);
    }
    
    return buffer;
}

/**
 * Process a raw BF16 buffer, scale it by maxVal, and encode it into 8-bit signed integers.
 * Keeps memory peak very low.
 * @param {Buffer} bf16Buffer - Raw BF16 bytes
 * @param {number} maxVal - Max absolute value of the tensor for scaling
 * @returns {Buffer} - Int8 packed bytes
 */
function encodeBF16To5Cell8Bit(bf16Buffer, maxVal) {
    const numElements = bf16Buffer.length >> 1;
    const outBuffer = Buffer.alloc(numElements);
    
    // SharedArrayBuffer / TypedArray trick to bitcast uint32 to float32
    const bitBuffer = new ArrayBuffer(4);
    const bitU32 = new Uint32Array(bitBuffer);
    const bitF32 = new Float32Array(bitBuffer);
    
    const view = new DataView(bf16Buffer.buffer, bf16Buffer.byteOffset, bf16Buffer.byteLength);
    const safeMaxVal = maxVal || 1.0;
    
    for (let i = 0; i < numElements; i++) {
        const bf16Bits = view.getUint16(i * 2, true);
        bitU32[0] = bf16Bits << 16;
        const f32Val = bitF32[0];
        
        let scaled = Math.round((f32Val / safeMaxVal) * 127.0);
        if (Number.isNaN(scaled)) scaled = 0;
        const byteVal = Math.max(-127, Math.min(127, scaled));
        
        outBuffer.writeInt8(byteVal, i);
    }
    
    return outBuffer;
}

/**
 * Optimized Dequantization.
 * @param {Buffer} i7Buffer
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
    dequantizeFromI7,
    encodeBF16To5Cell8Bit
};