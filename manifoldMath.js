// @ts-check

/**
 * Manifold Math operations for BFloat16 <-> I7 conversion.
 * 
 * Safetensors stores data in Little-Endian format.
 * BFloat16 is 16 bits: 1 sign bit, 8 exponent bits, 7 mantissa bits.
 * It has the exact same upper 16 bits as an IEEE-754 Float32.
 */

const LUT_SIZE = 256;

// Precomputed Math Shortcuts Lookup Table for I7 to unscaled Float32
// I7 is implemented as an 8-bit signed integer (1 sign bit, 7 magnitude bits).
const I7_TO_FLOAT32_LUT = new Float32Array(LUT_SIZE);
for (let i = -128; i < 128; i++) {
    // We map the 8-bit two's complement integer to its array index (0-255)
    I7_TO_FLOAT32_LUT[i & 0xFF] = i;
}

/**
 * Converts a Little-Endian BFloat16 buffer into a Float32Array.
 * @param {Buffer} bf16Buffer 
 * @returns {Float32Array}
 */
function bf16ToFloat32Array(bf16Buffer) {
    if (bf16Buffer.length % 2 !== 0) {
        throw new Error("BFloat16 buffer length must be a multiple of 2");
    }
    const numElements = bf16Buffer.length / 2;
    const f32Array = new Float32Array(numElements);
    const floatView = new DataView(new ArrayBuffer(4));

    for (let i = 0; i < numElements; i++) {
        // Read the 16 bits from Little-Endian
        const b0 = bf16Buffer[i * 2]; // LSB
        const b1 = bf16Buffer[i * 2 + 1]; // MSB
        
        // Assemble into the upper 16 bits of a 32-bit float
        const uint32Val = (b1 << 24) | (b0 << 16);
        floatView.setUint32(0, uint32Val, false); // big-endian internal view
        f32Array[i] = floatView.getFloat32(0, false);
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
    // 1. Find max absolute value for symmetric quantization
    let maxAbs = 0;
    for (let i = 0; i < f32Array.length; i++) {
        const absVal = Math.abs(f32Array[i]);
        if (absVal > maxAbs) maxAbs = absVal;
    }

    // 2. Compute scale (max I7 value is 127)
    // Avoid division by zero
    const scale = maxAbs === 0 ? 1 : maxAbs / 127;
    const invScale = maxAbs === 0 ? 1 : 127 / maxAbs;

    // 3. Create I7 buffer (4 bytes for scale + 1 byte per element)
    const i7Buffer = Buffer.alloc(4 + f32Array.length);
    i7Buffer.writeFloatLE(scale, 0);

    // 4. Quantize
    for (let i = 0; i < f32Array.length; i++) {
        let quantized = Math.round(f32Array[i] * invScale);
        
        // Clamp to [-127, 127] standard I7 range
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;
        
        i7Buffer.writeInt8(quantized, 4 + i);
    }

    return i7Buffer;
}

/**
 * Dequantizes an I7 buffer back to Float32 using the Math Shortcuts LUT.
 * @param {Buffer} i7Buffer 
 * @returns {Float32Array}
 */
function dequantizeFromI7(i7Buffer) {
    if (i7Buffer.length < 4) throw new Error("Invalid I7 buffer");
    const scale = i7Buffer.readFloatLE(0);
    const numElements = i7Buffer.length - 4;
    
    const f32Array = new Float32Array(numElements);
    
    for (let i = 0; i < numElements; i++) {
        // Read Int8 as unsigned byte to index the LUT directly
        const byteVal = i7Buffer.readUInt8(4 + i);
        // Math shortcut via LUT
        f32Array[i] = I7_TO_FLOAT32_LUT[byteVal] * scale;
    }
    
    return f32Array;
}

module.exports = {
    I7_TO_FLOAT32_LUT,
    bf16ToFloat32Array,
    quantizeToI7,
    dequantizeFromI7
};
