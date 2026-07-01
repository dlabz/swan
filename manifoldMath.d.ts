export const I7_TO_FLOAT32_LUT: Float32Array<ArrayBuffer>;
/**
 * Converts a Little-Endian BFloat16 buffer into a Float32Array.
 * @param {Buffer} bf16Buffer
 * @returns {Float32Array}
 */
export function bf16ToFloat32Array(bf16Buffer: Buffer): Float32Array;
/**
 * Projects a Float32Array into the I7 Manifold format.
 * Uses symmetric quantization to Int8 (1 sign, 7 magnitude).
 * Returns a buffer containing:
 * - 4 bytes: Float32 Scale factor (Little-Endian)
 * - N bytes: I7 quantized data
 * @param {Float32Array} f32Array
 * @returns {Buffer}
 */
export function quantizeToI7(f32Array: Float32Array): Buffer;
/**
 * Dequantizes an I7 buffer back to Float32 using the Math Shortcuts LUT.
 * @param {Buffer} i7Buffer
 * @returns {Float32Array}
 */
export function dequantizeFromI7(i7Buffer: Buffer): Float32Array;
