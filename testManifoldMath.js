// @ts-check
const assert = require('assert');
const { bf16ToFloat32Array, quantizeToI7, dequantizeFromI7 } = require('./manifoldMath');

function runTests() {
    console.log("Running Manifold Math Tests...");

    // 1. Create a dummy BF16 buffer representing [1.0, -1.0, 0.5, -0.5]
    // 1.0 in float32 is 0x3F800000, so in BF16 it's 0x3F80. Little Endian: 80 3F
    // -1.0 is 0xBF800000 -> 0xBF80. LE: 80 BF
    // 0.5 is 0x3F000000 -> 0x3F00. LE: 00 3F
    // -0.5 is 0xBF000000 -> 0xBF00. LE: 00 BF
    const bf16Buffer = Buffer.from([
        0x80, 0x3F,
        0x80, 0xBF,
        0x00, 0x3F,
        0x00, 0xBF
    ]);

    // Test bf16 to float32
    const f32Array = bf16ToFloat32Array(bf16Buffer);
    assert.strictEqual(f32Array[0], 1.0);
    assert.strictEqual(f32Array[1], -1.0);
    assert.strictEqual(f32Array[2], 0.5);
    assert.strictEqual(f32Array[3], -0.5);
    console.log("✓ bf16ToFloat32Array passed");

    // Test quantizeToI7
    const i7Buffer = quantizeToI7(f32Array);
    
    // Max abs is 1.0. Scale = 1.0 / 127 = 0.007874015...
    const scale = i7Buffer.readFloatLE(0);
    assert(Math.abs(scale - (1.0 / 127)) < 1e-6);
    
    // Check quantized values
    const q1 = i7Buffer.readInt8(4);
    const q2 = i7Buffer.readInt8(5);
    const q3 = i7Buffer.readInt8(6);
    const q4 = i7Buffer.readInt8(7);
    
    assert.strictEqual(q1, 127); // 1.0 / scale
    assert.strictEqual(q2, -127);
    assert.strictEqual(q3, 64);
    assert.strictEqual(q4, -63);

    console.log("✓ quantizeToI7 passed");

    // Test dequantizeFromI7 (LUT usage)
    const dequantized = dequantizeFromI7(i7Buffer);
    
    // Check values are close to original
    assert(Math.abs(dequantized[0] - 1.0) < 0.02);
    assert(Math.abs(dequantized[1] - -1.0) < 0.02);
    assert(Math.abs(dequantized[2] - 0.5) < 0.02);
    assert(Math.abs(dequantized[3] - -0.5) < 0.02);
    console.log("✓ dequantizeFromI7 passed");

    console.log("All tests passed successfully.");
}

runTests();
