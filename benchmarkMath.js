// @ts-check
const { performance } = require('perf_hooks');

const VECTOR_SIZE = 1536 * 4; // Typical Qwen hidden size
const ITERATIONS = 5000;

function runBenchmarks() {
    console.log(`Starting Benchmark: Vector Size = ${VECTOR_SIZE}, Iterations = ${ITERATIONS}`);

    // Generate random data
    const f32_A = new Float32Array(VECTOR_SIZE);
    const f32_B = new Float32Array(VECTOR_SIZE);
    
    // Fill with random values between -1 and 1
    for (let i = 0; i < VECTOR_SIZE; i++) {
        f32_A[i] = (Math.random() * 2) - 1;
        f32_B[i] = (Math.random() * 2) - 1;
    }

    // Quantize to I7 (Int8)
    const scaleA = 1.0 / 127;
    const scaleB = 1.0 / 127;
    const i7_A = new Int8Array(VECTOR_SIZE);
    const i7_B = new Int8Array(VECTOR_SIZE);

    for (let i = 0; i < VECTOR_SIZE; i++) {
        i7_A[i] = Math.round(f32_A[i] / scaleA);
        i7_B[i] = Math.round(f32_B[i] / scaleB);
    }

    // Output Arrays
    const f32_Out = new Float32Array(VECTOR_SIZE);
    const i7_Out = new Float32Array(VECTOR_SIZE); // Output of quantized ops is usually accumulated into float32

    console.log("\n--- Dot Product (Matrix-Vector equivalent) ---");
    // Float32 Dot Product
    let start = performance.now();
    let sumF32 = 0;
    for (let iter = 0; iter < ITERATIONS; iter++) {
        let sum = 0;
        for (let i = 0; i < VECTOR_SIZE; i++) {
            sum += f32_A[i] * f32_B[i];
        }
        sumF32 = sum; // prevent optimization
    }
    const f32DotTime = performance.now() - start;
    console.log(`Float32 Dot Product: ${f32DotTime.toFixed(2)} ms`);

    // I7 Dot Product
    start = performance.now();
    let sumI7 = 0;
    const combinedScale = scaleA * scaleB;
    for (let iter = 0; iter < ITERATIONS; iter++) {
        let sum = 0;
        for (let i = 0; i < VECTOR_SIZE; i++) {
            // Integer multiplication and accumulation
            sum += i7_A[i] * i7_B[i];
        }
        sumI7 = sum * combinedScale;
    }
    const i7DotTime = performance.now() - start;
    console.log(`I7 Dot Product:      ${i7DotTime.toFixed(2)} ms`);
    console.log(`Speedup:             ${(f32DotTime / i7DotTime).toFixed(2)}x`);

    console.log("\n--- Vector Addition (Element-wise) ---");
    // Float32 Addition
    start = performance.now();
    for (let iter = 0; iter < ITERATIONS; iter++) {
        for (let i = 0; i < VECTOR_SIZE; i++) {
            f32_Out[i] = f32_A[i] + f32_B[i];
        }
    }
    const f32AddTime = performance.now() - start;
    console.log(`Float32 Addition:    ${f32AddTime.toFixed(2)} ms`);

    // I7 Addition (assuming we dequantize on the fly via LUT or direct scale multiplication)
    // For addition, typically scales are different, so we scale them during addition.
    start = performance.now();
    for (let iter = 0; iter < ITERATIONS; iter++) {
        for (let i = 0; i < VECTOR_SIZE; i++) {
            i7_Out[i] = (i7_A[i] * scaleA) + (i7_B[i] * scaleB);
        }
    }
    const i7AddTime = performance.now() - start;
    console.log(`I7 Addition:         ${i7AddTime.toFixed(2)} ms`);
    console.log(`Speedup:             ${(f32AddTime / i7AddTime).toFixed(2)}x`);

    console.log("\n--- Vector Multiplication (Element-wise) ---");
    // Float32 Multiplication
    start = performance.now();
    for (let iter = 0; iter < ITERATIONS; iter++) {
        for (let i = 0; i < VECTOR_SIZE; i++) {
            f32_Out[i] = f32_A[i] * f32_B[i];
        }
    }
    const f32MulTime = performance.now() - start;
    console.log(`Float32 Multiply:    ${f32MulTime.toFixed(2)} ms`);

    // I7 Multiplication
    start = performance.now();
    for (let iter = 0; iter < ITERATIONS; iter++) {
        for (let i = 0; i < VECTOR_SIZE; i++) {
            i7_Out[i] = (i7_A[i] * i7_B[i]) * combinedScale;
        }
    }
    const i7MulTime = performance.now() - start;
    console.log(`I7 Multiply:         ${i7MulTime.toFixed(2)} ms`);
    console.log(`Speedup:             ${(f32MulTime / i7MulTime).toFixed(2)}x`);
}

runBenchmarks();
