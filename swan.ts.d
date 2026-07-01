/**
 * swan.d.ts - Core Type Definitions for SWAN Engine
 */

export interface AppConfig {
    readonly HIDDEN_SIZE: number;
    readonly BF16_BYTES: number;
    readonly CHUNK_SIZE_MB: number;
}

export interface InferenceState {
    inputTokens: number[];
    hiddenState: Float32Array;
    maxActivation: number;
    bestTokenId: number;
    processedBytes: number;
}

export interface Logger {
    log(msg: string, type?: 'log' | 'error' | 'system'): void;
    flush(): void;
}

/**
 * Kaitai Struct Representation for the Qwen Safetensors format
 * * meta:
 * id: bfloat16_tensor
 * title: Brain Floating Point (16-bit) Vector
 * endian: le
 * seq:
 * - id: elements
 * type: bfloat16
 * repeat: eos
 * types:
 * bfloat16:
 * seq:
 * - id: mantissa
 * type: b7
 * - id: exponent
 * type: b8
 * - id: sign
 * type: b1
 */
export type BFloat16Buffer = Uint8Array;
