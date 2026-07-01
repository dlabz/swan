# Qwen 2.5 Coder (1.5B Instruct) - Model Documentation

Welcome to the BlackSwan local model repository. This directory (`/data/model/bf16/`) serves as the cache for the raw, unprojected model weights downloaded directly from HuggingFace.

## Model Specifications
* **Architecture**: Qwen 2.5 Coder
* **Size**: 1.5 Billion Parameters
* **Type**: Instruct-tuned LLM
* **Original Format**: HuggingFace Safetensors
* **Data Type**: BFloat16 (`BF16`)
* **Endianness**: Little-Endian (LE)

## Storage and Format: Safetensors & BFloat16
Safetensors is a secure and fast file format designed for storing tensors. The files contained in this directory are the direct monolithic `.safetensors` files. 

By default, Safetensors stores the binary tensor chunks in **Little-Endian (LE)** byte order. The data type heavily utilized by Qwen 2.5 is **BFloat16**. 
- A standard `Float32` uses 32 bits (1 sign, 8 exponent, 23 fraction).
- `BFloat16` truncates the fraction down to 7 bits (1 sign, 8 exponent, 7 fraction).
- This means `BFloat16` has the exact same dynamic range as a standard 32-bit float, but fits into just 2 bytes, halving the memory footprint without sacrificing overflow/underflow boundaries.

## The I7 Manifold Quantization
While BFloat16 cuts memory in half compared to Float32, the Swan Engine aggressively optimizes this further into a custom **I7 Manifold Format** (housed in `/data/model/i7/`). 

### How It Works
The projection pipeline (`manifoldController.js`) extracts exact, individual tensor slices from the `.safetensors` blobs and pipes them through our mathematical quantizer (`manifoldMath.js`):

1. **Symmetric Int8 Quantization**: Each tensor block finds its absolute maximum value. The entire tensor is then scaled proportionally to fit within the `[-127, 127]` boundaries of a standard signed 8-bit integer.
2. **Storage Layout**: Each resulting `.bin` shard begins with a **4-byte Little-Endian Float32 scale factor**, immediately followed by the raw 8-bit (`Int8`) integers. 
3. **Efficiency**: This strictly bounds model inference to a format exactly **50% smaller** than BFloat16, acting as a massive multiplier for memory bandwidth and cache locality. 

## Inference & Math Shortcuts
LLM Inference is notoriously memory-bandwidth bound. While the V8 JavaScript engine struggles to accelerate raw 8-bit hardware mathematics (often unpacking arrays into `Float64` behind the scenes), the memory-fetching bottleneck is vastly relieved.

To speed up the dequantization overhead during forward passes, the engine utilizes a **Math Shortcuts Lookup Table (LUT)** (`I7_TO_FLOAT32_LUT`). 
- Instead of casting and multiplying floats per parameter, the engine precomputes all 256 possible boundaries of an 8-bit signed integer (`-128` to `127`).
- During inference, the engine performs an `O(1)` array lookup to map the raw binary byte directly back to its original Float32 magnitude before applying the tensor's shared scale factor.

## Pipeline Architecture
- **`modelDownloader.js`**: Streams the monolithic `.safetensors` via HF, performing real-time CRC32 integrity checks, and uses HTTP Range Requests to strip the JSON metadata headers without downloading the full files into memory.
- **`manifoldController.js`**: Orchestrates the parsing of the offset maps, slicing the massive files into granular per-layer chunks.
- **`manifoldMath.js`**: The beating heart of the numerical engine. Converts Big-Endian/Little-Endian alignments, handles the `BF16 -> Float32 -> I7` mathematical translations, and manages the Lookup Tables.