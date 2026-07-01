SWAN Inference Engine 🦢
SWAN is a high-performance, memory-safe Large Language Model inference engine designed to run massive AI models directly in the browser via chunked streaming and Origin Private File System (OPFS) storage.
Architectural Overview
The engine bypasses traditional browser memory limits (which often crash around 1GB of VRAM/Heap) by treating the hard drive as a streaming tape memory.
1. Backend Extraction (⁠manifoldController.js⁠)
￼ Parses ⁠.safetensors⁠ headers.
￼ Slices the monolithic model into individual tensor ⁠.bin⁠ files.
￼ Preserves raw ⁠bfloat16⁠ data without FP32 upcasting to save bandwidth.
2. Smart OPFS Caching (⁠SwanStorage⁠)
￼ Issues HTTP ⁠Range⁠ requests (max 5MB chunks) to prevent proxy timeouts.
￼ Pipes network streams directly into the browser's C++ disk layer via ⁠FileSystemWritableFileStream⁠.
￼ Prevents JavaScript heap exhaustion.
3. In-Browser Transformer Math (⁠SwanMath⁠)
￼ Dynamically decodes ⁠bfloat16⁠ Little Endian binary into IEEE-754 ⁠Float32⁠ arrays using zero-copy byte shifting.
￼ Streams massive projection matrices (e.g., ⁠lm_head⁠) block-by-block.
￼ Yields to the main thread to allow browser Garbage Collection.
Precision Types
The model heavily utilizes ⁠bfloat16⁠. Standard JavaScript ⁠DataView⁠ does not support this type natively, so we implement a Kaitai-documented custom byte-shifter to reconstruct 32-bit floats on the fly.