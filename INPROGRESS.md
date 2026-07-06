# In-Progress & Handoff State

This document captures the current state of the SWAN engine WebGPU inference pipeline, recent bugfixes, and debugging findings for the local migration.

## Current Status: 🟢 Pipeline Compiles & Executes
All WebGPU compute shaders compile successfully, and the forward-pass execution flow is structurally correct. Memory leaks and arithmetic scaling issues have been resolved.

---

## 🛠 Recent Core Fixes

### 1. Dequantization Scaling Error (127× Factor)
* **Problem**: The compute shaders were using the built-in WGSL `unpack4x8snorm()` function. This function automatically converts signed 8-bit integers (`-128` to `127`) to float32 by dividing them by `127.0`. However, our offline quantizer (`manifoldMath.js`) already scaled the weights down to fit `-127` to `127` using `scale = maxVal / 127.0`. The dequantization formula should have been `raw_int8 * scale`. By using `unpack4x8snorm()`, we got `(raw_int8 / 127) * (maxVal / 127)`, making all activations **127 times too small** and causing logits to collapse to exactly `0.00`.
* **Fix**: Replaced all `unpack4x8snorm` calls in [wgsl_shaders.js](file:///workspaces/swan/public/js/wgsl_shaders.js) with a custom `unpack4xi8(v: u32) -> vec4<f32>` function. This function extracts raw signed 8-bit integers without the additional division.

### 2. Discrete GPU Target selection (MacBook Pro)
* **Problem**: When running on machines with switchable/hybrid graphics (like Intel UHD + AMD Radeon ATI 5500M), WebGPU defaults to the low-power integrated GPU.
* **Fix**: Configured `SwanMathEngine.init()` in [swanMath.js](file:///workspaces/swan/public/js/swanMath.js) to explicitly prioritize the high-performance adapter:
  ```javascript
  let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  ```

### 3. VRAM Leakage (Transient Buffers)
* **Problem**: Every step of the token generation loop was allocating new uniform/config WebGPU buffers that were never released. This would eventually crash the browser GPU process.
* **Fix**: Added a tracking array `transientBuffers` inside the forward pass. Every temporary buffer created via `tempBuf()` is explicitly `.destroy()`ed immediately after the command encoder is submitted to the GPU queue:
  ```javascript
  for (const buf of transientBuffers) buf.destroy();
  ```

### 4. GPU Validation and Out-Of-Memory Reporting
* **Problem**: WebGPU validation errors were silent or crashed the page abruptly.
* **Fix**: Wrapped the compute pipeline executions in `pushErrorScope()` and `popErrorScope()` blocks to capture and print any GPU errors directly to the browser console.

---

## 🔍 Local Setup Diagnostics
When booting the app locally on your laptop:
1. Open the browser console. You should see a log: `[GPU] Adapter: <vendor> / <architecture> / <device>`. Verify this reports your discrete GPU (AMD Radeon / ATI 5500M).
2. On prompt execution (`INFERENCE`), look for the console log showing `[pos=0] Top-5 logits: ...`. The values should now be realistic floats, rather than `0.00`.
