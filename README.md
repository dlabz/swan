# SWAN: WebGPU Inference Pipeline

SWAN is a self-contained, high-performance client-side WebGPU inference engine designed to execute quantized large language models (like Qwen2.5/Qwen1.5) directly inside the browser. It features low-level WGSL compute shaders, symmetric Int7/Int8 quantization, and an optimized KV cache.

---

## 🛠 Local Installation

### Prerequisites
* **Node.js** (v18+)
* **Google Chrome** or any Chromium browser with WebGPU support.
* For switchable graphics laptops (e.g. MacBook Pro with Intel + AMD Radeon GPU), start Chrome with the high-performance GPU flag forced:
  ```bash
  # macOS Force High-Performance GPU Chrome Launch
  open -a "Google Chrome" --args --force_high_performance_gpu
  ```

### Getting Started
1. **Clone the repo locally**:
   ```bash
   git clone https://github.com/dlabz/swan.git
   cd swan
   ```
2. **Install dependencies**:
   ```bash
   npm install
   ```
3. **Start the backend server**:
   ```bash
   npm start
   ```
   * The server runs on secure HTTPS (`https://localhost:3000`) required for WebGPU secure context.
   * It also hosts a plain HTTP server on `http://localhost:3001` for debugging tools that reject self-signed certificates.
4. **Access the application**:
   Open Chrome and navigate to `https://localhost:3000`.

---

## 📦 Docker Deployment

Because WebGPU execution happens entirely **inside the client's browser**, the backend server only acts as a static host for the web app assets and model weights. This means **you do not need complex GPU pass-through drivers (like NVIDIA Container Toolkit) on the server/Docker side**.

### 1. Build the Docker Image
```bash
docker build -t swan-engine .
```

### 2. Run the Container
Map the HTTPS (3000) and HTTP (3001) ports to your host machine:
```bash
docker run -d \
  -p 3000:3000 \
  -p 3001:3001 \
  -v $(pwd)/data:/app/data \
  --name swan-app \
  swan-engine
```
*The `-v` flag mounts the local `data/` directory containing your model files, saving you from redownloading them if you recreate the container.*

---

## 🔄 Cross-Device Syncing of Model Weights

The model weights (~1GB+) are stored in `data/model/`. Downloading them over port-forwarding or slow connections on every new environment setup is inefficient.

### Best Ways to Sync Model Files Across Devices:

#### Option A: Syncing via local network (rsync)
If you have already downloaded the model weights on one machine (e.g. Codespace or your laptop) and want to transfer them directly to another local machine:
```bash
rsync -avz --progress user@remote-ip:/path/to/swan/data/model/ ./data/model/
```

#### Option B: Docker Volume Mount
If you run the app inside Docker, make sure to mount your host's local model directory:
```bash
-v /Users/shared/models/swan:/app/data/model
```
This ensures different container builds or instances instantly read the already-downloaded weights without fetching them again.

#### Option C: Browser Caching (OPFS)
Chrome caches the weights inside its local sandbox via **OPFS (Origin Private File System)**. Even if you restart your Docker container or Node server, Chrome will pull the model directly from the laptop's physical SSD rather than re-downloading it from the server, unless you manually clear "Site Data".