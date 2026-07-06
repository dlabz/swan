# SWAN Engine - TODO List

This list contains outstanding items to improve the performance, user interface, and overall stability of the SWAN engine WebGPU inference pipeline.

## 🚀 High Priority (Performance & Correctness)
- [ ] **Implement KV Cache Pruning / Rolling Window**: Current cache size is hardcoded to a max sequence length of 512 tokens. Implement a sliding window or dynamic allocation for longer contexts.
- [ ] **Benchmarking & Profiling**: Measure the time spent in the forward pass per token (ms/token) on the discrete ATI 5500M GPU vs CPU/Integrated GPU.
- [ ] **Validate Tokenizer Output**: Compare output logits for known prompts against PyTorch/HF reference models to ensure mathematically identical outputs down to quantization limits.

## 🎨 User Interface & Experience
- [ ] **Rich Premium CSS Dashboard**: Replace the basic terminal interface with a modern, glassmorphic dark-theme UI.
- [ ] **Streaming Token Decoder**: Update the inference button to stream generated words token-by-token into a chat bubble rather than posting terminal log lines.
- [ ] **Interactive Visual Graph**: Visualize live VRAM cache usage and attention weight activation maps during generation.

## 📦 Developer Experience & Portability
- [ ] **Local Model Asset Sync Script**: Write a shell script to quickly sync or symlink the `data/model` folder from local storage instead of redownloading.
- [ ] **Dockerized Setup**: Configure a multi-platform Docker container to run the Node.js server seamlessly.
- [ ] **PWA / Service Worker Caching**: Add service worker configuration to cache JS vendor libs and tokenizer files locally, ensuring 100% offline capability even if browser cache is disabled.
