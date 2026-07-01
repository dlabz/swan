const { downloadAndIndexModel } = require('./modelDownloader');

async function runQwenTest() {
    // Qwen2.5-Coder-1.5B shards (example URL structure)
    const shards = [
        { name: 'model-00001-of-00002.safetensors', url: 'https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/model-00001-of-00002.safetensors' },
        { name: 'model-00002-of-00002.safetensors', url: 'https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/model-00002-of-00002.safetensors' }
    ];

    console.log("Starting Qwen pipeline...");

    for (const shard of shards) {
        try {
            // Note: In real production, get the CRC from HF "LFS" metadata
            await downloadAndIndexModel(shard.url, shard.name);
            console.log(`Shard ${shard.name} ready.`);
        } catch (e) {
            console.error(`Failed: ${e.message}`);
        }
    }
}

runQwenTest();
