const { downloadAndIndexModel } = require('./modelDownloader');

async function runQwenTest() {
    const REPO_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct";
    
    console.log(`Starting discovery for ${REPO_ID}...`);

    try {
        await downloadAndIndexModel(REPO_ID);
        console.log("\n--- All files cached successfully ---");
    } catch (e) {
        console.error(`\nTEST FAILED: ${e.message}`);
    }
}

runQwenTest();