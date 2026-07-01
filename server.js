const express = require('express');
const fs = require('fs/promises');
const path = require('path');
const { downloadAndIndexModel } = require('./modelDownloader');
const { projectAllLayers, I7_DIR } = require('./manifoldController');

const app = express();
const F32_DIR = path.resolve(__dirname, 'data/model/f32');

app.use(express.static('public'));

// --- ADDED STATIC CACHING HEADERS ---
// Max-Age of 1 Year (31536000 seconds), marking files as immutable
const staticCacheOptions = {
    maxAge: '1y',
    immutable: true,
    lastModified: true,
    etag: true
};

app.use('/model/f32', express.static(F32_DIR, staticCacheOptions));
app.use('/model/i7', express.static(I7_DIR, staticCacheOptions));

// ---------------------------------------------------------
// SERVER-SENT EVENTS (SSE) MANAGER
// ---------------------------------------------------------
let clients = [];

function broadcast(data) {
    clients.forEach(client => client.write(`data: ${JSON.stringify(data)}\n\n`));
}

const originalLog = console.log;
const originalError = console.error;

console.log = (...args) => {
    originalLog(...args);
    broadcast({ type: 'log', message: args.join(' ') });
};
console.error = (...args) => {
    originalError(...args);
    broadcast({ type: 'error', message: args.join(' ') });
};

app.get('/api/events', (req, res) => {
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders(); 

    clients.push(res);
    req.on('close', () => {
        clients = clients.filter(client => client !== res);
    });
});

// Check System Readiness
app.get('/api/check', async (req, res) => {
    try {
        const files = await fs.readdir(I7_DIR);
        const hasCoreLayer = files.includes('projected_embed_tokens.bin');
        res.json({ ready: hasCoreLayer });
    } catch {
        res.json({ ready: false });
    }
});


app.get('/api/install', async (req, res) => {
    try {
        await downloadAndIndexModel("Qwen/Qwen2.5-Coder-1.5B-Instruct");
        await projectAllLayers();
        
        // ADD THIS CHECK: Verify the file actually exists before saying success
        const fs = require('fs');
        const target = path.resolve(__dirname, 'data/model/i7/projected_embed_tokens.bin');
        if (fs.existsSync(target)) {
            res.json({ status: 'success' });
        } else {
            throw new Error("Extraction finished, but file is missing.");
        }
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});


app.listen(3000, () => {
    originalLog('Swan Engine active on port 3000');
    originalLog(`- F32 Cache: ${F32_DIR}`);
    originalLog(`- I7 Manifold: ${I7_DIR}`);
});
