const express = require('express');
const fs = require('fs/promises');
const path = require('path');
const { downloadAndIndexModel } = require('./modelDownloader');
const { projectAllLayers, I7_DIR } = require('./manifoldController');

const app = express();
const F32_DIR = path.resolve(__dirname, 'data/model/f32');

app.use(express.static('public'));

// 1. Dual Model Serving: Serve both raw safetensors (F32) and projected F32 (i7)
app.use('/model/f32', express.static(F32_DIR));
app.use('/model/i7', express.static(I7_DIR));

// ---------------------------------------------------------
// SERVER-SENT EVENTS (SSE) MANAGER
// ---------------------------------------------------------
let clients = [];

function broadcast(data) {
    clients.forEach(client => client.write(`data: ${JSON.stringify(data)}\n\n`));
}

// Hook into console to automatically broadcast logs to the frontend
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

// SSE Endpoint
app.get('/api/events', (req, res) => {
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders(); // Establish connection immediately

    clients.push(res);
    req.on('close', () => {
        clients = clients.filter(client => client !== res);
    });
});
// ---------------------------------------------------------

// Check System Readiness
app.get('/api/check', async (req, res) => {
    try {
        const files = await fs.readdir(I7_DIR);
        // Look for embed_tokens since Qwen uses tied weights
        const hasCoreLayer = files.includes('projected_embed_tokens.bin');
        res.json({ ready: hasCoreLayer });
    } catch {
        res.json({ ready: false });
    }
});



// Start Installation Pipeline
app.get('/api/install', async (req, res) => {
    // Return immediately to prevent browser timeout; run process in background
    res.json({ status: 'STARTED', message: 'Installation running. Connect to /api/events for progress.' });

    try {
        broadcast({ type: 'system', message: 'Starting SWAN Engine Installation Pipeline...' });
        
        // 1. Download and Index (Using HF Repo ID)
        broadcast({ type: 'system', message: 'PHASE 1: Download & Indexing' });
        await downloadAndIndexModel("Qwen/Qwen2.5-Coder-1.5B-Instruct");

        // 2. Project Layers
        broadcast({ type: 'system', message: 'PHASE 2: Manifold Projection' });
        await projectAllLayers();

        broadcast({ type: 'complete', message: 'System Ready.' });
    } catch (e) {
        broadcast({ type: 'error', message: `Installation Failed: ${e.message}` });
    }
});

app.listen(3000, () => {
    originalLog('Swan Engine active on port 3000');
    originalLog(`- F32 Cache: ${F32_DIR}`);
    originalLog(`- I7 Manifold: ${I7_DIR}`);
});