// @ts-check
/**
 * @typedef {import('express').Response} ExpressResponse
 * @typedef {import('express').Request} ExpressRequest
 */
const express = require('express');
const cors = require('cors');
const fs = require('fs/promises');
const path = require('path');
const { downloadAndIndexModel } = require('./modelDownloader');
const { projectAllLayers, I7_DIR } = require('./manifoldController');

const app = express();
app.use(cors());
const BF16_DIR = path.resolve(__dirname, 'data/model/bf16');

/**
 * Serve the web application frontend.
 * Accessible at: /
 */
app.use(express.static('public'));

// --- ADDED STATIC CACHING HEADERS ---
// Max-Age of 1 Year (31536000 seconds), marking files as immutable
const staticCacheOptions = {
    maxAge: '1y',
    immutable: true,
    lastModified: true,
    etag: true
};

/**
 * Serve raw BF16 unprojected model weights.
 * Accessible at: /model/bf16
 * Cached for 1 year as files are immutable.
 */
app.use('/model/bf16', express.static(BF16_DIR, staticCacheOptions));

/**
 * Serve projected Int7 (I7) model weights and related artifacts.
 * Accessible at: /model/i7
 * Cached for 1 year as files are immutable.
 */
app.use('/model/i7', express.static(I7_DIR, staticCacheOptions));

// ---------------------------------------------------------
// SERVER-SENT EVENTS (SSE) MANAGER
// ---------------------------------------------------------
/** @type {ExpressResponse[]} */
let clients = [];

/**
 * Broadcasts data to all connected SSE clients.
 * @param {{ type: string, message: string } | any} data - The data to broadcast
 */
function broadcast(data) {
    clients.forEach(client => client.write(`data: ${JSON.stringify(data)}\n\n`));
}

const originalLog = console.log;
const originalError = console.error;

/**
 * @param {...any} args
 */
console.log = (...args) => {
    originalLog(...args);
    broadcast({ type: 'log', message: args.join(' ') });
};

/**
 * @param {...any} args
 */
console.error = (...args) => {
    originalError(...args);
    broadcast({ type: 'error', message: args.join(' ') });
};

/**
 * Server-Sent Events (SSE) endpoint to stream server logs to the client.
 * Accessible at: GET /api/events
 * @param {ExpressRequest} req
 * @param {ExpressResponse} res
 */
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

/**
 * Check if the projected model is fully extracted and ready.
 * Accessible at: GET /api/check
 * @param {ExpressRequest} req
 * @param {ExpressResponse} res
 * @returns {Promise<void>} JSON response indicating readiness: { ready: boolean }
 */
app.get('/api/check', async (req, res) => {
    try {
        const files = await fs.readdir(I7_DIR);
        const hasCoreLayer = files.includes('projected_embed_tokens.bin');
        res.json({ ready: hasCoreLayer });
    } catch {
        res.json({ ready: false });
    }
});


/**
 * Download the model and project it to Int7.
 * Accessible at: GET /api/install
 * @param {ExpressRequest} req
 * @param {ExpressResponse} res
 * @returns {Promise<void>} JSON response with status, e.g., { status: 'success' }
 */
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
    originalLog(`- BF16 Cache: ${BF16_DIR}`);
    originalLog(`- I7 Manifold: ${I7_DIR}`);
});
