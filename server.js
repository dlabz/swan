// @ts-check
/**
 * @typedef {import('express').Response} ExpressResponse
 * @typedef {import('express').Request} ExpressRequest
 */
const express = require('express');
const cors = require('cors');
const fs = require('fs/promises');
const fsSync = require('fs');
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
const staticCacheOptions = {
    maxAge: '1y',
    immutable: true,
    lastModified: true,
    etag: true
};

app.use('/model/bf16', express.static(BF16_DIR, staticCacheOptions));
app.use('/model/i7', express.static(I7_DIR, staticCacheOptions));

// --- FILE LISTING UTILS ---
/** @param {string} dir */
async function getDirListing(dir) {
    try {
        return await fs.readdir(dir);
    } catch (err) {
        return [];
    }
}

app.get('/api/files/bf16', async (req, res) => {
    res.json(await getDirListing(BF16_DIR));
});

app.get('/api/files/i7', async (req, res) => {
    res.json(await getDirListing(I7_DIR));
});

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

/**
 * Check if the projected model is fully extracted and ready.
 * Accessible at: GET /api/check
 */
app.get('/api/check', async (req, res) => {
    try {
        const files = await fs.readdir(I7_DIR);
        const hasCoreLayer = files.some(f => f.includes('model.embed_tokens'));
        res.json({ ready: hasCoreLayer });
    } catch {
        res.json({ ready: false });
    }
});

/**
 * Download the model and project it to Int7.
 */
app.get('/api/install', async (req, res) => {
    try {
        await downloadAndIndexModel("Qwen/Qwen2.5-Coder-1.5B-Instruct");
        await projectAllLayers();
        
        const files = await fs.readdir(I7_DIR);
        if (files.length > 0) {
            res.json({ status: 'success', count: files.length });
        } else {
            throw new Error("Extraction finished, but no files were created.");
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