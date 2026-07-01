require('express-async-errors');
const express = require('express');
const fs = require('fs/promises');
const path = require('path');
const { fetchAndProject, I7_DIR } = require('./manifoldController');

const app = express();
app.use(express.static('public'));

// Serve the projected manifold files directly to the front-end
app.use('/model/i7', express.static(I7_DIR));

app.get('/api/check', async (req, res) => {
    const exists = await fs.access(path.join(I7_DIR, 'weights_000.bin'))
        .then(() => true).catch(() => false);
    res.json({ ready: exists });
});

app.get('/api/install', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
        res.write(`data: ${JSON.stringify({ status: 'FETCHING_FROM_HF' })}\n\n`);
        await fetchAndProject('weights_000.bin');
        
        res.write(`data: ${JSON.stringify({ status: 'READY' })}\n\n`);
        res.end();
    } catch (e) {
        res.write(`data: ${JSON.stringify({ status: 'ERROR', message: e.message })}\n\n`);
        res.end();
    }
});

app.listen(3000, () => console.log('Swan Engine v3.0 active on port 3000'));
