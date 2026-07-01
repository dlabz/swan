const express = require('express');
const fs = require('fs/promises');
const path = require('path');
const { processAllLayers, I7_DIR } = require('./manifoldController');

const app = express();
app.use(express.static('public'));

// Serve projected model files
app.use('/model/i7', express.static(I7_DIR));

// Check if ALL projected files exist
app.get('/api/check', async (req, res) => {
    try {
        const files = await fs.readdir(I7_DIR);
        // Ensure at least the lm_head and some layers exist
        const hasLmHead = files.includes('projected_lm_head.bin');
        res.json({ ready: hasLmHead });
    } catch {
        res.json({ ready: false });
    }
});

// Full installation: triggers batch projection of all layers
app.get('/api/install', async (req, res) => {
    try {
        console.log('Starting full model manifold projection...');
        // processAllLayers() handles the sequence defined in MANIFEST
        await processAllLayers(); 
        res.json({ status: 'READY', message: 'All 28 layers projected' });
    } catch (e) {
        console.error('Installation error:', e);
        res.status(500).json({ status: 'ERROR', message: e.message });
    }
});

app.listen(3000, () => console.log('Swan Engine active on port 3000'));