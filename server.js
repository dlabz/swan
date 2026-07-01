require('express-async-errors');
const express = require('express');
const fs = require('fs/promises');
const path = require('path');
const { fetchAndProject, I7_DIR } = require('./manifoldController');

const app = express();

// Ensure the directory exists to prevent "root path required" error
if (!I7_DIR) {
    console.error("Critical: I7_DIR is undefined!");
    process.exit(1);
}

app.use(express.static('public'));
app.use('/model/i7', express.static(I7_DIR));

app.get('/api/check', async (req, res) => {
    try {
        await fs.access(path.join(I7_DIR, 'projected_model.safetensors'));
        res.json({ ready: true });
    } catch {
        res.json({ ready: false });
    }
});

app.get('/api/install', async (req, res) => {
    try {
        await fetchAndProject(); 
        res.json({ status: 'READY' });
    } catch (e) {
        res.status(500).json({ status: 'ERROR', message: e.message });
    }
});

app.listen(3000, () => console.log('Swan Engine active on port 3000'));
