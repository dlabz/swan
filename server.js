const express = require('express');
const fs = require('fs/promises');
const path = require('path');
const { fetchAndProject, I7_DIR } = require('./manifoldController');

const app = express();
app.use(express.static('public'));

// IMPORTANT: Point static to the exact folder your files are in
app.use('/model/i7', express.static(path.resolve(__dirname, 'data/model/i7')));

app.get('/api/check', async (req, res) => {
    try {
        await fs.access(path.resolve(__dirname, 'data/model/i7/projected_config.json'));
        res.json({ ready: true });
    } catch {
        res.json({ ready: false });
    }
});

app.get('/api/install', async (req, res) => {
    try {
        await fetchAndProject("config.json"); 
        res.json({ status: 'READY' });
    } catch (e) {
        res.status(500).json({ status: 'ERROR', message: e.message });
    }
});

app.listen(3000, () => console.log('Swan Engine active on port 3000'));
