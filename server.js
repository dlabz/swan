
const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();

app.use(express.static('public'));

app.get('/model/:type/:filename', (req, res) => {
    const { type, filename } = req.params;
    const filePath = path.join(__dirname, 'data', 'model', type, filename);
    const range = req.headers.range;

    if (!fs.existsSync(filePath)) return res.status(404).send("File not found");

    const fileSize = fs.statSync(filePath).size;
    const start = Number(range.replace(/\D/g, ""));
    const end = Math.min(start + (512 * 1024), fileSize - 1);
    
    res.writeHead(206, {
        "Content-Range": `bytes ${start}-${end}/${fileSize}`,
        "Content-Length": end - start + 1,
        "Content-Type": "application/octet-stream"
    });

    fs.createReadStream(filePath, { start, end }).pipe(res);
});

app.listen(3000, () => console.log('Swan Engine active on port 3000'));
