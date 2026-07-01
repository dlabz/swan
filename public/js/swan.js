// @ts-check
import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1/dist/transformers.min.js';

export const SwanStorage = {
    async getModelFile(filename, url) {
        const root = await navigator.storage.getDirectory();
        try {
            const handle = await root.getFileHandle(filename);
            return await handle.getFile();
        } catch {
            const res = await fetch(url);
            const blob = await res.blob();
            const handle = await root.getFileHandle(filename, { create: true });
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
            return await handle.getFile();
        }
    }
};

export const SwanPipeline = {
    async ensureReady(ui) {
        ui.log("Checking pipeline status...");
        const res = await fetch('/api/check');
        const { ready } = await res.json();
        
        if (!ready) {
            ui.log("Model assets missing. Installing (this may take a moment)...", "system");
            const installRes = await fetch('/api/install');
            if (!installRes.ok) throw new Error("Installation failed on server.");
            ui.log("Assets verified.", "system");
        } else {
            ui.log("Model assets verified.", "system");
        }
    }
};

export const SwanUI = {
    el: document.getElementById('terminal'),
    log(msg, type = 'log') {
        const color = type === 'error' ? 'text-red-500' : type === 'system' ? 'text-blue-400' : 'text-green-400';
        this.el.innerHTML += `<div class="${color} font-mono text-xs">> ${msg}</div>`;
        this.el.scrollTop = this.el.scrollHeight;
    }
};