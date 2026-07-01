// @ts-check

/** * MODULE: SwanStorage
 * Handles OPFS binary persistence.
 */
export const SwanStorage = {
    async getModelFile(filename, url) {
        const root = await navigator.storage.getDirectory();
        try {
            const handle = await root.getFileHandle(filename);
            return await handle.getFile();
        } catch {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status} on ${url}`);
            const blob = await res.blob();
            const handle = await root.getFileHandle(filename, { create: true });
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
            return await handle.getFile();
        }
    },
    async wipe() {
        const root = await navigator.storage.getDirectory();
        for await (const name of root.keys()) await root.removeEntry(name, { recursive: true });
    }
};

/** * MODULE: SwanMath
 * High-performance BFloat16/FP32 operations.
 */
export const SwanMath = {
    /** @param {Uint8Array} buf */
    decodeBfloat16(buf) {
        const f32 = new Float32Array(buf.length / 2);
        const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
        for (let i = 0; i < f32.length; i++) {
            // BFloat16 is top 16 bits of FP32
            view.setUint32(i * 4, view.getUint16(i * 2, true) << 16, true);
        }
        return f32;
    }
};

/** * MODULE: SwanUI
 * Minimalistic terminal interface.
 */
export const SwanUI = {
    el: document.getElementById('terminal'),
    log(msg, type = 'log') {
        const color = type === 'error' ? 'text-red-500' : 'text-green-400';
        this.el.innerHTML += `<div class="${color} font-mono text-xs">> ${msg}</div>`;
        this.el.scrollTop = this.el.scrollHeight;
    }
};