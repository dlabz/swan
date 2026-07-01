<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SWAN | Resilient Engine</title>
    <script src="https://cdn.jsdelivr.net/npm/eruda"></script>
    <script>eruda.init();</script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-black text-green-400 font-mono p-4 flex flex-col h-screen">
    <div class="max-w-3xl mx-auto w-full flex flex-col h-full">
        <header class="mb-4 border-b border-green-800 pb-4 shrink-0 flex justify-between items-end">
            <div>
                <h1 class="text-2xl font-bold tracking-widest text-white">SWAN INFERENCE</h1>
                <div id="status" class="text-xs text-yellow-500 font-bold">System Online</div>
            </div>
            <button onclick="SwanStorage.clearCache()" class="text-xs border border-red-800 text-red-500 hover:bg-red-900 px-3 py-1 transition">Wipe Cache</button>
        </header>

        <div id="terminal" class="bg-gray-900 p-4 rounded border border-green-800 mb-4 flex-1 overflow-y-auto text-xs whitespace-pre-wrap"></div>

        <div id="controls" class="hidden flex gap-2 shrink-0">
            <input type="text" id="prompt" class="flex-1 bg-gray-900 border border-green-800 p-3 text-white outline-none" placeholder="Enter query...">
            <button onclick="SwanCore.runInference()" class="bg-green-900 px-6 py-2 hover:bg-green-700 text-white font-bold transition" id="sendBtn">SEND</button>
        </div>
    </div>

    <script type="module">
        import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1/dist/transformers.min.js';

        const Config = { HIDDEN_SIZE: 1536, BF16_BYTES: 2 };

        const SwanUI = {
            terminal: document.getElementById('terminal'),
            status: document.getElementById('status'),
            controls: document.getElementById('controls'),
            btn: document.getElementById('sendBtn'),
            log(msg, type = 'log') {
                const color = type === 'error' ? 'text-red-500' : type === 'system' ? 'text-blue-400' : 'text-green-500';
                this.terminal.innerHTML += `<div class="${color}">> ${msg}</div>`;
                this.terminal.scrollTop = this.terminal.scrollHeight;
            },
            setReady() { this.controls.classList.remove('hidden'); }
        };

        const SwanStorage = {
            async getModelFile(filename, url) {
                const opfsRoot = await navigator.storage.getDirectory();
                try {
                    const handle = await opfsRoot.getFileHandle(filename);
                    return await handle.getFile();
                } catch {
                    SwanUI.log(`Fetching ${filename}...`, 'system');
                    const res = await fetch(url);
                    const blob = await res.blob();
                    const handle = await opfsRoot.getFileHandle(filename, { create: true });
                    const writable = await handle.createWritable();
                    await writable.write(blob);
                    await writable.close();
                    return await handle.getFile();
                }
            },
            async clearCache() {
                const root = await navigator.storage.getDirectory();
                // @ts-ignore
                for await (const name of root.keys()) await root.removeEntry(name, { recursive: true });
                window.location.reload();
            }
        };

        const SwanMath = {
            decodeBfloat16(bf16Bytes) {
                const f32 = new Float32Array(bf16Bytes.length / 2);
                const view = new DataView(bf16Bytes.buffer);
                for (let i = 0; i < f32.length; i++) {
                    // Shift Bfloat16 bytes into upper 16 bits of F32
                    view.setUint32(i * 4, view.getUint16(i * 2, true) << 16, true);
                }
                return f32;
            },
            rmsnorm(x, weight) {
                let ss = 0;
                for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
                let inv = 1.0 / Math.sqrt(ss / x.length + 1e-6);
                for (let i = 0; i < x.length; i++) x[i] *= inv * weight[i];
                return x;
            }
        };

        const SwanCore = {
            async init() {
                SwanCore.tokenizer = await AutoTokenizer.from_pretrained('Xenova/Qwen1.5-0.5B');
                SwanUI.setReady();
                SwanUI.log("Engine Ready.", "system");
            },
            async runInference() {
                try {
                    const input = document.getElementById('prompt').value;
                    const tokens = SwanCore.tokenizer.encode(input);
                    const embedFile = await SwanStorage.getModelFile('embed_tokens.bin', '/model/i7/embed_tokens.bin');
                    
                    const slice = embedFile.slice(tokens[0] * Config.HIDDEN_SIZE * 2, (tokens[0] + 1) * Config.HIDDEN_SIZE * 2);
                    let hiddenState = SwanMath.decodeBfloat16(new Uint8Array(await slice.arrayBuffer()));

                    const normFile = await SwanStorage.getModelFile('input_layernorm.weight.bin', '/model/i7/input_layernorm.weight.bin');
                    const normWeight = SwanMath.decodeBfloat16(new Uint8Array(await normFile.arrayBuffer()));
                    
                    hiddenState = SwanMath.rmsnorm(hiddenState, normWeight);
                    
                    SwanUI.log(`Output generated from ${hiddenState.length} dim state.`, 'system');
                    // Egress logic goes here...
                } catch(e) { SwanUI.log(e.message, 'error'); }
            }
        };
        window.SwanCore = SwanCore;
        setTimeout(SwanCore.init, 500);
    </script>
</body>
</html>