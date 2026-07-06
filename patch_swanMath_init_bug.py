with open('public/js/swanMath.js', 'r') as f:
    code = f.read()

# Replace loadAllWeights
old_load_weights = """        // Dummy bias buffer for projections that don't have biases
        this.dummyBias = await this.uploadBuffer(new Uint32Array([0]), GPUBufferUsage.STORAGE);
        this.ui.log("All weights loaded successfully.", "system");
    }"""

new_load_weights = """        // Dummy bias buffer for projections that don't have biases
        this.dummyBias = await this.uploadBuffer(new Uint32Array([0]), GPUBufferUsage.STORAGE);

        // Constant Scale Buffers
        const createScaleBuf = (val) => {
            const buf = this.device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(buf, 0, new Float32Array([val]));
            return buf;
        };

        this.embedScaleBuf = createScaleBuf(this.weights['model.embed_tokens.weight'].scale);
        this.wNormFinalScaleBuf = createScaleBuf(this.weights['model.norm.weight'].scale);

        this.wNormScaleBuf = [];
        this.wNorm2ScaleBuf = [];
        for (let l = 0; l < this.LAYERS; l++) {
            this.wNormScaleBuf.push(createScaleBuf(this.weights[`model.layers.${l}.input_layernorm.weight`].scale));
            this.wNorm2ScaleBuf.push(createScaleBuf(this.weights[`model.layers.${l}.post_attention_layernorm.weight`].scale));
        }

        this.ui.log("All weights loaded successfully.", "system");
    }"""

code = code.replace(old_load_weights, new_load_weights)

# Replace the Scale Buffers allocation block in allocateBuffers
old_scale_allocation = """        // Constant Scale Buffers
        const createScaleBuf = (val) => {
            const buf = this.device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(buf, 0, new Float32Array([val]));
            return buf;
        };

        this.embedScaleBuf = createScaleBuf(this.weights['model.embed_tokens.weight'].scale);
        this.wNormFinalScaleBuf = createScaleBuf(this.weights['model.norm.weight'].scale);

        this.wNormScaleBuf = [];
        this.wNorm2ScaleBuf = [];
        for (let l = 0; l < this.LAYERS; l++) {
            this.wNormScaleBuf.push(createScaleBuf(this.weights[`model.layers.${l}.input_layernorm.weight`].scale));
            this.wNorm2ScaleBuf.push(createScaleBuf(this.weights[`model.layers.${l}.post_attention_layernorm.weight`].scale));
        }"""

new_scale_allocation = """        this.embedScaleBuf = null;
        this.wNormFinalScaleBuf = null;
        this.wNormScaleBuf = [];
        this.wNorm2ScaleBuf = [];"""

code = code.replace(old_scale_allocation, new_scale_allocation)

with open('public/js/swanMath.js', 'w') as f:
    f.write(code)

print("swanMath.js init bug fixed!")
