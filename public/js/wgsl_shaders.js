// Helper WGSL snippet: unpack 4 raw signed i8 values from a u32, without the
// /127 normalization that unpack4x8snorm applies. Correct dequant: int8 * scale.
const UNPACK_I8_FN = `
    fn unpack4xi8(v: u32) -> vec4<f32> {
        let b0 = i32(v & 0xFFu);
        let b1 = i32((v >> 8u) & 0xFFu);
        let b2 = i32((v >> 16u) & 0xFFu);
        let b3 = i32((v >> 24u) & 0xFFu);
        // Sign-extend from 8 bits
        let s0 = select(b0, b0 - 256, b0 >= 128);
        let s1 = select(b1, b1 - 256, b1 >= 128);
        let s2 = select(b2, b2 - 256, b2 >= 128);
        let s3 = select(b3, b3 - 256, b3 >= 128);
        return vec4<f32>(f32(s0), f32(s1), f32(s2), f32(s3));
    }
`;

export const WGSL = {
    EmbedLookup: `
        struct Config {
            dim: u32,
            token_id: u32,
        }
        @group(0) @binding(0) var<storage, read> embed_matrix : array<u32>;
        @group(0) @binding(1) var<storage, read_write> output_vec : array<f32>;
        @group(0) @binding(2) var<uniform> config : Config;
        @group(0) @binding(3) var<uniform> scale : f32;

        ${UNPACK_I8_FN}

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx >= config.dim) { return; }
            
            let u32_idx = (config.token_id * config.dim / 4u) + (idx / 4u);
            let val_u32 = embed_matrix[u32_idx];
            
            let unpacked = unpack4xi8(val_u32);
            let rem = idx % 4u;
            var w_val = 0.0;
            if (rem == 0u) { w_val = unpacked.x; }
            else if (rem == 1u) { w_val = unpacked.y; }
            else if (rem == 2u) { w_val = unpacked.z; }
            else { w_val = unpacked.w; }
            
            output_vec[idx] = w_val * scale;
        }
    `,

    RMSNorm: `
        struct Config {
            dim: u32,
            eps: f32,
        }
        @group(0) @binding(0) var<storage, read> input_vec : array<f32>;
        @group(0) @binding(1) var<storage, read> weight_vec : array<u32>;
        @group(0) @binding(2) var<storage, read_write> output_vec : array<f32>;
        @group(0) @binding(3) var<uniform> config : Config;
        @group(0) @binding(4) var<uniform> weight_scale : f32;

        ${UNPACK_I8_FN}

        var<workgroup> shared_sum: array<f32, 256>;

        @compute @workgroup_size(256)
        fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
            let dim = config.dim;
            let tid = local_id.x;
            
            var local_sq_sum = 0.0;
            for (var i = tid; i < dim; i += 256u) {
                let val = input_vec[i];
                local_sq_sum += val * val;
            }
            shared_sum[tid] = local_sq_sum;
            workgroupBarrier();

            for (var offset = 128u; offset > 0u; offset /= 2u) {
                if (tid < offset) {
                    shared_sum[tid] += shared_sum[tid + offset];
                }
                workgroupBarrier();
            }

            let inv_rms = 1.0 / sqrt((shared_sum[0] / f32(dim)) + config.eps);

            for (var i = tid; i < dim; i += 256u) {
                let p_val = weight_vec[i / 4u];
                let unpacked = unpack4xi8(p_val);
                var w_val = 0.0;
                let rem = i % 4u;
                if (rem == 0u) { w_val = unpacked.x; }
                else if (rem == 1u) { w_val = unpacked.y; }
                else if (rem == 2u) { w_val = unpacked.z; }
                else { w_val = unpacked.w; }
                
                output_vec[i] = input_vec[i] * inv_rms * (w_val * weight_scale);
            }
        }
    `,

    MatMul: `
        struct Config {
            in_dim: u32,
            out_dim: u32,
        }
        @group(0) @binding(0) var<storage, read> input_vec : array<f32>;
        @group(0) @binding(1) var<storage, read> weight_matrix : array<u32>;
        @group(0) @binding(2) var<storage, read> bias_vec : array<u32>;
        @group(0) @binding(3) var<storage, read_write> output_vec : array<f32>;
        @group(0) @binding(4) var<uniform> config : Config;
        @group(0) @binding(5) var<uniform> scales : vec2<f32>; // x = weight_scale, y = bias_scale
        @group(0) @binding(6) var<uniform> has_bias : u32;

        ${UNPACK_I8_FN}

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let out_idx = global_id.x;
            if (out_idx >= config.out_dim) { return; }

            var sum = 0.0;
            let in_dim = config.in_dim;
            let row_start_u32 = (out_idx * in_dim) / 4u;
            let u32_count = in_dim / 4u;

            for (var i = 0u; i < u32_count; i++) {
                let p_val = weight_matrix[row_start_u32 + i];
                let unpacked = unpack4xi8(p_val);
                
                let in_base = i * 4u;
                sum += unpacked.x * input_vec[in_base];
                sum += unpacked.y * input_vec[in_base + 1u];
                sum += unpacked.z * input_vec[in_base + 2u];
                sum += unpacked.w * input_vec[in_base + 3u];
            }
            
            sum *= scales.x;

            if (has_bias == 1u) {
                let b_idx_u32 = out_idx / 4u;
                let rem = out_idx % 4u;
                let b_val = bias_vec[b_idx_u32];
                let b_unpacked = unpack4xi8(b_val);
                var bias = 0.0;
                if (rem == 0u) { bias = b_unpacked.x; }
                else if (rem == 1u) { bias = b_unpacked.y; }
                else if (rem == 2u) { bias = b_unpacked.z; }
                else { bias = b_unpacked.w; }
                sum += bias * scales.y;
            }

            output_vec[out_idx] = sum;
        }
    `,

    Add: `
        struct Config { dim: u32 }
        @group(0) @binding(0) var<storage, read_write> target_vec : array<f32>;
        @group(0) @binding(1) var<storage, read> add_vec : array<f32>;
        @group(0) @binding(2) var<uniform> config : Config;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x;
            if (i >= config.dim) { return; }
            target_vec[i] += add_vec[i];
        }
    `,

    SiLU_Mul: `
        struct Config { dim: u32 }
        @group(0) @binding(0) var<storage, read_write> gate_vec : array<f32>;
        @group(0) @binding(1) var<storage, read> up_vec : array<f32>;
        @group(0) @binding(2) var<uniform> config : Config;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x;
            if (i >= config.dim) { return; }
            let x = gate_vec[i];
            let silu = x / (1.0 + exp(-x));
            gate_vec[i] = silu * up_vec[i];
        }
    `,

    RoPE: `
        struct Config {
            dim: u32,
            head_dim: u32,
            pos: u32,
            base: f32,
        }
        @group(0) @binding(0) var<storage, read_write> q_vec : array<f32>;
        @group(0) @binding(1) var<storage, read_write> k_vec : array<f32>;
        @group(0) @binding(2) var<uniform> config : Config;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x * 2u;
            if (i >= config.dim) { return; }

            let head_dim = config.head_dim;
            let head_idx = i % head_dim;
            
            let freq = f32(config.pos) / pow(config.base, f32(head_idx) / f32(head_dim));
            let cos_val = cos(freq);
            let sin_val = sin(freq);

            let q0 = q_vec[i];
            let q1 = q_vec[i + 1u];
            q_vec[i] = q0 * cos_val - q1 * sin_val;
            q_vec[i + 1u] = q0 * sin_val + q1 * cos_val;

            if (i < 256u) {
                let k0 = k_vec[i];
                let k1 = k_vec[i + 1u];
                k_vec[i] = k0 * cos_val - k1 * sin_val;
                k_vec[i + 1u] = k0 * sin_val + k1 * cos_val;
            }
        }
    `,

    Attention: `
        struct Config {
            num_heads: u32,
            kv_heads: u32,
            head_dim: u32,
            pos: u32,
        }
        @group(0) @binding(0) var<storage, read> q_vec : array<f32>;
        @group(0) @binding(1) var<storage, read_write> k_cache : array<f32>;
        @group(0) @binding(2) var<storage, read_write> v_cache : array<f32>;
        @group(0) @binding(3) var<storage, read_write> out_vec : array<f32>;
        @group(0) @binding(4) var<storage, read> k_new : array<f32>;
        @group(0) @binding(5) var<storage, read> v_new : array<f32>;
        @group(0) @binding(6) var<uniform> config : Config;

        @compute @workgroup_size(12)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let head = global_id.x;
            let isActive = head < config.num_heads;
            
            let head_dim = config.head_dim;
            let kv_heads = config.kv_heads;
            let pos = config.pos;
            let kv_head = head / (config.num_heads / kv_heads);
            
            if (isActive && head < kv_heads) {
                let cache_offset = pos * kv_heads * head_dim + head * head_dim;
                let new_offset = head * head_dim;
                for (var i = 0u; i < head_dim; i++) {
                    k_cache[cache_offset + i] = k_new[new_offset + i];
                    v_cache[cache_offset + i] = v_new[new_offset + i];
                }
            }
            
            workgroupBarrier();
            
            if (isActive) {
                var scores : array<f32, 512>;
                if (pos < 512u) {
                    let q_offset = head * head_dim;
                    let scale = 1.0 / sqrt(f32(head_dim));
                    
                    var max_score = -1e20;
                    for (var t = 0u; t <= pos; t++) {
                        var dot = 0.0;
                        let k_offset = t * kv_heads * head_dim + kv_head * head_dim;
                        for (var i = 0u; i < head_dim; i++) {
                            dot += q_vec[q_offset + i] * k_cache[k_offset + i];
                        }
                        let score = dot * scale;
                        scores[t] = score;
                        if (score > max_score) {
                            max_score = score;
                        }
                    }
                    
                    var sum_exp = 0.0;
                    for (var t = 0u; t <= pos; t++) {
                        let exp_val = exp(scores[t] - max_score);
                        scores[t] = exp_val;
                        sum_exp += exp_val;
                    }
                    
                    let out_offset = head * head_dim;
                    for (var i = 0u; i < head_dim; i++) {
                        var val = 0.0;
                        for (var t = 0u; t <= pos; t++) {
                            let v_offset = t * kv_heads * head_dim + kv_head * head_dim;
                            val += (scores[t] / sum_exp) * v_cache[v_offset + i];
                        }
                        out_vec[out_offset + i] = val;
                    }
                }
            }
        }
    `
};
