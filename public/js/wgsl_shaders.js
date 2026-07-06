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
            seq_len: u32,
        }
        @group(0) @binding(0) var<storage, read> embed_matrix : array<u32>;
        @group(0) @binding(1) var<storage, read_write> output_matrix : array<f32>;
        @group(0) @binding(2) var<storage, read> tokens : array<u32>;
        @group(0) @binding(3) var<uniform> config : Config;
        @group(0) @binding(4) var<uniform> scale : f32;

        ${UNPACK_I8_FN}

        @compute @workgroup_size(64, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            let tok_idx = global_id.y;
            if (idx >= config.dim || tok_idx >= config.seq_len) { return; }
            
            let token_id = tokens[tok_idx];
            let u32_idx = (token_id * config.dim / 4u) + (idx / 4u);
            let val_u32 = embed_matrix[u32_idx];
            
            let unpacked = unpack4xi8(val_u32);
            let rem = idx % 4u;
            var w_val = 0.0;
            if (rem == 0u) { w_val = unpacked.x; }
            else if (rem == 1u) { w_val = unpacked.y; }
            else if (rem == 2u) { w_val = unpacked.z; }
            else { w_val = unpacked.w; }
            
            output_matrix[tok_idx * config.dim + idx] = w_val * scale;
        }
    `,

    RMSNorm: `
        struct Config {
            dim: u32,
            eps: f32,
            seq_len: u32,
        }
        @group(0) @binding(0) var<storage, read> input_matrix : array<f32>;
        @group(0) @binding(1) var<storage, read> weight_vec : array<u32>;
        @group(0) @binding(2) var<storage, read_write> output_matrix : array<f32>;
        @group(0) @binding(3) var<uniform> config : Config;
        @group(0) @binding(4) var<uniform> weight_scale : f32;

        ${UNPACK_I8_FN}

        var<workgroup> shared_sum: array<f32, 256>;

        @compute @workgroup_size(256, 1, 1)
        fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>) {
            let dim = config.dim;
            let tid = local_id.x;
            let tok_idx = group_id.y;
            if (tok_idx >= config.seq_len) { return; }
            
            let row_offset = tok_idx * dim;
            
            var local_sq_sum = 0.0;
            for (var i = tid; i < dim; i += 256u) {
                let val = input_matrix[row_offset + i];
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
                
                output_matrix[row_offset + i] = input_matrix[row_offset + i] * inv_rms * (w_val * weight_scale);
            }
        }
    `,

    MatMul: `
        struct Config {
            in_dim: u32,
            out_dim: u32,
            seq_len: u32,
        }
        @group(0) @binding(0) var<storage, read> input_matrix : array<f32>;
        @group(0) @binding(1) var<storage, read> weight_matrix : array<u32>;
        @group(0) @binding(2) var<storage, read> bias_vec : array<u32>;
        @group(0) @binding(3) var<storage, read_write> output_matrix : array<f32>;
        @group(0) @binding(4) var<uniform> config : Config;
        @group(0) @binding(5) var<uniform> scales : vec2<f32>; // x = weight_scale, y = bias_scale
        @group(0) @binding(6) var<uniform> has_bias : u32;

        ${UNPACK_I8_FN}

        @compute @workgroup_size(64, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let out_idx = global_id.x;
            if (out_idx >= config.out_dim) { return; }

            let in_dim = config.in_dim;
            let seq_len = config.seq_len;
            let row_start_u32 = (out_idx * in_dim) / 4u;
            let u32_count = in_dim / 4u;

            let CHUNK_SIZE = 32u;
            
            for (var t_start = 0u; t_start < seq_len; t_start += CHUNK_SIZE) {
                var sums : array<f32, 32>;
                for (var c = 0u; c < CHUNK_SIZE; c++) {
                    sums[c] = 0.0;
                }
                
                let cur_chunk = select(CHUNK_SIZE, seq_len - t_start, t_start + CHUNK_SIZE > seq_len);
                
                for (var i = 0u; i < u32_count; i++) {
                    let p_val = weight_matrix[row_start_u32 + i];
                    let unpacked = unpack4xi8(p_val);
                    
                    for (var c = 0u; c < CHUNK_SIZE; c++) {
                        if (c >= cur_chunk) { break; }
                        let tok_idx = t_start + c;
                        let in_base = tok_idx * in_dim + (i * 4u);
                        sums[c] += unpacked.x * input_matrix[in_base];
                        sums[c] += unpacked.y * input_matrix[in_base + 1u];
                        sums[c] += unpacked.z * input_matrix[in_base + 2u];
                        sums[c] += unpacked.w * input_matrix[in_base + 3u];
                    }
                }
                
                var bias = 0.0;
                if (has_bias == 1u) {
                    let b_idx_u32 = out_idx / 4u;
                    let rem = out_idx % 4u;
                    let b_val = bias_vec[b_idx_u32];
                    let b_unpacked = unpack4xi8(b_val);
                    if (rem == 0u) { bias = b_unpacked.x; }
                    else if (rem == 1u) { bias = b_unpacked.y; }
                    else if (rem == 2u) { bias = b_unpacked.z; }
                    else { bias = b_unpacked.w; }
                    bias *= scales.y;
                }

                for (var c = 0u; c < CHUNK_SIZE; c++) {
                    if (c >= cur_chunk) { break; }
                    let tok_idx = t_start + c;
                    output_matrix[tok_idx * config.out_dim + out_idx] = sums[c] * scales.x + bias;
                }
            }
        }
    `,

    Add: `
        struct Config { 
            dim: u32,
            seq_len: u32,
        }
        @group(0) @binding(0) var<storage, read_write> target_vec : array<f32>;
        @group(0) @binding(1) var<storage, read> add_vec : array<f32>;
        @group(0) @binding(2) var<uniform> config : Config;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x;
            if (i >= config.dim * config.seq_len) { return; }
            target_vec[i] += add_vec[i];
        }
    `,

    SiLU_Mul: `
        struct Config { 
            dim: u32,
            seq_len: u32,
        }
        @group(0) @binding(0) var<storage, read_write> gate_vec : array<f32>;
        @group(0) @binding(1) var<storage, read> up_vec : array<f32>;
        @group(0) @binding(2) var<uniform> config : Config;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x;
            if (i >= config.dim * config.seq_len) { return; }
            let x = gate_vec[i];
            let silu = x / (1.0 + exp(-x));
            gate_vec[i] = silu * up_vec[i];
        }
    `,

    RoPE: `
        struct Config {
            dim: u32,
            head_dim: u32,
            base_pos: u32,
            base: f32,
            seq_len: u32,
        }
        @group(0) @binding(0) var<storage, read_write> q_matrix : array<f32>;
        @group(0) @binding(1) var<storage, read_write> k_matrix : array<f32>;
        @group(0) @binding(2) var<uniform> config : Config;

        @compute @workgroup_size(64, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x;
            let tok_idx = global_id.y;
            if (i >= config.dim / 2u || tok_idx >= config.seq_len) { return; }

            let pos = config.base_pos + tok_idx;
            let head_dim = config.head_dim;
            let half_dim = head_dim / 2u;
            
            let head_idx = i / half_dim;
            let d = i % half_dim;
            
            let exponent = f32(d * 2u) / f32(head_dim);
            let freq = f32(pos) / pow(config.base, exponent);
            let cos_val = cos(freq);
            let sin_val = sin(freq);

            let q_row_offset = tok_idx * config.dim;
            let q_offset = q_row_offset + (head_idx * head_dim);
            let q0 = q_matrix[q_offset + d];
            let q1 = q_matrix[q_offset + d + half_dim];
            q_matrix[q_offset + d] = q0 * cos_val - q1 * sin_val;
            q_matrix[q_offset + d + half_dim] = q0 * sin_val + q1 * cos_val;

            // K matrix has 256 dim
            if (i < 128u) {
                let k_row_offset = tok_idx * 256u;
                let kv_head_idx = i / half_dim;
                let k_offset = k_row_offset + (kv_head_idx * head_dim);
                
                let k0 = k_matrix[k_offset + d];
                let k1 = k_matrix[k_offset + d + half_dim];
                k_matrix[k_offset + d] = k0 * cos_val - k1 * sin_val;
                k_matrix[k_offset + d + half_dim] = k0 * sin_val + k1 * cos_val;
            }
        }
    `,

    WriteCache: `
        struct Config {
            kv_dim: u32,
            base_pos: u32,
            seq_len: u32,
            max_seq_len: u32,
        }
        @group(0) @binding(0) var<storage, read> src : array<f32>;
        @group(0) @binding(1) var<storage, read_write> dst : array<f32>;
        @group(0) @binding(2) var<uniform> config : Config;
        
        @compute @workgroup_size(64, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x;
            let tok_idx = global_id.y;
            if (i >= config.kv_dim || tok_idx >= config.seq_len) { return; }
            
            let pos = (config.base_pos + tok_idx) % config.max_seq_len;
            let dst_offset = pos * config.kv_dim + i;
            let src_offset = tok_idx * config.kv_dim + i;
            dst[dst_offset] = src[src_offset];
        }
    `,

    Attention: `
        struct Config {
            num_heads: u32,
            kv_heads: u32,
            head_dim: u32,
            base_pos: u32,
            seq_len: u32,
        }
        @group(0) @binding(0) var<storage, read> q_matrix : array<f32>;
        @group(0) @binding(1) var<storage, read> k_cache : array<f32>;
        @group(0) @binding(2) var<storage, read> v_cache : array<f32>;
        @group(0) @binding(3) var<storage, read_write> out_matrix : array<f32>;
        @group(0) @binding(4) var<uniform> config : Config;

        @compute @workgroup_size(12, 1, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let head = global_id.x;
            let q_tok_idx = global_id.y;
            
            if (head >= config.num_heads || q_tok_idx >= config.seq_len) { return; }
            
            let head_dim = config.head_dim;
            let kv_heads = config.kv_heads;
            let base_pos = config.base_pos;
            let kv_head = head / (config.num_heads / kv_heads);
            let maxSeqLen = 512u;
            
            var scores : array<f32, 512>;
            let q_row_offset = q_tok_idx * config.num_heads * head_dim;
            let q_offset = q_row_offset + (head * head_dim);
            let scale = 1.0 / sqrt(f32(head_dim));
            
            var max_score = -1e20;
            let abs_pos = base_pos + q_tok_idx;
            let context_len = select(abs_pos + 1u, maxSeqLen, abs_pos >= maxSeqLen);
            
            for (var t_idx = 0u; t_idx < context_len; t_idx++) {
                var start_T = 0u;
                if (abs_pos >= maxSeqLen) {
                    start_T = abs_pos - maxSeqLen + 1u;
                }
                let T = start_T + t_idx;
                let cache_pos = T % maxSeqLen;
                
                var dot = 0.0;
                let k_offset = cache_pos * kv_heads * head_dim + kv_head * head_dim;
                for (var i = 0u; i < head_dim; i++) {
                    dot += q_matrix[q_offset + i] * k_cache[k_offset + i];
                }
                let score = dot * scale;
                scores[t_idx] = score;
                if (score > max_score) { max_score = score; }
            }
            
            var sum_exp = 0.0;
            for (var t_idx = 0u; t_idx < context_len; t_idx++) {
                let exp_val = exp(scores[t_idx] - max_score);
                scores[t_idx] = exp_val;
                sum_exp += exp_val;
            }
            
            let out_row_offset = q_tok_idx * config.num_heads * head_dim;
            let out_offset = out_row_offset + (head * head_dim);
            
            for (var i = 0u; i < head_dim; i++) {
                var val = 0.0;
                var start_T = 0u;
                if (abs_pos >= maxSeqLen) {
                    start_T = abs_pos - maxSeqLen + 1u;
                }
                for (var t_idx = 0u; t_idx < context_len; t_idx++) {
                    let T = start_T + t_idx;
                    let cache_pos = T % maxSeqLen;
                    let v_offset = cache_pos * kv_heads * head_dim + kv_head * head_dim;
                    val += (scores[t_idx] / sum_exp) * v_cache[v_offset + i];
                }
                out_matrix[out_offset + i] = val;
            }
        }
    `
};
