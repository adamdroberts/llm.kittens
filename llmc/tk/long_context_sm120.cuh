/*
long_context_sm120.cuh — ThunderKittens KV-management kernels for SM120.

  - h2o_select         pick top-k positions by accumulated score
  - h2o_gather         gather selected K, V rows
  - snapkv_score       compute prompt-aware scores for SnapKV
  - landmark_select    select landmark + tail KV positions
  - infini_update      update Σ k v^T memory state
  - infini_retrieve    retrieve from memory state
  - sink_select        select first-n_sink + last-window positions
*/
#pragma once

#include "tk_common.cuh"

namespace llmk::long_context_sm120 {

using namespace ::kittens;

__global__ void h2o_select_kernel(int* keep_idx, const float* scores, int S, int k_keep) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    constexpr int MAX_KEEP = 4096;
    if (k_keep > MAX_KEEP) return;
    float ts[MAX_KEEP]; int ti[MAX_KEEP];
    for (int i = 0; i < k_keep; ++i) { ts[i] = -INFINITY; ti[i] = 0; }
    for (int p = 0; p < S; ++p) {
        float v = scores[p];
        if (v > ts[k_keep - 1]) {
            int pos = k_keep - 1;
            while (pos > 0 && ts[pos - 1] < v) { ts[pos] = ts[pos - 1]; ti[pos] = ti[pos - 1]; --pos; }
            ts[pos] = v; ti[pos] = p;
        }
    }
    for (int i = 1; i < k_keep; ++i) {
        int v = ti[i]; int j = i;
        while (j > 0 && ti[j - 1] > v) { ti[j] = ti[j - 1]; --j; }
        ti[j] = v;
    }
    for (int i = 0; i < k_keep; ++i) keep_idx[i] = ti[i];
}

__global__ void gather_kv_kernel(bf16* K_out, bf16* V_out,
                                 const bf16* K_in, const bf16* V_in,
                                 const int* keep_idx, int k_keep, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (d >= D || t >= k_keep) return;
    int src = keep_idx[t];
    K_out[t * D + d] = K_in[src * D + d];
    V_out[t * D + d] = V_in[src * D + d];
}

inline void launch_h2o(bf16* K_out, bf16* V_out, int* keep_idx,
                       const bf16* K_in, const bf16* V_in, const float* scores,
                       int S, int k_keep, int D, cudaStream_t stream) {
    h2o_select_kernel<<<1, 1, 0, stream>>>(keep_idx, scores, S, k_keep);
    const int bs = 128;
    dim3 grid(CEIL_DIV(D, bs), k_keep);
    gather_kv_kernel<<<grid, bs, 0, stream>>>(K_out, V_out, K_in, V_in, keep_idx, k_keep, D);
    cudaCheck(cudaGetLastError());
}

__global__ void snapkv_score_kernel(float* scores, const bf16* obs_q, const bf16* K_in,
                                    int W, int S, int D) {
    int t = blockIdx.x;
    if (t >= S || threadIdx.x != 0) return;
    float acc = 0.0f;
    for (int w = 0; w < W; ++w) {
        float dot = 0.0f;
        const bf16* qr = obs_q + w * D;
        const bf16* kr = K_in + t * D;
        for (int d = 0; d < D; ++d) dot += __bfloat162float(qr[d]) * __bfloat162float(kr[d]);
        acc += dot;
    }
    scores[t] = acc / (float)W;
}
inline void launch_snapkv(bf16* K_out, bf16* V_out, int* keep_idx,
                          const bf16* obs_q, const bf16* K_in, const bf16* V_in,
                          int W, int S, int k_keep, int D,
                          float* scratch_scores, cudaStream_t stream) {
    snapkv_score_kernel<<<S, 1, 0, stream>>>(scratch_scores, obs_q, K_in, W, S, D);
    launch_h2o(K_out, V_out, keep_idx, K_in, V_in, scratch_scores, S, k_keep, D, stream);
}

__global__ void landmark_select_kernel(int* keep_idx, int S, int span, int tail) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_landmarks = S / span;
    int total = n_landmarks + tail;
    if (i >= total) return;
    if (i < n_landmarks) {
        keep_idx[i] = (i + 1) * span - 1;
    } else {
        int t = i - n_landmarks;
        keep_idx[i] = S - tail + t;
    }
}
inline void launch_landmark(bf16* K_out, bf16* V_out, int* keep_idx,
                            const bf16* K_in, const bf16* V_in,
                            int S, int D, int span, int tail, cudaStream_t stream) {
    int n_landmarks = S / span;
    int k_out = n_landmarks + tail;
    landmark_select_kernel<<<CEIL_DIV(k_out, 64), 64, 0, stream>>>(keep_idx, S, span, tail);
    const int bs = 128;
    dim3 grid(CEIL_DIV(D, bs), k_out);
    gather_kv_kernel<<<grid, bs, 0, stream>>>(K_out, V_out, K_in, V_in, keep_idx, k_out, D);
    cudaCheck(cudaGetLastError());
}

// Infini-attention memory update.
__global__ void infini_update_kernel(bf16* mem, bf16* z, const bf16* K_seg, const bf16* V_seg, int S, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= D || j >= D) return;
    float acc = __bfloat162float(mem[i * D + j]);
    for (int t = 0; t < S; ++t) {
        acc += __bfloat162float(K_seg[t * D + i]) * __bfloat162float(V_seg[t * D + j]);
    }
    mem[i * D + j] = __float2bfloat16(acc);
    if (j == 0) {
        float zacc = __bfloat162float(z[i]);
        for (int t = 0; t < S; ++t) zacc += __bfloat162float(K_seg[t * D + i]);
        z[i] = __float2bfloat16(zacc);
    }
}
inline void launch_infini_update(bf16* mem, bf16* z, const bf16* K_seg, const bf16* V_seg,
                                 int S, int D, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(D, 16), CEIL_DIV(D, 16));
    infini_update_kernel<<<grid, block, 0, stream>>>(mem, z, K_seg, V_seg, S, D);
    cudaCheck(cudaGetLastError());
}

__global__ void infini_retrieve_kernel(bf16* y, const bf16* q, const bf16* mem, const bf16* z, int D) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= D) return;
    float qz = 0.0f;
    for (int i = 0; i < D; ++i) qz += __bfloat162float(q[i]) * __bfloat162float(z[i]);
    float qm = 0.0f;
    for (int i = 0; i < D; ++i) qm += __bfloat162float(q[i]) * __bfloat162float(mem[i * D + o]);
    y[o] = __float2bfloat16(qm / (qz + 1e-6f));
}
inline void launch_infini_retrieve(bf16* y, const bf16* q, const bf16* mem, const bf16* z, int D,
                                   cudaStream_t stream) {
    const int bs = 64;
    infini_retrieve_kernel<<<CEIL_DIV(D, bs), bs, 0, stream>>>(y, q, mem, z, D);
    cudaCheck(cudaGetLastError());
}

__global__ void sink_select_kernel(int* keep_idx, int S, int n_sink, int window) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_sink + window;
    if (i >= total) return;
    if (i < n_sink) keep_idx[i] = i;
    else            keep_idx[i] = S - window + (i - n_sink);
}
inline void launch_sink(bf16* K_out, bf16* V_out, int* keep_idx,
                        const bf16* K_in, const bf16* V_in,
                        int S, int D, int n_sink, int window, cudaStream_t stream) {
    int k_out = n_sink + window;
    sink_select_kernel<<<CEIL_DIV(k_out, 64), 64, 0, stream>>>(keep_idx, S, n_sink, window);
    const int bs = 128;
    dim3 grid(CEIL_DIV(D, bs), k_out);
    gather_kv_kernel<<<grid, bs, 0, stream>>>(K_out, V_out, K_in, V_in, keep_idx, k_out, D);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::long_context_sm120
