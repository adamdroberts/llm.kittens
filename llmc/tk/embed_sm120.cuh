/*
embed_sm120.cuh — ThunderKittens embedding lookups for SM120.

  - token_embedding_forward    (out[b,s,:] = wte[tokens[b,s], :])
  - token_embedding_backward   (scatter-add via packed-pair CAS)
  - abs_pos_embedding_forward  (out[b,s,:] = wpe[s,:])
  - small_embed_lookup         (generic small table)
  - patch_flatten              (Conv2d output → [B, H*W, C])
*/
#pragma once

#include "tk_common.cuh"

namespace llmk::embed_sm120 {

using namespace ::kittens;

// Token embedding forward.
__global__ void token_embed_fwd_kernel(bf16* out, const int* tokens, const bf16* wte,
                                       int B, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int b = blockIdx.z;
    if (d >= D) return;
    int t = tokens[b * S + s];
    out[((b * S) + s) * D + d] = wte[t * D + d];
}
inline void launch_token_fwd(bf16* out, const int* tokens, const bf16* wte,
                             int B, int S, int D, cudaStream_t stream) {
    const int bs = 128;
    dim3 grid(CEIL_DIV(D, bs), S, B);
    token_embed_fwd_kernel<<<grid, bs, 0, stream>>>(out, tokens, wte, B, S, D);
    cudaCheck(cudaGetLastError());
}

// Token embedding backward (packed-pair CAS atomic add).
__global__ void token_embed_bwd_kernel(bf16* dwte, const bf16* dout, const int* tokens,
                                       int B, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    int s = blockIdx.y;
    int b = blockIdx.z;
    int t = tokens[b * S + s];
    float v = __bfloat162float(dout[((b * S) + s) * D + d]);
    unsigned int* addr = (unsigned int*)(dwte + t * D + (d & ~1));
    unsigned int old, assumed;
    do {
        old = *addr;
        __nv_bfloat162 packed;
        memcpy(&packed, &old, sizeof(packed));
        float lo = __bfloat162float(packed.x);
        float hi = __bfloat162float(packed.y);
        if ((d & 1) == 0) lo += v; else hi += v;
        packed.x = __float2bfloat16(lo);
        packed.y = __float2bfloat16(hi);
        unsigned int desired;
        memcpy(&desired, &packed, sizeof(desired));
        assumed = old;
        old = atomicCAS(addr, assumed, desired);
    } while (old != assumed);
}
inline void launch_token_bwd(bf16* dwte, const bf16* dout, const int* tokens,
                             int B, int S, int D, cudaStream_t stream) {
    const int bs = 128;
    dim3 grid(CEIL_DIV(D, bs), S, B);
    token_embed_bwd_kernel<<<grid, bs, 0, stream>>>(dwte, dout, tokens, B, S, D);
    cudaCheck(cudaGetLastError());
}

// Absolute position embedding forward.
__global__ void abs_pos_fwd_kernel(bf16* out, const bf16* wpe, int B, int S, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    int b = blockIdx.z;
    if (d >= D) return;
    out[((b * S) + s) * D + d] = wpe[s * D + d];
}
inline void launch_abs_pos_fwd(bf16* out, const bf16* wpe, int B, int S, int D, cudaStream_t stream) {
    const int bs = 128;
    dim3 grid(CEIL_DIV(D, bs), S, B);
    abs_pos_fwd_kernel<<<grid, bs, 0, stream>>>(out, wpe, B, S, D);
    cudaCheck(cudaGetLastError());
}

// Generic small embedding lookup (hash / bucket).
__global__ void small_lookup_kernel(bf16* out, const bf16* table, const int* indices,
                                    int rows, int D, int N_table) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (d >= D || row >= rows) return;
    int t = indices[row];
    if (t < 0 || t >= N_table) t = 0;
    out[row * D + d] = table[t * D + d];
}
inline void launch_small_lookup(bf16* out, const bf16* table, const int* indices,
                                int rows, int D, int N_table, cudaStream_t stream) {
    const int bs = 128;
    dim3 grid(CEIL_DIV(D, bs), rows);
    small_lookup_kernel<<<grid, bs, 0, stream>>>(out, table, indices, rows, D, N_table);
    cudaCheck(cudaGetLastError());
}

// Patch flatten (vision): [B, C, H_p, W_p] → [B, H_p*W_p, C].
__global__ void patch_flatten_kernel(bf16* out, const bf16* x, int B, int C, int H_p, int W_p) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int sp = blockIdx.y;
    int b  = blockIdx.z;
    if (c >= C) return;
    int h = sp / W_p;
    int w = sp % W_p;
    int src = ((b * C + c) * H_p + h) * W_p + w;
    int dst = (b * H_p * W_p + sp) * C + c;
    out[dst] = x[src];
}
inline void launch_patch_flatten(bf16* out, const bf16* x, int B, int C, int H_p, int W_p,
                                 cudaStream_t stream) {
    const int bs = 128;
    dim3 grid(CEIL_DIV(C, bs), H_p * W_p, B);
    patch_flatten_kernel<<<grid, bs, 0, stream>>>(out, x, B, C, H_p, W_p);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::embed_sm120
