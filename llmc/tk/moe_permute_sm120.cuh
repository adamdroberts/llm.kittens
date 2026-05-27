/*
moe_permute_sm120.cuh — ThunderKittens MoE token permute / unpermute for SM120.

Permute (token → expert-grouped layout):
  Inputs:
    tokens:        [rows, dim]            bf16
    topk_indices:  [rows, K]              int32
  Outputs:
    permuted:      [rows*K, dim]          bf16   (each token replicated per top-k)
    sort_order:    [rows*K]               int32  (orig (row,k) flattened index)
    expert_offsets:[E + 1]                int32  (CSR-style)

Unpermute (scatter weighted expert outputs back):
  Inputs:
    permuted_out:  [rows*K, dim]          bf16
    sort_order:    [rows*K]               int32
    topk_weights:  [rows, K]              bf16
  Output:
    out:           [rows, dim]            bf16   (zeroed by caller)

Strategy: each row × top-k slot is one CTA. CTA copies the dim vector to its
permuted destination using TK row tiles. The expert assignment is computed
host-side or via a small device counter; we use the standard CSR pattern.
*/
#pragma once

#include "tk_common.cuh"

namespace llmk::moe_permute {

using namespace ::kittens;

#ifndef LLMK_SM120_MOE_TILE
#define LLMK_SM120_MOE_TILE 128   // dim chunk per warp
#endif

template <int DIM>
struct permute_globals {
    using vec_tile = sv_bf<DIM>;
    using gl_t     = gl<bf16, -1, -1, -1, -1, vec_tile>;
    gl_t  tokens;
    gl_t  permuted;
    int*  topk_indices;     // [rows, K]
    int*  sort_order;       // [rows*K]
    int*  expert_offsets;   // [E+1]
    int*  expert_cursors;   // [E]  (atomic counter scratch)
    int   rows;
    int   K;
    int   E;
};

template <int DIM>
__global__ void permute_kernel(const __grid_constant__ permute_globals<DIM> g) {
    int slot = blockIdx.x;
    int total = g.rows * g.K;
    if (slot >= total) return;

    int e = g.topk_indices[slot];
    if (e < 0 || e >= g.E) return;

    int local_idx = atomicAdd(&g.expert_cursors[e], 1);
    int dst_row = g.expert_offsets[e] + local_idx;
    int src_row = slot / g.K;

    using v = sv_bf<DIM>;
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    v& buf = al.allocate<v>();

    warp::load(buf, g.tokens, {0, 0, src_row, 0});
    __syncwarp();
    warp::store(g.permuted, buf, {0, 0, dst_row, 0});

    if (threadIdx.x == 0) g.sort_order[dst_row] = slot;
}

template <int DIM>
inline void launch_permute(bf16* permuted, int* sort_order, const bf16* tokens,
                           const int* topk_indices, const int* expert_offsets,
                           int* scratch_cursors, int rows, int K, int E,
                           cudaStream_t stream) {
    using G = permute_globals<DIM>;
    typename G::gl_t tok_arg {const_cast<bf16*>(tokens),   1u, 1u, (unsigned)rows,    (unsigned)DIM};
    typename G::gl_t perm_arg{permuted,                    1u, 1u, (unsigned)(rows*K),(unsigned)DIM};
    G g{tok_arg, perm_arg, const_cast<int*>(topk_indices), sort_order,
        const_cast<int*>(expert_offsets), scratch_cursors, rows, K, E};
    cudaCheck(cudaMemsetAsync(scratch_cursors, 0, sizeof(int) * E, stream));
    int total = rows * K;
    permute_kernel<DIM><<<total, ::kittens::WARP_THREADS, sizeof(sv_bf<DIM>) + 16, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Unpermute: scatter-add weighted expert outputs back to original positions.
// ----------------------------------------------------------------------------

template <int DIM>
struct unpermute_globals {
    using vec_tile = sv_bf<DIM>;
    using gl_t     = gl<bf16, -1, -1, -1, -1, vec_tile>;
    gl_t  permuted_out;
    gl_t  out;
    int*  sort_order;
    bf16* topk_weights;
    int   rows;
    int   K;
};

template <int DIM>
__global__ void unpermute_kernel(const __grid_constant__ unpermute_globals<DIM> g) {
    int slot = blockIdx.x;
    int total = g.rows * g.K;
    if (slot >= total) return;
    int orig = g.sort_order[slot];
    int row  = orig / g.K;
    int k    = orig % g.K;
    float w  = __bfloat162float(g.topk_weights[row * g.K + k]);

    using v = sv_bf<DIM>;
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    v& src = al.allocate<v>();

    warp::load(src, g.permuted_out, {0, 0, slot, 0});
    __syncwarp();

    // Atomic-add via per-lane CAS on packed bf16 pairs.
    bf16* out_ptr = (bf16*)&g.out[{0, 0, row, 0}];
    int lane = threadIdx.x;
    for (int i = lane; i < DIM; i += blockDim.x) {
        float partial = __bfloat162float(src[i]) * w;
        // packed pair CAS
        unsigned int* aligned = (unsigned int*)(out_ptr + (i & ~1));
        unsigned int old, assumed;
        do {
            old = *aligned;
            __nv_bfloat162 packed;
            memcpy(&packed, &old, sizeof(packed));
            float lo = __bfloat162float(packed.x);
            float hi = __bfloat162float(packed.y);
            if ((i & 1) == 0) lo += partial; else hi += partial;
            packed.x = __float2bfloat16(lo);
            packed.y = __float2bfloat16(hi);
            unsigned int desired;
            memcpy(&desired, &packed, sizeof(desired));
            assumed = old;
            old = atomicCAS(aligned, assumed, desired);
        } while (old != assumed);
    }
}

template <int DIM>
inline void launch_unpermute(bf16* out, const bf16* permuted_out,
                             const int* sort_order, const bf16* topk_weights,
                             int rows, int K, cudaStream_t stream) {
    using G = unpermute_globals<DIM>;
    typename G::gl_t perm_arg{const_cast<bf16*>(permuted_out), 1u, 1u, (unsigned)(rows*K), (unsigned)DIM};
    typename G::gl_t out_arg {out,                              1u, 1u, (unsigned)rows,     (unsigned)DIM};
    G g{perm_arg, out_arg, const_cast<int*>(sort_order),
        const_cast<bf16*>(topk_weights), rows, K};
    int total = rows * K;
    unpermute_kernel<DIM><<<total, ::kittens::WARP_THREADS, sizeof(sv_bf<DIM>) + 16, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline void launch_permute_dispatch(bf16* permuted, int* sort_order, const bf16* tokens,
                                    const int* topk_indices, const int* expert_offsets,
                                    int* scratch_cursors, int rows, int K, int dim, int E,
                                    cudaStream_t stream) {
    if (dim == 64)        launch_permute<64>  (permuted, sort_order, tokens, topk_indices, expert_offsets, scratch_cursors, rows, K, E, stream);
    else if (dim == 128)  launch_permute<128> (permuted, sort_order, tokens, topk_indices, expert_offsets, scratch_cursors, rows, K, E, stream);
    else if (dim == 256)  launch_permute<256> (permuted, sort_order, tokens, topk_indices, expert_offsets, scratch_cursors, rows, K, E, stream);
    else if (dim == 512)  launch_permute<512> (permuted, sort_order, tokens, topk_indices, expert_offsets, scratch_cursors, rows, K, E, stream);
    else if (dim == 768)  launch_permute<768> (permuted, sort_order, tokens, topk_indices, expert_offsets, scratch_cursors, rows, K, E, stream);
    else if (dim == 1024) launch_permute<1024>(permuted, sort_order, tokens, topk_indices, expert_offsets, scratch_cursors, rows, K, E, stream);
    else {
        fprintf(stderr, "moe_permute_sm120: unsupported dim=%d\n", dim);
        exit(EXIT_FAILURE);
    }
}

inline void launch_unpermute_dispatch(bf16* out, const bf16* permuted_out,
                                      const int* sort_order, const bf16* topk_weights,
                                      int rows, int K, int dim, cudaStream_t stream) {
    if (dim == 64)        launch_unpermute<64>  (out, permuted_out, sort_order, topk_weights, rows, K, stream);
    else if (dim == 128)  launch_unpermute<128> (out, permuted_out, sort_order, topk_weights, rows, K, stream);
    else if (dim == 256)  launch_unpermute<256> (out, permuted_out, sort_order, topk_weights, rows, K, stream);
    else if (dim == 512)  launch_unpermute<512> (out, permuted_out, sort_order, topk_weights, rows, K, stream);
    else if (dim == 768)  launch_unpermute<768> (out, permuted_out, sort_order, topk_weights, rows, K, stream);
    else if (dim == 1024) launch_unpermute<1024>(out, permuted_out, sort_order, topk_weights, rows, K, stream);
    else {
        fprintf(stderr, "moe_unpermute_sm120: unsupported dim=%d\n", dim);
        exit(EXIT_FAILURE);
    }
}

}  // namespace llmk::moe_permute
