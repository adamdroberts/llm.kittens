/*
attention_gqa_h100.cuh — ThunderKittens H100 bf16 GQA forward wrapper.

This is the first TK-backed slice of M6. It adapts the MHA forward template to
grouped-query attention by launching over query heads and mapping each query
head to its shared KV head with `kv_head_idx = q_head_idx / (n_q / n_kv)`.
For shapes where both TK forward and backward are available, RoPE can be
applied to Q/K after TMA load and before WGMMA; fallback shapes still use the
materialized RoPE path in llmc/attention_gqa.cuh.
*/
#pragma once

#include "attention_h100.cuh"
#include "tk_common.cuh"

namespace llmk::attention_gqa {

using namespace ::kittens;

constexpr int GQA_FWD_CONSUMER_WARPGROUPS = 2;
constexpr int GQA_FWD_PRODUCER_WARPGROUPS = 1;
constexpr int GQA_FWD_NUM_WARPGROUPS =
    GQA_FWD_CONSUMER_WARPGROUPS + GQA_FWD_PRODUCER_WARPGROUPS;
constexpr int GQA_FWD_NUM_WORKERS = GQA_FWD_NUM_WARPGROUPS * ::kittens::WARPGROUP_WARPS;

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = 64;
    constexpr static int qo_height = 4 * 16;
    constexpr static int kv_height = 8 * 16;
    constexpr static int stages = 4;
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = 128;
    constexpr static int qo_height = 4 * 16;
    constexpr static int kv_height = 8 * 16;
    constexpr static int stages = 2;
};

template<int D>
struct fwd_globals {
    using q_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const bf16* cos;
    const bf16* sin;
    const int N;
    const int hr;
};

template<typename tile>
__device__ static inline void rope_tile_forward(tile& x, const bf16* cos, const bf16* sin,
                                                int row_start, int lane, int stride) {
    constexpr int D = tile::cols;
    constexpr int half = D / 2;
    static_assert(D == 64 || D == 128);

    for (int idx = lane; idx < tile::rows * half; idx += stride) {
        int row = idx / half;
        int pair = idx % half;
        float x1 = (float)x[{row, pair}];
        float x2 = (float)x[{row, pair + half}];
        float c = (float)cos[(row_start + row) * half + pair];
        float s = (float)sin[(row_start + row) * half + pair];
        x[{row, pair}] = (bf16)(x1 * c - x2 * s);
        x[{row, pair + half}] = (bf16)(x2 * c + x1 * s);
    }
}

template<int D, bool is_causal>
__global__ __launch_bounds__(GQA_FWD_NUM_WORKERS * ::kittens::WARP_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid();
    int warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;

    using q_tile = st_bf<K::qo_height, K::tile_width>;
    using k_tile = st_bf<K::kv_height, K::tile_width>;
    using v_tile = st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile = st_bf<K::qo_height, K::tile_width>;

    q_tile (&q_smem)[GQA_FWD_CONSUMER_WARPGROUPS] =
        al.allocate<q_tile, GQA_FWD_CONSUMER_WARPGROUPS>();
    k_tile (&k_smem)[K::stages] = al.allocate<k_tile, K::stages>();
    v_tile (&v_smem)[K::stages] = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[GQA_FWD_CONSUMER_WARPGROUPS] =
        al.allocate<l_col_vec, GQA_FWD_CONSUMER_WARPGROUPS>();
    auto (*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

    int batch_idx = static_cast<int>(blockIdx.z);
    int q_head_idx = static_cast<int>(blockIdx.y);
    int block_idx = static_cast<int>(blockIdx.x);
    int kv_blocks = g.N / K::kv_height;
    int kv_head_idx = q_head_idx / g.hr;
    int seq_idx = block_idx * GQA_FWD_CONSUMER_WARPGROUPS;

    __shared__ kittens::semaphore qsmem_semaphore;
    __shared__ kittens::semaphore k_smem_arrived[K::stages];
    __shared__ kittens::semaphore v_smem_arrived[K::stages];
    __shared__ kittens::semaphore compute_done[K::stages];

    if (threadIdx.x == 0) {
        init_semaphore(qsmem_semaphore, 0, 1);
        for (int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1);
            init_semaphore(v_smem_arrived[j], 0, 1);
            init_semaphore(compute_done[j], GQA_FWD_CONSUMER_WARPGROUPS, 0);
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < GQA_FWD_CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {batch_idx, q_head_idx, seq_idx + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            coord<k_tile> kv_tile_idx = {batch_idx, kv_head_idx, j, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
        }
    }
    __syncthreads();

    int pipe_idx = K::stages - 1;

    if (warpgroupid == GQA_FWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<32>();

        int kv_iters;
        if constexpr (is_causal) {
            kv_iters = (seq_idx * (K::qo_height / kittens::TILE_ROW_DIM<bf16>)) - 1
                + (GQA_FWD_CONSUMER_WARPGROUPS * (K::qo_height / kittens::TILE_ROW_DIM<bf16>));
            kv_iters = ((kv_iters / (K::kv_height / kittens::TILE_ROW_DIM<bf16>)) == 0)
                ? 0
                : ((kv_iters / (K::kv_height / kittens::TILE_ROW_DIM<bf16>)) - 1);
        } else {
            kv_iters = kv_blocks - 2;
        }

        if (warpid == GQA_FWD_NUM_WORKERS - 4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                coord<k_tile> kv_tile_idx = {batch_idx, kv_head_idx, kv_idx + 1, 0};
                warp::tma::expect_bytes(k_smem_arrived[(kv_idx + 1) % K::stages], sizeof(k_tile));
                warp::tma::load_async(k_smem[(kv_idx + 1) % K::stages], g.k, kv_tile_idx,
                                      k_smem_arrived[(kv_idx + 1) % K::stages]);
                warp::tma::expect_bytes(v_smem_arrived[(kv_idx + 1) % K::stages], sizeof(v_tile));
                warp::tma::load_async(v_smem[(kv_idx + 1) % K::stages], g.v, kv_tile_idx,
                                      v_smem_arrived[(kv_idx + 1) % K::stages]);

                wait(compute_done[kv_idx % K::stages], (kv_idx / K::stages) % 2);
            }
        }
    } else {
        warpgroup::increase_registers<160>();

        rt_fl<16, K::kv_height> att_block;
        rt_bf<16, K::kv_height> att_block_mma;
        rt_fl<16, K::tile_width> o_reg;

        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

        warp::neg_infty(max_vec);
        warp::zero(norm_vec);
        warp::zero(o_reg);

        int kv_iters;
        if constexpr (is_causal) {
            kv_iters = (seq_idx * 4) - 1 + (GQA_FWD_CONSUMER_WARPGROUPS * 4);
            kv_iters = (kv_iters / 8);
        } else {
            kv_iters = kv_blocks - 1;
        }

        wait(qsmem_semaphore, 0);
        if (g.cos != nullptr && g.sin != nullptr) {
            rope_tile_forward(q_smem[warpgroupid], g.cos, g.sin,
                              (seq_idx + warpgroupid) * K::qo_height,
                              warpgroup::laneid(), kittens::WARPGROUP_WARPS * kittens::WARP_THREADS);
            warpgroup::sync(warpgroupid + 4);
        }

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
            wait(k_smem_arrived[kv_idx % K::stages], (kv_idx / K::stages) % 2);
            if (g.cos != nullptr && g.sin != nullptr) {
                group<GQA_FWD_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>::sync(10);
                if (warpgroupid == 0) {
                    rope_tile_forward(k_smem[kv_idx % K::stages], g.cos, g.sin,
                                      kv_idx * K::kv_height,
                                      warpgroup::laneid(),
                                      kittens::WARPGROUP_WARPS * kittens::WARP_THREADS);
                }
                group<GQA_FWD_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>::sync(10);
            }
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[kv_idx % K::stages]);

            warp::copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) {
                warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.125f);
            } else {
                warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.08838834764f);
            }

            warpgroup::mma_async_wait();

            if constexpr (is_causal) {
                const int q_blk = (seq_idx * (K::qo_height / kittens::TILE_ROW_DIM<bf16>)) + warpid;
                int k_blk = (kv_idx * (K::kv_height / kittens::TILE_ROW_DIM<bf16>));

                #pragma unroll
                for (; k_blk == (kv_iters - 1) * (K::kv_height / kittens::TILE_ROW_DIM<bf16>)
                       || k_blk == (kv_iters) * (K::kv_height / kittens::TILE_ROW_DIM<bf16>);
                     k_blk += 10000) {
                    #pragma unroll
                    for (auto j = 0; j < (K::kv_height / kittens::TILE_ROW_DIM<bf16>); j++) {
                        auto k_idx = k_blk + j;
                        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);

                        if (k_idx > q_blk) {
                            warp::neg_infty(attn_subtile);
                        } else if (k_idx == q_blk) {
                            warp::make_causal(attn_subtile, attn_subtile,
                                              kittens::base_types::constants<float>::neg_infty());
                        }
                        __syncwarp();
                    }
                }
            }

            warp::row_max(max_vec, att_block, max_vec);

            if constexpr (D == 64) {
                warp::mul(att_block, att_block, 1.44269504089f * 0.125f);
                warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);
            } else {
                warp::mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
                warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
            }

            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
            warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
            warp::row_sum(norm_vec, att_block, norm_vec);
            warp::add(att_block, att_block, 0.f);
            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, max_vec_last_scaled);

            wait(v_smem_arrived[kv_idx % K::stages], (kv_idx / K::stages) % 2);

            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[kv_idx % K::stages]);
            warpgroup::mma_async_wait();

            if (warpgroup::laneid() == 0) arrive(compute_done[kv_idx % K::stages], 1);
        }

        warp::div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(o_smem[warpgroupid], o_reg);
        warpgroup::sync(warpgroupid + 4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {batch_idx, q_head_idx, seq_idx + warpgroupid, 0};
            warp::tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        warp::mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
        warp::log(norm_vec, norm_vec);
        warp::add(norm_vec, norm_vec, max_vec_scaled);

        if constexpr (D == 64) {
            warp::mul(norm_vec, norm_vec, -8.0f);
        } else {
            warp::mul(norm_vec, norm_vec, -11.313708499f);
        }

        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync(warpgroupid + 4);

        if (warpid % 4 == 0) {
            coord<l_col_vec> tile_idx = {batch_idx, q_head_idx, 0, seq_idx + warpgroupid};
            warp::tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
        }
        warp::tma::store_async_wait();
    }
}

inline constexpr int fwd_sequence_granularity() {
    return GQA_FWD_CONSUMER_WARPGROUPS * ::kittens::TILE_ROW_DIM<bf16> * 4;
}

template<int D>
inline constexpr int fwd_min_sequence_length() {
    return (fwd_attend_ker_tile_dims<D>::stages - 1) *
           fwd_attend_ker_tile_dims<D>::kv_height;
}

template<int D>
inline void launch_forward_causal_impl(bf16* q, bf16* k, bf16* v, const bf16* cos, const bf16* sin,
                                       float* l, bf16* o, int B, int NH, int NKVH, int T,
                                       cudaStream_t stream) {
    static_assert(D == 64 || D == 128);
    assert(NKVH > 0);
    assert(NH % NKVH == 0);
    assert(T % fwd_sequence_granularity() == 0 &&
           "TK GQA forward requires T to be divisible by 128");
    assert(T >= fwd_min_sequence_length<D>() &&
           "TK GQA forward requires enough KV tiles for stage prefetch");

    using globals = fwd_globals<D>;
    using q_global = typename globals::q_gl;
    using k_global = typename globals::k_gl;
    using v_global = typename globals::v_gl;
    using l_global = typename globals::l_gl;
    using o_global = typename globals::o_gl;

    q_global q_arg{q, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    k_global k_arg{k, static_cast<unsigned int>(B), static_cast<unsigned int>(NKVH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    v_global v_arg{v, static_cast<unsigned int>(B), static_cast<unsigned int>(NKVH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    l_global l_arg{l, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   1U, static_cast<unsigned int>(T)};
    o_global o_arg{o, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};

    globals g{q_arg, k_arg, v_arg, l_arg, o_arg, cos, sin, T, NH / NKVH};

    const unsigned long mem_size = kittens::MAX_SHARED_MEMORY - 1024;
    static bool smem_attr_set = false;
    if (!smem_attr_set) {
        cudaCheck(cudaFuncSetAttribute(
            fwd_attend_ker<D, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size));
        smem_attr_set = true;
    }

    dim3 grid(T / fwd_sequence_granularity(), NH, B);
    fwd_attend_ker<D, true><<<grid, 32 * GQA_FWD_NUM_WORKERS, mem_size, stream>>>(g);
}

template<int D>
inline void launch_forward_causal(bf16* q, bf16* k, bf16* v, float* l, bf16* o,
                                  int B, int NH, int NKVH, int T, cudaStream_t stream) {
    launch_forward_causal_impl<D>(q, k, v, nullptr, nullptr, l, o, B, NH, NKVH, T, stream);
}

template<int D>
inline void launch_forward_causal_rope(bf16* q, bf16* k, bf16* v,
                                       const bf16* cos, const bf16* sin,
                                       float* l, bf16* o,
                                       int B, int NH, int NKVH, int T, cudaStream_t stream) {
    launch_forward_causal_impl<D>(q, k, v, cos, sin, l, o, B, NH, NKVH, T, stream);
}

inline void launch_forward_causal(bf16* q, bf16* k, bf16* v, float* l, bf16* o,
                                  int B, int NH, int NKVH, int T, int HS,
                                  cudaStream_t stream) {
    if (HS == 64) {
        launch_forward_causal<64>(q, k, v, l, o, B, NH, NKVH, T, stream);
    } else if (HS == 128) {
        launch_forward_causal<128>(q, k, v, l, o, B, NH, NKVH, T, stream);
    } else {
        fprintf(stderr, "attention_gqa_forward: TK GQA only supports head_dim 64 or 128, got %d\n", HS);
        exit(EXIT_FAILURE);
    }
}

inline void launch_forward_causal_rope(bf16* q, bf16* k, bf16* v,
                                       const bf16* cos, const bf16* sin,
                                       float* l, bf16* o,
                                       int B, int NH, int NKVH, int T, int HS,
                                       cudaStream_t stream) {
    if (HS == 64) {
        launch_forward_causal_rope<64>(q, k, v, cos, sin, l, o, B, NH, NKVH, T, stream);
    } else if (HS == 128) {
        launch_forward_causal_rope<128>(q, k, v, cos, sin, l, o, B, NH, NKVH, T, stream);
    } else {
        fprintf(stderr, "attention_gqa_forward: TK GQA only supports head_dim 64 or 128, got %d\n", HS);
        exit(EXIT_FAILURE);
    }
}

inline bool has_tk_forward(int T, int head_dim, int n_q_heads, int n_kv_heads) {
    if (T <= 0 || n_q_heads <= 0 || n_kv_heads <= 0) { return false; }
    if (n_q_heads % n_kv_heads != 0) { return false; }
    if (T % fwd_sequence_granularity() != 0) { return false; }
    if (head_dim == 64) {
        return T >= fwd_min_sequence_length<64>();
    }
    if (head_dim == 128) {
        return T >= fwd_min_sequence_length<128>();
    }
    return false;
}

inline bool has_tk_kernel(int T, int head_dim, int n_q_heads, int n_kv_heads) {
    return has_tk_forward(T, head_dim, n_q_heads, n_kv_heads);
}

inline bool has_tk_backward(int T, int head_dim, int n_q_heads, int n_kv_heads) {
    if (!has_tk_forward(T, head_dim, n_q_heads, n_kv_heads)) { return false; }
    return T % llmk::attention::bwd_sequence_granularity() == 0;
}

inline void launch_backward_causal(bf16* q, bf16* k, bf16* v, bf16* o,
                                   float* l, bf16* og, float* d,
                                   float* qg, float* kg, float* vg,
                                   int B, int NH, int NKVH, int T, int HS,
                                   cudaStream_t stream,
                                   const bf16* cos = nullptr, const bf16* sin = nullptr) {
    if (HS == 64) {
        llmk::attention::launch_backward_causal_gqa<64>(
            q, k, v, o, l, og, d, qg, kg, vg, B, NH, NKVH, T, stream, cos, sin);
    } else if (HS == 128) {
        llmk::attention::launch_backward_causal_gqa<128>(
            q, k, v, o, l, og, d, qg, kg, vg, B, NH, NKVH, T, stream, cos, sin);
    } else {
        fprintf(stderr, "attention_gqa_backward: TK GQA only supports head_dim 64 or 128, got %d\n", HS);
        exit(EXIT_FAILURE);
    }
}

} // namespace llmk::attention_gqa
