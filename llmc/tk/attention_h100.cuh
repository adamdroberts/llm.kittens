/*
attention_h100.cuh — ThunderKittens H100 bf16 MHA forward wrapper.

Source: ThunderKittens/kernels/attention/mha_h100/mha_h100.cu
(`fwd_attend_ker`). This header keeps the TK template-heavy code behind the
llmc/tk boundary; llmc/attention.cuh exposes the C-style API used by the
trainer.
*/
#pragma once

#include "tk_common.cuh"

namespace llmk::attention {

using namespace ::kittens;

constexpr int FWD_CONSUMER_WARPGROUPS = 3;
constexpr int FWD_PRODUCER_WARPGROUPS = 1;
constexpr int FWD_NUM_WARPGROUPS = FWD_CONSUMER_WARPGROUPS + FWD_PRODUCER_WARPGROUPS;
constexpr int FWD_NUM_WORKERS = FWD_NUM_WARPGROUPS * ::kittens::WARPGROUP_WARPS;

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

    const int N;
    const int hr;
};

template<int D, bool is_causal>
__global__ __launch_bounds__(FWD_NUM_WORKERS * ::kittens::WARP_THREADS, 1)
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

    q_tile (&q_smem)[FWD_CONSUMER_WARPGROUPS] = al.allocate<q_tile, FWD_CONSUMER_WARPGROUPS>();
    k_tile (&k_smem)[K::stages] = al.allocate<k_tile, K::stages>();
    v_tile (&v_smem)[K::stages] = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[FWD_CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, FWD_CONSUMER_WARPGROUPS>();
    auto (*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

    int kv_blocks = g.N / K::kv_height;
    int batch_idx = static_cast<int>(blockIdx.z);
    int q_head_idx = static_cast<int>(blockIdx.y);
    int block_idx = static_cast<int>(blockIdx.x);
    int kv_head_idx = q_head_idx / g.hr;
    int seq_idx = block_idx * FWD_CONSUMER_WARPGROUPS;

    __shared__ kittens::semaphore qsmem_semaphore;
    __shared__ kittens::semaphore k_smem_arrived[K::stages];
    __shared__ kittens::semaphore v_smem_arrived[K::stages];
    __shared__ kittens::semaphore compute_done[K::stages];

    if (threadIdx.x == 0) {
        init_semaphore(qsmem_semaphore, 0, 1);
        for (int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1);
            init_semaphore(v_smem_arrived[j], 0, 1);
            init_semaphore(compute_done[j], FWD_CONSUMER_WARPGROUPS, 0);
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < FWD_CONSUMER_WARPGROUPS; wg++) {
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

    if (warpgroupid == FWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<32>();

        int kv_iters;
        if constexpr (is_causal) {
            kv_iters = (seq_idx * (K::qo_height / kittens::TILE_ROW_DIM<bf16>)) - 1
                + (FWD_CONSUMER_WARPGROUPS * (K::qo_height / kittens::TILE_ROW_DIM<bf16>));
            kv_iters = ((kv_iters / (K::kv_height / kittens::TILE_ROW_DIM<bf16>)) == 0)
                ? 0
                : ((kv_iters / (K::kv_height / kittens::TILE_ROW_DIM<bf16>)) - 1);
        } else {
            kv_iters = kv_blocks - 2;
        }

        if (warpid == FWD_NUM_WORKERS - 4) {
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
            kv_iters = (seq_idx * 4) - 1 + (FWD_CONSUMER_WARPGROUPS * 4);
            kv_iters = (kv_iters / 8);
        } else {
            kv_iters = kv_blocks - 1;
        }

        wait(qsmem_semaphore, 0);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
            wait(k_smem_arrived[kv_idx % K::stages], (kv_idx / K::stages) % 2);
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
    return FWD_CONSUMER_WARPGROUPS * ::kittens::TILE_ROW_DIM<bf16> * 4;
}

template<int D>
inline void launch_forward_causal(bf16* q, bf16* k, bf16* v, float* l, bf16* o,
                                  int B, int NH, int T, cudaStream_t stream) {
    static_assert(D == 64 || D == 128);
    assert(T % fwd_sequence_granularity() == 0 &&
           "TK MHA forward requires T to be divisible by 192");

    using globals = fwd_globals<D>;
    using q_global = typename globals::q_gl;
    using k_global = typename globals::k_gl;
    using v_global = typename globals::v_gl;
    using l_global = typename globals::l_gl;
    using o_global = typename globals::o_gl;

    q_global q_arg{q, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    k_global k_arg{k, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    v_global v_arg{v, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    l_global l_arg{l, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   1U, static_cast<unsigned int>(T)};
    o_global o_arg{o, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};

    globals g{q_arg, k_arg, v_arg, l_arg, o_arg, T, 1};

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
    fwd_attend_ker<D, true><<<grid, 32 * FWD_NUM_WORKERS, mem_size, stream>>>(g);
}

inline void launch_forward_causal(bf16* q, bf16* k, bf16* v, float* l, bf16* o,
                                  int B, int NH, int T, int HS, cudaStream_t stream) {
    if (HS == 64) {
        launch_forward_causal<64>(q, k, v, l, o, B, NH, T, stream);
    } else if (HS == 128) {
        launch_forward_causal<128>(q, k, v, l, o, B, NH, T, stream);
    } else {
        fprintf(stderr, "attention_forward: TK MHA only supports head_dim 64 or 128, got %d\n", HS);
        exit(EXIT_FAILURE);
    }
}

// -------------------------------------------------------------------------------------------------
// Backward preparation kernel
// -------------------------------------------------------------------------------------------------

template<int D>
struct bwd_prep_globals {
    using og_tile = st_bf<4 * 16, D>;
    using o_tile  = st_bf<4 * 16, D>;
    using d_tile  = col_vec<st_fl<4 * 16, D>>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    og_gl og;
    o_gl  o;
    d_gl  d;
};

template<int D>
__global__ __launch_bounds__(4 * ::kittens::WARP_THREADS, (D == 64) ? 2 : 1)
void bwd_attend_prep_ker(const __grid_constant__ bwd_prep_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();
    int batch_idx = static_cast<int>(blockIdx.z);
    int head_idx = static_cast<int>(blockIdx.y);
    int block_idx = static_cast<int>(blockIdx.x);

    using og_tile = st_bf<4 * 16, D>;
    using o_tile  = st_bf<4 * 16, D>;
    using d_tile  = col_vec<st_fl<4 * 16, D>>;

    og_tile (&og_smem)[4] = al.allocate<og_tile, 4>();
    o_tile  (&o_smem) [4] = al.allocate<o_tile , 4>();
    d_tile  (&d_smem) [4] = al.allocate<d_tile , 4>();

    rt_fl<4 * 16, D> og_reg, o_reg;
    col_vec<rt_fl<4 * 16, D>> d_reg;

    __shared__ kittens::semaphore smem_semaphore;

    if (threadIdx.x == 0) {
        init_semaphore(smem_semaphore, 0, 1);
        tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * 4 * 2);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) {
            coord<o_tile> tile_idx = {batch_idx, head_idx, (block_idx * 4) + w, 0};
            warp::tma::load_async(o_smem[w],  g.o,  tile_idx, smem_semaphore);
            warp::tma::load_async(og_smem[w], g.og, tile_idx, smem_semaphore);
        }
    }

    wait(smem_semaphore, 0);
    warp::load(o_reg, o_smem[warpid]);
    warp::load(og_reg, og_smem[warpid]);
    warp::mul(og_reg, og_reg, o_reg);
    warp::row_sum(d_reg, og_reg);
    warp::store(d_smem[warpid], d_reg);
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) {
            coord<d_tile> tile_idx = {batch_idx, head_idx, 0, (block_idx * 4) + w};
            warp::tma::store_async(g.d, d_smem[w], tile_idx);
        }
    }
    warp::tma::store_async_wait();
}

// -------------------------------------------------------------------------------------------------
// Backward main kernel
// -------------------------------------------------------------------------------------------------

template<int D> struct bwd_attend_ker_tile_dims {};
template<> struct bwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = 64;
    constexpr static int tile_h     = 4 * 16;
    constexpr static int tile_h_qo  = 4 * 16;
    constexpr static int blocks_sm  = 1;
};
template<> struct bwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = 128;
    constexpr static int tile_h     = 4 * 16;
    constexpr static int tile_h_qo  = 4 * 16;
    constexpr static int blocks_sm  = 1;
};

constexpr int BWD_CONSUMER_WARPGROUPS = 2;
constexpr int BWD_PRODUCER_WARPGROUPS = 1;
constexpr int BWD_NUM_WARPGROUPS      = BWD_CONSUMER_WARPGROUPS + BWD_PRODUCER_WARPGROUPS;
constexpr int BWD_NUM_WORKERS         = BWD_NUM_WARPGROUPS * ::kittens::WARPGROUP_WARPS;

template<int D>
struct bwd_globals {
    using G = bwd_attend_ker_tile_dims<D>;

    using q_tile  =         st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using v_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using og_tile =         st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile =         st_fl<G::tile_h_qo, G::tile_width>;
    using kg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using vg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using l_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
    using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
    using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;
    using l_gl  = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    q_gl  q;
    k_gl  k;
    v_gl  v;
    og_gl og;
    qg_gl qg;
    kg_gl kg;
    vg_gl vg;
    l_gl  l;
    d_gl  d;

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

__device__ static inline void stream_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_col = 16 * i + 2 * (kittens::laneid() % 4);
        reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
        reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
    }
}

__device__ static inline void stream_sub_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_col = 16 * i + 2 * (laneid() % 4);
        reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
        reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
    }
}

template<int tile_h_qo, int tile_h>
__device__ static inline void causal_mask(auto &reg_tile, int qo_idx) {
    int q_blk = qo_idx * (tile_h_qo / kittens::TILE_ROW_DIM<bf16>);
    int block_idx = static_cast<int>(blockIdx.x);
    int k_blk = (block_idx * BWD_CONSUMER_WARPGROUPS * (tile_h / kittens::TILE_ROW_DIM<bf16>))
              + ((kittens::warpid() / kittens::WARPGROUP_WARPS) * (tile_h / kittens::TILE_ROW_DIM<bf16>))
              + (kittens::warpid() % kittens::WARPGROUP_WARPS);

    for (int j = 0; j < (tile_h_qo / kittens::TILE_ROW_DIM<bf16>); j++) {
        int q_idx = q_blk + j;
        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(reg_tile.tiles[0][j]);
        if (q_idx < k_blk) {
            warp::neg_infty(attn_subtile);
        } else if (q_idx == k_blk) {
            warp::make_causal_t(attn_subtile, attn_subtile,
                                kittens::base_types::constants<float>::neg_infty());
        }
    }
}

template<bool is_causal, int tile_h_qo, int tile_h, int tile_width, int D>
__device__ static inline void compute_bwd_loop(
        kittens::semaphore *vec_b, kittens::semaphore *q_b, kittens::semaphore *o_b,
        rt_fl<16, 64> &s_block_t, rt_fl<16, 64> &dp_block_t,
        rt_fl<16, 64> &p_block_t, rt_fl<16, 64> &ds_block_t,
        rt_bf<16, 64> &p_block_t_mma, rt_bf<16, 64> &ds_block_t_mma,
        rt_fl<16, tile_width> &kg_reg, rt_fl<16, tile_width> &vg_reg,
        auto &q_smem, auto &k_smem, auto &v_smem,
        auto &og_smem, auto &ds_smem, auto &l_smem, auto &d_smem,
        const bf16* cos, const bf16* sin,
        int qo_idx, int q_start, int tic, int toc) {
    wait(vec_b[tic], ((qo_idx - q_start) / 2) % 2);
    stream_tile(s_block_t, l_smem, tic);
    wait(q_b[tic], ((qo_idx - q_start) / 2) % 2);
    if (cos != nullptr && sin != nullptr) {
        group<BWD_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>::sync(11);
        if (kittens::warpid() / kittens::WARPGROUP_WARPS == 0) {
            rope_tile_forward(q_smem[tic], cos, sin,
                              qo_idx * tile_h_qo,
                              warpgroup::laneid(),
                              kittens::WARPGROUP_WARPS * kittens::WARP_THREADS);
        }
        group<BWD_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>::sync(11);
    }

    warpgroup::mma_ABt(s_block_t, k_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], q_smem[tic]);
    warpgroup::mma_commit_group();

    wait(o_b[tic], ((qo_idx - q_start) / 2) % 2);
    warpgroup::mm_ABt(dp_block_t, v_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], og_smem[tic]);
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    if constexpr (D == 64) {
        warp::mul(s_block_t, s_block_t, 1.44269504089f * 0.125f);
    } else {
        warp::mul(s_block_t, s_block_t, 1.44269504089f * 0.08838834764f);
    }

    if constexpr (is_causal) {
        causal_mask<tile_h_qo, tile_h>(s_block_t, qo_idx);
    }

    warp::exp2(s_block_t, s_block_t);
    warp::copy(p_block_t, s_block_t);
    warp::copy(p_block_t_mma, s_block_t);
    stream_sub_tile(dp_block_t, d_smem, tic);
    warp::mul(ds_block_t, p_block_t, dp_block_t);

    if constexpr (D == 64) {
        warp::mul(ds_block_t, ds_block_t, 0.125f);
    } else {
        warp::mul(ds_block_t, ds_block_t, 0.08838834764f);
    }

    warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[tic]);
    warpgroup::mma_commit_group();

    warp::copy(ds_block_t_mma, ds_block_t);
    warpgroup::store(ds_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], ds_block_t);
    warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[tic]);
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
    group<8>::sync(10);
}

template<typename kg_tile, typename vg_tile>
__device__ static inline void kv_store(auto &kg_smem, auto &kg_reg,
                                       auto &vg_smem, auto &vg_reg,
                                       auto &dst, auto &bar, int kv_head_idx, int toc) {
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int block_idx = static_cast<int>(blockIdx.x);
    group<8>::sync(10);
    warpgroup::store(kg_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], kg_reg);

    group<4>::sync(warpgroup::groupid() + 4);
    if (kittens::warpid() % 4 == 0) {
        coord<kg_tile> tile_idx = {batch_idx, kv_head_idx,
                                   (block_idx * BWD_CONSUMER_WARPGROUPS)
                                   + (kittens::warpid() / kittens::WARPGROUP_WARPS), 0};
        warp::tma::store_add_async(dst.kg, kg_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], tile_idx);
        warp::tma::store_commit_group();
    }

    wait(bar, toc);
    warpgroup::store(vg_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], vg_reg);
    group<4>::sync(warpgroup::groupid() + 4);

    if (kittens::warpid() % 4 == 0) {
        coord<vg_tile> tile_idx = {batch_idx, kv_head_idx,
                                   (block_idx * BWD_CONSUMER_WARPGROUPS)
                                   + (kittens::warpid() / kittens::WARPGROUP_WARPS), 0};
        warp::tma::store_add_async(dst.vg, vg_smem[kittens::warpid() / kittens::WARPGROUP_WARPS], tile_idx);
        warp::tma::store_commit_group();
    }
    warp::tma::store_async_wait();
}

template<int D, bool is_causal>
__global__ __launch_bounds__(BWD_NUM_WORKERS * kittens::WARP_THREADS,
                             bwd_attend_ker_tile_dims<D>::blocks_sm)
void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int N = g.N, hr = g.hr;
    using G = bwd_attend_ker_tile_dims<D>;

    using kg_tile   = st_fl<G::tile_h, G::tile_width>;
    using vg_tile   = st_fl<G::tile_h, G::tile_width>;
    using k_tile    = st_bf<G::tile_h, G::tile_width>;
    using v_tile    = st_bf<G::tile_h, G::tile_width>;
    using q_tile    = st_bf<G::tile_h_qo, G::tile_width>;
    using og_tile   = st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile   = st_fl<G::tile_h_qo, G::tile_width>;
    using l_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;

    k_tile  (&k_smem) [BWD_CONSUMER_WARPGROUPS] = al.allocate<k_tile, BWD_CONSUMER_WARPGROUPS>();
    v_tile  (&v_smem) [BWD_CONSUMER_WARPGROUPS] = al.allocate<v_tile, BWD_CONSUMER_WARPGROUPS>();
    q_tile  (&q_smem) [2] = al.allocate<q_tile,  2>();
    og_tile (&og_smem)[2] = al.allocate<og_tile, 2>();
    qg_tile (&qg_smem)    = al.allocate<qg_tile>();
    l_tile  (&l_smem)[2] = al.allocate<l_tile, 2>();
    d_tile  (&d_smem)[2] = al.allocate<d_tile, 2>();
    kg_tile (*kg_smem) = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]);
    vg_tile (*vg_smem) = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]);
    attn_tile (&ds_smem)[BWD_CONSUMER_WARPGROUPS] = al.allocate<attn_tile, BWD_CONSUMER_WARPGROUPS>();

    const int warpid      = kittens::warpid();
    const int warpgroupid = warpid / kittens::WARPGROUP_WARPS;
    const int qo_blocks   = N / G::tile_h_qo;
    const int batch_idx   = static_cast<int>(blockIdx.z);
    const int q_head_idx  = static_cast<int>(blockIdx.y);
    const int block_idx   = static_cast<int>(blockIdx.x);
    const int kv_head_idx = q_head_idx / hr;

    __shared__ kittens::semaphore kv_b, q_b[2], o_b[2], vec_b[2];
    __shared__ kittens::semaphore compute_done[2], qg_ready;

    int tic = 0, toc = 1;
    const int q_start = is_causal ? (block_idx * 2) : 0;

    if (threadIdx.x == 0) {
        init_semaphore(kv_b, 0, 1);
        init_semaphore(qg_ready, 1, 0);
        for (int s = 0; s < 2; s++) {
            init_semaphore(q_b[s], 0, 1);
            init_semaphore(o_b[s], 0, 1);
            init_semaphore(vec_b[s], 0, 1);
            init_semaphore(compute_done[s], 1, 0);
        }

        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * BWD_CONSUMER_WARPGROUPS);
        for (int w = 0; w < BWD_CONSUMER_WARPGROUPS; w++) {
            coord<k_tile> tile_idx = {batch_idx, kv_head_idx,
                                      (block_idx * BWD_CONSUMER_WARPGROUPS) + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }

        coord<q_tile> tile_idx = {batch_idx, q_head_idx, q_start, 0};
        tma::expect_bytes(q_b[tic], sizeof(q_smem[0]));
        tma::load_async(q_smem[tic], g.q, tile_idx, q_b[tic]);
        tma::expect_bytes(o_b[tic], sizeof(og_smem[0]));
        tma::load_async(og_smem[tic], g.og, tile_idx, o_b[tic]);

        coord<l_tile> vec_idx = {batch_idx, q_head_idx, 0, q_start};
        tma::expect_bytes(vec_b[tic], sizeof(l_smem[0]) + sizeof(d_smem[0]));
        tma::load_async(l_smem[tic], g.l, vec_idx, vec_b[tic]);
        tma::load_async(d_smem[tic], g.d, vec_idx, vec_b[tic]);
    }
    __syncthreads();

    if (warpgroupid == BWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<24>();

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                if (qo_idx + 1 < qo_blocks) {
                    coord<q_tile> tile_idx = {batch_idx, q_head_idx, qo_idx + 1, 0};
                    warp::tma::expect_bytes(q_b[toc], sizeof(q_smem[0]));
                    warp::tma::load_async(q_smem[toc], g.q, tile_idx, q_b[toc]);
                    warp::tma::expect_bytes(o_b[toc], sizeof(og_smem[0]));
                    warp::tma::load_async(og_smem[toc], g.og, tile_idx, o_b[toc]);

                    coord<l_tile> vec_idx = {batch_idx, q_head_idx, 0, qo_idx + 1};
                    warp::tma::expect_bytes(vec_b[toc], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                    warp::tma::load_async(l_smem[toc], g.l, vec_idx, vec_b[toc]);
                    warp::tma::load_async(d_smem[toc], g.d, vec_idx, vec_b[toc]);
                }

                wait(compute_done[tic], ((qo_idx - q_start) / 2) % 2);
            }
        } else if (warpid % WARPGROUP_WARPS == 1) {
            for (auto qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                wait(compute_done[tic], ((qo_idx - q_start) / 2) % 2);

                coord<qg_tile> tile_idx = {batch_idx, q_head_idx, qo_idx, 0};
                warp::tma::store_add_async(g.qg, qg_smem, tile_idx);
                warp::tma::store_async_wait();

                if (laneid() == 0) arrive(qg_ready);
            }
        }
    } else {
        rt_fl<16, G::tile_width> kg_reg, vg_reg;
        rt_fl<16, 64> s_block_t, p_block_t;
        rt_fl<16, 64> ds_block_t, dp_block_t;
        rt_bf<16, 64> ds_block_t_mma, p_block_t_mma;

        warp::zero(kg_reg);
        warp::zero(vg_reg);

        if (warpgroupid == 0) {
            warpgroup::increase_registers<256>();
            wait(kv_b, 0);
            if (g.cos != nullptr && g.sin != nullptr) {
                group<BWD_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>::sync(10);
                rope_tile_forward(k_smem[0], g.cos, g.sin,
                                  (block_idx * BWD_CONSUMER_WARPGROUPS) * G::tile_h,
                                  warpgroup::laneid(),
                                  kittens::WARPGROUP_WARPS * kittens::WARP_THREADS);
                rope_tile_forward(k_smem[1], g.cos, g.sin,
                                  ((block_idx * BWD_CONSUMER_WARPGROUPS) + 1) * G::tile_h,
                                  warpgroup::laneid(),
                                  kittens::WARPGROUP_WARPS * kittens::WARP_THREADS);
                group<BWD_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>::sync(10);
            }
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                compute_bwd_loop<is_causal, G::tile_h_qo, G::tile_h, G::tile_width, D>(
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                    kg_reg, vg_reg,
                    q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                    g.cos, g.sin,
                    qo_idx, q_start, tic, toc);

                rt_fl<16, G::tile_width> qg_reg;
                warpgroup::mm_AtB(qg_reg, ds_smem[0], k_smem[0]);
                warpgroup::mma_AtB(qg_reg, ds_smem[1], k_smem[1]);
                warpgroup::mma_commit_group();

                wait(qg_ready, toc);
                if (qo_idx > 0) warp::tma::store_async_wait();

                warpgroup::mma_async_wait();
                warpgroup::store(qg_smem, qg_reg);
                group<4>::sync(warpgroup::groupid() + 4);

                if (warpgroup::laneid() == 0) arrive(compute_done[tic]);
            }
            kv_store<kg_tile, vg_tile>(kg_smem, kg_reg, vg_smem, vg_reg, g, qg_ready, kv_head_idx, toc);
        } else {
            warpgroup::increase_registers<224>();
            wait(kv_b, 0);
            if (g.cos != nullptr && g.sin != nullptr) {
                group<BWD_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>::sync(10);
                group<BWD_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS>::sync(10);
            }
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                compute_bwd_loop<is_causal, G::tile_h_qo, G::tile_h, G::tile_width, D>(
                    vec_b, q_b, o_b,
                    s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                    kg_reg, vg_reg,
                    q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                    g.cos, g.sin,
                    qo_idx, q_start, tic, toc);
            }
            kv_store<kg_tile, vg_tile>(kg_smem, kg_reg, vg_smem, vg_reg, g, qg_ready, kv_head_idx, toc);
        }
    }
}

inline constexpr int bwd_sequence_granularity() {
    return 4 * ::kittens::TILE_ROW_DIM<bf16> * 4;
}
// H100 backward supports HS=64 and HS=128.
inline bool bwd_supports_head_dim(int HS) { return HS == 64 || HS == 128; }

template<int D>
inline void launch_backward_causal_gqa(bf16* q, bf16* k, bf16* v, bf16* o,
                                       float* l, bf16* og, float* d,
                                       float* qg, float* kg, float* vg,
                                       int B, int NH, int NKVH, int T, cudaStream_t stream,
                                       const bf16* cos = nullptr, const bf16* sin = nullptr) {
    static_assert(D == 64 || D == 128);
    assert(NKVH > 0);
    assert(NH % NKVH == 0);
    assert(T % bwd_sequence_granularity() == 0 &&
           "TK MHA backward requires T to be divisible by 256");

    using prep_globals = bwd_prep_globals<D>;
    using prep_og_global = typename prep_globals::og_gl;
    using prep_o_global  = typename prep_globals::o_gl;
    using prep_d_global  = typename prep_globals::d_gl;

    prep_og_global prep_og_arg{og, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                               static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    prep_o_global prep_o_arg{o, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                             static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    prep_d_global prep_d_arg{d, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                             1U, static_cast<unsigned int>(T)};
    prep_globals prep_g{prep_og_arg, prep_o_arg, prep_d_arg};

    using prep_og_tile = st_bf<64, D>;
    using prep_o_tile  = st_bf<64, D>;
    using prep_d_tile  = col_vec<st_fl<64, D>>;
    size_t prep_mem_size = sizeof(prep_og_tile) * 4 + sizeof(prep_o_tile) * 4
                         + sizeof(prep_d_tile) * 4 + 4096;

    cudaCheck(cudaFuncSetAttribute(
        bwd_attend_prep_ker<D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        prep_mem_size));

    dim3 grid_prep(T / bwd_sequence_granularity(), NH, B);
    bwd_attend_prep_ker<D><<<grid_prep, 4 * kittens::WARP_THREADS,
                            prep_mem_size, stream>>>(prep_g);
    cudaCheck(cudaGetLastError());

    using globals = bwd_globals<D>;
    using q_global  = typename globals::q_gl;
    using k_global  = typename globals::k_gl;
    using v_global  = typename globals::v_gl;
    using og_global = typename globals::og_gl;
    using qg_global = typename globals::qg_gl;
    using kg_global = typename globals::kg_gl;
    using vg_global = typename globals::vg_gl;
    using l_global  = typename globals::l_gl;
    using d_global  = typename globals::d_gl;

    q_global q_arg{q, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    k_global k_arg{k, static_cast<unsigned int>(B), static_cast<unsigned int>(NKVH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    v_global v_arg{v, static_cast<unsigned int>(B), static_cast<unsigned int>(NKVH),
                   static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    og_global og_arg{og, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                     static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    qg_global qg_arg{qg, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                     static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    kg_global kg_arg{kg, static_cast<unsigned int>(B), static_cast<unsigned int>(NKVH),
                     static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    vg_global vg_arg{vg, static_cast<unsigned int>(B), static_cast<unsigned int>(NKVH),
                     static_cast<unsigned int>(T), static_cast<unsigned int>(D)};
    l_global l_arg{l, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   1U, static_cast<unsigned int>(T)};
    d_global d_arg{d, static_cast<unsigned int>(B), static_cast<unsigned int>(NH),
                   1U, static_cast<unsigned int>(T)};

    globals g{q_arg, k_arg, v_arg, og_arg, qg_arg, kg_arg, vg_arg, l_arg, d_arg, cos, sin, T, NH / NKVH};

    constexpr int main_smem = D == 64 ? 117760 : 183296;
    cudaCheck(cudaFuncSetAttribute(
        bwd_attend_ker<D, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        main_smem));

    dim3 grid_bwd(T / (4 * BWD_CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16>), NH, B);
    bwd_attend_ker<D, true><<<grid_bwd, kittens::WARP_THREADS * BWD_NUM_WORKERS,
                             main_smem, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

template<int D>
inline void launch_backward_causal(bf16* q, bf16* k, bf16* v, bf16* o,
                                   float* l, bf16* og, float* d,
                                   float* qg, float* kg, float* vg,
                                   int B, int NH, int T, cudaStream_t stream) {
    launch_backward_causal_gqa<D>(q, k, v, o, l, og, d, qg, kg, vg,
                                  B, NH, NH, T, stream);
}

inline void launch_backward_causal(bf16* q, bf16* k, bf16* v, bf16* o,
                                   float* l, bf16* og, float* d,
                                   float* qg, float* kg, float* vg,
                                   int B, int NH, int T, int HS, cudaStream_t stream) {
    if (HS == 64) {
        launch_backward_causal<64>(q, k, v, o, l, og, d, qg, kg, vg, B, NH, T, stream);
    } else if (HS == 128) {
        launch_backward_causal<128>(q, k, v, o, l, og, d, qg, kg, vg, B, NH, T, stream);
    } else {
        fprintf(stderr, "attention_backward: TK MHA only supports head_dim 64 or 128, got %d\n", HS);
        exit(EXIT_FAILURE);
    }
}

} // namespace llmk::attention
