/*
gemm_h100.cuh — ThunderKittens bf16 H100 GEMM, ported into header form.

Source: ThunderKittens/kernels/gemm/bf16_h100/bf16_h100_gemm.cu (lines 1-92, 97-106).

The kernel template
`matmul_template<M_BLOCK, N_BLOCK, SUPER_M, A_TRANSPOSED, B_TRANSPOSED,
APPLY_BIAS, APPLY_GELU, STORE_PRE_GELU>`
computes
    C = A · B        when B_TRANSPOSED=false
    C = A · B^T      when B_TRANSPOSED=true
    C = A^T · B      when A_TRANSPOSED=true and B_TRANSPOSED=false
where C is M×N. Non-transposed A is M×K row-major; transposed A is stored K×M
row-major. Non-transposed B is K×N row-major; transposed B is stored N×K
row-major. The `B_TRANSPOSED=true` path is the model-weight forward path used
by llm.c parameter files, whose dense weights are stored as (OC, C). The
`A_TRANSPOSED=true` path is used by matmul backward dWeight. The opt-in
finish-path flags add bias, write a pre-GELU auxiliary buffer, and write GELU
output for GPT-2's MLP up-projection path. The kernel uses
the LCF (Load-Compute-Finish) prototype with persistent grid (GRID_SIZE = 132 =
H100 SM count), TMA async loads, and WGMMA on the consumer warpgroup.

Shape constraints (must hold at launch time):
  * M divisible by  64*M_BLOCK    (default 2 → 128)
  * N divisible by  64*N_BLOCK    (default 4 → 256)
  * K divisible by  64
  * For our GPT-2/Llama-3 inner dims, all hold; the LM-head projection's N
    dimension is the trickiest case — see padding logic in the wrapper.

This file is meant to be #included only from llmc/matmul.cuh (and the GEMM
smoke test). Do not bring `kittens::` into scope outside.
*/
#pragma once

#include <type_traits>
#include "tk_common.cuh"

namespace llmk::gemm {

using namespace ::kittens;
using namespace ::kittens::prototype;
using namespace ::kittens::prototype::lcf;

template <int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using base_tile     = st_bf<64, 64>;
    using global_layout = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals      {
        global_layout A, B, C, P;
        const bf16* bias;
    };
    struct input_block  { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK * base_tile::cols> accum; };
};

template <int _M_BLOCK = 2, int _N_BLOCK = 4, int _SUPER_M = 12,
          bool _A_TRANSPOSED = false, bool _B_TRANSPOSED = false,
          bool _APPLY_BIAS = false, bool _APPLY_GELU = false, bool _STORE_PRE_GELU = false>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK;
    static constexpr int N_BLOCK = _N_BLOCK;
    static constexpr int SUPER_M = _SUPER_M;
    static constexpr bool A_TRANSPOSED = _A_TRANSPOSED;
    static constexpr bool B_TRANSPOSED = _B_TRANSPOSED;
    static constexpr bool APPLY_BIAS = _APPLY_BIAS;
    static constexpr bool APPLY_GELU = _APPLY_GELU;
    static constexpr bool STORE_PRE_GELU = _STORE_PRE_GELU;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64 * N_BLOCK>;
    using bt_tile   = st_bf<64 * N_BLOCK, 64>;
    using b_mma_tile = std::conditional_t<B_TRANSPOSED, bt_tile, wide_tile>;

    static constexpr int NUM_CONSUMER_WARPS       = M_BLOCK * 4;
    static constexpr int INPUT_PIPE_STAGES        = 4;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1;

    template<typename tile>
    __device__ static inline void apply_bias(tile& x, const bf16* bias, int col_start) {
        if (bias == nullptr) { return; }
        for (int idx = warpgroup::laneid(); idx < tile::num_elements;
             idx += kittens::WARPGROUP_WARPS * kittens::WARP_THREADS) {
            int row = idx / tile::cols;
            int col = idx % tile::cols;
            float v = (float)x[{row, col}];
            v += (float)bias[col_start + col];
            x[{row, col}] = (bf16)v;
        }
    }

    template<typename tile>
    __device__ static inline void apply_gelu(tile& x) {
        for (int idx = warpgroup::laneid(); idx < tile::num_elements;
             idx += kittens::WARPGROUP_WARPS * kittens::WARP_THREADS) {
            int row = idx / tile::cols;
            int col = idx % tile::cols;
            float v = (float)x[{row, col}];
            float cube = 0.044715f * v * v * v;
            v = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + cube)));
            x[{row, col}] = (bf16)v;
        }
    }

    template <bool PERSISTENT_GRID = true>
    __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERSISTENT_GRID
                        ? 132
                        : M * N / (M_BLOCK * N_BLOCK * layout::base_tile::num_elements));
    }

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK * 64);
        int Cblocks = args.globals.C.cols() / (N_BLOCK * 64);
        int super_rows   = (Rblocks / SUPER_M) * SUPER_M;
        int final_rows   = Rblocks - super_rows;
        int super_repeat = SUPER_M * Cblocks;
        int task_id = args.task_iter * gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks) {
            args.common.coord = { SUPER_M * (task_id / super_repeat) + task_id % SUPER_M,
                                  (task_id % super_repeat) / SUPER_M };
        } else if (task_id < Rblocks * Cblocks) {
            int remainder_id = task_id - super_rows * Cblocks;
            args.common.coord = { super_rows + (remainder_id % final_rows),
                                  remainder_id / final_rows };
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = (A_TRANSPOSED ? args.globals.A.rows() : args.globals.A.cols()) / 64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS / 4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x * M_BLOCK + id,
                              args.common.coord.y * N_BLOCK };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < M_BLOCK; i++) {
                    if constexpr (A_TRANSPOSED) {
                        tma::load_async(args.input.a[i], args.globals.A,
                                        { args.iter, args.common.coord.x + i },
                                        args.inputs_arrived);
                    } else {
                        tma::load_async(args.input.a[i], args.globals.A,
                                        { args.common.coord.x + i, args.iter },
                                        args.inputs_arrived);
                    }
                }
                for (int i = 0; i < N_BLOCK; i++) {
                    if constexpr (B_TRANSPOSED) {
                        tma::load_async(args.input.b[i], args.globals.B,
                                        { args.common.coord.y + i, args.iter },
                                        args.inputs_arrived);
                    } else {
                        tma::load_async(args.input.b[i], args.globals.B,
                                        { args.iter, args.common.coord.y + i },
                                        args.inputs_arrived);
                    }
                }
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            kittens::warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            if constexpr (A_TRANSPOSED && B_TRANSPOSED) {
                warpgroup::mma_AtBt(
                    args.state.accum,
                    args.input.a[warpgroup::groupid()],
                    reinterpret_cast<b_mma_tile&>(args.input.b));
            } else if constexpr (A_TRANSPOSED) {
                warpgroup::mma_AtB(
                    args.state.accum,
                    args.input.a[warpgroup::groupid()],
                    reinterpret_cast<b_mma_tile&>(args.input.b));
            } else if constexpr (B_TRANSPOSED) {
                warpgroup::mma_ABt(
                    args.state.accum,
                    args.input.a[warpgroup::groupid()],
                    reinterpret_cast<b_mma_tile&>(args.input.b));
            } else {
                warpgroup::mma_AB(
                    args.state.accum,
                    args.input.a[warpgroup::groupid()],
                    reinterpret_cast<b_mma_tile&>(args.input.b));
            }
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            wide_tile& c_wide = reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]);
            warpgroup::store(c_wide, args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if constexpr (APPLY_BIAS) {
                apply_bias<wide_tile>(c_wide, args.globals.bias, args.common.coord.y * 64);
                warpgroup::sync(warpgroup::groupid() + 4);
            }
            if constexpr (STORE_PRE_GELU) {
                if (warpgroup::laneid() == 0) {
                    for (int i = 0; i < N_BLOCK; i++) {
                        tma::store_async(args.globals.P,
                                         args.finish.c[warpgroup::groupid()][i],
                                         { args.common.coord.x, args.common.coord.y + i });
                        tma::store_async_read_wait();
                    }
                }
            }
            if constexpr (APPLY_GELU) {
                apply_gelu<wide_tile>(c_wide);
                warpgroup::sync(warpgroup::groupid() + 4);
            }
            if (warpgroup::laneid() == 0) {
                for (int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.C,
                                     args.finish.c[warpgroup::groupid()][i],
                                     { args.common.coord.x, args.common.coord.y + i });
                    tma::store_async_read_wait();
                }
            }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

// Default tuning. (2,4,8) is what TK's own benchmark uses for square matrices
// at N=4096. For the GPT-2 family this is correct for every dense matmul
// EXCEPT the LM head (V_padded must be a multiple of 256). For Llama-3 it's
// correct everywhere (V=128256 is a multiple of 256).
using matmul_default = matmul_template<2, 4, 8>;
using matmul_default_nt = matmul_template<2, 4, 8, false, true>;
using matmul_default_tn = matmul_template<2, 4, 8, true, false>;
using matmul_default_nt_bias = matmul_template<2, 4, 8, false, true, true, false, false>;
using matmul_default_nt_bias_gelu = matmul_template<2, 4, 8, false, true, true, true, true>;

// Smaller block — useful when N is only a multiple of 128 (e.g. the GPT-2
// LM head with V_padded=50304). M still must be a multiple of 128.
using matmul_small_n = matmul_template<2, 2, 8>;
using matmul_small_n_nt = matmul_template<2, 2, 8, false, true>;
using matmul_small_n_tn = matmul_template<2, 2, 8, true, false>;
using matmul_small_n_nt_bias = matmul_template<2, 2, 8, false, true, true, false, false>;
using matmul_small_n_nt_bias_gelu = matmul_template<2, 2, 8, false, true, true, true, true>;

// Launch helper. Caller must guarantee M, N, K satisfy the divisibility
// constraints implied by `mmt::M_BLOCK`, `mmt::N_BLOCK`, and 64 (for K).
template <typename mmt>
inline void launch(bf16* d_A, bf16* d_B, bf16* d_C, int M, int N, int K,
                   cudaStream_t stream = 0, bf16* d_pre_gelu = nullptr,
                   const bf16* d_bias = nullptr) {
    using global_layout = typename mmt::layout::global_layout;
    using globals       = typename mmt::layout::globals;
    const size_t M_ = static_cast<size_t>(M);
    const size_t N_ = static_cast<size_t>(N);
    const size_t K_ = static_cast<size_t>(K);
    global_layout Ag{ d_A, nullptr, nullptr,
                      mmt::A_TRANSPOSED ? K_ : M_,
                      mmt::A_TRANSPOSED ? M_ : K_ };
    global_layout Bg{ d_B, nullptr, nullptr,
                      mmt::B_TRANSPOSED ? N_ : K_,
                      mmt::B_TRANSPOSED ? K_ : N_ };
    global_layout Cg{ d_C, nullptr, nullptr, M_, N_ };
    global_layout Pg{ d_pre_gelu == nullptr ? d_C : d_pre_gelu, nullptr, nullptr, M_, N_ };
    globals G{ Ag, Bg, Cg, Pg, d_bias };

    static bool smem_attr_set = false;
    auto kfn = ::kittens::prototype::lcf::kernel<mmt>;
    if (!smem_attr_set) {
        cudaCheck(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kfn),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            MAX_SHARED_MEMORY - 1024));
        smem_attr_set = true;
    }

    dim3 grid_dim  = mmt::grid(M, N, K);
    dim3 block_dim = dim3(::kittens::prototype::detail::NUM_THREADS_v<mmt>);
    kfn<<<grid_dim, block_dim, MAX_SHARED_MEMORY - 1024, stream>>>(G);
}

} // namespace llmk::gemm
