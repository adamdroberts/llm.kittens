/*
rope_tk.cuh — ThunderKittens bf16 RoPE wrappers.

Source: ThunderKittens/kernels/rotary/rotary.cu. This fork removes the Torch
extension/harness glue and exposes stream launchers for llm.kittens' raw-pointer
API. The backward pass is the inverse rotation, equivalent to reusing the same
kernel with `sin -> -sin`.
*/
#pragma once

#include "tk_common.cuh"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

namespace llmk::rope {

using namespace ::kittens;
using namespace ::kittens::prototype;
using namespace ::kittens::prototype::lcsf;

template <int _headdim, int _warps>
struct rotary_layout {
    static constexpr int headdim = _headdim;
    static constexpr int warps = _warps;
    using seq_tile = st_bf<16, headdim>;
    using seq_global = gl<bf16, -1, -1, -1, headdim, seq_tile>;
    using rope_global = gl<bf16, 1, 1, -1, headdim / 2>;

    struct globals {
        seq_global o, x;
        rope_global sin, cos;
        int batches;
    };
    struct input_block { seq_tile x[warps]; };
    struct output_block { seq_tile o[warps]; };
    struct producer_state { int active_warps; };
    struct consumer_state { rt_fl<16, headdim / 2> sin, cos; };
};

template <int _headdim, bool Inverse>
struct rotary_template {
    static constexpr int headdim = _headdim;
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int OUTPUT_PIPE_STAGES = 3;
    static constexpr int INPUT_PIPE_STAGES = 3;
    using layout = rotary_layout<headdim, NUM_CONSUMER_WARPS>;

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        const int batch_block = static_cast<int>(blockIdx.y);
        if (args.task_iter == 0) {
            args.num_iters = min(
                args.globals.batches,
                (int)(args.globals.x.batch() - batch_block * args.globals.batches)
            ) * args.globals.x.depth();
        } else {
            args.num_iters = -1;
        }
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
            const int row_block = static_cast<int>(blockIdx.x);
            args.state.active_warps = min(
                (int)NUM_CONSUMER_WARPS,
                (int)(args.globals.x.rows() / 16 - row_block * NUM_CONSUMER_WARPS)
            );
        }

        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == args.iter % 4) {
                const int batch_block = static_cast<int>(blockIdx.y);
                const int row_block = static_cast<int>(blockIdx.x);
                kittens::coord idx = {
                    batch_block * args.globals.batches + args.iter / args.globals.x.depth(),
                    args.iter % args.globals.x.depth(),
                    row_block * NUM_CONSUMER_WARPS,
                    0
                };
                warp::tma::expect_bytes(args.inputs_arrived, sizeof(typename layout::seq_tile) * args.state.active_warps);
                for (int i = 0; i < args.state.active_warps; i++) {
                    warp::tma::load_async(args.input.x[i], args.globals.x, {idx.b, idx.d, idx.r + i, idx.c}, args.inputs_arrived);
                }
                if (laneid() == 0) { arrive(args.inputs_arrived, 3); }
                __syncwarp();
            }
        }

        __device__ static void store(producer_store_args<layout> args) {
            if (warpgroup::warpid() == args.iter % 4) {
                const int batch_block = static_cast<int>(blockIdx.y);
                const int row_block = static_cast<int>(blockIdx.x);
                kittens::coord idx = {
                    batch_block * args.globals.batches + args.iter / args.globals.x.depth(),
                    args.iter % args.globals.x.depth(),
                    row_block * NUM_CONSUMER_WARPS,
                    0
                };
                for (int i = 0; i < args.state.active_warps; i++) {
                    warp::tma::store_async(args.globals.o, args.output.o[i], {idx.b, idx.d, idx.r + i, idx.c});
                }
                warp::tma::store_async_read_wait();
                if (laneid() == 0) { arrive(args.outputs_finished, 4); }
                __syncwarp();
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS / 4>();
            const int row_block = static_cast<int>(blockIdx.x);
            kittens::coord idx = {row_block * NUM_CONSUMER_WARPS + warpid(), 0};
            warp::load(args.state.sin, args.globals.sin, idx);
            warp::load(args.state.cos, args.globals.cos, idx);
        }

        __device__ static void compute(consumer_compute_args<layout> args) {
            rt_fl<16, headdim> x;
            rt_fl<16, headdim / 2> x1, x2, x1_cos, x2_cos, x1_sin, x2_sin;
            warp::load(x, args.input.x[warpid()]);
            if (laneid() == 0) { arrive(args.inputs_finished); }
            __syncwarp();

            for (int i = 0; i < headdim / 32; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    x1.tiles[0][i].data[j] = x.tiles[0][i].data[j];
                    x2.tiles[0][i].data[j] = x.tiles[0][i + headdim / 32].data[j];
                }
            }

            warp::mul(x1_cos, x1, args.state.cos);
            warp::mul(x2_cos, x2, args.state.cos);
            warp::mul(x1_sin, x1, args.state.sin);
            warp::mul(x2_sin, x2, args.state.sin);

            if constexpr (Inverse) {
                warp::add(x1, x1_cos, x2_sin);
                warp::mul(x1_sin, x1_sin, -1.f);
                warp::add(x2, x2_cos, x1_sin);
            } else {
                warp::mul(x2_sin, x2_sin, -1.f);
                warp::add(x1, x1_cos, x2_sin);
                warp::add(x2, x2_cos, x1_sin);
            }

            for (int i = 0; i < headdim / 32; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    x.tiles[0][i].data[j] = x1.tiles[0][i].data[j];
                    x.tiles[0][i + headdim / 32].data[j] = x2.tiles[0][i].data[j];
                }
            }
            warp::store(args.output.o[warpid()], x);
            __syncwarp();
            if (laneid() == 0) { arrive(args.outputs_arrived); }
        }

        __device__ static void finish(consumer_finish_args<layout> args) {
            if (laneid() == 0) { arrive(args.finish_finished); }
        }
    };
};

template <int D, bool Inverse>
inline void launch_impl(bf16* out, const bf16* x, const bf16* cos, const bf16* sin,
                        int B, int H, int T, cudaStream_t stream) {
    static_assert(D == 64 || D == 128);
    assert(B > 0 && H > 0 && T > 0);
    assert(T % 16 == 0);

    using rope_t = rotary_template<D, Inverse>;
    constexpr int BATCHES_PER_BLOCK = 4;

    using seq_globals = typename rope_t::layout::seq_global;
    using rope_globals = typename rope_t::layout::rope_global;
    using globals = typename rope_t::layout::globals;

    seq_globals out_arg{out, static_cast<unsigned int>(B), static_cast<unsigned int>(H),
                        static_cast<unsigned int>(T), nullptr};
    seq_globals x_arg{const_cast<bf16*>(x), static_cast<unsigned int>(B), static_cast<unsigned int>(H),
                      static_cast<unsigned int>(T), nullptr};
    rope_globals sin_arg{const_cast<bf16*>(sin), nullptr, nullptr, static_cast<unsigned int>(T), nullptr};
    rope_globals cos_arg{const_cast<bf16*>(cos), nullptr, nullptr, static_cast<unsigned int>(T), nullptr};
    globals g{out_arg, x_arg, sin_arg, cos_arg, BATCHES_PER_BLOCK};

    constexpr unsigned long mem_size = MAX_SHARED_MEMORY - 2048;
    constexpr int ROWS_PER_BLOCK = rope_t::NUM_CONSUMER_WARPS * rope_t::layout::seq_tile::rows;

    static bool smem_attr_set = false;
    if (!smem_attr_set) {
        cudaCheck(cudaFuncSetAttribute(
            prototype::lcsf::kernel<rope_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size));
        smem_attr_set = true;
    }

    dim3 grid(CEIL_DIV(T, ROWS_PER_BLOCK), CEIL_DIV(B, BATCHES_PER_BLOCK));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<rope_t>);
    kittens::prototype::lcsf::kernel<rope_t><<<grid, block, mem_size, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

inline bool supports_head_dim(int head_dim) {
    return head_dim == 64 || head_dim == 128;
}

inline void launch_forward(bf16* out, const bf16* x, const bf16* cos, const bf16* sin,
                           int B, int H, int T, int head_dim, cudaStream_t stream) {
    switch (head_dim) {
        case 64:  launch_impl<64, false>(out, x, cos, sin, B, H, T, stream); break;
        case 128: launch_impl<128, false>(out, x, cos, sin, B, H, T, stream); break;
        default:
            fprintf(stderr, "rope_forward: TK RoPE unsupported head_dim=%d\n", head_dim);
            exit(EXIT_FAILURE);
    }
}

inline void launch_backward(bf16* dx, const bf16* dout, const bf16* cos, const bf16* sin,
                            int B, int H, int T, int head_dim, cudaStream_t stream) {
    switch (head_dim) {
        case 64:  launch_impl<64, true>(dx, dout, cos, sin, B, H, T, stream); break;
        case 128: launch_impl<128, true>(dx, dout, cos, sin, B, H, T, stream); break;
        default:
            fprintf(stderr, "rope_backward: TK RoPE unsupported head_dim=%d\n", head_dim);
            exit(EXIT_FAILURE);
    }
}

} // namespace llmk::rope
