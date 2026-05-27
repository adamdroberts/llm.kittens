/*
distributed_ext.cuh — distributed primitives beyond the ZeRO path in zero.cuh.

Wraps NCCL collectives needed for MoE / TP / PP / CP and provides simple
CUDA-side glue (e.g. shard slicing) used by tensor parallelism.

For tensor parallelism we ship the shard-slice + post-allreduce add helpers;
the underlying GEMM still lives in matmul.cuh, and the NCCL collective is
called by host code via the helpers below.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "zero.cuh"  // existing ncclCheck etc.

// ============================================================================
// all_to_all wrapper (MoE expert-parallel).
//
//   send_buf, recv_buf: same shape; partitioned by world_size.
//   count_per_rank:     elements per rank in each direction.
//
// Each rank sends its block i to rank i and receives block coming from rank i.
// ============================================================================

#ifdef MULTI_GPU
inline void all_to_all_floatx(floatX* recv_buf, const floatX* send_buf, size_t count_per_rank,
                              ncclComm_t comm, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int world_size;
    ncclCheck(ncclCommCount(comm, &world_size));
    ncclCheck(ncclGroupStart());
    for (int r = 0; r < world_size; ++r) {
        ncclCheck(ncclSend(send_buf + r * count_per_rank, count_per_rank,
                           ncclBfloat16, r, comm, stream));
        ncclCheck(ncclRecv(recv_buf + r * count_per_rank, count_per_rank,
                           ncclBfloat16, r, comm, stream));
    }
    ncclCheck(ncclGroupEnd());
}
#else
inline void all_to_all_floatx(floatX* recv_buf, const floatX* send_buf, size_t count_per_rank,
                              void* /*comm*/, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemcpyAsync(recv_buf, send_buf, sizeof(floatX) * count_per_rank, cudaMemcpyDeviceToDevice, stream);
}
#endif

// ============================================================================
// Column-parallel linear (TP):
//
//   In TP, the weight matrix W[out, in] is sharded along its `out` axis across
//   ranks. Each rank holds W_shard[out_local, in] and produces a partial
//   output y_shard[batch, out_local]. The forward is just a matmul on the
//   shard — no collective needed.
//
//   The backward does an all-reduce of dW_shard and dx (gradient flowing back
//   through the column-split linear contributes to all replicas of x).
//
// We provide:
//   * a shard-slice helper that selects out_local rows of W from the full W
//     (used during model load),
//   * a post-backward "reduce_dx" wrapper that all-reduces dx across the TP
//     group.
// ============================================================================

__global__ void column_parallel_slice_kernel(floatX* W_shard, const floatX* W_full,
                                             int out_local, int in_dim, int rank_offset) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= out_local || i >= in_dim) return;
    W_shard[o * in_dim + i] = W_full[(rank_offset + o) * in_dim + i];
}
void column_parallel_slice(floatX* W_shard, const floatX* W_full,
                           int out_local, int in_dim, int rank_offset, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(out_local, 16), CEIL_DIV(in_dim, 16));
    column_parallel_slice_kernel<<<grid, block, 0, stream>>>(W_shard, W_full, out_local, in_dim, rank_offset);
    cudaCheck(cudaGetLastError());
}

#ifdef MULTI_GPU
inline void tp_allreduce_dx(floatX* dx, size_t N, ncclComm_t comm, cudaStream_t stream) {
    NVTX_RANGE_FN();
    ncclCheck(ncclAllReduce(dx, dx, N, ncclBfloat16, ncclSum, comm, stream));
}
#else
inline void tp_allreduce_dx(floatX* /*dx*/, size_t /*N*/, void* /*comm*/, cudaStream_t /*stream*/) {
    NVTX_RANGE_FN();
}
#endif

// ============================================================================
// Row-parallel linear (TP):
//
//   W is sharded along the `in` axis. Each rank computes y_partial = x_shard @
//   W_shard^T, then the partials are all-reduced to give y.
//
// We provide:
//   * shard-slice helper for the input-axis split,
//   * post-matmul allreduce wrapper.
// ============================================================================

__global__ void row_parallel_slice_kernel(floatX* W_shard, const floatX* W_full,
                                          int out_dim, int in_local, int in_rank_offset, int in_total) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= out_dim || i >= in_local) return;
    W_shard[o * in_local + i] = W_full[o * in_total + (in_rank_offset + i)];
}
void row_parallel_slice(floatX* W_shard, const floatX* W_full,
                        int out_dim, int in_local, int in_rank_offset, int in_total, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(out_dim, 16), CEIL_DIV(in_local, 16));
    row_parallel_slice_kernel<<<grid, block, 0, stream>>>(W_shard, W_full, out_dim, in_local, in_rank_offset, in_total);
    cudaCheck(cudaGetLastError());
}

#ifdef MULTI_GPU
inline void tp_allreduce_output(floatX* y, size_t N, ncclComm_t comm, cudaStream_t stream) {
    NVTX_RANGE_FN();
    ncclCheck(ncclAllReduce(y, y, N, ncclBfloat16, ncclSum, comm, stream));
}
#else
inline void tp_allreduce_output(floatX* /*y*/, size_t /*N*/, void* /*comm*/, cudaStream_t /*stream*/) {
    NVTX_RANGE_FN();
}
#endif

// ============================================================================
// Sequence parallelism: shard the activation along the sequence axis across
// the TP group. Norm/dropout run on the local shard; the boundaries are
// handled by an all-gather inside the TP region.
//
// We provide an all-gather helper for the SP shard along axis=1 of [B, S/N, D]
// to produce [B, S, D].
// ============================================================================

#ifdef MULTI_GPU
inline void sp_allgather(floatX* recv_full, const floatX* send_shard, size_t shard_count,
                         ncclComm_t comm, cudaStream_t stream) {
    NVTX_RANGE_FN();
    ncclCheck(ncclAllGather(send_shard, recv_full, shard_count, ncclBfloat16, comm, stream));
}
#else
inline void sp_allgather(floatX* recv_full, const floatX* send_shard, size_t shard_count,
                         void* /*comm*/, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemcpyAsync(recv_full, send_shard, sizeof(floatX) * shard_count, cudaMemcpyDeviceToDevice, stream);
}
#endif

// ============================================================================
// Pipeline parallelism: simple send / recv wrappers between adjacent ranks.
// Real 1F1B / zero-bubble schedules are orchestrated host-side; these are the
// primitive ops.
// ============================================================================

#ifdef MULTI_GPU
inline void pp_send(const floatX* buf, size_t N, int dst_rank, ncclComm_t comm, cudaStream_t stream) {
    NVTX_RANGE_FN();
    ncclCheck(ncclSend(buf, N, ncclBfloat16, dst_rank, comm, stream));
}
inline void pp_recv(floatX* buf, size_t N, int src_rank, ncclComm_t comm, cudaStream_t stream) {
    NVTX_RANGE_FN();
    ncclCheck(ncclRecv(buf, N, ncclBfloat16, src_rank, comm, stream));
}
#else
inline void pp_send(const floatX*, size_t, int, void*, cudaStream_t) {}
inline void pp_recv(floatX*, size_t, int, void*, cudaStream_t) {}
#endif

// ============================================================================
// Context parallelism (ring attention coord): each rank holds a Q,K,V shard
// along sequence axis; we rotate K,V across ranks so every Q sees every K,V.
// The local-step kernel is in attention_ext.cuh; here we provide the ring
// rotate communication.
// ============================================================================

#ifdef MULTI_GPU
inline void cp_ring_send_recv(floatX* recv_buf, const floatX* send_buf, size_t N,
                              int next_rank, int prev_rank, ncclComm_t comm, cudaStream_t stream) {
    NVTX_RANGE_FN();
    ncclCheck(ncclGroupStart());
    ncclCheck(ncclSend(send_buf, N, ncclBfloat16, next_rank, comm, stream));
    ncclCheck(ncclRecv(recv_buf, N, ncclBfloat16, prev_rank, comm, stream));
    ncclCheck(ncclGroupEnd());
}
#else
inline void cp_ring_send_recv(floatX* recv_buf, const floatX* send_buf, size_t N,
                              int /*next*/, int /*prev*/, void*, cudaStream_t stream) {
    NVTX_RANGE_FN();
    cudaMemcpyAsync(recv_buf, send_buf, sizeof(floatX) * N, cudaMemcpyDeviceToDevice, stream);
}
#endif
