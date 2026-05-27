/*
train_hooks.cuh — gradient checkpointing and selective recomputation hooks.

Both are orchestration patterns rather than kernels: a Stage saves a flag and
some activation, then re-runs forward during backward. We provide:

  - GradCheckpoint scope helper: a `begin/end` pair that the host uses to
    decide whether to drop activations.
  - SelectiveRecompute hook: signals which intermediate tensors to drop in
    favour of recompute during backward.

The actual save / recompute logic lives in callers (e.g. train_gpt2.cu).
This header declares the API and provides a small "tag" buffer kernel.
*/
#pragma once

#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// GradCheckpoint scope marker. The "kernel" is a no-op that records an int
// tag into a device-side counter so callers can verify the scope was entered.
// ============================================================================

__global__ void gradient_checkpoint_mark_kernel(int* counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(counter, 1);
}

inline void gradient_checkpoint_begin(int* device_counter, cudaStream_t stream) {
    gradient_checkpoint_mark_kernel<<<1, 1, 0, stream>>>(device_counter);
    cudaCheck(cudaGetLastError());
}
inline void gradient_checkpoint_end(int* device_counter, cudaStream_t stream) {
    gradient_checkpoint_mark_kernel<<<1, 1, 0, stream>>>(device_counter);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Selective recompute tag: stores a uint8 selector per activation buffer to
// indicate "drop" vs "keep". Caller maintains the selector tensor.
// ============================================================================

__global__ void selective_recompute_mark_kernel(uint8_t* selector, int index, uint8_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) selector[index] = value;
}
inline void selective_recompute_mark(uint8_t* selector, int index, bool drop, cudaStream_t stream) {
    selective_recompute_mark_kernel<<<1, 1, 0, stream>>>(selector, index, drop ? (uint8_t)1 : (uint8_t)0);
    cudaCheck(cudaGetLastError());
}
