/*
fused_classifier_sm120.cuh — ThunderKittens masked fused classifier (CE + grad)
for SM120.

Variant of the existing `fused_classifier.cuh` that:
  - Accepts a per-row loss_mask (bf16, 1.0 for include / 0.0 to skip) so SFT
    response-only loss can be computed in the same fused pass.
  - Optionally writes dlogits at the same time (set `WriteDlogits=true`).

One warp per row. Same online-max + sumexp + target lookup pattern as the
existing kernel.
*/
#pragma once

#include "tk_common.cuh"
#include <cmath>

namespace llmk::fused_classifier_masked {

using namespace ::kittens;

struct globals {
    bf16*       logits;       // [rows, vocab] (in/out — overwritten with dlogits if WriteDlogits)
    float*      losses;       // [rows]
    const int*  targets;      // [rows]
    const bf16* loss_mask;    // [rows]
    float*      mean_loss;    // [1]
    int rows;
    int vocab;
    int ignore_index;
    bool write_dlogits;
};

__global__ void fused_classifier_masked_kernel(globals g) {
    int row = blockIdx.x;
    int target = g.targets[row];
    float mask = __bfloat162float(g.loss_mask[row]);
    bool active = (target != g.ignore_index) && (mask > 0.f);

    bf16* row_logits = g.logits + row * g.vocab;

    // Row max
    float local_max = -INFINITY;
    for (int v = threadIdx.x; v < g.vocab; v += blockDim.x) {
        float val = __bfloat162float(row_logits[v]);
        if (val > local_max) local_max = val;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, off);
        if (other > local_max) local_max = other;
    }
    float row_max = __shfl_sync(0xFFFFFFFF, local_max, 0);

    // sumexp
    float local_sum = 0.f;
    for (int v = threadIdx.x; v < g.vocab; v += blockDim.x) {
        local_sum += expf(__bfloat162float(row_logits[v]) - row_max);
    }
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);
    float sumexp = __shfl_sync(0xFFFFFFFF, local_sum, 0);
    float log_z  = row_max + logf(sumexp);

    if (threadIdx.x == 0) {
        if (active) {
            float target_logit = __bfloat162float(row_logits[target]);
            float loss = log_z - target_logit;
            g.losses[row] = loss * mask;
        } else {
            g.losses[row] = 0.f;
        }
    }

    if (g.write_dlogits) {
        // dlogits[v] = (softmax(v) - 1{v==target}) * mask / N_valid
        // The total N_valid is computed in a follow-up reduce; here we
        // write (softmax - one_hot) * mask for later normalisation.
        for (int v = threadIdx.x; v < g.vocab; v += blockDim.x) {
            float p = expf(__bfloat162float(row_logits[v]) - row_max) / sumexp;
            float grad = (active ? (p - (v == target ? 1.f : 0.f)) * mask : 0.f);
            row_logits[v] = __float2bfloat16(grad);
        }
    }
}

inline void launch(bf16* logits, float* losses,
                   const int* targets, const bf16* loss_mask, float* mean_loss,
                   int rows, int vocab, int ignore_index, bool write_dlogits,
                   cudaStream_t stream) {
    globals g{logits, losses, targets, loss_mask, mean_loss,
              rows, vocab, ignore_index, write_dlogits};
    fused_classifier_masked_kernel<<<rows, ::kittens::WARP_THREADS, 0, stream>>>(g);
    cudaCheck(cudaGetLastError());
}

}  // namespace llmk::fused_classifier_masked
