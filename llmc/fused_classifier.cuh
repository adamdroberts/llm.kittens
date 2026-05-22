/*
Fused Classifier:
- Forwards the Cross Entropy Loss
- Never materializes the full normalized logits, only at the target label
- (fusion) Also kicks off the backward pass, because everything is already loaded
*/
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

#if defined(KITTENS_SM120) && !defined(LLMK_SM120_CLASSIFIER_BLOCK_SIZE)
#define LLMK_SM120_CLASSIFIER_BLOCK_SIZE 1024
#endif
#if defined(KITTENS_SM120) && !defined(LLMK_SM120_CLASSIFIER_LOSS_ONLY_BLOCK_SIZE)
#define LLMK_SM120_CLASSIFIER_LOSS_ONLY_BLOCK_SIZE LLMK_SM120_CLASSIFIER_BLOCK_SIZE
#endif

// ----------------------------------------------------------------------------
// CUDA kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

__device__ inline float classifier_exp(float x) {
#if defined(KITTENS_SM120) && defined(LLMK_SM120_CLASSIFIER_EXP2)
    return exp2f(x * 1.4426950408889634f);
#else
    return expf(x);
#endif
}

__device__ inline float classifier_block_reduce_max(float val, float* shared) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warpReduceMax(val);
    if (lane_id == 0) {
        shared[warp_id] = warp_val;
    }
    __syncthreads();

    warp_val = (lane_id < num_warps) ? shared[lane_id] : -INFINITY;
    float block_val = warpReduceMax(warp_val);
    __syncthreads();
    return block_val;
}

__device__ inline float classifier_block_reduce_sum(float val, float* shared) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warpReduceSum(val);
    if (lane_id == 0) {
        shared[warp_id] = warp_val;
    }
    __syncthreads();

    warp_val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
    float block_val = warpReduceSum(warp_val);
    __syncthreads();
    return block_val;
}

__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P,
                                                    float* max_shared, float* sum_shared) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
#if defined(KITTENS_SM120)
    // The upstream reverse/tail traversal hangs on RTX 5090 once the block has
    // four or more warps for the small unaligned smoke shape. Use a simpler
    // two-pass traversal on SM120: one vectorized pass for the row max, then a
    // second vectorized pass for the sumexp under the block max.
    const int vecs = V / x128::size;
    const int tail_start = vecs * x128::size;
    for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
        x128 packed_x = load128(x + i * x128::size);
        for (int k = 0; k < x128::size; ++k) {
            thread_maxval = fmaxf(thread_maxval, (float)packed_x[k]);
        }
    }
    for (int i = tail_start + threadIdx.x; i < V; i += blockDim.x) {
        thread_maxval = fmaxf(thread_maxval, (float)x[i]);
    }

    float block_maxval = classifier_block_reduce_max(thread_maxval, max_shared);

    for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
        x128 packed_x = load128(x + i * x128::size);
        for (int k = 0; k < x128::size; ++k) {
            thread_sumval += classifier_exp((float)packed_x[k] - block_maxval);
        }
    }
    for (int i = tail_start + threadIdx.x; i < V; i += blockDim.x) {
        thread_sumval += classifier_exp((float)x[i] - block_maxval);
    }

    float block_sumval = classifier_block_reduce_sum(thread_sumval, sum_shared);
    return SoftmaxParams{1.f / block_sumval, block_maxval};
#else
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= classifier_exp((old_maxval - thread_maxval));
            thread_sumval += classifier_exp(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        x128 packed_x = load128(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= classifier_exp((old_maxval - thread_maxval));
            thread_sumval += classifier_exp(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    (void)max_shared;
    (void)sum_shared;
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
    thread_sumval *= classifier_exp(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
#endif
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
//
// NOTE: __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS) was removed for the
// llm.kittens fork: with sm_90a + nvcc -O3 the launch_bounds hint causes the
// kernel to hang (cudaDeviceSynchronize times out at 300s) even with
// minBlocks=1. The kernel is correct without the hint and the actual
// occupancy is fine for our use cases. Restore once we identify whether this
// is an nvcc 12.8 bug specific to compute_90a.
template <bool WriteDLogits = true, bool WriteProbs = false>
__global__ void
    fused_classifier_kernel5(floatX* logits, float* losses, floatX* probs,
                                const float dloss, const int* targets,
                                int B, int T, int V, int P, std::bool_constant<WriteDLogits>) {
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx = gridDim.x - (blockIdx.x+1); // reverse order for cache hits on matmul data
    int ix = targets[idx];

    __shared__ float max_shared[WARP_SIZE];
    __shared__ float sum_shared[WARP_SIZE];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P, max_shared, sum_shared);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = classifier_exp((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] -= logf(prob);
    }

    if constexpr (!WriteDLogits && !WriteProbs) {
        return;
    }

    // without this synchronization point we have a race condition:
    // the logits used above to compute the loss are concurrently (race) modified to carry backward pass grads.
    // since the "logits" are overwritten to be in the [-1, 1] range and sp.Offset is sometimes smaller than -90
    // we errouneously end up computing exp^(90+) which gives us infinities in the loss! this is the fix.
    __syncthreads();

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = load128(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = classifier_exp((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteDLogits){
            // reduce cache persistence for the overwritten logits
            // to maximise probability that logits remain in cache between prepare_softmax and here
            store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * x128::size, packed_probs);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = classifier_exp((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteDLogits){
            __stcs(logits + idx * P + i, (floatX)dlogit);
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }

}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type, bool WriteDLogits>
void fused_classifier(Type* logits, float* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, cudaStream_t stream) {
    NVTX_RANGE_FN();
    // Was 1024 in upstream llm.c; reduced to 256 for the H100 build because
    // the 1024-thread variant hangs on cudaDeviceSynchronize under sm_90a
    // (300s timeout) — see notes above the kernel definition. SM120 currently
    // hung at 128+ threads in the generic blockReduce path on SM120. The
    // classifier-local reduction above keeps explicit shared arrays and a
    // trailing sync for each reduction, so SM120 can use a larger block again.
#if defined(KITTENS_SM120)
    const int block_size = WriteDLogits ? LLMK_SM120_CLASSIFIER_BLOCK_SIZE : LLMK_SM120_CLASSIFIER_LOSS_ONLY_BLOCK_SIZE;
#else
    const int block_size = 256;
#endif
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, (floatX*)NULL, dloss, targets, B, T, V, P, write_dlogits);
    cudaCheck(cudaGetLastError());
}
