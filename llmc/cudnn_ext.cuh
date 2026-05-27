/*
cudnn_ext.cuh — cuDNN back-ends for activations, norms, convolutions, and
the cuDNN attention path.

The implementations live in `cudnn_ext.cu` (which sets up the cuDNN frontend
graph and calls cudnnExecute). Declarations only here.

cuDNN coverage:
  * activations:   sigmoid, tanh, relu, elu, gelu, swish (mostly cudnnPointwiseFwd)
  * normalisation: LayerNorm, RMSNorm, GroupNorm (cudnnNormalizationForward)
  * attention:     SDPA fwd/bwd (cudnnGraph attention API)
  * convolutions:  Conv1d / Conv2d / depthwise (cudnnConvolutionForward)
*/
#pragma once

#include "cuda_common.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ============================================================================
// Pointwise activations via cuDNN.
// ============================================================================

enum class CudnnActMode : int {
    Sigmoid = 0, Tanh = 1, Relu = 2, Elu = 3,
    Gelu = 4, Silu = 5, Softplus = 6, HardSwish = 7,
};

void cudnn_act_forward(__nv_bfloat16* out, const __nv_bfloat16* x, CudnnActMode mode,
                       int N, cudaStream_t stream);
void cudnn_act_backward(__nv_bfloat16* dx, const __nv_bfloat16* dout, const __nv_bfloat16* x,
                        CudnnActMode mode, int N, cudaStream_t stream);

// ============================================================================
// Normalisation via cuDNN.
// ============================================================================

void cudnn_layernorm_forward(__nv_bfloat16* y, float* mean, float* rstd,
                             const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                             int N, int C, float eps, cudaStream_t stream);
void cudnn_layernorm_backward(__nv_bfloat16* dx, __nv_bfloat16* dweight, __nv_bfloat16* dbias,
                              const __nv_bfloat16* dy, const __nv_bfloat16* x,
                              const __nv_bfloat16* weight, const float* mean, const float* rstd,
                              int N, int C, cudaStream_t stream);
void cudnn_rmsnorm_forward(__nv_bfloat16* y, float* rstd,
                           const __nv_bfloat16* x, const __nv_bfloat16* weight,
                           int N, int C, float eps, cudaStream_t stream);
void cudnn_groupnorm_forward(__nv_bfloat16* y, float* mean, float* rstd,
                             const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                             int B, int C, int S, int groups, float eps, cudaStream_t stream);

// ============================================================================
// Attention (cuDNN graph API).
// ============================================================================

void cudnn_sdpa_forward(__nv_bfloat16* out, const __nv_bfloat16* q, const __nv_bfloat16* k,
                        const __nv_bfloat16* v, const __nv_bfloat16* bias /*nullable*/,
                        int B, int H, int S_q, int S_k, int D, bool is_causal, float dropout_p,
                        cudaStream_t stream);
void cudnn_sdpa_backward(__nv_bfloat16* dq, __nv_bfloat16* dk, __nv_bfloat16* dv,
                         const __nv_bfloat16* dout, const __nv_bfloat16* q,
                         const __nv_bfloat16* k, const __nv_bfloat16* v,
                         int B, int H, int S_q, int S_k, int D, bool is_causal,
                         cudaStream_t stream);

// ============================================================================
// Convolution via cuDNN.
// ============================================================================

void cudnn_conv1d_forward(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                          int B, int C_in, int C_out, int S_in, int K,
                          int stride, int pad, cudaStream_t stream);
void cudnn_conv1d_depthwise_forward(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                                    int B, int C, int S_in, int K,
                                    int stride, int pad, cudaStream_t stream);
void cudnn_conv2d_forward(__nv_bfloat16* y, const __nv_bfloat16* x, const __nv_bfloat16* w,
                          int B, int C_in, int C_out, int H, int W, int KH, int KW,
                          int stride_h, int stride_w, int pad_h, int pad_w,
                          cudaStream_t stream);
