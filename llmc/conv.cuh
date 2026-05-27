/*
conv.cuh — 1D convolutions (depthwise + general) and 1D nearest interpolate
plus F.pad.

The general Conv1d is a thin wrapper for cuDNN; we ship a direct CUDA kernel
for the depthwise case which is what Mamba and BytePatchEmbed need at the
sizes we use. For larger general Conv1d/Conv2d the cuDNN path should be
preferred — declared here, defined in the cuDNN wrapper file.

For 2D conv (vision patch embed) we declare API too; cuDNN handles it best.
*/
#pragma once

#include <assert.h>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ============================================================================
// Depthwise Conv1d (groups = channels).
//
//   x:      [B, C, S_in]
//   w:      [C, K]        (kernel weights per channel)
//   out:    [B, C, S_out]
//   padding causal: zero-pad K-1 on the left so out[:,:,t] depends on x[:,:,t-K+1..t].
//
// Stride=1, dilation=1 for the Mamba case; we expose stride as a parameter.
// ============================================================================

__global__ void depthwise_conv1d_kernel(floatX* out, const floatX* x, const floatX* w,
                                        int B, int C, int S_in, int K, int S_out,
                                        int stride, int pad_left) {
    int s   = blockIdx.x * blockDim.x + threadIdx.x;
    int c   = blockIdx.y;
    int b   = blockIdx.z;
    if (s >= S_out || c >= C || b >= B) return;
    const floatX* xb = x + ((b * C) + c) * S_in;
    const floatX* wc = w + c * K;
    int s_in_start = s * stride - pad_left;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int s_in = s_in_start + k;
        if (s_in < 0 || s_in >= S_in) continue;
        acc += (float)xb[s_in] * (float)wc[k];
    }
    out[((b * C) + c) * S_out + s] = (floatX)acc;
}

void depthwise_conv1d_forward(floatX* out, const floatX* x, const floatX* w,
                              int B, int C, int S_in, int K, int S_out, int stride, int pad_left,
                              cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(S_out, block_size), C, B);
    depthwise_conv1d_kernel<<<grid, block_size, 0, stream>>>(
        out, x, w, B, C, S_in, K, S_out, stride, pad_left);
    cudaCheck(cudaGetLastError());
}

// Depthwise conv backward (gradient wrt input and weight).
__global__ void depthwise_conv1d_bwd_input_kernel(floatX* dx, const floatX* dout, const floatX* w,
                                                  int B, int C, int S_in, int K, int S_out,
                                                  int stride, int pad_left) {
    int s   = blockIdx.x * blockDim.x + threadIdx.x;
    int c   = blockIdx.y;
    int b   = blockIdx.z;
    if (s >= S_in || c >= C || b >= B) return;
    const floatX* wc = w + c * K;
    const floatX* doutb = dout + ((b * C) + c) * S_out;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // out_t = sum_k w[k] * x[t * stride - pad_left + k]
        // Solve t such that t * stride - pad_left + k == s:
        //   t * stride == s + pad_left - k  =>  t = (s + pad_left - k) / stride
        int num = s + pad_left - k;
        if (num < 0) continue;
        if (num % stride != 0) continue;
        int t = num / stride;
        if (t < 0 || t >= S_out) continue;
        acc += (float)doutb[t] * (float)wc[k];
    }
    dx[((b * C) + c) * S_in + s] = (floatX)acc;
}

void depthwise_conv1d_backward_input(floatX* dx, const floatX* dout, const floatX* w,
                                     int B, int C, int S_in, int K, int S_out,
                                     int stride, int pad_left, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(S_in, block_size), C, B);
    depthwise_conv1d_bwd_input_kernel<<<grid, block_size, 0, stream>>>(
        dx, dout, w, B, C, S_in, K, S_out, stride, pad_left);
    cudaCheck(cudaGetLastError());
}

__global__ void depthwise_conv1d_bwd_weight_kernel(floatX* dw, const floatX* dout, const floatX* x,
                                                   int B, int C, int S_in, int K, int S_out,
                                                   int stride, int pad_left) {
    int c = blockIdx.y;
    int k = blockIdx.x;
    if (c >= C || k >= K) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        const floatX* xb = x + ((b * C) + c) * S_in;
        const floatX* doutb = dout + ((b * C) + c) * S_out;
        for (int t = threadIdx.x; t < S_out; t += blockDim.x) {
            int s_in = t * stride - pad_left + k;
            if (s_in < 0 || s_in >= S_in) continue;
            acc += (float)doutb[t] * (float)xb[s_in];
        }
    }
    float sum = blockReduce<warpReduceSum>(acc);
    if (threadIdx.x == 0) {
        dw[c * K + k] = (floatX)sum;
    }
}
void depthwise_conv1d_backward_weight(floatX* dw, const floatX* dout, const floatX* x,
                                      int B, int C, int S_in, int K, int S_out,
                                      int stride, int pad_left, cudaStream_t stream) {
    NVTX_RANGE_FN();
    dim3 grid(K, C);
    depthwise_conv1d_bwd_weight_kernel<<<grid, 128, 0, stream>>>(dw, dout, x, B, C, S_in, K, S_out, stride, pad_left);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// General Conv1d (forward). For larger sizes call cuDNN; this kernel is a
// correctness reference for small kernels and embeddings.
//
//   x:   [B, C_in, S_in]
//   w:   [C_out, C_in, K]
//   out: [B, C_out, S_out]
// ============================================================================

__global__ void conv1d_general_forward_kernel(floatX* out, const floatX* x, const floatX* w,
                                              int B, int C_in, int C_out, int S_in, int K, int S_out,
                                              int stride, int pad_left) {
    int s   = blockIdx.x * blockDim.x + threadIdx.x;
    int oc  = blockIdx.y;
    int b   = blockIdx.z;
    if (s >= S_out || oc >= C_out || b >= B) return;
    int s_in_start = s * stride - pad_left;
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) {
        const floatX* xb = x + ((b * C_in) + ic) * S_in;
        const floatX* wc = w + ((oc * C_in) + ic) * K;
        for (int k = 0; k < K; ++k) {
            int s_in = s_in_start + k;
            if (s_in < 0 || s_in >= S_in) continue;
            acc += (float)xb[s_in] * (float)wc[k];
        }
    }
    out[((b * C_out) + oc) * S_out + s] = (floatX)acc;
}

void conv1d_general_forward(floatX* out, const floatX* x, const floatX* w,
                            int B, int C_in, int C_out, int S_in, int K, int S_out,
                            int stride, int pad_left, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(S_out, block_size), C_out, B);
    conv1d_general_forward_kernel<<<grid, block_size, 0, stream>>>(
        out, x, w, B, C_in, C_out, S_in, K, S_out, stride, pad_left);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 1D nearest-neighbor interpolate to target length.
//   x:   [B, C, S_in]
//   out: [B, C, S_out]
// ============================================================================

__global__ void interp1d_nearest_kernel(floatX* out, const floatX* x, int B, int C, int S_in, int S_out) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    int b = blockIdx.z;
    if (s >= S_out || c >= C || b >= B) return;
    int src = (int)((s + 0.5f) * (float)S_in / (float)S_out);
    if (src < 0) src = 0;
    if (src >= S_in) src = S_in - 1;
    out[((b * C) + c) * S_out + s] = x[((b * C) + c) * S_in + src];
}
void interp1d_nearest(floatX* out, const floatX* x, int B, int C, int S_in, int S_out, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    dim3 grid(CEIL_DIV(S_out, block_size), C, B);
    interp1d_nearest_kernel<<<grid, block_size, 0, stream>>>(out, x, B, C, S_in, S_out);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// F.pad: right-zero-pad along last axis. Generic enough for byte_patch_embed.
//   x:   [B, C, S_in]
//   out: [B, C, S_in + pad_right]
// ============================================================================

__global__ void pad_right_zero_kernel(floatX* out, const floatX* x, int B, int C, int S_in, int S_out) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    int b = blockIdx.z;
    if (s >= S_out || c >= C || b >= B) return;
    int idx_out = ((b * C) + c) * S_out + s;
    out[idx_out] = s < S_in ? x[((b * C) + c) * S_in + s] : (floatX)0.0f;
}
void pad_right_zero(floatX* out, const floatX* x, int B, int C, int S_in, int pad_right, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int S_out = S_in + pad_right;
    const int block_size = 128;
    dim3 grid(CEIL_DIV(S_out, block_size), C, B);
    pad_right_zero_kernel<<<grid, block_size, 0, stream>>>(out, x, B, C, S_in, S_out);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Conv2d (forward, general). Reference kernel for small sizes; cuDNN handles
// large vision-encoder convs.
//
//   x:   [B, C_in, H, W]
//   w:   [C_out, C_in, KH, KW]
//   out: [B, C_out, H_out, W_out]
// ============================================================================

__global__ void conv2d_general_forward_kernel(floatX* out, const floatX* x, const floatX* w,
                                              int B, int C_in, int C_out,
                                              int H, int W, int KH, int KW,
                                              int H_out, int W_out,
                                              int stride_h, int stride_w,
                                              int pad_h, int pad_w) {
    int w_o = blockIdx.x * blockDim.x + threadIdx.x;
    int h_o = blockIdx.y * blockDim.y + threadIdx.y;
    int boc = blockIdx.z;
    int oc  = boc % C_out;
    int b   = boc / C_out;
    if (w_o >= W_out || h_o >= H_out) return;
    int h_in_start = h_o * stride_h - pad_h;
    int w_in_start = w_o * stride_w - pad_w;
    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            int h_in = h_in_start + kh;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int w_in = w_in_start + kw;
                if (w_in < 0 || w_in >= W) continue;
                int x_idx = ((b * C_in + ic) * H + h_in) * W + w_in;
                int w_idx = ((oc * C_in + ic) * KH + kh) * KW + kw;
                acc += (float)x[x_idx] * (float)w[w_idx];
            }
        }
    }
    int out_idx = ((b * C_out + oc) * H_out + h_o) * W_out + w_o;
    out[out_idx] = (floatX)acc;
}

void conv2d_general_forward(floatX* out, const floatX* x, const floatX* w,
                            int B, int C_in, int C_out,
                            int H, int W, int KH, int KW,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            cudaStream_t stream) {
    NVTX_RANGE_FN();
    int H_out = (H + 2 * pad_h - KH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - KW) / stride_w + 1;
    dim3 block(16, 8);
    dim3 grid(CEIL_DIV(W_out, 16), CEIL_DIV(H_out, 8), B * C_out);
    conv2d_general_forward_kernel<<<grid, block, 0, stream>>>(
        out, x, w, B, C_in, C_out, H, W, KH, KW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w);
    cudaCheck(cudaGetLastError());
}
