#include "tk/tk_common.cuh"
#include "llmc/adamw.cuh"

// Faster AdamW using x128 (8-word) vectorized loads/stores
template <typename Tp, typename Tg>
__global__ void adamw_kernel_x128(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                                  float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                                  float grad_scale, unsigned int seed) {
    const size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= num_parameters) return;

    params_memory += blockIdx.y * w_stride;
    if (master_params_memory) master_params_memory += blockIdx.y * s_stride;
    grads_memory += blockIdx.y * g_stride;
    m_memory += blockIdx.y * s_stride;
    v_memory += blockIdx.y * s_stride;

    x128 g_val = load128(grads_memory + idx);
    x128 m_val = load128(m_memory + idx);
    x128 v_val = load128(v_memory + idx);
    
    // We can't easily vectorize the math without breaking the lerp logic or precision
    // but the LOADS and STORES are the bottleneck.
    
    for (int k = 0; k < x128::size; ++k) {
        float grad = grad_scale * (float)g_val[k];
        float m = (float)m_val[k];
        float v = (float)v_val[k];
        
        m = llmc_lerp(grad, m, beta1);
        v = llmc_lerp(grad * grad, v, beta2);
        
        m_val[k] = (float)m;
        v_val[k] = (float)v;
        
        float m_hat = m / beta1_correction;
        float v_hat = v / beta2_correction;
        
        float old_param = (master_params_memory != NULL) ? master_params_memory[idx + k] : (float)params_memory[idx + k];
        float param = old_param - (learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * old_param));
        
        stochastic_rounding(param, &params_memory[idx + k], seed + idx + k);
        if (master_params_memory != NULL) { master_params_memory[idx + k] = param; }
    }
    
    store128(m_memory + idx, m_val);
    store128(v_memory + idx, v_val);
}
