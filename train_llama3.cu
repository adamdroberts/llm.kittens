/*
Llama-3 trainer.

This file owns the Llama config, checkpoint parser, parameter layout, and the
M6 training loop. The attention path dispatches TK GQA forward where supported
and falls back to the slow GQA correctness baseline for unsupported forward
shapes and backward.
*/
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llmc/utils.h"
#include "llmc/dataloader.h"
#include "llmc/rand.h"
#include "llmc/sampler.h"
#include "llmc/logger.h"
#include "llmc/schedulers.h"
#include "llmc/outlier_detector.h"
#include "llmc/cuda_common.h"
#include "llmc/cuda_utils.cuh"
#include "llmc/encoder.cuh"
#include "llmc/matmul.cuh"
#include "llmc/rmsnorm.cuh"
#include "llmc/attention_gqa.cuh"
#include "llmc/swiglu.cuh"
#include "llmc/fused_classifier.cuh"
#include "llmc/adamw.cuh"
#include "llmc/global_norm.cuh"
#include "llmc/zero.cuh"

constexpr int LLAMA3_MAGIC = 20240803;
constexpr int LLAMA_STATE_MAGIC = 20240804;
constexpr int LLAMA_HEADER_SIZE = 256;

cudaDeviceProp deviceProp;
cudaStream_t main_stream;
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;

char filename_buffer[512];

typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int channels;
    int hidden_dim;
    int multiple_of;
    float norm_eps;
    float rope_theta;
    int use_scaled_rope;
    int max_gen_batch_size;
    int version_major;
    int version_minor;
} LlamaConfig;

constexpr const int LLAMA_NUM_PARAMETER_TENSORS = 10;
typedef struct {
    floatX* wte;      // (V, C)
    floatX* ln1w;     // (L, C)
    floatX* qkvw;     // (L, (NH + 2*NKVH)*HS, C)
    floatX* attprojw; // (L, C, C)
    floatX* ln2w;     // (L, C)
    floatX* fcw_up;   // (L, hidden_dim, C), Python c_fc / Meta w3
    floatX* fcw_gate; // (L, hidden_dim, C), Python c_fc2 / Meta w1
    floatX* fcprojw;  // (L, C, hidden_dim)
    floatX* lnfw;     // (C)
    floatX* lm_head;  // (V, C), untied
} LlamaParameterTensors;
static_assert(sizeof(LlamaParameterTensors) == LLAMA_NUM_PARAMETER_TENSORS * sizeof(void*),
              "Inconsistent LlamaParameterTensors size");

static const char* LLAMA_PARAMETER_TENSOR_NAMES[LLAMA_NUM_PARAMETER_TENSORS] = {
    "Llama wte",
    "Llama ln1w per layer",
    "Llama qkvw per layer",
    "Llama attprojw per layer",
    "Llama ln2w per layer",
    "Llama fcw_up per layer",
    "Llama fcw_gate per layer",
    "Llama fcprojw per layer",
    "Llama lnfw",
    "Llama lm_head",
};

typedef struct {
    LlamaConfig config;
    LlamaParameterTensors params;
    size_t param_elements[LLAMA_NUM_PARAMETER_TENSORS];
    size_t param_sizeof[LLAMA_NUM_PARAMETER_TENSORS];
    void* params_memory;
    void* param_shards_memory; // ZeRO-3 authoritative local parameter shard, BF16
    size_t num_parameters;
    size_t num_parameters_bytes;
    LlamaParameterTensors grads;
    void* grads_memory;
    float* m_memory;
    float* v_memory;
    float* master_weights;
    unsigned long long rng_state;
    unsigned long long rng_state_last_update;
    int use_master_weights;
    bool init_state;
} LlamaModel;

ShardInfo llama_get_tensor_at_layer(const LlamaModel* model, int layer_id, int param_tensor_id);

struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};

#define LLAMA_TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)}

constexpr const int LLAMA_NUM_ACTIVATION_TENSORS = 25;
typedef struct {
    floatX* encoded;        // (B, T, C)
    floatX* ln1;            // (L, B, T, C)
    float* ln1_rstd;        // (L, B, T)
    floatX* qkvr;           // (L, B, T, (NH + 2*NKVH)*HS), permuted Q/K/V storage
    floatX* atty;           // (L, B, T, C)
    float* att_lse;         // (L, B, NH, T)
    floatX* residual2;      // (L, B, T, C)
    floatX* ln2;            // (L, B, T, C)
    float* ln2_rstd;        // (L, B, T)
    floatX* fch_up;         // (L, B, T, hidden_dim)
    floatX* fch_gate;       // (L, B, T, hidden_dim)
    floatX* fch_swiglu;     // (L, B, T, hidden_dim)
    floatX* residual3;      // (L, B, T, C)
    floatX* lnf;            // (B, T, C)
    float* lnf_rstd;        // (B, T)
    float* losses;          // (B, T)
    floatX* output;         // logits in forward, dlogits/scratch in backward
    floatX* scratch_btc;    // (B, T, C)
    floatX* scratch_hidden; // (B, T, hidden_dim)
    floatX* scratch_hidden2;// (B, T, hidden_dim)
    floatX* matmul_scratch; // largest dWeight tensor for accumulated TK A^T*B
    floatX* att_bwd_x;      // TK GQA bwd permuted output + doutput scratch
    float* att_bwd_f;       // TK GQA bwd d/qg/kg/vg float scratch
    floatX* rope_cos;       // (T, HS/2)
    floatX* rope_sin;       // (T, HS/2)
} LlamaActivationTensors;

typedef struct {
    LlamaActivationTensors acts;
    TensorSpec specs[LLAMA_NUM_ACTIVATION_TENSORS];
    void* memory;
    int batch_size;
    int seq_len;
    int* inputs;
    int* targets;
    float mean_loss;
    float* accumulated_mean_loss;
    float* cpu_losses;
    int* workload_indices;
    int4* bucket_info;
} LlamaRuntimeState;

static int round_up_to_multiple(int value, int multiple) {
    return multiple * ((value + multiple - 1) / multiple);
}

static int llama_hidden_dim_from_config(const LlamaConfig& config) {
    if (config.hidden_dim > 0) {
        return config.hidden_dim;
    }
    if (config.channels == 2048 && config.num_layers == 16) {
        return 8192;
    }
    if (config.channels == 4096 && config.num_layers == 32) {
        return 14336;
    }
    int hidden = (4 * config.channels * 2) / 3;
    hidden = (13 * hidden) / 10; // mirrors the 1.3 multiplier used by train_llama3.py for 8B.
    return round_up_to_multiple(hidden, config.multiple_of);
}

__global__ void llama_embedding_forward_kernel(floatX* out, const int* inp,
                                               const floatX* wte, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int total = B * T * C;
    if (idx >= total) { return; }

    int bt = idx / C;
    int c = idx % C;
    int token = inp[bt];
    x128 packed = load128cs(wte + token * C + c);
    store128(out + idx, packed);
}

__global__ void llama_embedding_backward_kernel(floatX* dwte, const floatX* dout,
                                                const int* inp, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int total = B * T * C;
    if (idx >= total) { return; }

    int bt = idx / C;
    int c = idx % C;
    int token = inp[bt];
    floatX* dst = dwte + token * C + c;
    x128 packed_dst = load128(dst);
    x128 packed_dout = load128cs(dout + idx);
    for (int k = 0; k < x128::size; ++k) {
        packed_dst[k] = (floatX)((float)packed_dst[k] + (float)packed_dout[k]);
    }
    store128(dst, packed_dst);
}

__global__ void llama_add_inplace_kernel(floatX* dst, const floatX* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) { return; }
    dst[idx] = (floatX)((float)dst[idx] + (float)src[idx]);
}

__global__ void llama_residual_add_kernel(floatX* out, const floatX* a, const floatX* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) { return; }
    out[idx] = (floatX)((float)a[idx] + (float)b[idx]);
}

static void llama_add_inplace(floatX* dst, const floatX* src, size_t n, cudaStream_t stream) {
    const int block = 256;
    llama_add_inplace_kernel<<<CEIL_DIV(n, (size_t)block), block, 0, stream>>>(dst, src, n);
    cudaCheck(cudaGetLastError());
}

static void llama_residual_add(floatX* out, const floatX* a, const floatX* b, size_t n, cudaStream_t stream) {
    const int block = 256;
    llama_residual_add_kernel<<<CEIL_DIV(n, (size_t)block), block, 0, stream>>>(out, a, b, n);
    cudaCheck(cudaGetLastError());
}

static void llama_embedding_forward(floatX* out, const int* inp, const floatX* wte,
                                    int B, int T, int C, cudaStream_t stream) {
    const int block = 256;
    int total = B * T * C;
    llama_embedding_forward_kernel<<<CEIL_DIV(total, (int)(block * x128::size)), block, 0, stream>>>(
        out, inp, wte, B, T, C);
    cudaCheck(cudaGetLastError());
}

static void llama_embedding_backward(floatX* dwte, floatX* scratch,
                                     int* workload_indices, int4* bucket_info,
                                     const floatX* dout, const int* inp, const int* inputs_cpu,
                                     int B, int T, int C, unsigned int seed,
                                     cudaStream_t stream) {
    int num_c_groups = CEIL_DIV(C, x128::size * WARP_SIZE);
    assert((size_t)B * T * num_c_groups * (sizeof(int4) + sizeof(int))
           <= (size_t)B * T * C * sizeof(floatX));

    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < (uint64_t)B * T; ++bt) {
        for (uint64_t c_group = 0; c_group < (uint64_t)num_c_groups; ++c_group) {
            uint64_t data = bt + (c_group << 32ULL) + ((uint64_t)inputs_cpu[bt] << 42ULL);
            buckets[c_group + (uint64_t)num_c_groups * (uint64_t)inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sorted_buckets(buckets.begin(), buckets.end());
    std::sort(sorted_buckets.begin(), sorted_buckets.end(),
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a,
                 const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int bucket_index = 0;
    int workload_index = 0;
    for (const auto& bucket : sorted_buckets) {
        bucket_info[bucket_index].x = workload_index;
        bucket_info[bucket_index].y = bucket.second.size();
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL << 20ULL) - 1);
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL << 10ULL) - 1);
        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL << 31ULL) - 1ULL));
        }
        bucket_index++;
    }

    int4* d_bucket_info = (int4*)scratch;
    int* d_workload_indices = (int*)((char*)scratch + (size_t)B * T * num_c_groups * sizeof(int4));
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, bucket_index * sizeof(int4),
                              cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
    wte_backward_kernel<256><<<bucket_index, 256, 0, stream>>>(
        dwte, d_bucket_info, d_workload_indices, dout, inp, seed, B, T, C);
    cudaCheck(cudaGetLastError());
}

static float llama_rope_scaled_frequency(float freq) {
    constexpr float PI = 3.14159265358979323846f;
    const float scale_factor = 8.0f;
    const float low_freq_factor = 1.0f;
    const float high_freq_factor = 4.0f;
    const float old_context_len = 8192.0f;
    const float low_freq_wavelen = old_context_len / low_freq_factor;
    const float high_freq_wavelen = old_context_len / high_freq_factor;
    float wavelen = 2.0f * PI / freq;
    if (wavelen < high_freq_wavelen) {
        return freq;
    }
    if (wavelen > low_freq_wavelen) {
        return freq / scale_factor;
    }
    float smooth = (old_context_len / wavelen - low_freq_factor) /
                   (high_freq_factor - low_freq_factor);
    return (1.0f - smooth) * freq / scale_factor + smooth * freq;
}

static void llama_build_rope_cache(floatX* cos_dev, floatX* sin_dev,
                                   const LlamaConfig& config, int T, cudaStream_t stream) {
    int head_dim = config.channels / config.num_heads;
    int half_dim = head_dim / 2;
    size_t elems = (size_t)T * half_dim;
    floatX* cos_cpu = (floatX*)mallocCheck(elems * sizeof(floatX));
    floatX* sin_cpu = (floatX*)mallocCheck(elems * sizeof(floatX));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < half_dim; ++i) {
            float exponent = (float)(2 * i) / (float)head_dim;
            float freq = 1.0f / powf(config.rope_theta, exponent);
            if (config.use_scaled_rope) {
                freq = llama_rope_scaled_frequency(freq);
            }
            float angle = (float)t * freq;
            cos_cpu[(size_t)t * half_dim + i] = (floatX)cosf(angle);
            sin_cpu[(size_t)t * half_dim + i] = (floatX)sinf(angle);
        }
    }
    cudaCheck(cudaMemcpyAsync(cos_dev, cos_cpu, elems * sizeof(floatX), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(sin_dev, sin_cpu, elems * sizeof(floatX), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    free(cos_cpu);
    free(sin_cpu);
}

void fill_in_llama_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, LlamaConfig config) {
    size_t V = config.padded_vocab_size;
    size_t C = config.channels;
    size_t L = config.num_layers;
    size_t HS = config.channels / config.num_heads;
    size_t qkv_width = (config.num_heads + 2 * config.num_kv_heads) * HS;
    size_t hidden = llama_hidden_dim_from_config(config);

    param_sizes[0] = V * C;               // wte
    param_sizes[1] = L * C;               // ln1w
    param_sizes[2] = L * qkv_width * C;   // qkvw
    param_sizes[3] = L * C * C;           // attprojw
    param_sizes[4] = L * C;               // ln2w
    param_sizes[5] = L * hidden * C;      // fcw_up
    param_sizes[6] = L * hidden * C;      // fcw_gate
    param_sizes[7] = L * C * hidden;      // fcprojw
    param_sizes[8] = C;                   // lnfw
    param_sizes[9] = V * C;               // lm_head

    for (int i = 0; i < LLAMA_NUM_PARAMETER_TENSORS; ++i) {
        param_sizeof[i] = sizeof(floatX);
    }
}

floatX* malloc_and_point_llama_parameters(LlamaParameterTensors* params,
                                          const size_t* param_elements,
                                          size_t* out_num_parameters,
                                          size_t* out_num_parameters_bytes) {
    size_t num_parameters = 0;
    for (int i = 0; i < LLAMA_NUM_PARAMETER_TENSORS; ++i) {
        num_parameters += param_elements[i];
    }
    size_t num_bytes = num_parameters * sizeof(floatX);
    floatX* params_memory = NULL;
    cudaCheck(cudaMalloc((void**)&params_memory, num_bytes));

    floatX** ptrs[] = {
        &params->wte, &params->ln1w, &params->qkvw, &params->attprojw, &params->ln2w,
        &params->fcw_up, &params->fcw_gate, &params->fcprojw, &params->lnfw, &params->lm_head
    };
    char* memory_iterator = (char*)params_memory;
    for (int i = 0; i < LLAMA_NUM_PARAMETER_TENSORS; ++i) {
        assert(((uintptr_t)memory_iterator % llmk::TK_ALIGN) == 0 &&
               "TK parameter tensors must start at 128-byte-aligned offsets");
        *(ptrs[i]) = (floatX*)memory_iterator;
        memory_iterator += param_elements[i] * sizeof(floatX);
    }
    assert(((uintptr_t)memory_iterator % llmk::TK_ALIGN) == 0 &&
           "TK parameter allocation must end at a 128-byte-aligned offset");

    *out_num_parameters = num_parameters;
    *out_num_parameters_bytes = num_bytes;
    return params_memory;
}

void llama_allocate_weights(LlamaModel* model) {
    model->config.hidden_dim = llama_hidden_dim_from_config(model->config);
    fill_in_llama_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    memset(&model->params, 0, sizeof(model->params));
    memset(&model->grads, 0, sizeof(model->grads));
    model->num_parameters = 0;
    for (int i = 0; i < LLAMA_NUM_PARAMETER_TENSORS; ++i) {
        model->num_parameters += model->param_elements[i];
    }
    model->num_parameters_bytes = model->num_parameters * sizeof(floatX);
}

void llama_materialize_parameters(LlamaModel* model) {
    assert(model->params_memory == NULL);
    model->params_memory = malloc_and_point_llama_parameters(
        &model->params, model->param_elements, &model->num_parameters, &model->num_parameters_bytes);
}

void llama_allocate_zero3_parameter_shards(LlamaModel* model, MultiGpuConfig* config) {
    if (!zero_shards_parameters(config)) {
        return;
    }
    if (model->params_memory == NULL) {
        fprintf(stderr, "Need full Llama parameter buffer before initializing ZeRO-3 shards\n");
        exit(EXIT_FAILURE);
    }
    assert(model->param_shards_memory == NULL);
    cudaCheck(cudaMalloc(&model->param_shards_memory,
                         config->shard_num_parameters * sizeof(floatX)));

    floatX* shards = (floatX*)model->param_shards_memory;
    floatX* full = (floatX*)model->params_memory;
    for (int i = 0; i < LLAMA_NUM_PARAMETER_TENSORS; ++i) {
        int num_layers = (1 <= i && i <= 7) ? model->config.num_layers : 1;
        ShardInfo tensor = llama_get_tensor_at_layer(model, 0, i);
        ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, config, 1);
        ptrdiff_t full_offset = tensor.offset + shard.offset;
        ptrdiff_t shard_offset = tensor.offset / config->num_processes;
        zero_copy_parameter_shards_from_full(
            shards + shard_offset,
            full + full_offset,
            shard.size,
            shard.size,
            tensor.size,
            num_layers,
            main_stream);
    }
    cudaCheck(cudaDeviceSynchronize());
    printf0("allocated %zu MiB for ZeRO-3 Llama parameter shards\n",
            (config->shard_num_parameters * sizeof(floatX)) >> 20);
}

void llama_init_common(LlamaModel* model) {
    memset(model, 0, sizeof(*model));
    model->rng_state = 13371337 + multi_gpu_config.process_rank;
    model->rng_state_last_update = model->rng_state;
    model->use_master_weights = 1;
    model->init_state = true;
}

void fill_in_llama_activation_sizes(const LlamaActivationTensors* data,
                                    TensorSpec (&tensors)[LLAMA_NUM_ACTIVATION_TENSORS],
                                    size_t B, size_t T, LlamaConfig config) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t NKVH = config.num_kv_heads;
    size_t C = config.channels;
    size_t HS = C / NH;
    size_t qkv_width = (NH + 2 * NKVH) * HS;
    size_t hidden = llama_hidden_dim_from_config(config);
    size_t BT = B * T;
    size_t max_dweight = max(Vp * C, max(qkv_width * C, max(hidden * C, C * hidden)));

    tensors[0] = LLAMA_TENSOR_SPEC(data->encoded, BT * C);
    tensors[1] = LLAMA_TENSOR_SPEC(data->ln1, L * BT * C);
    tensors[2] = LLAMA_TENSOR_SPEC(data->ln1_rstd, L * BT);
    tensors[3] = LLAMA_TENSOR_SPEC(data->qkvr, L * BT * qkv_width);
    tensors[4] = LLAMA_TENSOR_SPEC(data->atty, L * BT * C);
    tensors[5] = LLAMA_TENSOR_SPEC(data->att_lse, L * B * NH * T);
    tensors[6] = LLAMA_TENSOR_SPEC(data->residual2, L * BT * C);
    tensors[7] = LLAMA_TENSOR_SPEC(data->ln2, L * BT * C);
    tensors[8] = LLAMA_TENSOR_SPEC(data->ln2_rstd, L * BT);
    tensors[9] = LLAMA_TENSOR_SPEC(data->fch_up, L * BT * hidden);
    tensors[10] = LLAMA_TENSOR_SPEC(data->fch_gate, L * BT * hidden);
    tensors[11] = LLAMA_TENSOR_SPEC(data->fch_swiglu, L * BT * hidden);
    tensors[12] = LLAMA_TENSOR_SPEC(data->residual3, L * BT * C);
    tensors[13] = LLAMA_TENSOR_SPEC(data->lnf, BT * C);
    tensors[14] = LLAMA_TENSOR_SPEC(data->lnf_rstd, BT);
    tensors[15] = LLAMA_TENSOR_SPEC(data->losses, BT);
    tensors[16] = LLAMA_TENSOR_SPEC(data->output, BT * max(Vp, max(qkv_width, max(hidden, C))));
    tensors[17] = LLAMA_TENSOR_SPEC(data->scratch_btc, BT * C);
    tensors[18] = LLAMA_TENSOR_SPEC(data->scratch_hidden, BT * hidden);
    tensors[19] = LLAMA_TENSOR_SPEC(data->scratch_hidden2, BT * hidden);
    tensors[20] = LLAMA_TENSOR_SPEC(data->matmul_scratch, max_dweight);
    tensors[21] = LLAMA_TENSOR_SPEC(data->att_bwd_x, 2 * BT * C);
    tensors[22] = LLAMA_TENSOR_SPEC(data->att_bwd_f, B * NH * T + BT * C + 2 * BT * NKVH * HS);
    tensors[23] = LLAMA_TENSOR_SPEC(data->rope_cos, T * (HS / 2));
    tensors[24] = LLAMA_TENSOR_SPEC(data->rope_sin, T * (HS / 2));
}

void* malloc_and_point_llama_activations(TensorSpec (&tensors)[LLAMA_NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0;
    for (size_t i = 0; i < LLAMA_NUM_ACTIVATION_TENSORS; ++i) {
        size_t tensor_bytes = tensors[i].size * sizeof_dtype(tensors[i].type);
        if (tensor_bytes == 0) { continue; }
        bytes = llmk::tk_align(bytes);
        bytes += tensor_bytes;
    }
    bytes = llmk::tk_align(bytes);
    printf0("allocating %d MiB for Llama activations\n", (int)round(bytes / (1024 * 1024)));

    void* memory = NULL;
    cudaCheck(cudaMalloc(&memory, bytes));
    cudaCheck(cudaMemset(memory, 0, bytes));
    char* iterator = (char*)memory;
    for (size_t i = 0; i < LLAMA_NUM_ACTIVATION_TENSORS; ++i) {
        if (tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;
            continue;
        }
        uintptr_t aligned = ((uintptr_t)iterator + llmk::TK_ALIGN - 1)
                          / llmk::TK_ALIGN * llmk::TK_ALIGN;
        iterator = (char*)aligned;
        assert(((uintptr_t)iterator % llmk::TK_ALIGN) == 0 &&
               "TK activation tensors must start at 128-byte-aligned offsets");
        *(tensors[i].ptr) = iterator;
        iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
    }
    return memory;
}

void llama_runtime_allocate(LlamaRuntimeState* state, const LlamaModel* model, int B, int T) {
    memset(state, 0, sizeof(*state));
    state->batch_size = B;
    state->seq_len = T;
    state->mean_loss = -1.0f;
    fill_in_llama_activation_sizes(&state->acts, state->specs, B, T, model->config);
    state->memory = malloc_and_point_llama_activations(state->specs);
    cudaCheck(cudaMalloc((void**)&state->inputs, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&state->targets, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&state->accumulated_mean_loss, sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&state->cpu_losses, (size_t)B * T * sizeof(float)));
    size_t num_c_groups = CEIL_DIV(model->config.channels, (WARP_SIZE * x128::size));
    state->workload_indices = (int*)mallocCheck(sizeof(int) * B * T * num_c_groups);
    state->bucket_info = (int4*)mallocCheck(sizeof(int4) * B * T * num_c_groups);
    llama_build_rope_cache(state->acts.rope_cos, state->acts.rope_sin, model->config, T, main_stream);
}

void llama_allocate_optimizer_state(LlamaModel* model) {
    printf0("allocating %d MiB for Llama parameter gradients\n",
            (int)round(model->num_parameters_bytes / (1024 * 1024)));
    assert(model->grads_memory == NULL);
    model->grads_memory = malloc_and_point_llama_parameters(
        &model->grads, model->param_elements, &model->num_parameters, &model->num_parameters_bytes);

    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    printf0("allocating %zu MiB for Llama AdamW optimizer state m\n",
            (shard_num_parameters * sizeof(float)) >> 20);
    printf0("allocating %zu MiB for Llama AdamW optimizer state v\n",
            (shard_num_parameters * sizeof(float)) >> 20);
    int memory_status = 0;
    memory_status |= cudaMallocConditionallyManaged((void**)&model->m_memory,
                                                    shard_num_parameters * sizeof(float));
    memory_status |= cudaMallocConditionallyManaged((void**)&model->v_memory,
                                                    shard_num_parameters * sizeof(float));
    if (model->use_master_weights) {
        printf0("allocating %zu MiB for Llama master weights\n",
                (shard_num_parameters * sizeof(float)) >> 20);
        memory_status |= cudaMallocConditionallyManaged((void**)&model->master_weights,
                                                        shard_num_parameters * sizeof(float));
    }
    int reduced_memory_status = (int)multi_gpu_cpu_float_sum((float)memory_status, &multi_gpu_config);
    if (reduced_memory_status >= 1) {
        printf0("WARNING: Fell back to cudaMallocManaged for Llama optimizer state on %d GPUs\n",
                reduced_memory_status);
    }
}

void llama3_set_hyperparameters(LlamaConfig* config, const char* descriptor) {
    memset(config, 0, sizeof(*config));
    config->vocab_size = 128256;
    config->padded_vocab_size = 128256;
    config->multiple_of = 1024;
    config->norm_eps = 1e-5f;
    config->rope_theta = 500000.0f;
    config->max_gen_batch_size = 4;

    if (strcmp(descriptor, "llama3:1B") == 0) {
        config->max_seq_len = 2048;
        config->num_layers = 16;
        config->num_heads = 16;
        config->num_kv_heads = 8;
        config->channels = 2048;
        config->hidden_dim = 8192;
        config->use_scaled_rope = 0;
        config->version_major = 3;
        config->version_minor = 0;
    } else if (strcmp(descriptor, "llama3:8B") == 0) {
        config->max_seq_len = 2048;
        config->num_layers = 32;
        config->num_heads = 32;
        config->num_kv_heads = 8;
        config->channels = 4096;
        config->hidden_dim = 14336;
        config->use_scaled_rope = 0;
        config->version_major = 3;
        config->version_minor = 0;
    } else if (strcmp(descriptor, "llama3.1:8B") == 0) {
        config->max_seq_len = 8192;
        config->num_layers = 32;
        config->num_heads = 32;
        config->num_kv_heads = 8;
        config->channels = 4096;
        config->hidden_dim = 14336;
        config->use_scaled_rope = 1;
        config->version_major = 3;
        config->version_minor = 1;
    } else {
        fprintf(stderr, "Unsupported Llama descriptor: %s\n", descriptor);
        exit(EXIT_FAILURE);
    }

    assert(config->num_heads % config->num_kv_heads == 0);
    assert(config->channels % config->num_heads == 0);
}

void llama3_config_from_checkpoint_header(LlamaConfig* config, const int* header) {
    if (header[0] != LLAMA3_MAGIC) {
        fprintf(stderr, "Bad Llama checkpoint magic: got %d, expected %d\n", header[0], LLAMA3_MAGIC);
        exit(EXIT_FAILURE);
    }
    if (header[1] != 5) {
        fprintf(stderr, "Only BF16 Llama checkpoints (version 5) are supported; got version %d\n", header[1]);
        exit(EXIT_FAILURE);
    }

    memset(config, 0, sizeof(*config));
    config->max_seq_len = header[2];
    config->vocab_size = header[3];
    config->padded_vocab_size = header[3];
    config->num_layers = header[4];
    config->num_heads = header[5];
    config->num_kv_heads = header[6];
    config->channels = header[7];
    config->multiple_of = header[9] == 0 ? 1024 : header[9];
    config->norm_eps = 1e-5f;
    config->rope_theta = header[11] == 0 ? 500000.0f : (float)header[11];
    config->use_scaled_rope = header[12];
    config->max_gen_batch_size = header[13] == 0 ? 4 : header[13];
    config->version_major = header[14];
    config->version_minor = header[15];
    config->hidden_dim = llama_hidden_dim_from_config(*config);

    assert(config->num_heads % config->num_kv_heads == 0);
    assert(config->channels % config->num_heads == 0);
}

void llama_build_from_descriptor(LlamaModel* model, const char* descriptor) {
    llama3_set_hyperparameters(&model->config, descriptor);
    llama_allocate_weights(model);
}

void llama_random_init_weights(LlamaModel* model) {
    assert(model->params_memory != NULL);
    mt19937_state init_rng;
    manual_seed(&init_rng, 42);
    floatX* params_cpu = (floatX*)mallocCheck(model->num_parameters_bytes);
    memset(params_cpu, 0, model->num_parameters_bytes);
    float residual_scale = 1.0f / sqrtf(2.0f * model->config.num_layers);
    size_t offset = 0;
    for (int i = 0; i < LLAMA_NUM_PARAMETER_TENSORS; ++i) {
        size_t n = model->param_elements[i];
        bool is_norm = (i == 1 || i == 4 || i == 8);
        bool is_weight = (i == 0 || i == 2 || i == 3 || i == 5 || i == 6 || i == 7 || i == 9);
        if (is_norm) {
            for (size_t j = 0; j < n; ++j) {
                params_cpu[offset + j] = (floatX)1.0f;
            }
        } else if (is_weight) {
            float scale = (i == 3 || i == 7) ? 0.02f * residual_scale : 0.02f;
            float* fp32 = (float*)mallocCheck(n * sizeof(float));
            normal_(fp32, n, 0.0f, scale, &init_rng);
            for (size_t j = 0; j < n; ++j) {
                params_cpu[offset + j] = (floatX)fp32[j];
            }
            free(fp32);
        }
        offset += n;
    }
    cudaCheck(cudaMemcpy(model->params_memory, params_cpu, model->num_parameters_bytes, cudaMemcpyHostToDevice));
    free(params_cpu);
}

void llama_build_from_checkpoint_header(LlamaModel* model, const char* checkpoint_path) {
    FILE* model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[LLAMA_HEADER_SIZE];
    freadCheck(model_header, sizeof(int), LLAMA_HEADER_SIZE, model_file);
    fseekCheck(model_file, 0, SEEK_END);
    int64_t file_size_bytes = ftell(model_file);
    fcloseCheck(model_file);
    llama3_config_from_checkpoint_header(&model->config, model_header);
    llama_allocate_weights(model);
    int64_t expected_file_size = (int64_t)LLAMA_HEADER_SIZE * (int64_t)sizeof(int)
                               + (int64_t)model->num_parameters_bytes;
    if (file_size_bytes != expected_file_size) {
        fprintf(stderr,
                "Bad Llama checkpoint size for %s: got %lld bytes, expected %lld bytes "
                "(%d-byte header + %zu parameter bytes)\n",
                checkpoint_path,
                (long long)file_size_bytes,
                (long long)expected_file_size,
                LLAMA_HEADER_SIZE * (int)sizeof(int),
                model->num_parameters_bytes);
        exit(EXIT_FAILURE);
    }
}

void llama_load_checkpoint_weights(LlamaModel* model, const char* checkpoint_path) {
    FILE* model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[LLAMA_HEADER_SIZE];
    freadCheck(model_header, sizeof(int), LLAMA_HEADER_SIZE, model_file);
    LlamaConfig parsed;
    llama3_config_from_checkpoint_header(&parsed, model_header);
    assert(parsed.max_seq_len == model->config.max_seq_len);
    assert(parsed.vocab_size == model->config.vocab_size);
    assert(parsed.num_layers == model->config.num_layers);
    assert(parsed.num_heads == model->config.num_heads);
    assert(parsed.num_kv_heads == model->config.num_kv_heads);
    assert(parsed.channels == model->config.channels);
    assert(model->params_memory != NULL);
    file_to_device(model->params_memory, model_file, model->num_parameters_bytes, IO_BUF_SIZE, main_stream);
    fcloseCheck(model_file);
    cudaCheck(cudaDeviceSynchronize());
}

void llama_write_to_checkpoint(LlamaModel* model, const char* checkpoint_path) {
    printf0("Writing Llama model to %s\n", checkpoint_path);
    FILE* model_file = fopenCheck(checkpoint_path, "wb");
    int header[LLAMA_HEADER_SIZE];
    memset(header, 0, sizeof(header));
    header[0] = LLAMA3_MAGIC;
    header[1] = 5;
    header[2] = model->config.max_seq_len;
    header[3] = model->config.vocab_size;
    header[4] = model->config.num_layers;
    header[5] = model->config.num_heads;
    header[6] = model->config.num_kv_heads;
    header[7] = model->config.channels;
    header[8] = model->config.hidden_dim;
    header[9] = model->config.multiple_of;
    header[11] = (int)model->config.rope_theta;
    header[12] = model->config.use_scaled_rope;
    header[13] = model->config.max_gen_batch_size;
    header[14] = model->config.version_major;
    header[15] = model->config.version_minor;
    fwriteCheck(header, sizeof(int), LLAMA_HEADER_SIZE, model_file);
    device_to_file(model_file, model->params_memory, model->num_parameters_bytes,
                   IO_BUF_SIZE, main_stream);
    fcloseCheck(model_file);
}

void llama_update(LlamaModel* model, float learning_rate, float beta1, float beta2,
                  float eps, float weight_decay, float grad_scale, int t,
                  MultiGpuConfig* config, bool init_from_master_only);

void llama_save_state(const char* filename, int step, LlamaModel* model, DataLoader* loader) {
    printf("Writing Llama state to %s\n", filename);
    FILE* state_file = fopenCheck(filename, "wb");
    int state_header[LLAMA_HEADER_SIZE];
    memset(state_header, 0, sizeof(state_header));
    state_header[0] = LLAMA_STATE_MAGIC;
    state_header[1] = 1;
    state_header[2] = multi_gpu_config.num_processes;
    state_header[3] = multi_gpu_config.process_rank;
    state_header[4] = model->use_master_weights;
    state_header[5] = loader->should_shuffle;
    state_header[10] = step;
    *((unsigned long long*)&state_header[20]) = model->rng_state;
    *((unsigned long long*)&state_header[22]) = model->rng_state_last_update;
    *((size_t*)&state_header[30]) = loader->current_shard_idx;
    *((size_t*)&state_header[32]) = loader->current_sample_idx;
    fwriteCheck(state_header, sizeof(int), LLAMA_HEADER_SIZE, state_file);

    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    device_to_file(state_file, model->m_memory, shard_num_parameters * sizeof(float),
                   IO_BUF_SIZE, main_stream);
    device_to_file(state_file, model->v_memory, shard_num_parameters * sizeof(float),
                   IO_BUF_SIZE, main_stream);
    if (model->use_master_weights) {
        device_to_file(state_file, model->master_weights,
                       shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    }

    if (loader->should_shuffle) {
        fwriteCheck(&loader->glob_result.gl_pathc, sizeof(size_t), 1, state_file);
        fwriteCheck(loader->shard_indices, sizeof(int),
                    loader->glob_result.gl_pathc, state_file);
        fwriteCheck(&loader->shard_num_samples, sizeof(size_t), 1, state_file);
        fwriteCheck(loader->intra_shard_indices, sizeof(int),
                    loader->shard_num_samples, state_file);
        fwriteCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    fcloseCheck(state_file);
}

void llama_load_state(int* step, LlamaModel* model, DataLoader* loader, const char* filename) {
    FILE* state_file = fopenCheck(filename, "rb");
    int state_header[LLAMA_HEADER_SIZE];
    freadCheck(state_header, sizeof(int), LLAMA_HEADER_SIZE, state_file);
    assert(state_header[0] == LLAMA_STATE_MAGIC);
    assert(state_header[1] == 1);
    assert(state_header[2] == multi_gpu_config.num_processes);
    assert(state_header[3] == multi_gpu_config.process_rank);
    int use_master_weights = state_header[4];
    int should_shuffle = state_header[5];
    *step = state_header[10];
    model->rng_state = *((unsigned long long*)&state_header[20]);
    model->rng_state_last_update = *((unsigned long long*)&state_header[22]);
    size_t current_shard_idx = *((size_t*)&state_header[30]);
    size_t current_sample_idx = *((size_t*)&state_header[32]);

    if (use_master_weights == 1 && !model->use_master_weights) {
        printf0("Warning: Llama master weights are present in state, but not enabled for current run.\n");
    } else if (use_master_weights == 0 && model->use_master_weights) {
        fprintf(stderr, "Error: Llama master weights requested, but not present in state file.\n");
        exit(EXIT_FAILURE);
    }

    model->init_state = false;
    assert(model->m_memory != NULL);
    assert(model->v_memory != NULL);
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    file_to_device(model->m_memory, state_file, shard_num_parameters * sizeof(float),
                   IO_BUF_SIZE, main_stream);
    file_to_device(model->v_memory, state_file, shard_num_parameters * sizeof(float),
                   IO_BUF_SIZE, main_stream);
    if (model->use_master_weights) {
        assert(model->master_weights != NULL);
        file_to_device(model->master_weights, state_file,
                       shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
        model->rng_state = model->rng_state_last_update;
        llama_update(model, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0,
                     &multi_gpu_config, /*init_from_master_only=*/true);
        model->rng_state = *((unsigned long long*)&state_header[20]);
    }

    loader->should_shuffle = should_shuffle;
    if (should_shuffle == 1) {
        size_t glob_result_gl_pathc;
        freadCheck(&glob_result_gl_pathc, sizeof(size_t), 1, state_file);
        assert(glob_result_gl_pathc == loader->glob_result.gl_pathc);
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        freadCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        size_t shard_num_samples;
        freadCheck(&shard_num_samples, sizeof(size_t), 1, state_file);
        assert(shard_num_samples == loader->shard_num_samples);
        loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
        freadCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        freadCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    dataloader_resume(loader, current_shard_idx, current_sample_idx);
    fcloseCheck(state_file);
}

void llama_write_checkpoint(const char* output_log_dir, int step, LlamaModel* model,
                            DataLoader* train_loader, MultiGpuConfig* config) {
    printf0("Writing Llama checkpoint at step %d\n", step);
    int rank = config->process_rank;
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        llama_write_to_checkpoint(model, filename_buffer);
    }
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin",
             output_log_dir, step, rank);
    llama_save_state(filename_buffer, step, model, train_loader);
    multi_gpu_barrier(config);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        FILE* done_file = fopenCheck(filename_buffer, "w");
        fcloseCheck(done_file);
    }
}

void llama_delete_checkpoint(const char* output_log_dir, int step, MultiGpuConfig* config) {
    printf0("Deleting Llama checkpoint at step %d\n", step);
    int rank = config->process_rank;
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        remove(filename_buffer);
    }
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin",
             output_log_dir, step, rank);
    remove(filename_buffer);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        remove(filename_buffer);
    }
}

void print_llama_config(const LlamaModel* model, const char* source) {
    const LlamaConfig& cfg = model->config;
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| Llama config source   | %-50s |\n", source);
    printf("| sequence length       | %-50d |\n", cfg.max_seq_len);
    printf("| vocab size            | %-50d |\n", cfg.vocab_size);
    printf("| layers                | %-50d |\n", cfg.num_layers);
    printf("| heads q / kv          | %-23d / %-24d |\n", cfg.num_heads, cfg.num_kv_heads);
    printf("| channels              | %-50d |\n", cfg.channels);
    printf("| head dim              | %-50d |\n", cfg.channels / cfg.num_heads);
    printf("| ffn hidden dim        | %-50d |\n", cfg.hidden_dim);
    printf("| scaled RoPE           | %-50s |\n", cfg.use_scaled_rope ? "yes" : "no");
    printf("| parameter tensors     | %-50d |\n", LLAMA_NUM_PARAMETER_TENSORS);
    printf("| parameters            | %-50zu |\n", model->num_parameters);
    printf("| parameter bytes       | %-50zu |\n", model->num_parameters_bytes);
    printf("+-----------------------+----------------------------------------------------+\n");
}

void llama_forward(LlamaModel* model, LlamaRuntimeState* state, const int* inputs, size_t B, size_t T) {
    NVTX_RANGE_FN();
    if (model->params_memory == NULL) {
        fprintf(stderr, "Llama model parameters are not materialized on the GPU.\n");
        exit(EXIT_FAILURE);
    }
    if (B > (size_t)state->batch_size || T > (size_t)state->seq_len) {
        fprintf(stderr, "Llama runtime allocated for B=%d T=%d, got B=%zu T=%zu\n",
                state->batch_size, state->seq_len, B, T);
        exit(EXIT_FAILURE);
    }

    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t NKVH = model->config.num_kv_heads;
    const size_t C = model->config.channels;
    const size_t HS = C / NH;
    const size_t qkv_width = (NH + 2 * NKVH) * HS;
    const size_t hidden = model->config.hidden_dim;
    const size_t BT = B * T;

    cudaCheck(cudaMemcpy(state->inputs, inputs, BT * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(inputs, BT, V);

    LlamaParameterTensors params = model->params;
    LlamaActivationTensors acts = state->acts;
    llama_embedding_forward(acts.encoded, state->inputs, params.wte, B, T, C, main_stream);

    for (int l = 0; l < (int)L; ++l) {
        NvtxRange layer_range("Llama layer", l);
        floatX* residual = (l == 0) ? acts.encoded : acts.residual3 + (l - 1) * BT * C;

        floatX* l_ln1 = acts.ln1 + l * BT * C;
        float* l_ln1_rstd = acts.ln1_rstd + l * BT;
        floatX* l_qkvr = acts.qkvr + l * BT * qkv_width;
        floatX* l_atty = acts.atty + l * BT * C;
        float* l_att_lse = acts.att_lse + l * B * NH * T;
        floatX* l_residual2 = acts.residual2 + l * BT * C;
        floatX* l_ln2 = acts.ln2 + l * BT * C;
        float* l_ln2_rstd = acts.ln2_rstd + l * BT;
        floatX* l_fch_up = acts.fch_up + l * BT * hidden;
        floatX* l_fch_gate = acts.fch_gate + l * BT * hidden;
        floatX* l_fch_swiglu = acts.fch_swiglu + l * BT * hidden;
        floatX* l_residual3 = acts.residual3 + l * BT * C;

        floatX* l_ln1w = params.ln1w + l * C;
        floatX* l_qkvw = params.qkvw + l * qkv_width * C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_fcw_up = params.fcw_up + l * hidden * C;
        floatX* l_fcw_gate = params.fcw_gate + l * hidden * C;
        floatX* l_fcprojw = params.fcprojw + l * C * hidden;

        rmsnorm_forward(l_ln1, l_ln1_rstd, residual, l_ln1w, BT, C, model->config.norm_eps, main_stream);
        matmul_forward((floatX*)acts.output, l_ln1, l_qkvw, NULL, B, T, C, qkv_width, main_stream);
        attention_gqa_forward(l_atty, l_qkvr, l_att_lse, (floatX*)acts.output,
                              acts.rope_cos, acts.rope_sin, B, T, C, NH, NKVH, main_stream,
                              (floatX*)acts.output);
        matmul_forward((floatX*)acts.output, l_atty, l_attprojw, NULL, B, T, C, C, main_stream);
        llama_residual_add(l_residual2, residual, (floatX*)acts.output, BT * C, main_stream);

        rmsnorm_forward(l_ln2, l_ln2_rstd, l_residual2, l_ln2w, BT, C, model->config.norm_eps, main_stream);
        matmul_forward(l_fch_up, l_ln2, l_fcw_up, NULL, B, T, C, hidden, main_stream);
        matmul_forward(l_fch_gate, l_ln2, l_fcw_gate, NULL, B, T, C, hidden, main_stream);
        swiglu_forward(l_fch_swiglu, l_fch_gate, l_fch_up, BT * hidden, main_stream);
        matmul_forward((floatX*)acts.output, l_fch_swiglu, l_fcprojw, NULL, B, T, hidden, C, main_stream);
        llama_residual_add(l_residual3, l_residual2, (floatX*)acts.output, BT * C, main_stream);
    }

    floatX* final_residual = acts.residual3 + (L - 1) * BT * C;
    rmsnorm_forward(acts.lnf, acts.lnf_rstd, final_residual, params.lnfw,
                    BT, C, model->config.norm_eps, main_stream);
    matmul_forward(acts.output, acts.lnf, params.lm_head, NULL, B, T, C, Vp, main_stream);
    cudaCheck(cudaDeviceSynchronize());
}

float llama_validate(LlamaModel* model, LlamaRuntimeState* state,
                     const int* inputs, const int* targets, size_t B, size_t T) {
    assert(targets != NULL);
    llama_forward(model, state, inputs, B, T);
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const float dloss = 1.0f / (B * T);
    cudaCheck(cudaMemset(state->acts.losses, 0, B * T * sizeof(float)));
    cudaCheck(cudaMemcpy(state->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B * T, V);
    fused_classifier(state->acts.output, state->acts.losses, dloss, state->targets,
                     B, T, V, Vp, False, main_stream);
    cudaCheck(cudaMemcpy(state->cpu_losses, state->acts.losses, B * T * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float mean_loss = 0.0f;
    for (int i = 0; i < (int)(B * T); ++i) {
        mean_loss += state->cpu_losses[i];
    }
    mean_loss /= (float)(B * T);
    cudaCheck(cudaDeviceSynchronize());
    return mean_loss;
}

void llama_backward_and_reduce(LlamaModel* model, LlamaRuntimeState* state,
                               int* inputs, const int* targets,
                               int grad_accum_steps, int micro_step) {
    NVTX_RANGE_FN();
    if (model->grads_memory == NULL) {
        fprintf(stderr, "Need to allocate Llama gradients before backward\n");
        exit(EXIT_FAILURE);
    }
    bool last_step = micro_step == grad_accum_steps - 1;
    if (micro_step == 0) {
        cudaCheck(cudaMemsetAsync(state->acts.losses, 0,
                                  (size_t)state->batch_size * state->seq_len * sizeof(float),
                                  main_stream));
        cudaCheck(cudaMemsetAsync(model->grads_memory, 0, model->num_parameters_bytes, main_stream));
    }

    const int B = state->batch_size;
    const int T = state->seq_len;
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t NKVH = model->config.num_kv_heads;
    const size_t C = model->config.channels;
    const size_t HS = C / NH;
    const size_t qkv_width = (NH + 2 * NKVH) * HS;
    const size_t hidden = model->config.hidden_dim;
    const size_t BT = (size_t)B * T;

    LlamaParameterTensors params = model->params;
    LlamaParameterTensors grads = model->grads;
    LlamaActivationTensors acts = state->acts;

    const float dloss = 1.0f / (float)(BT * grad_accum_steps);
    const bool dweight_accumulate = micro_step != 0;
    cudaCheck(cudaMemcpy(state->targets, targets, BT * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, BT, V);
    fused_classifier(acts.output, acts.losses, dloss, state->targets, B, T, V, Vp, True, main_stream);

    float* scratchF = (float*)acts.output;
    floatX* scratchX = (floatX*)acts.output;
    floatX* dresidual = acts.scratch_btc;
    floatX* dtmp_btc = acts.residual3 + (L - 1) * BT * C;
    floatX* dhidden = acts.scratch_hidden;
    floatX* dhidden_aux = acts.scratch_hidden2;
    floatX* matmul_scratch = acts.matmul_scratch;
    size_t matmul_scratch_elements = state->specs[20].size;

    matmul_backward(dresidual, grads.lm_head, NULL, acts.output, acts.lnf,
                    params.lm_head, NULL, B, T, C, Vp, main_stream,
                    dweight_accumulate, matmul_scratch, matmul_scratch_elements);

    floatX* final_residual = acts.residual3 + (L - 1) * BT * C;
    rmsnorm_backward(dtmp_btc, grads.lnfw, dresidual, final_residual, params.lnfw,
                     acts.lnf_rstd, BT, C, main_stream);
    cudaCheck(cudaMemcpyAsync(dresidual, dtmp_btc, BT * C * sizeof(floatX),
                              cudaMemcpyDeviceToDevice, main_stream));

    for (int l = (int)L - 1; l >= 0; --l) {
        NvtxRange layer_range("Llama backward layer", l);
        floatX* residual = (l == 0) ? acts.encoded : acts.residual3 + (l - 1) * BT * C;

        floatX* l_ln1 = acts.ln1 + l * BT * C;
        float* l_ln1_rstd = acts.ln1_rstd + l * BT;
        floatX* l_qkvr = acts.qkvr + l * BT * qkv_width;
        floatX* l_atty = acts.atty + l * BT * C;
        float* l_att_lse = acts.att_lse + l * B * NH * T;
        floatX* l_residual2 = acts.residual2 + l * BT * C;
        floatX* l_ln2 = acts.ln2 + l * BT * C;
        float* l_ln2_rstd = acts.ln2_rstd + l * BT;
        floatX* l_fch_up = acts.fch_up + l * BT * hidden;
        floatX* l_fch_gate = acts.fch_gate + l * BT * hidden;
        floatX* l_fch_swiglu = acts.fch_swiglu + l * BT * hidden;

        floatX* l_ln1w = params.ln1w + l * C;
        floatX* l_qkvw = params.qkvw + l * qkv_width * C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_fcw_up = params.fcw_up + l * hidden * C;
        floatX* l_fcw_gate = params.fcw_gate + l * hidden * C;
        floatX* l_fcprojw = params.fcprojw + l * C * hidden;

        floatX* dl_ln1w = grads.ln1w + l * C;
        floatX* dl_qkvw = grads.qkvw + l * qkv_width * C;
        floatX* dl_attprojw = grads.attprojw + l * C * C;
        floatX* dl_ln2w = grads.ln2w + l * C;
        floatX* dl_fcw_up = grads.fcw_up + l * hidden * C;
        floatX* dl_fcw_gate = grads.fcw_gate + l * hidden * C;
        floatX* dl_fcprojw = grads.fcprojw + l * C * hidden;

        matmul_backward(dhidden, dl_fcprojw, NULL, dresidual, l_fch_swiglu,
                        l_fcprojw, scratchF, B, T, hidden, C, main_stream,
                        dweight_accumulate, matmul_scratch, matmul_scratch_elements);
        floatX* dgate = l_fch_swiglu;
        floatX* dup = dhidden_aux;
        swiglu_backward(dgate, dup, dhidden, l_fch_gate, l_fch_up, BT * hidden, main_stream);
        matmul_backward(dtmp_btc, dl_fcw_gate, NULL, dgate, l_ln2,
                        l_fcw_gate, scratchF, B, T, C, hidden, main_stream,
                        dweight_accumulate, matmul_scratch, matmul_scratch_elements);
        matmul_backward(scratchX, dl_fcw_up, NULL, dup, l_ln2,
                        l_fcw_up, scratchF, B, T, C, hidden, main_stream,
                        dweight_accumulate, matmul_scratch, matmul_scratch_elements);
        llama_add_inplace(dtmp_btc, scratchX, BT * C, main_stream);
        rmsnorm_backward(scratchX, dl_ln2w, dtmp_btc, l_residual2, l_ln2w,
                         l_ln2_rstd, BT, C, main_stream);
        llama_add_inplace(dresidual, scratchX, BT * C, main_stream);

        matmul_backward(dtmp_btc, dl_attprojw, NULL, dresidual, l_atty,
                        l_attprojw, scratchF, B, T, C, C, main_stream,
                        dweight_accumulate, matmul_scratch, matmul_scratch_elements);
        attention_gqa_backward(scratchX, dhidden_aux, acts.att_bwd_f, acts.att_bwd_x,
                               dtmp_btc, l_atty, l_qkvr,
                               l_att_lse, acts.rope_cos, acts.rope_sin,
                               B, T, C, NH, NKVH, main_stream,
                               attention_gqa_uses_tk_tile_rope(
                                   acts.rope_cos, acts.rope_sin, (floatX*)acts.output,
                                   T, C, NH, NKVH));
        matmul_backward(dtmp_btc, dl_qkvw, NULL, scratchX, l_ln1,
                        l_qkvw, scratchF, B, T, C, qkv_width, main_stream,
                        dweight_accumulate, matmul_scratch, matmul_scratch_elements);
        rmsnorm_backward(scratchX, dl_ln1w, dtmp_btc, residual, l_ln1w,
                         l_ln1_rstd, BT, C, main_stream);
        llama_add_inplace(dresidual, scratchX, BT * C, main_stream);

        if (last_step) {
            floatX* const pointers[] = {
                dl_ln1w, dl_qkvw, dl_attprojw, dl_ln2w, dl_fcw_up, dl_fcw_gate, dl_fcprojw
            };
            const size_t nelem[] = {
                C, qkv_width * C, C * C, C, hidden * C, hidden * C, C * hidden
            };
            multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
        }
    }

    llama_embedding_backward(grads.wte, scratchX, state->workload_indices, state->bucket_info,
                             dresidual, state->inputs, inputs, B, T, C,
                             random_u32(&model->rng_state), main_stream);

    if (last_step) {
        global_sum_deterministic(state->accumulated_mean_loss, acts.losses, BT, main_stream);
#if MULTI_GPU
        ncclCheck(ncclAllReduce(state->accumulated_mean_loss, state->accumulated_mean_loss,
                                1, ncclFloat, ncclAvg,
                                multi_gpu_config.nccl_comm, main_stream));
#endif
        cudaCheck(cudaMemcpyAsync(&state->mean_loss, state->accumulated_mean_loss,
                                  sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        floatX* const pointers[] = {grads.wte, grads.lnfw, grads.lm_head};
        const size_t nelem[] = {Vp * C, C, Vp * C};
        multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
    }

    cudaCheck(cudaDeviceSynchronize());
    if (last_step) {
        state->mean_loss /= (float)(BT * grad_accum_steps);
    } else {
        state->mean_loss = -1.0f;
    }
}

ShardInfo llama_get_tensor_at_layer(const LlamaModel* model, int layer_id, int param_tensor_id) {
    ptrdiff_t offset = 0;
    for (int i = 0; i < param_tensor_id; ++i) {
        offset += (ptrdiff_t)model->param_elements[i];
    }
    size_t size = model->param_elements[param_tensor_id];
    if (1 <= param_tensor_id && param_tensor_id <= 7) {
        size /= model->config.num_layers;
        offset += (ptrdiff_t)(layer_id * size);
    }
    return {offset, size};
}

void llama_validate_zero_tensor_sharding(const LlamaModel* model, const MultiGpuConfig* config) {
    for (int i = 0; i < LLAMA_NUM_PARAMETER_TENSORS; ++i) {
        ShardInfo tensor = llama_get_tensor_at_layer(model, 0, i);
        validate_zero_tensor_shardable(LLAMA_PARAMETER_TENSOR_NAMES[i], tensor.size, config);
    }
}

float llama_calculate_grad_norm(LlamaModel* model, LlamaRuntimeState* state,
                                MultiGpuConfig* config) {
    floatX* grads_memory = (floatX*)model->grads_memory;
    float* grad_norm_squared = (float*)state->acts.output;
    float grad_norm_squared_cpu = 0.0f;
    int num_slices[2] = {1, model->config.num_layers};
    int max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    if (zero_uses_reduce_scatter_gradients(config)) {
        for (int i = 0; i < LLAMA_NUM_PARAMETER_TENSORS; ++i) {
            ShardInfo tensor = llama_get_tensor_at_layer(model, 0, i);
            ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, config, 1);
            ptrdiff_t offset = tensor.offset + shard.offset;
            bool first = (i == 0);
            if (i == 0 || i >= 8) {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size,
                                    0, 1, max_num_block_sums, first, main_stream);
            } else {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size,
                                    tensor.size, model->config.num_layers,
                                    max_num_block_sums, first, main_stream);
            }
        }
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
#if MULTI_GPU
        ncclCheck(ncclAllReduce(grad_norm_squared, grad_norm_squared, 1,
                                ncclFloat, ncclSum, config->nccl_comm, main_stream));
#endif
    } else {
        global_norm_squared(grad_norm_squared, grads_memory, model->num_parameters,
                            0, 1, max_num_block_sums, true, main_stream);
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
    }
    cudaCheck(cudaMemcpy(&grad_norm_squared_cpu, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));
    return sqrtf(grad_norm_squared_cpu);
}

void llama_update(LlamaModel* model, float learning_rate, float beta1, float beta2,
                  float eps, float weight_decay, float grad_scale, int t,
                  MultiGpuConfig* config, bool init_from_master_only = false) {
    NVTX_RANGE_FN();
    if (model->grads_memory == NULL || model->m_memory == NULL || model->v_memory == NULL) {
        fprintf(stderr, "Need to allocate Llama optimizer state before update\n");
        exit(EXIT_FAILURE);
    }
    if (zero_shards_parameters(config) && model->param_shards_memory == NULL) {
        fprintf(stderr, "Need to allocate ZeRO-3 Llama parameter shards before update\n");
        exit(EXIT_FAILURE);
    }
    bool init_state = model->init_state;
    if (init_state) {
        model->init_state = false;
        cudaCheck(cudaMemset(model->m_memory, 0, config->shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, config->shard_num_parameters * sizeof(float)));
    }
    model->rng_state_last_update = model->rng_state;

    for (int i = 0; i < LLAMA_NUM_PARAMETER_TENSORS; ++i) {
        unsigned int seed = random_u32(&model->rng_state);
        int num_layers = (1 <= i && i <= 7) ? model->config.num_layers : 1;
        ShardInfo tensor = llama_get_tensor_at_layer(model, 0, i);
        ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, config, 1);
        ptrdiff_t local_offset_full = tensor.offset + shard.offset;
        ptrdiff_t local_offset_partial = tensor.offset / config->num_processes;
        bool decay_tensor = (i == 0 || i == 2 || i == 3 || i == 5 || i == 6 || i == 7 || i == 9);
        float wd = decay_tensor ? weight_decay : 0.0f;

        floatX* param_ptr = zero_shards_parameters(config)
                           ? (floatX*)model->param_shards_memory + local_offset_partial
                           : (floatX*)model->params_memory + local_offset_full;
        floatX* grad_ptr = (floatX*)model->grads_memory + local_offset_full;
        ptrdiff_t param_stride = zero_shards_parameters(config) ? shard.size : tensor.size;
        ptrdiff_t opt_state_offset = config->zero_stage < 1 ? local_offset_full : local_offset_partial;
        float* m_ptr = model->m_memory + opt_state_offset;
        float* v_ptr = model->v_memory + opt_state_offset;
        float* master_ptr = model->master_weights == NULL ? NULL : model->master_weights + opt_state_offset;

        if (init_state && master_ptr != NULL) {
            size_t grid_size = CEIL_DIV(shard.size, 512);
            copy_and_cast_kernel<<<dim3(grid_size, num_layers), 512, 0, main_stream>>>(
                master_ptr, param_ptr, shard.size, shard.size, param_stride);
            cudaCheck(cudaGetLastError());
        }
        if (init_from_master_only) {
            init_from_master(param_ptr, master_ptr, shard.size, param_stride, shard.size,
                             num_layers, seed, main_stream);
        } else {
            adamw_update(param_ptr, master_ptr, grad_ptr, m_ptr, v_ptr,
                         shard.size, param_stride, tensor.size, shard.size, num_layers,
                         learning_rate, beta1, beta2, t, eps, wd, grad_scale, seed, main_stream);
        }

        if (zero_shards_parameters(config)) {
            zero_all_gather_parameter_shards_to_full(
                (floatX*)model->params_memory + tensor.offset,
                (floatX*)model->param_shards_memory + local_offset_partial,
                shard.size,
                tensor.size,
                shard.size,
                num_layers,
                config,
                main_stream);
        } else if (zero_shards_optimizer_state(config)) {
#if MULTI_GPU
            multi_gpu_sync_nccl_stream_from_compute(config, main_stream);
            ncclCheck(ncclGroupStart());
            for (int l = 0; l < num_layers; ++l) {
                ncclCheck(ncclAllGather(param_ptr + l * tensor.size,
                                        (floatX*)model->params_memory + tensor.offset + l * tensor.size,
                                        shard.size, ncclFloatX, config->nccl_comm, config->nccl_stream));
            }
            ncclCheck(ncclGroupEnd());
#endif
        }
    }
    cudaCheck(cudaDeviceSynchronize());
}

void llama_free(LlamaModel* model, LlamaRuntimeState* state) {
    cudaFreeCheck(&model->params_memory);
    cudaFreeCheck(&model->param_shards_memory);
    cudaFreeCheck(&model->grads_memory);
    cudaFreeCheck(&model->m_memory);
    cudaFreeCheck(&model->v_memory);
    cudaFreeCheck(&model->master_weights);
    cudaFreeCheck(&state->memory);
    cudaFreeCheck(&state->inputs);
    cudaFreeCheck(&state->targets);
    cudaFreeCheck(&state->accumulated_mean_loss);
    if (state->cpu_losses != NULL) {
        cudaCheck(cudaFreeHost(state->cpu_losses));
        state->cpu_losses = NULL;
    }
    free(state->workload_indices);
    free(state->bucket_info);
}

void llama_common_start(bool print_device_info = true) {
    cudaCheck(cudaGetDeviceProperties(&deviceProp, multi_gpu_config.local_device_idx));
    if (print_device_info) {
        printf0("[System]\n");
        printf0("Device %d: %s\n", multi_gpu_config.local_device_idx, deviceProp.name);
    }
    cudaCheck(cudaStreamCreate(&main_stream));
    nvtxNameCudaStreamA(main_stream, "main stream");
}

void llama_common_free() {
    cudaCheck(cudaStreamDestroy(main_stream));
}

void llama_error_usage() {
    fprintf(stderr, "Usage: ./train_llama3cu [options]\n");
    fprintf(stderr, "Options mirror train_gpt2cu where possible:\n");
    fprintf(stderr, "  -e <string> input .bin filename or descriptor: llama3:1B, llama3:8B, llama3.1:8B\n");
    fprintf(stderr, "  -i/-j <string> train/val data filename patterns\n");
    fprintf(stderr, "  -o <string> output log dir\n");
    fprintf(stderr, "  -b <int> micro batch size, -t <int> sequence length, -d <int> total batch tokens\n");
    fprintf(stderr, "  -x <int> max steps, -l <float> learning rate, -u <int> warmup, -q <float> final LR fraction\n");
    fprintf(stderr, "  -z <int> ZeRO stage, runtime stages: 0,1,2,3; -pn/-pr/-pg/-pi/-pf/-ps multi-node settings\n");
    exit(EXIT_FAILURE);
}

MultiGpuConfig llama_host_only_multi_gpu_config(int num_processes, int process_rank, int gpus_per_node) {
    if (num_processes < 1) {
        fprintf(stderr, "Invalid process count %d for Llama dry run.\n", num_processes);
        exit(EXIT_FAILURE);
    }
    if (process_rank < 0 || process_rank >= num_processes) {
        fprintf(stderr, "Invalid process rank %d for %d processes in Llama dry run.\n",
                process_rank, num_processes);
        exit(EXIT_FAILURE);
    }
    if (gpus_per_node < 1) {
        fprintf(stderr, "Invalid GPUs-per-node count %d for Llama dry run.\n", gpus_per_node);
        exit(EXIT_FAILURE);
    }
    MultiGpuConfig config = {};
    config.process_rank = process_rank;
    config.num_processes = num_processes;
    config.local_device_idx = process_rank % gpus_per_node;
    config.zero_stage = 0;
    config.shard_num_parameters = 0;
    return config;
}

int main(int argc, char *argv[]) {
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* model_source = "llama3:1B";
    const char* output_log_dir = NULL;
    const char* lr_scheduler_type = "cosine";
    int B = 4;
    int T = 2048;
    int total_batch_size = -1;
    int checkpoint_every = 0;
    int checkpoints_keep = 0;
    int major_checkpoint_every = 0;
    int resume = 0;
    int max_steps = 0;
    int val_loss_every = 20;
    int val_max_steps = 20;
    int sample_every = 20;
    int genT = 64;
    int hellaswag_eval = 0;
    int recompute = 1;
    int zero_stage = 0;
    int warmup_iterations = 0;
    float learning_rate = 3e-4f;
    float final_learning_rate_frac = 1.0f;
    float weight_decay = 0.0f;
    float skip_update_lossz = 0.0f;
    float skip_update_gradz = 0.0f;
    int num_processes = 1;
    int process_rank = 0;
    int gpus_per_node = 8;
    char nccl_init_method[256] = "mpi";
    char server_ip[256] = "";
    char fs_path[256] = "";

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { llama_error_usage(); }
        if (argv[i][0] != '-') { llama_error_usage(); }
        if (!(strlen(argv[i]) == 2 || strlen(argv[i]) == 3)) { llama_error_usage(); }

        if (argv[i][1] == 'i') { train_data_pattern = argv[i + 1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i + 1]; }
        else if (argv[i][1] == 'e') { model_source = argv[i + 1]; }
        else if (argv[i][1] == 'o') { output_log_dir = argv[i + 1]; }
        else if (argv[i][1] == 'b') { B = atoi(argv[i + 1]); }
        else if (argv[i][1] == 't') { T = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'd') { total_batch_size = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'x') { max_steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 's' && argv[i][2] == '\0') { sample_every = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == '\0') { genT = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'h') { hellaswag_eval = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'r') { recompute = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'z') { zero_stage = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == '\0') { learning_rate = atof(argv[i + 1]); }
        else if (argv[i][1] == 'u') { warmup_iterations = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'q') { final_learning_rate_frac = atof(argv[i + 1]); }
        else if (argv[i][1] == 'c') { weight_decay = atof(argv[i + 1]); }
        else if (argv[i][1] == 'k') { lr_scheduler_type = argv[i + 1]; }
        else if (argv[i][1] == 'n' && argv[i][2] == '\0') { checkpoint_every = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'k') { checkpoints_keep = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'm') { major_checkpoint_every = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'y') { resume = atoi(argv[i + 1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'l') { skip_update_lossz = atof(argv[i + 1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'g') { skip_update_gradz = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'n') { num_processes = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'r') { process_rank = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'g') { gpus_per_node = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'i') { strcpy(nccl_init_method, argv[i + 1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'f') { strcpy(fs_path, argv[i + 1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 's') { strcpy(server_ip, argv[i + 1]); }
        else { llama_error_usage(); }
    }
    validate_zero_stage_request(zero_stage);

    int resuming = 0;
    int resume_max_step = find_max_step(output_log_dir);
    char resume_model_path[512];
    if (resume == 1) {
        assert(output_log_dir != NULL);
        if (resume_max_step != -1) {
            resuming = 1;
            snprintf(resume_model_path, sizeof(resume_model_path),
                     "%s/model_%08d.bin", output_log_dir, resume_max_step);
            model_source = resume_model_path;
        }
    }

    LlamaModel model;
    llama_init_common(&model);
    if (ends_with_bin(model_source)) {
        llama_build_from_checkpoint_header(&model, model_source);
    } else {
        llama_build_from_descriptor(&model, model_source);
    }

    if (max_steps == 0) {
        print_llama_config(&model, model_source);
        MultiGpuConfig dry_config = llama_host_only_multi_gpu_config(num_processes, process_rank, gpus_per_node);
        multi_gpu_config = dry_config;
        set_zero_configs(&dry_config, zero_stage, model.num_parameters);
        llama_validate_zero_tensor_sharding(&model, &dry_config);
        if (zero_stage > 0) {
            printf0("train_llama3cu dry run: ZeRO-%d shard layout validated for %d process(es); "
                    "local shard parameters=%zu.\n",
                    dry_config.zero_stage, dry_config.num_processes, dry_config.shard_num_parameters);
        }
        printf0("train_llama3cu dry run: checkpoint/config parsed and payload sizes validated. Pass -x >0 to train.\n");
        return EXIT_SUCCESS;
    }

    validate_zero_runtime_request(zero_stage);

    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node,
                                             server_ip, fs_path, nccl_init_method);
    llama_common_start(false);
    model.rng_state = 13371337 + multi_gpu_config.process_rank;
    model.rng_state_last_update = model.rng_state;

    llama_materialize_parameters(&model);
    if (resuming == 1) {
        llama_load_checkpoint_weights(&model, resume_model_path);
    } else if (ends_with_bin(model_source)) {
        llama_load_checkpoint_weights(&model, model_source);
    } else {
        llama_random_init_weights(&model);
    }

    int tokens_per_fwdbwd = B * T * multi_gpu_config.num_processes;
    if (total_batch_size == -1) {
        total_batch_size = tokens_per_fwdbwd;
    }
    assert(total_batch_size % tokens_per_fwdbwd == 0);
    int grad_accum_steps = total_batch_size / tokens_per_fwdbwd;

    print_llama_config(&model, model_source);
    printf0("train data pattern    : %s\n", train_data_pattern);
    printf0("val data pattern      : %s\n", val_data_pattern);
    printf0("output log dir        : %s\n", output_log_dir == NULL ? "NULL" : output_log_dir);
    printf0("B/T/total batch       : %d / %d / %d\n", B, T, total_batch_size);
    printf0("grad accum steps      : %d\n", grad_accum_steps);
    printf0("scheduler/LR/warmup   : %s / %.8f / %d\n", lr_scheduler_type, learning_rate, warmup_iterations);
    printf0("final LR fraction     : %.6f\n", final_learning_rate_frac);
    printf0("weight decay          : %.6f\n", weight_decay);
    printf0("max steps             : %d\n", max_steps);
    printf0("eval/sample/genT      : %d / %d / %d\n", val_loss_every, sample_every, genT);
    printf0("val max steps         : %d\n", val_max_steps);
    printf0("hellaswag eval        : %d\n", hellaswag_eval);
    printf0("recompute / ZeRO      : %d / %d\n", recompute, zero_stage);
    printf0("checkpointing         : every=%d keep=%d major_every=%d resume=%d\n",
            checkpoint_every, checkpoints_keep, major_checkpoint_every, resume);
    printf0("outlier skip z        : loss=%.3f grad=%.3f\n", skip_update_lossz, skip_update_gradz);
    printf0("distributed           : processes=%d rank=%d gpus_per_node=%d init=%s fs=%s server=%s\n",
            multi_gpu_config.num_processes, multi_gpu_config.process_rank, gpus_per_node,
            nccl_init_method, fs_path, server_ip);
    printf0("Llama GQA uses TK forward where supported; CUDA fallback/backward remains active.\n");

    if (T > model.config.max_seq_len) {
        fprintf(stderr, "Sequence length T=%d exceeds model max_seq_len=%d\n", T, model.config.max_seq_len);
        exit(EXIT_FAILURE);
    }
    if (T % 16 != 0) {
        fprintf(stderr, "Llama RoPE requires T %% 16 == 0, got T=%d\n", T);
        exit(EXIT_FAILURE);
    }
    if ((B * T) % 128 != 0) {
        fprintf(stderr, "TK GEMM path requires B*T %% 128 == 0, got B*T=%d\n", B * T);
        exit(EXIT_FAILURE);
    }

    set_zero_configs(&multi_gpu_config, zero_stage, model.num_parameters);
    llama_validate_zero_tensor_sharding(&model, &multi_gpu_config);
    llama_allocate_zero3_parameter_shards(&model, &multi_gpu_config);
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T,
                    multi_gpu_config.process_rank, multi_gpu_config.num_processes, 1);
    dataloader_init(&val_loader, val_data_pattern, B, T,
                    multi_gpu_config.process_rank, multi_gpu_config.num_processes, 0);

    int train_num_batches = max_steps;
    if (train_num_batches < 0) {
        train_num_batches = train_loader.num_tokens / total_batch_size;
    }
    int val_num_batches = val_max_steps;
    if (val_num_batches < 0) {
        val_num_batches = val_loader.num_tokens / tokens_per_fwdbwd;
    }
    printf0("train_num_batches     : %d\n", train_num_batches);
    printf0("val_num_batches       : %d\n", val_num_batches);

    EvalLoader eval_loader;
    const char* hellaswag_path = "dev/data/hellaswag/hellaswag_val_llama3.bin";
    const bool hellaswag_available = access(hellaswag_path, F_OK) == 0;
    const bool run_hellaswag = hellaswag_eval && hellaswag_available;
    if (run_hellaswag) {
        evalloader_init(&eval_loader, hellaswag_path, B, T,
                        multi_gpu_config.process_rank, multi_gpu_config.num_processes);
    } else if (hellaswag_eval) {
        printf0("Llama HellaSwag eval not found at %s, skipping.\n", hellaswag_path);
    }

    if (output_log_dir != NULL && multi_gpu_config.process_rank == 0) {
        create_dir_if_not_exists(output_log_dir);
    }
    Logger logger;
    logger_init(&logger, output_log_dir, multi_gpu_config.process_rank, resuming);

    LlamaRuntimeState state;
    llama_runtime_allocate(&state, &model, B, T);
    llama_allocate_optimizer_state(&model);
    int step = 0;
    if (resuming == 1) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin",
                 output_log_dir, resume_max_step, multi_gpu_config.process_rank);
        llama_load_state(&step, &model, &train_loader, filename_buffer);
    }

    LearningRateScheduler lr_scheduler;
    lr_scheduler_init(&lr_scheduler, lr_scheduler_type, learning_rate,
                      warmup_iterations, train_num_batches, final_learning_rate_frac);

    OutlierDetector loss_outlier_detector, grad_norm_outlier_detector;
    init_detector(&loss_outlier_detector);
    init_detector(&grad_norm_outlier_detector);

    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaProfilerStart());
    double total_sum_iteration_time_s = 0.0;
    float ema_tokens_per_second = 0.0f;
    for (; step <= train_num_batches; ++step) {
        NvtxRange step_range("Llama train step", step);
        bool last_step = step == train_num_batches;

        if (step % val_loss_every == 0 || last_step) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; ++i) {
                dataloader_next_batch(&val_loader);
                val_loss += llama_validate(&model, &state, val_loader.inputs, val_loader.targets, B, T);
            }
            val_loss /= val_num_batches;
            val_loss = multi_gpu_cpu_float_sum(val_loss, &multi_gpu_config) / multi_gpu_config.num_processes;
            printf0("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        if (run_hellaswag && ((step > 0 && step % val_loss_every == 0) || last_step)) {
            float eval_acc_norm = 0.0f;
            evalloader_reset(&eval_loader);
            for (int i = 0; i < eval_loader.num_batches; ++i) {
                evalloader_next_batch(&eval_loader);
                llama_validate(&model, &state, eval_loader.inputs, eval_loader.targets, B, T);
                int correct = evalloader_stat_losses(&eval_loader, state.cpu_losses);
                eval_acc_norm += (float)correct;
            }
            eval_acc_norm = multi_gpu_cpu_float_sum(eval_acc_norm, &multi_gpu_config);
            printf0("HellaSwag: %d/%d = %f\n",
                    (int)eval_acc_norm, eval_loader.num_examples,
                    eval_acc_norm / eval_loader.num_examples);
            logger_log_eval(&logger, step, eval_acc_norm / eval_loader.num_examples);
        }

        if ((checkpoint_every > 0 && output_log_dir != NULL) &&
            ((step > 0 && step % checkpoint_every == 0) || last_step)) {
            llama_write_checkpoint(output_log_dir, step, &model, &train_loader, &multi_gpu_config);
            int step_delete = step - checkpoints_keep * checkpoint_every;
            if (checkpoints_keep > 0 && step_delete > 0 &&
                (major_checkpoint_every == 0 || step_delete % major_checkpoint_every != 0)) {
                llama_delete_checkpoint(output_log_dir, step_delete, &multi_gpu_config);
            }
        }

        if (last_step) { break; }

        cudaCheck(cudaEventRecord(start));
        for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
            dataloader_next_batch(&train_loader);
            llama_forward(&model, &state, train_loader.inputs, B, T);
            llama_backward_and_reduce(&model, &state, train_loader.inputs, train_loader.targets,
                                      grad_accum_steps, micro_step);
        }
        float zloss = (float)update_detector(&loss_outlier_detector, (double)state.mean_loss);
        float step_learning_rate = get_learning_rate(&lr_scheduler, step);
        float grad_norm = llama_calculate_grad_norm(&model, &state, &multi_gpu_config);
        float zgrad = (float)update_detector(&grad_norm_outlier_detector, (double)grad_norm);
        if (isfinite(zloss) && skip_update_lossz != 0.0f && zloss > skip_update_lossz) {
            printf0("skipping update due to loss z-score of %f\n", zloss);
        } else if (isfinite(zgrad) && skip_update_gradz != 0.0f && zgrad > skip_update_gradz) {
            printf0("skipping update due to grad z-score of %f\n", zgrad);
        } else {
            float grad_clip = 1.0f;
            float grad_scale = (grad_norm > grad_clip) ? grad_clip / grad_norm : 1.0f;
            llama_update(&model, step_learning_rate, 0.9f, 0.95f, 1e-8f,
                         weight_decay, grad_scale, step + 1, &multi_gpu_config);
        }
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(end));

        float time_elapsed_ms;
        cudaCheck(cudaEventElapsedTime(&time_elapsed_ms, start, end));
        size_t tokens_processed = (size_t)multi_gpu_config.num_processes * B * T * grad_accum_steps;
        float tokens_per_second = tokens_processed / time_elapsed_ms * 1000.0f;
        float bias_corrected_ema_tokens_per_second = tokens_per_second;
        if (step > 0) {
            total_sum_iteration_time_s += time_elapsed_ms / 1000.0;
            ema_tokens_per_second = 0.95f * ema_tokens_per_second + 0.05f * tokens_per_second;
            bias_corrected_ema_tokens_per_second = ema_tokens_per_second / (1.0f - powf(0.95f, step));
        }
        printf0("step %4d/%d | loss %7.6f (%+.2fz)| norm %6.4f (%+.2fz)| lr %.2e | %.2f ms | %.0f tok/s\n",
                step + 1, train_num_batches, state.mean_loss, zloss, grad_norm, zgrad,
                step_learning_rate, time_elapsed_ms, bias_corrected_ema_tokens_per_second);
        logger_log_train(&logger, step, state.mean_loss, step_learning_rate, grad_norm);
        if (step == 3) { cudaProfilerStop(); }
    }

    if (train_num_batches > 1) {
        printf0("total average iteration time: %f ms\n",
                total_sum_iteration_time_s / (train_num_batches - 1) * 1000.0);
    }
    cudaCheck(cudaEventDestroy(end));
    cudaCheck(cudaEventDestroy(start));
    if (run_hellaswag) { evalloader_free(&eval_loader); }
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    llama_free(&model, &state);
    multi_gpu_config_free(&multi_gpu_config);
    llama_common_free();
    return EXIT_SUCCESS;
}
