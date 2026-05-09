/*
Convenience binary for profiling the CUDA kernels in one GPT-2 training step.

Compile:

    make profile_gpt2cu NO_MULTI_GPU=1

Then run with Nsight Compute, for example:

    sudo ncu --set full --import-source yes -o profile -f ./profile_gpt2cu

This writes profile.ncu-rep, which can be opened in the Nsight Compute UI.
*/

#define TESTING
#include "train_gpt2.cu"

int main(int argc, char *argv[]) {
    int gelu_fusion = 0;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--gelu-fusion") == 0 || strcmp(argv[i], "-ge") == 0) && i + 1 < argc) {
            gelu_fusion = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [--gelu-fusion 0|1]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (gelu_fusion < 0 || gelu_fusion > 1) {
        fprintf(stderr, "--gelu-fusion must be 0 or 1 for profiling\n");
        return EXIT_FAILURE;
    }

    char nccl_init_method[256] = "fs";
    int num_processes = 1;
    int process_rank = 0;
    int gpus_per_node = 1;
    char server_ip[256] = "";
    char fs_path[256] = "/tmp/llmk_profile_gpt2_nccl";
    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);
    common_start(true, true);

    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, "gpt2_124M_bf16.bin");

    // If this OOMs, reduce B first. Keep T as a power of two for cleaner traces.
    int B = 24;
    int T = 1024;
    printf("batch size: %d\n", B);
    printf("sequence length: %d\n", T);

    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    for (int i = 0; i < B * T; ++i) {
        x[i] = i % model.config.vocab_size;
        y[i] = i % model.config.vocab_size;
    }

    // One block is enough for profiling: transformer layers repeat the same kernels.
    model.config.num_layers = 1;
    model.gelu_fusion = gelu_fusion;
    printf("gelu fusion: %d\n", gelu_fusion);
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

    gpt2_allocate_state(&model, B, T);
    gpt2_forward(&model, x, B, T);
    gpt2_backward_and_reduce(&model, x, y, 1, 0);
    float grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
    float grad_scale = (grad_norm > 1.0f) ? 1.0f / grad_norm : 1.0f;
    gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, grad_scale, 1, &multi_gpu_config);
    cudaCheck(cudaDeviceSynchronize());

    free(x);
    free(y);
    gpt2_free(&model);
    common_free(model);
    return 0;
}
