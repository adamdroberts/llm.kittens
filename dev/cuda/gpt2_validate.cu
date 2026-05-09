#define TESTING
#include "train_gpt2.cu"

/*
Forward-only GPT-2 sanity gate for M2.

This intentionally stops before backward/AdamW. It loads the reference batch and
loss from gpt2_124M_debug_state.bin, calls gpt2_validate(), and checks that the
forward loss is within the same BF16 tolerance used by test_gpt2.cu.
*/

int main(int argc, char *argv[]) {
    char nccl_init_method[256] = "mpi";
    int num_processes = -1;
    int process_rank = -1;
    int gpus_per_node = -1;
    char server_ip[256] = "";
    char fs_path[256] = "";
    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node,
                                             server_ip, fs_path, nccl_init_method);
    common_start(false, true);

    #if defined(ENABLE_BF16)
    const char* load_filename = "gpt2_124M_bf16.bin";
    const float loss_diff_threshold = 0.05f;
    #else
    const char* load_filename = "gpt2_124M.bin";
    const float loss_diff_threshold = 1e-5f;
    #endif

    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, load_filename);

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { exit(EXIT_FAILURE); }
        if (!(strlen(argv[i]) == 2 || strlen(argv[i]) == 3)) { exit(EXIT_FAILURE); }
        if (argv[i][0] != '-') { exit(EXIT_FAILURE); }
        if (argv[i][1] == 'w') { model.use_master_weights = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'r') { model.recompute = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'e') { model.gelu_fusion = atoi(argv[i + 1]); }
    }

    FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) {
        fprintf(stderr, "Bad magic state file\n");
        exit(EXIT_FAILURE);
    }
    if (state_header[1] != 2) {
        fprintf(stderr, "Bad version in state file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }

    int B = state_header[2];
    int T = state_header[3];
    if (T < 0 || T > (int)model.config.max_seq_len) {
        fprintf(stderr, "Bad sequence length in state file: %d\n", T);
        exit(EXIT_FAILURE);
    }

    int* x = (int*)mallocCheck((size_t)B * T * sizeof(int));
    int* y = (int*)mallocCheck((size_t)B * T * sizeof(int));
    freadCheck(x, sizeof(int), (size_t)B * T, state_file);
    freadCheck(y, sizeof(int), (size_t)B * T, state_file);

    const size_t V = model.config.vocab_size;
    const size_t logits_count = (size_t)B * T * V;
    fseekCheck(state_file, (long)(logits_count * sizeof(float)), SEEK_CUR);

    float expected_loss = 0.0f;
    freadCheck(&expected_loss, sizeof(float), 1, state_file);
    fcloseCheck(state_file);

    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);
    printf("expected_loss: %.9f\n", expected_loss);

    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);
    gpt2_allocate_state(&model, B, T);

    float actual_loss = gpt2_validate(&model, x, y, B, T);
    float loss_diff = fabsf(actual_loss - expected_loss);
    int loss_ok = loss_diff < loss_diff_threshold;

    printf("actual_loss: %.9f\n", actual_loss);
    printf("loss_diff: %.9f threshold %.9f\n", loss_diff, loss_diff_threshold);
    printf("overall okay: %d\n", loss_ok);
    if (loss_ok) {
        printf("gpt2_validate OK\n");
    }

    gpt2_free(&model);
    common_free(model);
    free(x);
    free(y);
    return loss_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
