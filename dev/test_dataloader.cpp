#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "llmc/dataloader.h"

static const char* GPT_TRAIN_PATH = "/tmp/llmkittens_gpt_train.bin";
static const char* LLAMA_TRAIN_PATH = "/tmp/llmkittens_llama_train.bin";
static const char* GPT_EVAL_PATH = "/tmp/llmkittens_gpt_eval.bin";
static const char* LLAMA_EVAL_PATH = "/tmp/llmkittens_llama_eval.bin";

static void write_header(FILE* f, int magic, int version, int count, int extra) {
    int header[HEADER_SIZE];
    memset(header, 0, sizeof(header));
    header[0] = magic;
    header[1] = version;
    header[2] = count;
    header[3] = extra;
    fwriteCheck(header, sizeof(int), HEADER_SIZE, f);
}

template<typename T>
static void write_train_file(const char* path, int magic, int version, const std::vector<T>& tokens) {
    FILE* f = fopenCheck(path, "wb");
    write_header(f, magic, version, (int)tokens.size(), 0);
    fwriteCheck((void*)tokens.data(), sizeof(T), tokens.size(), f);
    fcloseCheck(f);
}

template<typename T>
static void write_eval_file(const char* path, int magic, int version, T start_example,
                            const std::vector<T>& stream) {
    FILE* f = fopenCheck(path, "wb");
    std::vector<T> payload = stream;
    payload[0] = start_example;
    payload[1] = (T)(payload.size() * sizeof(T));
    write_header(f, magic, version, 1, (int)payload[1]);
    fwriteCheck((void*)payload.data(), sizeof(T), payload.size(), f);
    fcloseCheck(f);
}

static void write_smoke_files() {
    write_train_file<uint16_t>(
        GPT_TRAIN_PATH,
        DATALOADER_GPT2_MAGIC,
        DATALOADER_GPT2_VERSION,
        {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21});

    write_train_file<uint32_t>(
        LLAMA_TRAIN_PATH,
        DATALOADER_LLAMA3_MAGIC,
        DATALOADER_LLAMA3_VERSION,
        {128000, 128001, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51});

    std::vector<uint16_t> gpt_eval = {
        0, 0, 0, 2, ASSUMED_NUM_COMPLETIONS,
        2, 100, 101,
        1, 102,
        1, 103,
        1, 104,
        1, 105,
    };
    write_eval_file<uint16_t>(
        GPT_EVAL_PATH,
        EVALLOADER_GPT2_MAGIC,
        EVALLOADER_GPT2_VERSION,
        UINT16_MAX,
        gpt_eval);

    std::vector<uint32_t> llama_eval = {
        0, 0, 0, 1, ASSUMED_NUM_COMPLETIONS,
        2, 128000, 128001,
        1, 128002,
        1, 128003,
        1, 128004,
        1, 128005,
    };
    write_eval_file<uint32_t>(
        LLAMA_EVAL_PATH,
        EVALLOADER_LLAMA3_MAGIC,
        EVALLOADER_LLAMA3_VERSION,
        UINT32_MAX,
        llama_eval);
}

static void check_train_loader(const char* path, int expected_magic, int expected_token_bytes,
                               int expected_first, int expected_shifted, int expected_target3,
                               int process_rank) {
    DataLoader loader;
    dataloader_init(&loader, path, 1, 4, process_rank, 2, 0);
    assert(loader.data_magic == expected_magic);
    assert(loader.token_bytes == expected_token_bytes);
    dataloader_next_batch(&loader);
    assert(loader.inputs[0] == expected_first);
    assert(loader.targets[0] == expected_shifted);
    assert(loader.targets[3] == expected_target3);
    dataloader_free(&loader);
}

static void check_eval_loader(const char* path, int expected_magic, int expected_token_bytes,
                              int expected_label, int expected_context0, int expected_completion0) {
    EvalLoader loader;
    evalloader_init(&loader, path, ASSUMED_NUM_COMPLETIONS, 16, 0, 1);
    assert(loader.data_magic == expected_magic);
    assert(loader.token_bytes == expected_token_bytes);
    evalloader_next_batch(&loader);
    assert(loader.label[0] == expected_label);
    assert(loader.inputs[0] == expected_context0);
    assert(loader.inputs[1] == expected_context0 + 1);
    assert(loader.inputs[2] == expected_completion0);
    assert(loader.targets[1] == expected_completion0);
    assert(loader.mask[1] == 1);
    assert(loader.num_completions == ASSUMED_NUM_COMPLETIONS);
    evalloader_free(&loader);
}

int main() {
    write_smoke_files();

    check_train_loader(GPT_TRAIN_PATH, DATALOADER_GPT2_MAGIC, (int)sizeof(uint16_t), 10, 11, 14, 0);
    check_train_loader(GPT_TRAIN_PATH, DATALOADER_GPT2_MAGIC, (int)sizeof(uint16_t), 14, 15, 18, 1);
    check_train_loader(LLAMA_TRAIN_PATH, DATALOADER_LLAMA3_MAGIC, (int)sizeof(uint32_t), 128000, 128001, 44, 0);
    check_train_loader(LLAMA_TRAIN_PATH, DATALOADER_LLAMA3_MAGIC, (int)sizeof(uint32_t), 44, 45, 48, 1);

    check_eval_loader(GPT_EVAL_PATH, EVALLOADER_GPT2_MAGIC, (int)sizeof(uint16_t), 2, 100, 102);
    check_eval_loader(LLAMA_EVAL_PATH, EVALLOADER_LLAMA3_MAGIC, (int)sizeof(uint32_t), 1, 128000, 128002);

    remove(GPT_TRAIN_PATH);
    remove(LLAMA_TRAIN_PATH);
    remove(GPT_EVAL_PATH);
    remove(LLAMA_EVAL_PATH);

    printf("DataLoader/EvalLoader smoke OK: GPT-2 uint16 and Llama-3 uint32 formats\n");
    return 0;
}
