/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#include "utils.h"
#include "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#include <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256
#define DATALOADER_GPT2_MAGIC 20240520
#define DATALOADER_GPT2_VERSION 1
#define DATALOADER_LLAMA3_MAGIC 20240801
#define DATALOADER_LLAMA3_VERSION 7

typedef struct {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t num_tokens; // total number of tokens
    size_t shard_num_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_result; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* tokens_file;
    // data buffers
    void* buffer; // we fread data from file into this buffer
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    int should_shuffle;
    int* shard_indices;
    int* intra_shard_indices;
    // sizes in bytes
    size_t total_batch_size_bytes;  // total across all processes
    size_t local_batch_offset_bytes;  // inner-sample offset for this process
    size_t header_bytes;  // header size in bytes
    int64_t file_size_bytes;
    // token file format
    int token_bytes;
    int data_magic;
    int data_version;
} DataLoader;

static inline const char* dataloader_format_name_(int magic) {
    if (magic == DATALOADER_GPT2_MAGIC) { return "gpt-2"; }
    if (magic == DATALOADER_LLAMA3_MAGIC) { return "llama-3"; }
    return "unknown";
}

static inline int dataloader_token_bytes_from_header_(int magic, int version, const char* filename) {
    if (magic == DATALOADER_GPT2_MAGIC) {
        if (version != DATALOADER_GPT2_VERSION) {
            printf("Bad version in data file %s: got %d for gpt-2, expected %d\n",
                   filename, version, DATALOADER_GPT2_VERSION);
            exit(EXIT_FAILURE);
        }
        return (int)sizeof(uint16_t);
    }
    if (magic == DATALOADER_LLAMA3_MAGIC) {
        if (version != DATALOADER_LLAMA3_VERSION) {
            printf("Bad version in data file %s: got %d for llama-3, expected %d\n",
                   filename, version, DATALOADER_LLAMA3_VERSION);
            exit(EXIT_FAILURE);
        }
        return (int)sizeof(uint32_t);
    }
    printf("Bad magic in the data file %s: got %d\n", filename, magic);
    printf("---> HINT: Expected gpt-2 magic %d v%d or llama-3 magic %d v%d.\n",
           DATALOADER_GPT2_MAGIC, DATALOADER_GPT2_VERSION,
           DATALOADER_LLAMA3_MAGIC, DATALOADER_LLAMA3_VERSION);
    printf("---> HINT: Re-run data preprocessing with the correct --model_desc.\n");
    exit(EXIT_FAILURE);
}

static inline void dataloader_set_format_(DataLoader* loader, int magic, int version, int token_bytes, const char* filename) {
    if (loader->token_bytes == 0) {
        loader->token_bytes = token_bytes;
        loader->data_magic = magic;
        loader->data_version = version;
        loader->total_batch_size_bytes = loader->num_processes * loader->B * loader->T * (size_t)loader->token_bytes;
        loader->local_batch_offset_bytes = loader->process_rank * loader->B * loader->T * (size_t)loader->token_bytes;
        return;
    }
    if (loader->token_bytes != token_bytes || loader->data_magic != magic || loader->data_version != version) {
        printf("Mixed data shard formats are not supported: %s is %s magic %d v%d, but loader is %s magic %d v%d\n",
               filename,
               dataloader_format_name_(magic), magic, version,
               dataloader_format_name_(loader->data_magic), loader->data_magic, loader->data_version);
        exit(EXIT_FAILURE);
    }
}

int64_t dataloader_load_shard_(DataLoader *loader, int shard_index) {
    if (loader->should_shuffle) {
        shard_index = loader->shard_indices[shard_index];
    }
    // use the first glob match as the filename for now
    const char* filename = loader->glob_result.gl_pathv[shard_index];
    // open the input file for reading. also only a single file can be opened at a time
    if (loader->tokens_file != NULL) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->tokens_file);
    int magic = header[0];
    int version = header[1];
    int token_bytes = dataloader_token_bytes_from_header_(magic, version, filename);
    dataloader_set_format_(loader, magic, version, token_bytes, filename);
    int64_t ntok = header[2]; // number of tokens in the file
    assert(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->tokens_file, 0, SEEK_END); // seek to end of file
    loader->file_size_bytes = ftell(loader->tokens_file); // read the offset, i.e. file size
    fseekCheck(loader->tokens_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_size = (int64_t)loader->header_bytes + ntok * loader->token_bytes;
    if (loader->file_size_bytes != expected_file_size) {
        printf("Error: file size is not as expected for %s\n", filename);
        printf("Expected %lld bytes, got %lld bytes\n",
               (long long)expected_file_size, (long long)loader->file_size_bytes);
        exit(EXIT_FAILURE);
    }
    // -1 token due to us taking B*T+1 tokens but moving by B*T tokens
    size_t total_batch_tokens = loader->num_processes * loader->B * loader->T;
    assert(total_batch_tokens > 0);
    loader->shard_num_samples = (size_t)((ntok - 1) / (int64_t)total_batch_tokens);
    return ntok;
}

void prepare_intra_shard_indices_(DataLoader *loader) {
    // shuffle the examples inside the shards
    if (loader->intra_shard_indices != NULL) {
        // in case shards have different number of samples / sizes
        free(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
    init_identity_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples);
    random_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples, &loader->shuffle_rng);
}

void dataloader_reset(DataLoader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    if (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    if (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

void dataloader_advance_(DataLoader *loader) {
    if (loader->current_shard_idx == loader->glob_result.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        return;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    if (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

void dataloader_init(DataLoader *loader,
                     const char* filename_pattern,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes,
                     int should_shuffle) {
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = NULL;
    loader->should_shuffle = should_shuffle;
    loader->header_bytes = HEADER_SIZE * sizeof(int);
    loader->total_batch_size_bytes = 0;
    loader->local_batch_offset_bytes = 0;
    loader->token_bytes = 0;
    loader->data_magic = 0;
    loader->data_version = 0;
    loader->buffer = NULL;
    loader->shard_indices = NULL;
    loader->intra_shard_indices = NULL;

    // glob to get the list of files matching the pattern, these are our data shards
    int glob_status = glob(filename_pattern, 0, NULL, &loader->glob_result);
    if (glob_status != 0) {
        printf("Error: failed to glob pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }
    if (loader->glob_result.gl_pathc == 0) {
        printf("Error: no files found matching the pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }

    if (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        init_identity_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc);
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_total = 0;
    for (int shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++) {
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_index);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        assert(shard_ntok >= (int64_t) (num_processes * B * T + 1));
        ntok_total += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->buffer = mallocCheck((B * T + 1) * (size_t)loader->token_bytes);
    loader->inputs = (int*)mallocCheck(B * T * sizeof(int));
    loader->targets = (int*)mallocCheck(B * T * sizeof(int));
    loader->num_tokens = ntok_total;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

void dataloader_load_batch(DataLoader* loader) {
    assert(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != NULL));
    assert(loader->current_sample_idx < loader->shard_num_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_offset_bytes = idx * loader->total_batch_size_bytes;
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 tokens from the file into buffer
    fseekCheck(loader->tokens_file, (long) current_offset, SEEK_SET);
    freadCheck(loader->buffer, (size_t)loader->token_bytes, B*T+1, loader->tokens_file);
    // decode the buffer into inputs and targets (cast to int)
    if (loader->token_bytes == (int)sizeof(uint16_t)) {
        uint16_t* buffer = (uint16_t*)loader->buffer;
        for (int i = 0; i < B*T; i++) {
            loader->inputs[i] = (int)buffer[i];
            loader->targets[i] = (int)buffer[i+1];
        }
    } else {
        uint32_t* buffer = (uint32_t*)loader->buffer;
        for (int i = 0; i < B*T; i++) {
            assert(buffer[i] <= INT32_MAX);
            assert(buffer[i+1] <= INT32_MAX);
            loader->inputs[i] = (int)buffer[i];
            loader->targets[i] = (int)buffer[i+1];
        }
    }
}

void dataloader_next_batch(DataLoader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    if (loader->current_sample_idx >= loader->shard_num_samples) {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    loader->current_sample_idx += 1;
}


void dataloader_resume(DataLoader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);
}

void dataloader_free(DataLoader *loader) {
    free(loader->buffer);
    free(loader->inputs);
    free(loader->targets);
    if (loader->should_shuffle) {
        free(loader->shard_indices);
        free(loader->intra_shard_indices);
    }
    fcloseCheck(loader->tokens_file);
    globfree(&loader->glob_result);
}

// ----------------------------------------------------------------------------
// Distributed Eval Loader
// Many evals (like) HellaSwag and MMLU are multiple-choice
// where there are 4 possible continuations and a label for the correct one
// We want to load and serve these style of evals
/*
Copy pasting the section on the eval datafile format, from data_common.py:
- First comes a header with 256 int32s
- The examples follow, each example is a stream of uint16_t (GPT-2) or uint32_t (Llama-3):
    - <START_EXAMPLE> delimiter of the max token value
    - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
    - <EXAMPLE_INDEX>, the index of the example in the dataset
    - <LABEL>, the index of the correct completion
    - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
    - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
    - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
*/

// for now, could relax later
#define ASSUMED_NUM_COMPLETIONS 4
#define EVALLOADER_GPT2_MAGIC 20240522
#define EVALLOADER_GPT2_VERSION 1
#define EVALLOADER_LLAMA3_MAGIC 20240802
#define EVALLOADER_LLAMA3_VERSION 7
// helper macro for ceildiv
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

typedef struct {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    void* buffer; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    int num_examples; // in total across all processes
    int num_batches; // to process the entire dataset across all processes
    int start_example_index; // the assignment of work for this process, start
    int end_example_index; // and end. start is inclusive, end is exclusive
    int current_example_index; // the next example we would read
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
    char* mask; // mask=1 at all completion token locations
    int* label; // the correct completion labels
    int num_completions; // number of completions for this example
    // token file format
    int token_bytes;
    int data_magic;
    int data_version;
    uint32_t start_example;
} EvalLoader;

static inline int evalloader_token_bytes_from_header_(int magic, int version, const char* filename) {
    if (magic == EVALLOADER_GPT2_MAGIC) {
        if (version != EVALLOADER_GPT2_VERSION) {
            printf("Bad version in eval file %s: got %d for gpt-2, expected %d\n",
                   filename, version, EVALLOADER_GPT2_VERSION);
            exit(EXIT_FAILURE);
        }
        return (int)sizeof(uint16_t);
    }
    if (magic == EVALLOADER_LLAMA3_MAGIC) {
        if (version != EVALLOADER_LLAMA3_VERSION) {
            printf("Bad version in eval file %s: got %d for llama-3, expected %d\n",
                   filename, version, EVALLOADER_LLAMA3_VERSION);
            exit(EXIT_FAILURE);
        }
        return (int)sizeof(uint32_t);
    }
    printf("Bad magic in eval file %s: got %d\n", filename, magic);
    printf("---> HINT: Expected gpt-2 eval magic %d v%d or llama-3 eval magic %d v%d.\n",
           EVALLOADER_GPT2_MAGIC, EVALLOADER_GPT2_VERSION,
           EVALLOADER_LLAMA3_MAGIC, EVALLOADER_LLAMA3_VERSION);
    exit(EXIT_FAILURE);
}

static inline uint32_t evalloader_start_example_for_magic_(int magic) {
    if (magic == EVALLOADER_GPT2_MAGIC) { return UINT16_MAX; }
    if (magic == EVALLOADER_LLAMA3_MAGIC) { return UINT32_MAX; }
    return 0;
}

static inline void evalloader_read_example_header_(EvalLoader* loader, uint32_t example_header[3]) {
    if (loader->token_bytes == (int)sizeof(uint16_t)) {
        uint16_t raw[3];
        freadCheck(raw, sizeof(uint16_t), 3, loader->eval_file);
        example_header[0] = raw[0];
        example_header[1] = raw[1];
        example_header[2] = raw[2];
    } else {
        freadCheck(example_header, sizeof(uint32_t), 3, loader->eval_file);
    }
}

static inline uint32_t evalloader_buffer_token_(EvalLoader* loader, size_t idx) {
    if (loader->token_bytes == (int)sizeof(uint16_t)) {
        return ((uint16_t*)loader->buffer)[idx];
    }
    return ((uint32_t*)loader->buffer)[idx];
}

static inline int evalloader_buffer_token_int_(EvalLoader* loader, size_t idx) {
    uint32_t token = evalloader_buffer_token_(loader, idx);
    assert(token <= INT32_MAX);
    return (int)token;
}

void evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    int examples_per_process = CEIL_DIV(loader->num_examples, loader->num_processes);
    int can_fit_examples = (int) (loader->B / ASSUMED_NUM_COMPLETIONS);
    if (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        printf("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_NUM_COMPLETIONS);
        printf("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        exit(EXIT_FAILURE);
    }
    loader->num_batches = CEIL_DIV(examples_per_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->start_example_index = examples_per_process * loader->process_rank;
    loader->end_example_index = examples_per_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    if (loader->end_example_index > loader->num_examples) {
        loader->end_example_index = loader->num_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t header_bytes = HEADER_SIZE * sizeof(int);
    fseekCheck(loader->eval_file, (long) header_bytes, SEEK_SET);
    for (int i = 0; i < loader->start_example_index; i++) {
        uint32_t example_header[3];
        // read 3 token-sized values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        evalloader_read_example_header_(loader, example_header);
        // validate the <START_EXAMPLE> delimiter
        assert(example_header[0] == loader->start_example); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        assert(example_header[2] == (uint32_t)i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t remaining_bytes = example_header[1] - (size_t)loader->token_bytes * 3;
        assert(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (long) remaining_bytes, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_index = loader->start_example_index;
}

void evalloader_init(EvalLoader *loader,
                     const char* filename,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes) {
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->eval_file);
    int magic = header[0];
    int version = header[1];
    loader->token_bytes = evalloader_token_bytes_from_header_(magic, version, filename);
    loader->data_magic = magic;
    loader->data_version = version;
    loader->start_example = evalloader_start_example_for_magic_(magic);
    loader->num_examples = header[2]; // number of examples in the file
    assert(loader->num_examples >= num_processes); // avoid headaches for now
    size_t longest_example_bytes = header[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are either uint16_t or uint32_t.
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    assert(longest_example_bytes > 0 && longest_example_bytes < (1+ASSUMED_NUM_COMPLETIONS)*T*(size_t)loader->token_bytes);

    // allocate all the space we'll need
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS);
    loader->buffer = mallocCheck(longest_example_bytes);
    loader->inputs = (int*)calloc(B * T, sizeof(int));
    loader->targets = (int*)calloc(B * T, sizeof(int));
    loader->mask = (char*)mallocCheck(B * T * sizeof(char));
    loader->label = (int*)mallocCheck(can_fit_examples * sizeof(int));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

void evalloader_next_example_(EvalLoader *loader, int example_batch_index) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    int batch_dim_offset = example_batch_index * ASSUMED_NUM_COMPLETIONS;
    // read the current example header
    uint32_t example_header[3];
    evalloader_read_example_header_(loader, example_header);
    // validate the <START_EXAMPLE> delimiter
    assert(example_header[0] == loader->start_example); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    assert(example_header[2] == (uint32_t)loader->current_example_index); // <EXAMPLE_INDEX> should match the loop index
    assert(example_header[2] >= (uint32_t)loader->start_example_index && example_header[2] < (uint32_t)loader->end_example_index);
    // read the rest of the example (we have space for 3 more token values in buffer, it's ok)
    size_t example_bytes = example_header[1] - (size_t)loader->token_bytes * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->buffer, sizeof(char), example_bytes, loader->eval_file);
    // process the example label
    int label = evalloader_buffer_token_int_(loader, 0);
    int can_fit_examples = (int) (loader->B / ASSUMED_NUM_COMPLETIONS);
    assert(label >= 0 && label < ASSUMED_NUM_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    assert(example_batch_index >= 0 && example_batch_index < can_fit_examples);
    loader->label[example_batch_index] = label; // store for output
    // process the number of completions
    int num_completions = evalloader_buffer_token_int_(loader, 1);
    assert(num_completions == ASSUMED_NUM_COMPLETIONS); // we expect 4 completions for now
    assert(batch_dim_offset + num_completions <= B); // we expect to fit in the batch
    loader->num_completions = num_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    int context_length = evalloader_buffer_token_int_(loader, 2);
    size_t context_tokens_start = 3; // where the tokens start
    assert(context_length > 0 && context_length < T); // context is non-empty and up to T
    for (int b = 0; b < num_completions; b++) {
        for (int i = 0; i < context_length; i++) {
            int boff = batch_dim_offset + b;
            int tok_cur = evalloader_buffer_token_int_(loader, context_tokens_start + i);
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    size_t completions_iter = 3 + context_length;
    for (int c = 0; c < num_completions; c++) {
        int coff = batch_dim_offset + c;
        int completion_length = evalloader_buffer_token_int_(loader, completions_iter);
        size_t completion_tokens_start = completions_iter + 1;
        assert(completion_length > 0 && context_length + completion_length < T); // things fit?
        for (int i = 0; i < completion_length; i++) {
            int tok_cur = evalloader_buffer_token_int_(loader, completion_tokens_start + i);
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_length + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_length + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->mask[coff * T + context_length + i - 1] = 1;
        }
        completions_iter += 1 + completion_length; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_index += 1;
}

void evalloader_next_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->mask, 0, B * T * sizeof(char));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS); // how many examples can we fit in the batch?
    for (int i = 0; i < can_fit_examples; i++) {
        if (loader->current_example_index >= loader->end_example_index) {
            break; // this process has exhausted its work, noop from here on
        }
        evalloader_next_example_(loader, i);
    }
}

int evalloader_stat_losses(EvalLoader *loader, float* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    int correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS);
    for (int i = 0; i < can_fit_examples; i++) {
        float min_loss = 0.0f;
        int min_loss_index = -1;
        char active = 0; // is this example active or fully empty?
        // iterate the completions in this example
        for (int b = 0; b < ASSUMED_NUM_COMPLETIONS; b++) {
            int boff = i * ASSUMED_NUM_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            float average_loss = 0.0f;
            int count = 0;
            for (int t = 0; t < T; t++) {
                char mask = loader->mask[boff * T + t];
                if (mask == 1) {
                    active = 1;
                    average_loss += losses[boff * T + t];
                    count++;
                }
            }
            if (count > 0) { average_loss /= count; }
            if (b == 0 || average_loss < min_loss) {
                min_loss = average_loss;
                min_loss_index = b;
            }
        }
        if (active && (min_loss_index == loader->label[i])) {
            correct += 1;
        }
    }
    return correct;
}

void evalloader_free(EvalLoader *loader) {
    free(loader->buffer);
    free(loader->inputs);
    free(loader->targets);
    free(loader->mask);
    free(loader->label);
    fcloseCheck(loader->eval_file);
}

#endif // DATALOADER_H
