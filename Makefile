# llm.kittens — H100 GPT-2/GPT-3/Llama-3 trainer on top of ThunderKittens.
# Adapted from llm.c's Makefile. Differences:
#   * sm_90a by default (TK H100 kernels need WGMMA + TMA — sm_90 alone is not enough)
#     plus opt-in SM100/SM103/SM120 Blackwell build modes
#   * c++20 instead of c++17 (TK requires it)
#   * BF16 only — FP16/FP32 paths removed; TK H100 GEMM/MHA are bf16
#   * cuBLAS, cuBLASLt, cuDNN all removed (every matmul/attn goes through TK)
#   * TK include paths added; TK_ROOT defaults to ../ThunderKittens

CC ?= clang
CXX ?= c++
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -Wno-unused-function -Wno-sign-compare -Wno-missing-field-initializers
LDFLAGS =
LDLIBS = -lm
INCLUDES =
CFLAGS_COND = -march=native

SHELL_UNAME = $(shell uname)
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@
CUDA_OUTPUT_FILE = -o $@

FORCE_NVCC_O ?= 3
DEVICE_ARCH ?= SM90

ifeq ($(DEVICE_ARCH),SM90)
  KITTENS_ARCH_DEFINE := -DKITTENS_SM90
  CUDA_GENCODE := -gencode arch=compute_90a,code=sm_90a
  DEVICE_ARCH_LABEL := sm_90a
  DEVICE_ARCH_CC := 90
else ifeq ($(DEVICE_ARCH),SM100)
  KITTENS_ARCH_DEFINE := -DKITTENS_SM100
  CUDA_GENCODE := -gencode arch=compute_100a,code=sm_100a
  DEVICE_ARCH_LABEL := sm_100a
  DEVICE_ARCH_CC := 100
else ifeq ($(DEVICE_ARCH),SM103)
  KITTENS_ARCH_DEFINE := -DKITTENS_SM103
  CUDA_GENCODE := -gencode arch=compute_103a,code=sm_103a
  DEVICE_ARCH_LABEL := sm_103a
  DEVICE_ARCH_CC := 103
else ifeq ($(DEVICE_ARCH),SM120)
  KITTENS_ARCH_DEFINE := -DKITTENS_SM120
  CUDA_GENCODE := -gencode arch=compute_120a,code=sm_120a
  DEVICE_ARCH_LABEL := sm_120a
  DEVICE_ARCH_CC := 120
else
  $(error Unsupported DEVICE_ARCH=$(DEVICE_ARCH). Use SM90, SM100, SM103, or SM120.)
endif

# ThunderKittens
TK_ROOT ?= $(abspath ../ThunderKittens)
ifeq ($(wildcard $(TK_ROOT)/include/kittens.cuh),)
  $(warning ThunderKittens not found at TK_ROOT=$(TK_ROOT) — set TK_ROOT to your ThunderKittens checkout)
endif

# llm.c reference checkout — used by parity tests in dev/cuda/probe_*_ref.cu.
LLMC_REF_ROOT ?= $(abspath ../llm.c)
ifeq ($(wildcard $(LLMC_REF_ROOT)/llmc/layernorm.cuh),)
  $(warning llm.c not found at LLMC_REF_ROOT=$(LLMC_REF_ROOT) — parity reference probes will not build)
endif

# NVCC flags — modeled after ThunderKittens/kernels/common.mk
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++20 -O$(FORCE_NVCC_O)
NVCC_FLAGS += --expt-extended-lambda --expt-relaxed-constexpr
NVCC_FLAGS += -forward-unknown-to-host-compiler
NVCC_FLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -ftemplate-backtrace-limit=0
NVCC_FLAGS += -I$(TK_ROOT)/include -I$(TK_ROOT)/prototype
NVCC_FLAGS += $(KITTENS_ARCH_DEFINE) $(CUDA_GENCODE)
NVCC_FLAGS += -DENABLE_BF16

NVCC_LDFLAGS = -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static
NVCC_INCLUDES =
NVCC_LDLIBS =

BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))

# nvidia-smi-based GPU sanity check (warn if not Hopper)
ifneq ($(CI),true)
  ifndef GPU_COMPUTE_CAPABILITY
    ifneq ($(shell which nvidia-smi 2>/dev/null),)
      GPU_COMPUTE_CAPABILITY := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/\.//g' | sort -n | head -n 1)
      ifneq ($(GPU_COMPUTE_CAPABILITY),)
        ifneq ($(GPU_COMPUTE_CAPABILITY),$(DEVICE_ARCH_CC))
          $(info ⚠ Detected GPU compute capability $(GPU_COMPUTE_CAPABILITY); DEVICE_ARCH=$(DEVICE_ARCH) targets $(DEVICE_ARCH_LABEL). Build will still proceed.)
        endif
      endif
    endif
  endif
endif

$(info ---------------------------------------------)
$(info llm.kittens build configuration)
$(info ---------------------------------------------)
$(info TK_ROOT          : $(TK_ROOT))
$(info NVCC arch        : $(DEVICE_ARCH_LABEL))
$(info Precision        : BF16 (locked))
$(info ---------------------------------------------)

NVCC := $(shell which nvcc 2>/dev/null)
NVCC_LDLIBS += -lnvidia-ml

# OpenMP — useful for the CPU-side dataloader / outlier detector
ifeq ($(NO_OMP), 1)
  $(info → OpenMP disabled)
else
  ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
    CFLAGS += -fopenmp -DOMP
    LDLIBS += -lgomp
    NVCC_FLAGS += -Xcompiler=-fopenmp
    $(info ✓ OpenMP found)
  else
    $(info ✗ OpenMP not found)
  endif
endif

# NCCL — multi-GPU
NCCL_DIR ?=
NCCL_INCLUDE_PATH_ORIGIN := $(origin NCCL_INCLUDE_PATH)
NCCL_LIB_PATH_ORIGIN := $(origin NCCL_LIB_PATH)
NCCL_INCLUDE_PATH ?= $(if $(NCCL_DIR),$(NCCL_DIR)/include,)
NCCL_LIB_PATH ?= $(if $(NCCL_DIR),$(if $(wildcard $(NCCL_DIR)/lib64),$(NCCL_DIR)/lib64,$(NCCL_DIR)/lib),)
NCCL_CUSTOM_REQUESTED := $(strip $(NCCL_DIR)$(if $(filter undefined,$(NCCL_INCLUDE_PATH_ORIGIN)),,$(NCCL_INCLUDE_PATH))$(if $(filter undefined,$(NCCL_LIB_PATH_ORIGIN)),,$(NCCL_LIB_PATH)))
NCCL_SYSTEM_FOUND := $(shell (ldconfig -p 2>/dev/null | grep -q 'libnccl\.so' || dpkg -l 2>/dev/null | grep -q nccl) && ([ -f /usr/include/nccl.h ] || [ -f /usr/local/cuda/include/nccl.h ]) && echo yes)
ifeq ($(NO_MULTI_GPU), 1)
  $(info → Multi-GPU (NCCL) disabled)
else ifneq ($(NCCL_CUSTOM_REQUESTED),)
  ifeq ($(NCCL_INCLUDE_PATH),)
    $(error NCCL include path is empty; set NCCL_INCLUDE_PATH or NCCL_DIR)
  endif
  ifeq ($(NCCL_LIB_PATH),)
    $(error NCCL library path is empty; set NCCL_LIB_PATH or NCCL_DIR)
  endif
  ifeq ($(wildcard $(NCCL_INCLUDE_PATH)/nccl.h),)
    $(error nccl.h not found at $(NCCL_INCLUDE_PATH)/nccl.h)
  endif
  ifeq ($(wildcard $(NCCL_LIB_PATH)/libnccl.so*),)
    $(error libnccl.so not found under $(NCCL_LIB_PATH))
  endif
  $(info ✓ NCCL enabled from NCCL_INCLUDE_PATH=$(NCCL_INCLUDE_PATH) NCCL_LIB_PATH=$(NCCL_LIB_PATH))
  NVCC_FLAGS += -DMULTI_GPU
  NVCC_INCLUDES += -I$(NCCL_INCLUDE_PATH)
  NVCC_LDFLAGS += -L$(NCCL_LIB_PATH)
  NVCC_LDLIBS += -lnccl
else
  ifeq ($(NCCL_SYSTEM_FOUND), yes)
    $(info ✓ NCCL found, multi-GPU enabled)
    NVCC_FLAGS += -DMULTI_GPU
    NVCC_LDLIBS += -lnccl
  else
    $(info ✗ NCCL not found, multi-GPU disabled)
    $(info     install libnccl2/libnccl-dev or set NCCL_DIR/NCCL_INCLUDE_PATH/NCCL_LIB_PATH)
  endif
endif

# OpenMPI — multi-node init
OPENMPI_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi
OPENMPI_LIB_PATH = $(OPENMPI_DIR)/lib/
OPENMPI_INCLUDE_PATH = $(OPENMPI_DIR)/include/
ifeq ($(NO_USE_MPI), 1)
  $(info → MPI disabled)
else ifeq ($(shell [ -d $(OPENMPI_LIB_PATH) ] && [ -d $(OPENMPI_INCLUDE_PATH) ] && echo "exists"), exists)
  $(info ✓ MPI enabled)
  NVCC_INCLUDES += -I$(OPENMPI_INCLUDE_PATH)
  NVCC_LDFLAGS += -L$(OPENMPI_LIB_PATH)
  NVCC_LDLIBS += -lmpi
  NVCC_FLAGS += -DUSE_MPI
else
  $(info ✗ MPI not found)
endif

$(info ---------------------------------------------)

.PHONY: all cuda_runtime_check test_dataloader test_matmul test_attention test_layernorm test_rope test_rmsnorm test_swiglu test_attention_gqa test_gelu test_fused_classifier test_encoder test_adamw test_global_norm test-kernels probe_layernorm_ref probe_layernorm_tk probe_gelu_ref probe_gelu_tk probe_encoder_ref probe_encoder_tk probe_global_norm_ref probe_global_norm_tk probe_swiglu probe_adamw_ref probe_adamw_tk probe_fused_classifier_ref probe_fused_classifier_tk probe_attention_ref probe_attention_tk probe_matmul_ref probe_matmul_tk probe_rmsnorm probe_rope probe_attention_gqa parity-kernels train_gpt2cu train_llama3cu gpt2_validate test_gpt2cu profile_gpt2cu clean

ifeq ($(NVCC),)
  $(error nvcc not found — install CUDA Toolkit 12.4+)
endif

# v1 progress: M1 plus the GPT-2 compile path are wired up. Runtime parity still
# depends on the remaining TK backward/performance kernels and H100 validation.

all: test_matmul train_gpt2cu test_gpt2cu

test_dataloader: dev/test_dataloader.cpp llmc/dataloader.h llmc/utils.h llmc/rand.h
	$(CXX) $(CXXFLAGS) -I. $< $(LDFLAGS) $(LDLIBS) -o $@

cuda_runtime_check: dev/cuda/cuda_runtime_check.cu
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/cuda_runtime_check.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

# GEMM smoke test — exercises llmc/matmul.cuh end-to-end against a naive bf16
# reference kernel. Requires H100 (sm_90a).
test_matmul: dev/cuda/test_matmul.cu llmc/matmul.cuh llmc/tk/gemm_h100.cuh llmc/tk/tk_common.cuh
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_matmul.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_attention: dev/cuda/test_attention.cu llmc/attention.cuh llmc/tk/attention_h100.cuh llmc/tk/tk_common.cuh
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_attention.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_layernorm: dev/cuda/test_layernorm.cu llmc/layernorm.cuh llmc/tk/layernorm_tk.cuh llmc/tk/tk_common.cuh
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_layernorm.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_attention_gqa: dev/cuda/test_attention_gqa.cu llmc/attention_gqa.cuh llmc/tk/attention_gqa_h100.cuh llmc/rope.cuh llmc/tk/rope_tk.cuh
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_attention_gqa.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_rope: dev/cuda/test_rope.cu llmc/rope.cuh llmc/tk/rope_tk.cuh llmc/tk/tk_common.cuh
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_rope.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_rmsnorm: dev/cuda/test_rmsnorm.cu llmc/rmsnorm.cuh llmc/tk/rmsnorm_tk.cuh llmc/tk/tk_common.cuh
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_rmsnorm.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_swiglu: dev/cuda/test_swiglu.cu llmc/swiglu.cuh llmc/cuda_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_swiglu.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_gelu: dev/cuda/test_gelu.cu llmc/gelu.cuh llmc/cuda_utils.cuh llmc/cuda_common.h
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_gelu.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_fused_classifier: dev/cuda/test_fused_classifier.cu llmc/fused_classifier.cuh llmc/cuda_utils.cuh llmc/cuda_common.h
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_fused_classifier.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_encoder: dev/cuda/test_encoder.cu llmc/encoder.cuh llmc/cuda_utils.cuh llmc/cuda_common.h
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_encoder.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_adamw: dev/cuda/test_adamw.cu llmc/adamw.cuh llmc/cuda_utils.cuh llmc/cuda_common.h
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_adamw.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

test_global_norm: dev/cuda/test_global_norm.cu llmc/global_norm.cuh llmc/cuda_utils.cuh llmc/cuda_common.h
	$(NVCC) $(NVCC_FLAGS) -I. dev/cuda/test_global_norm.cu $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

# Aggregate target — builds every per-kernel smoke binary. Use:
#   make -j test-kernels [DEVICE_ARCH=SM120]
# then drive them through the pytest harness under tests/.
test-kernels: test_matmul test_attention test_attention_gqa test_layernorm \
              test_rmsnorm test_rope test_swiglu test_gelu \
              test_fused_classifier test_encoder test_adamw test_global_norm

# ----------------------------------------------------------------------------
# Per-kernel parity probes — compare TK kernel against an authoritative
# baseline (llm.c for the GPT-2 stack; PyTorch for the Llama-only kernels).
# Each probe binary reads bf16/fp32 inputs from .npy files and writes outputs
# to .npy files; the pytest layer under tests/parity/ orchestrates inputs and
# diffs the outputs. See docs/testing.md "Per-kernel parity".

PARITY_NPY_INCLUDE = -I dev/third_party

# Reference probes: include order puts $(LLMC_REF_ROOT) FIRST so `#include
# "llmc/<name>.cuh"` resolves to the llm.c version. -I. is still added so the
# probe TU itself can find dev/third_party/npy/npy.h via PARITY_NPY_INCLUDE.
probe_layernorm_ref: dev/cuda/probe_layernorm_ref.cu $(LLMC_REF_ROOT)/llmc/layernorm.cuh
	$(NVCC) $(NVCC_FLAGS) -I$(LLMC_REF_ROOT) $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_gelu_ref: dev/cuda/probe_gelu_ref.cu $(LLMC_REF_ROOT)/llmc/gelu.cuh
	$(NVCC) $(NVCC_FLAGS) -I$(LLMC_REF_ROOT) $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_encoder_ref: dev/cuda/probe_encoder_ref.cu $(LLMC_REF_ROOT)/llmc/encoder.cuh
	$(NVCC) $(NVCC_FLAGS) -I$(LLMC_REF_ROOT) $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_global_norm_ref: dev/cuda/probe_global_norm_ref.cu $(LLMC_REF_ROOT)/llmc/global_norm.cuh
	$(NVCC) $(NVCC_FLAGS) -I$(LLMC_REF_ROOT) $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

# TK probes: -I. first so `#include "llmc/<name>.cuh"` resolves to the
# llm.kittens version (the TK-wrapped kernel).
probe_layernorm_tk: dev/cuda/probe_layernorm_tk.cu llmc/layernorm.cuh llmc/tk/layernorm_tk.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_gelu_tk: dev/cuda/probe_gelu_tk.cu llmc/gelu.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_encoder_tk: dev/cuda/probe_encoder_tk.cu llmc/encoder.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_global_norm_tk: dev/cuda/probe_global_norm_tk.cu llmc/global_norm.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_adamw_ref: dev/cuda/probe_adamw_ref.cu $(LLMC_REF_ROOT)/llmc/adamw.cuh
	$(NVCC) $(NVCC_FLAGS) -I$(LLMC_REF_ROOT) $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_fused_classifier_ref: dev/cuda/probe_fused_classifier_ref.cu $(LLMC_REF_ROOT)/llmc/fused_classifier.cuh
	$(NVCC) $(NVCC_FLAGS) -I$(LLMC_REF_ROOT) $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_attention_ref: dev/cuda/probe_attention_ref.cu $(LLMC_REF_ROOT)/llmc/attention.cuh $(LLMC_REF_ROOT)/llmc/matmul.cuh
	$(NVCC) $(NVCC_FLAGS) -I$(LLMC_REF_ROOT) $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -lcublas -lcublasLt -o $@

# matmul ref needs cuBLASLt — llm.c's matmul_forward_cublaslt requires the
# globals defined in llm.c/llmc/cublas_common.h.
probe_matmul_ref: dev/cuda/probe_matmul_ref.cu $(LLMC_REF_ROOT)/llmc/matmul.cuh
	$(NVCC) $(NVCC_FLAGS) -I$(LLMC_REF_ROOT) $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -lcublas -lcublasLt -o $@

probe_adamw_tk: dev/cuda/probe_adamw_tk.cu llmc/adamw.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_fused_classifier_tk: dev/cuda/probe_fused_classifier_tk.cu llmc/fused_classifier.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_attention_tk: dev/cuda/probe_attention_tk.cu llmc/attention.cuh llmc/tk/attention_h100.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_matmul_tk: dev/cuda/probe_matmul_tk.cu llmc/matmul.cuh llmc/tk/gemm_h100.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

# Family B: TK-only probes for kernels with no llm.c counterpart. Reference is
# computed in PyTorch on the Python side.
probe_swiglu: dev/cuda/probe_swiglu.cu llmc/swiglu.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_rmsnorm: dev/cuda/probe_rmsnorm.cu llmc/rmsnorm.cuh llmc/tk/rmsnorm_tk.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_rope: dev/cuda/probe_rope.cu llmc/rope.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

probe_attention_gqa: dev/cuda/probe_attention_gqa.cu llmc/attention_gqa.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $(PARITY_NPY_INCLUDE) $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) -o $@

parity-kernels: probe_layernorm_ref probe_layernorm_tk \
                probe_gelu_ref probe_gelu_tk \
                probe_encoder_ref probe_encoder_tk \
                probe_global_norm_ref probe_global_norm_tk \
                probe_adamw_ref probe_adamw_tk \
                probe_fused_classifier_ref probe_fused_classifier_tk \
                probe_attention_ref probe_attention_tk \
                probe_matmul_ref probe_matmul_tk \
                probe_swiglu probe_rmsnorm probe_rope probe_attention_gqa

train_gpt2cu: train_gpt2.cu llmc/matmul.cuh llmc/attention.cuh llmc/layernorm.cuh llmc/tk/layernorm_tk.cuh llmc/gelu.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_llama3cu: train_llama3.cu llmc/rmsnorm.cuh llmc/tk/rmsnorm_tk.cuh llmc/rope.cuh llmc/tk/rope_tk.cuh llmc/attention_gqa.cuh llmc/tk/attention_gqa_h100.cuh llmc/swiglu.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

gpt2_validate: dev/cuda/gpt2_validate.cu train_gpt2.cu llmc/matmul.cuh llmc/attention.cuh llmc/layernorm.cuh llmc/tk/layernorm_tk.cuh llmc/gelu.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2cu: test_gpt2.cu train_gpt2.cu llmc/matmul.cuh llmc/attention.cuh llmc/layernorm.cuh llmc/tk/layernorm_tk.cuh llmc/gelu.cuh
	$(NVCC) $(NVCC_FLAGS) -I. $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

profile_gpt2cu: profile_gpt2.cu train_gpt2.cu llmc/matmul.cuh llmc/attention.cuh llmc/layernorm.cuh llmc/tk/layernorm_tk.cuh llmc/gelu.cuh
	$(NVCC) $(NVCC_FLAGS) -I. -lineinfo $< $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

clean:
	$(REMOVE_FILES) cuda_runtime_check test_dataloader test_matmul test_attention test_layernorm test_rope test_rmsnorm test_swiglu test_attention_gqa test_gelu test_fused_classifier test_encoder test_adamw test_global_norm probe_layernorm_ref probe_layernorm_tk probe_gelu_ref probe_gelu_tk probe_encoder_ref probe_encoder_tk probe_global_norm_ref probe_global_norm_tk probe_adamw_ref probe_adamw_tk probe_fused_classifier_ref probe_fused_classifier_tk probe_attention_ref probe_attention_tk probe_matmul_ref probe_matmul_tk probe_swiglu probe_rmsnorm probe_rope probe_attention_gqa train_gpt2cu train_llama3cu gpt2_validate test_gpt2cu profile_gpt2cu
	rm -f $(BUILD_DIR)/*.o
