/*
tk_common.cuh — bridge between llm.c-style C signatures and ThunderKittens.

llm.c's training loop calls C-style functions whose arguments are raw `floatX*`
buffers (where `floatX = __nv_bfloat16` in v1). ThunderKittens is a C++17/20
template library where kernels take `kittens::gl<>` global-layout descriptors
and tile types like `kittens::st_bf<M,N>`. This header is the only file that
should bring the `kittens::` namespace into scope; everywhere else in llmc/
should hide the templates behind plain function-pointer signatures.

Rules of the road:
  - All llmc/*.cuh wrappers #include "tk/tk_common.cuh", never <kittens.cuh>
    directly. This keeps a single, controlled bridge.
  - bf16 is the only activation/parameter dtype on the GPU. FP32 master weights
    and AdamW state are fine — those don't ride through TK kernels.
  - TMA descriptors require 128-byte aligned global pointers. The parameter
    allocator in train_gpt2.cu must guarantee this; see TK_ALIGN below.
*/
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <kittens.cuh>
#include <prototype.cuh>

#include "../cuda_common.h"

// Hard guard: llm.kittens v1 is BF16 only.
static_assert(std::is_same_v<floatX, __nv_bfloat16>,
              "llm.kittens v1 requires PRECISION=BF16. ThunderKittens H100 "
              "GEMM/MHA kernels are bf16; FP16/FP32 paths are not implemented.");

// Hard guard: H100 only in v1.
#ifndef KITTENS_SM90
#error "llm.kittens v1 targets H100 (sm_90a). Build with -DKITTENS_SM90."
#endif

namespace llmk {

// Reinterpret-cast helpers. llm.c uses `__nv_bfloat16*` (alias for `floatX*`);
// TK's gl<> wants `bf16*` (alias for the same). They're the same memory; this
// just keeps the declarations clean.
using bf16 = ::kittens::bf16;
static_assert(sizeof(bf16) == sizeof(__nv_bfloat16),
              "kittens::bf16 must alias __nv_bfloat16");

inline bf16*       to_bf16(__nv_bfloat16* p)       { return reinterpret_cast<bf16*>(p); }
inline const bf16* to_bf16(const __nv_bfloat16* p) { return reinterpret_cast<const bf16*>(p); }

// TMA descriptors need 128-byte alignment of the underlying global allocation.
// llm.c aligned to 16 bytes (the cuBLAS + element-wise minimum); we tighten it.
constexpr size_t TK_ALIGN = 128;

// Round a byte count up to TK_ALIGN.
inline size_t tk_align(size_t bytes) {
    return ((bytes + TK_ALIGN - 1) / TK_ALIGN) * TK_ALIGN;
}

// TK kernels generally need opt-in for the full H100 dynamic shared-mem budget.
// Call this once per kernel function before launching.
template <typename KernelFn>
inline void tk_set_max_dynamic_smem(KernelFn* fn, int smem_bytes = -1) {
    if (smem_bytes < 0) {
        // Conservative: ask for the full budget minus a 1 KB safety pad. TK's
        // own kernels do the same (see TK kernels/attention/mha_h100/mha_h100.cu).
        cudaDeviceProp prop;
        int dev = 0;
        cudaCheck(cudaGetDevice(&dev));
        cudaCheck(cudaGetDeviceProperties(&prop, dev));
        smem_bytes = prop.sharedMemPerBlockOptin - 1024;
    }
    cudaCheck(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(fn),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
}

} // namespace llmk
