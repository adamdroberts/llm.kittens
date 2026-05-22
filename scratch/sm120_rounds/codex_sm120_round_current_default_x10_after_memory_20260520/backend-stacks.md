# SM120 Backend Stack Probe

| Stack | Status | Candidate use | Evidence | Next action |
|---|---|---|---|---|
| ThunderKittens 2.0 | available | native TK kernels and current SM120 packed-QKV attention path | TK_ROOT=/mnt/disk1/home/adam/dev/open-source/ThunderKittens<br>header=/mnt/disk1/home/adam/dev/open-source/ThunderKittens/include/kittens.cuh<br>prototype=/mnt/disk1/home/adam/dev/open-source/ThunderKittens/prototype/prototype.cuh | benchmark against cuBLASLt/plain CUDA by shape before promoting TK-only wins |
| Plain CUDA | available | plain CUDA baselines and C++ benchmarks | nvcc=/usr/local/cuda/bin/nvcc<br>Build cuda_13.2.r13.2/compiler.37668154_0 | run the SM120 round on the RTX 5090 target for runtime timings |
| GPU runtime | available | runtime timing and correctness execution | nvidia-smi/NVML metadata query did not return device metadata in this process context<br>target runtime availability is proven by explicit SM120 correctness and benchmark runs | use explicit correctness and benchmark logs as the runtime evidence source |
| cuBLAS | available | baseline GEMM comparison where cuBLASLt epilogues are not needed | header=/usr/local/cuda/include/cublas_v2.h<br>library=libcublas.so.13 | add explicit cuBLAS benchmark/parity rows before selecting it over cuBLASLt |
| cuBLASLt | available | current SM120 GEMM baseline and fused GEMM epilogues | header=/usr/local/cuda/include/cublasLt.h<br>library=libcublasLt.so.13 | keep benchmark rows shape-specific; do not switch global defaults from one isolated win |
| cuDNN | available | attention alternatives through detected headers/libs; GPT-2 BF16 shape support still needs benchmark proof | header=/home/adam/miniconda3/envs/llm-kittens/lib/python3.13/site-packages/nvidia/cudnn/include/cudnn.h<br>version=9.22.0<br>library=/home/adam/miniconda3/envs/llm-kittens/lib/python3.13/site-packages/nvidia/cudnn/lib/libcudnn.so.9 | prototype as an opt-in benchmark first; current v1 build contract intentionally avoids -lcudnn |
| Triton | available | attention, normalization, elementwise fusion, and GEMM candidates | python=/home/adam/miniconda3/envs/llm-kittens/bin/python<br>python_version=3.13.13<br>module=triton version=3.6.0 origin=/home/adam/miniconda3/envs/llm-kittens/lib/python3.13/site-packages/triton/__init__.py<br>requirements.txt lists triton | add stack-specific parity tests before trainer promotion |
| Torch | available | PyTorch operator kernels for exact family-by-family backend comparisons | python=/home/adam/miniconda3/envs/llm-kittens/bin/python<br>python_version=3.13.13<br>module=torch version=2.11.0+cu130 origin=/home/adam/miniconda3/envs/llm-kittens/lib/python3.13/site-packages/torch/__init__.py<br>requirements.txt lists torch | add stack-specific parity tests before trainer promotion |
| CuTeDSL | available | Blackwell GEMM and fused epilogue candidates | python=/home/adam/miniconda3/envs/llm-kittens/bin/python<br>python_version=3.13.13<br>module=cutlass version=4.5.1 origin=/home/adam/miniconda3/envs/llm-kittens/lib/python3.13/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/__init__.py<br>requirements.txt lists nvidia-cutlass-dsl | add stack-specific parity tests before trainer promotion |

## Family Applicability Matrix

| Family | Stack | Status | Reason | Next action |
|---|---|---|---|---|
| gemm_forward | ThunderKittens 2.0 | candidate | native TK GEMM rows are benchmarked by GPT-2 shape | keep only shape wins that preserve the TinyStories smoke |
| gemm_forward | cuBLAS | candidate | cuBLAS BF16 GEMM is a direct baseline; fused rows use cuBLAS plus explicit CUDA pointwise work | compare against cuBLASLt and TK for every required pass and shape |
| gemm_forward | cuBLASLt | baseline | current SM120 GEMM baseline, including fused epilogue candidates | keep selector decisions shape-specific |
| gemm_forward | cuDNN | not_applicable | cuDNN is not used as a GEMM provider in this project | none until a cuDNN GEMM-equivalent path is intentionally scoped |
| gemm_forward | Triton | candidate | Triton can express GEMM or fused epilogue variants if installed | add Triton parity tests before timing or trainer promotion |
| gemm_forward | Torch | candidate | Torch BF16 matmul/operator routes can provide an implementation-backed comparison point | benchmark exact GPT-2 shapes and only promote routes that can be called from the trainer path |
| gemm_forward | CuTeDSL | candidate | CuTeDSL can generate Blackwell GEMM or fused epilogue kernels if installed | add CuTeDSL parity tests before timing or trainer promotion |
| gemm_forward | Plain CUDA | fallback | plain CUDA is a correctness fallback for GEMM-scale work, not the expected performance winner | use only as a safety fallback unless focused timing proves otherwise |
| gemm_forward_fused_gelu | ThunderKittens 2.0 | candidate | native TK GEMM rows are benchmarked by GPT-2 shape | keep only shape wins that preserve the TinyStories smoke |
| gemm_forward_fused_gelu | cuBLAS | candidate | cuBLAS BF16 GEMM is a direct baseline; fused rows use cuBLAS plus explicit CUDA pointwise work | compare against cuBLASLt and TK for every required pass and shape |
| gemm_forward_fused_gelu | cuBLASLt | baseline | current SM120 GEMM baseline, including fused epilogue candidates | keep selector decisions shape-specific |
| gemm_forward_fused_gelu | cuDNN | not_applicable | cuDNN is not used as a GEMM provider in this project | none until a cuDNN GEMM-equivalent path is intentionally scoped |
| gemm_forward_fused_gelu | Triton | candidate | Triton can express GEMM or fused epilogue variants if installed | add Triton parity tests before timing or trainer promotion |
| gemm_forward_fused_gelu | Torch | candidate | Torch BF16 matmul/operator routes can provide an implementation-backed comparison point | benchmark exact GPT-2 shapes and only promote routes that can be called from the trainer path |
| gemm_forward_fused_gelu | CuTeDSL | candidate | CuTeDSL can generate Blackwell GEMM or fused epilogue kernels if installed | add CuTeDSL parity tests before timing or trainer promotion |
| gemm_forward_fused_gelu | Plain CUDA | fallback | plain CUDA is a correctness fallback for GEMM-scale work, not the expected performance winner | use only as a safety fallback unless focused timing proves otherwise |
| gemm_backward_dinput | ThunderKittens 2.0 | candidate | native TK GEMM rows are benchmarked by GPT-2 shape | keep only shape wins that preserve the TinyStories smoke |
| gemm_backward_dinput | cuBLAS | candidate | cuBLAS BF16 GEMM is a direct baseline; fused rows use cuBLAS plus explicit CUDA pointwise work | compare against cuBLASLt and TK for every required pass and shape |
| gemm_backward_dinput | cuBLASLt | baseline | current SM120 GEMM baseline, including fused epilogue candidates | keep selector decisions shape-specific |
| gemm_backward_dinput | cuDNN | not_applicable | cuDNN is not used as a GEMM provider in this project | none until a cuDNN GEMM-equivalent path is intentionally scoped |
| gemm_backward_dinput | Triton | candidate | Triton can express GEMM or fused epilogue variants if installed | add Triton parity tests before timing or trainer promotion |
| gemm_backward_dinput | Torch | candidate | Torch BF16 matmul/operator routes can provide an implementation-backed comparison point | benchmark exact GPT-2 shapes and only promote routes that can be called from the trainer path |
| gemm_backward_dinput | CuTeDSL | candidate | CuTeDSL can generate Blackwell GEMM or fused epilogue kernels if installed | add CuTeDSL parity tests before timing or trainer promotion |
| gemm_backward_dinput | Plain CUDA | fallback | plain CUDA is a correctness fallback for GEMM-scale work, not the expected performance winner | use only as a safety fallback unless focused timing proves otherwise |
| gemm_backward_dinput_fused_dgelu | ThunderKittens 2.0 | candidate | native TK GEMM rows are benchmarked by GPT-2 shape | keep only shape wins that preserve the TinyStories smoke |
| gemm_backward_dinput_fused_dgelu | cuBLAS | candidate | cuBLAS BF16 GEMM is a direct baseline; fused rows use cuBLAS plus explicit CUDA pointwise work | compare against cuBLASLt and TK for every required pass and shape |
| gemm_backward_dinput_fused_dgelu | cuBLASLt | baseline | current SM120 GEMM baseline, including fused epilogue candidates | keep selector decisions shape-specific |
| gemm_backward_dinput_fused_dgelu | cuDNN | not_applicable | cuDNN is not used as a GEMM provider in this project | none until a cuDNN GEMM-equivalent path is intentionally scoped |
| gemm_backward_dinput_fused_dgelu | Triton | candidate | Triton can express GEMM or fused epilogue variants if installed | add Triton parity tests before timing or trainer promotion |
| gemm_backward_dinput_fused_dgelu | Torch | candidate | Torch BF16 matmul/operator routes can provide an implementation-backed comparison point | benchmark exact GPT-2 shapes and only promote routes that can be called from the trainer path |
| gemm_backward_dinput_fused_dgelu | CuTeDSL | candidate | CuTeDSL can generate Blackwell GEMM or fused epilogue kernels if installed | add CuTeDSL parity tests before timing or trainer promotion |
| gemm_backward_dinput_fused_dgelu | Plain CUDA | fallback | plain CUDA is a correctness fallback for GEMM-scale work, not the expected performance winner | use only as a safety fallback unless focused timing proves otherwise |
| gemm_backward_dweight | ThunderKittens 2.0 | candidate | native TK GEMM rows are benchmarked by GPT-2 shape | keep only shape wins that preserve the TinyStories smoke |
| gemm_backward_dweight | cuBLAS | candidate | cuBLAS BF16 GEMM is a direct baseline; fused rows use cuBLAS plus explicit CUDA pointwise work | compare against cuBLASLt and TK for every required pass and shape |
| gemm_backward_dweight | cuBLASLt | baseline | current SM120 GEMM baseline, including fused epilogue candidates | keep selector decisions shape-specific |
| gemm_backward_dweight | cuDNN | not_applicable | cuDNN is not used as a GEMM provider in this project | none until a cuDNN GEMM-equivalent path is intentionally scoped |
| gemm_backward_dweight | Triton | candidate | Triton can express GEMM or fused epilogue variants if installed | add Triton parity tests before timing or trainer promotion |
| gemm_backward_dweight | Torch | candidate | Torch BF16 matmul/operator routes can provide an implementation-backed comparison point | benchmark exact GPT-2 shapes and only promote routes that can be called from the trainer path |
| gemm_backward_dweight | CuTeDSL | candidate | CuTeDSL can generate Blackwell GEMM or fused epilogue kernels if installed | add CuTeDSL parity tests before timing or trainer promotion |
| gemm_backward_dweight | Plain CUDA | fallback | plain CUDA is a correctness fallback for GEMM-scale work, not the expected performance winner | use only as a safety fallback unless focused timing proves otherwise |
| gemm_backward_dweight_accum | ThunderKittens 2.0 | candidate | native TK GEMM rows are benchmarked by GPT-2 shape | keep only shape wins that preserve the TinyStories smoke |
| gemm_backward_dweight_accum | cuBLAS | candidate | cuBLAS BF16 GEMM is a direct baseline; fused rows use cuBLAS plus explicit CUDA pointwise work | compare against cuBLASLt and TK for every required pass and shape |
| gemm_backward_dweight_accum | cuBLASLt | baseline | current SM120 GEMM baseline, including fused epilogue candidates | keep selector decisions shape-specific |
| gemm_backward_dweight_accum | cuDNN | not_applicable | cuDNN is not used as a GEMM provider in this project | none until a cuDNN GEMM-equivalent path is intentionally scoped |
| gemm_backward_dweight_accum | Triton | candidate | Triton can express GEMM or fused epilogue variants if installed | add Triton parity tests before timing or trainer promotion |
| gemm_backward_dweight_accum | Torch | candidate | Torch BF16 matmul/operator routes can provide an implementation-backed comparison point | benchmark exact GPT-2 shapes and only promote routes that can be called from the trainer path |
| gemm_backward_dweight_accum | CuTeDSL | candidate | CuTeDSL can generate Blackwell GEMM or fused epilogue kernels if installed | add CuTeDSL parity tests before timing or trainer promotion |
| gemm_backward_dweight_accum | Plain CUDA | fallback | plain CUDA is a correctness fallback for GEMM-scale work, not the expected performance winner | use only as a safety fallback unless focused timing proves otherwise |
| bias_add | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_add | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_add | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_add | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_add | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| bias_add | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| bias_add | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_add | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
| bias_gradient_reduce | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_gradient_reduce | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_gradient_reduce | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_gradient_reduce | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_gradient_reduce | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| bias_gradient_reduce | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| bias_gradient_reduce | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| bias_gradient_reduce | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
| gelu_forward | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_forward | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_forward | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_forward | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_forward | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| gelu_forward | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| gelu_forward | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_forward | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
| gelu_backward | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_backward | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_backward | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_backward | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_backward | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| gelu_backward | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| gelu_backward | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| gelu_backward | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
| attention_forward | ThunderKittens 2.0 | baseline | SM120 packed-QKV attention is the current focused benchmark path | compare only against stack-specific parity-tested alternatives |
| attention_forward | cuBLAS | not_applicable | cuBLAS is a GEMM library and does not implement causal attention | none |
| attention_forward | cuBLASLt | not_applicable | cuBLASLt is a GEMM library and does not implement causal attention | none |
| attention_forward | cuDNN | candidate | cuDNN headers/libraries are detected; GPT-2 BF16 attention shape support still needs benchmark proof | prototype as an opt-in attention benchmark before trainer promotion |
| attention_forward | Triton | candidate | Triton can express FlashAttention-style alternatives if installed | add parity checks against the TK packed-QKV reference |
| attention_forward | Torch | candidate | Torch scaled-dot-product attention can provide a backend comparison if the packed-QKV layout and saved-state needs are matched | add parity checks against the TK packed-QKV reference before timing |
| attention_forward | CuTeDSL | not_applicable | CuTeDSL is being considered for GEMM/codegen work here, not attention | none until an attention-specific CuTeDSL prototype is scoped |
| attention_forward | Plain CUDA | fallback | plain CUDA recompute attention remains a correctness fallback for unsupported shapes | time only as a fallback comparison, not a default candidate |
| attention_backward | ThunderKittens 2.0 | baseline | SM120 packed-QKV attention is the current focused benchmark path | compare only against stack-specific parity-tested alternatives |
| attention_backward | cuBLAS | not_applicable | cuBLAS is a GEMM library and does not implement causal attention | none |
| attention_backward | cuBLASLt | not_applicable | cuBLASLt is a GEMM library and does not implement causal attention | none |
| attention_backward | cuDNN | candidate | cuDNN headers/libraries are detected; GPT-2 BF16 attention shape support still needs benchmark proof | prototype as an opt-in attention benchmark before trainer promotion |
| attention_backward | Triton | candidate | Triton can express FlashAttention-style alternatives if installed | add parity checks against the TK packed-QKV reference |
| attention_backward | Torch | candidate | Torch scaled-dot-product attention can provide a backend comparison if the packed-QKV layout and saved-state needs are matched | add parity checks against the TK packed-QKV reference before timing |
| attention_backward | CuTeDSL | not_applicable | CuTeDSL is being considered for GEMM/codegen work here, not attention | none until an attention-specific CuTeDSL prototype is scoped |
| attention_backward | Plain CUDA | fallback | plain CUDA recompute attention remains a correctness fallback for unsupported shapes | time only as a fallback comparison, not a default candidate |
| layernorm_forward | ThunderKittens 2.0 | missing | the current TK LayerNorm wrapper is Hopper-only; SM120 routes LayerNorm through the CUDA baseline | port and parity-test an SM120 TK LayerNorm path before benchmarking it |
| layernorm_forward | cuBLAS | not_applicable | cuBLAS is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_forward | cuBLASLt | not_applicable | cuBLASLt is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_forward | cuDNN | not_applicable | cuDNN is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_forward | Triton | candidate | Triton is a practical candidate for normalization kernels if installed | add parity tests before timing |
| layernorm_forward | Torch | candidate | Torch LayerNorm kernels are useful comparison points; native rows do not expose saved mean/rstd | benchmark native and stats-producing variants separately before trainer promotion |
| layernorm_forward | CuTeDSL | not_applicable | CuTeDSL is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_forward | Plain CUDA | baseline | current SM120 LayerNorm benchmark baseline | keep as default until a focused benchmark and TinyStories smoke improve |
| layernorm_fused_residual_forward | ThunderKittens 2.0 | missing | the current TK LayerNorm wrapper is Hopper-only; SM120 routes LayerNorm through the CUDA baseline | port and parity-test an SM120 TK LayerNorm path before benchmarking it |
| layernorm_fused_residual_forward | cuBLAS | not_applicable | cuBLAS is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_fused_residual_forward | cuBLASLt | not_applicable | cuBLASLt is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_fused_residual_forward | cuDNN | not_applicable | cuDNN is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_fused_residual_forward | Triton | candidate | Triton is a practical candidate for normalization kernels if installed | add parity tests before timing |
| layernorm_fused_residual_forward | Torch | candidate | Torch LayerNorm kernels are useful comparison points; native rows do not expose saved mean/rstd | benchmark native and stats-producing variants separately before trainer promotion |
| layernorm_fused_residual_forward | CuTeDSL | not_applicable | CuTeDSL is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_fused_residual_forward | Plain CUDA | baseline | current SM120 LayerNorm benchmark baseline | keep as default until a focused benchmark and TinyStories smoke improve |
| layernorm_backward | ThunderKittens 2.0 | missing | the current TK LayerNorm wrapper is Hopper-only; SM120 routes LayerNorm through the CUDA baseline | port and parity-test an SM120 TK LayerNorm path before benchmarking it |
| layernorm_backward | cuBLAS | not_applicable | cuBLAS is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_backward | cuBLASLt | not_applicable | cuBLASLt is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_backward | cuDNN | not_applicable | cuDNN is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_backward | Triton | candidate | Triton is a practical candidate for normalization kernels if installed | add parity tests before timing |
| layernorm_backward | Torch | candidate | Torch LayerNorm kernels are useful comparison points; native rows do not expose saved mean/rstd | benchmark native and stats-producing variants separately before trainer promotion |
| layernorm_backward | CuTeDSL | not_applicable | CuTeDSL is not the scoped LayerNorm provider for this optimization round | none until a concrete LayerNorm implementation for this stack is scoped |
| layernorm_backward | Plain CUDA | baseline | current SM120 LayerNorm benchmark baseline | keep as default until a focused benchmark and TinyStories smoke improve |
| classifier_softmax_cross_entropy_dlogits | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| classifier_softmax_cross_entropy_dlogits | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| classifier_softmax_cross_entropy_dlogits | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| classifier_softmax_cross_entropy_dlogits | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| classifier_softmax_cross_entropy_dlogits | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| classifier_softmax_cross_entropy_dlogits | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| classifier_softmax_cross_entropy_dlogits | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| classifier_softmax_cross_entropy_dlogits | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
| adamw | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| adamw | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| adamw | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| adamw | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| adamw | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| adamw | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| adamw | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| adamw | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
| global_norm | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| global_norm | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| global_norm | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| global_norm | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| global_norm | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| global_norm | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| global_norm | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| global_norm | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
| encoder_forward | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| encoder_forward | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| encoder_forward | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| encoder_forward | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| encoder_forward | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| encoder_forward | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| encoder_forward | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| encoder_forward | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
| cuda_memset | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_memset | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_memset | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_memset | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_memset | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| cuda_memset | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| cuda_memset | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_memset | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
| cuda_copy_d2d | ThunderKittens 2.0 | not_applicable | ThunderKittens 2.0 is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_copy_d2d | cuBLAS | not_applicable | cuBLAS is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_copy_d2d | cuBLASLt | not_applicable | cuBLASLt is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_copy_d2d | cuDNN | not_applicable | cuDNN is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_copy_d2d | Triton | candidate | Triton can express selected pointwise or reduction fusions if installed | add stack-specific parity tests before timing |
| cuda_copy_d2d | Torch | candidate | Torch can provide operator-backed pointwise/reduction comparisons for selected runtime families | benchmark exact trainer shapes and account for composition overhead before promotion |
| cuda_copy_d2d | CuTeDSL | not_applicable | CuTeDSL is not a reasonable provider for this pointwise/reduction/runtime family | none |
| cuda_copy_d2d | Plain CUDA | baseline | current SM120 runtime-family baseline | keep until a fused candidate improves focused timing and trainer smoke |
