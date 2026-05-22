# Current SM120 Backend Selection

- native selection round: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522`
- optional-stack comparison round: `scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522`
- native training manifest: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/round-manifest.json`
- native training log: `scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522/train_gpt2cu.log`
- native selected rows: `43`
- extra native benchmark-only selections: `3`
- inactive native microbench selections: `6`
- optional non-trainer selected rows: `11`
- optional decision rows: `18`
- project-wide fastest rows: `50`
- project-wide Torch fastest rows: `9`
- project-wide Torch fastest rows partition: `0` trainer-used, `6` resolved, `3` extra
- project-wide Torch disposition rows with action/reason: `9`/`9`
- project-wide trainer-callable fastest rows: `40`
- project-wide fastest rows used by trainer: `33`
- project-wide fastest rows resolved away from trainer: `10`
- project-wide extra benchmark rows: `7`
- active promotion candidates: `0`

Use a native round with TinyStories training evidence as the current trainer backend mix. For each exact objective row, compare the optional-stack round against that native trainer row and publish the faster current observed row as the project-wide fastest selection, including Torch only where it still beats the current native evidence. Optional rows remain operator/reference evidence, or rejected trainer-callable microbench wins, unless a refreshed integration exposes a trainer call path and passes correctness plus TinyStories smoke gates. Every selected optional row without a trainer call path must have a matching inactive promotion decision before this artifact can be generated.

## Native Trainer Mix

| Stack          | Selected rows |
| -------------- | ------------- |
| CUDA           | 15            |
| CUDA kernel    | 3             |
| CUDA runtime   | 2             |
| TK packed-QKV  | 2             |
| cuBLAS         | 10            |
| cuBLASLt       | 10            |
| cuBLASLt fused | 1             |

## Optional-Stack Decisions

| Status                                | Rows |
| ------------------------------------- | ---- |
| benchmark_context_flip                | 1    |
| contract_mismatch                     | 1    |
| layout_rewrite_only                   | 2    |
| noise_floor_microbench_flip           | 1    |
| non_trainer_shape                     | 1    |
| partial_backward_only                 | 1    |
| profiler_only_runtime_row             | 3    |
| rejected_same_session_refresh         | 1    |
| rejected_slower_than_trainer_baseline | 1    |
| rejected_trainer_smoke                | 1    |
| rejected_x10_selector                 | 3    |
| rejected_x10_trainer_route            | 2    |

## Fastest Rows Not Used By Trainer

| Call path                       | Rows |
| ------------------------------- | ---- |
| libtorch_raw_pointer_prototype  | 2    |
| operator_or_reference_prototype | 2    |
| profiler_runtime_benchmark_only | 3    |
| trainer_or_cxx_route            | 3    |

### Decision Statuses

| Decision                   | Rows |
| -------------------------- | ---- |
| layout_rewrite_only        | 2    |
| profiler_only_runtime_row  | 3    |
| rejected_x10_selector      | 3    |
| rejected_x10_trainer_route | 2    |

- trainer/C++ callable resolved rows with stability evidence: `3`/`3`
- resolved rows linked to decision table: `10`/`10`
- non-trainer resolved rows with action metadata: `6`/`6`

## Project-Wide Torch Fastest Rows

| Suite     | Kernel                  | Shape                                   | Selected stack | Time (us) | Scope                               | Call path                       |
| --------- | ----------------------- | --------------------------------------- | -------------- | --------- | ----------------------------------- | ------------------------------- |
| attention | backward                | `B=64 T=1024 C=768 NH=12 HS=64`         | Torch          | 2160.624  | python separated-Q/K/V              | operator_or_reference_prototype |
| attention | forward                 | `B=64 T=1024 C=768 NH=12 HS=64`         | Torch          | 556.565   | python separated-Q/K/V              | operator_or_reference_prototype |
| layernorm | backward_dinput         | `N=65536 C=768`                         | Torch native   | 216.416   | partial backward prototype          | operator_or_reference_prototype |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state` | Torch          | 7284.800  | operator prototype                  | operator_or_reference_prototype |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`            | Torch          | 1198.336  | non-equivalent BF16-state reference | operator_or_reference_prototype |
| runtime   | cuda_copy_d2d           | `logits_elems=3296722944`               | Torch C++      | 8633.024  | C++ API prototype                   | profiler_runtime_benchmark_only |
| runtime   | cuda_memset             | `grad_elems=124475904`                  | Torch C++      | 148.206   | C++ API prototype                   | libtorch_raw_pointer_prototype  |
| runtime   | cuda_memset             | `hidden_elems=50331648`                 | Torch C++      | 59.861    | C++ API prototype                   | libtorch_raw_pointer_prototype  |
| runtime   | cuda_memset             | `logits_elems=3296722944`               | Torch C++      | 3911.808  | C++ API prototype                   | profiler_runtime_benchmark_only |

## Project-Wide Torch Fastest Row Disposition

| Suite     | Kernel                  | Shape                                   | Selected stack | Disposition     | Class/Reason                                                                                           | Action/Gate                                                                                                                  |
| --------- | ----------------------- | --------------------------------------- | -------------- | --------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| attention | backward                | `B=64 T=1024 C=768 NH=12 HS=64`         | Torch          | resolved_away   | layout rewrite                                                                                         | refresh same-session baseline, add a packed-QKV or separated-Q/K/V trainer path, then compare against TK and run TinyStories |
| attention | forward                 | `B=64 T=1024 C=768 NH=12 HS=64`         | Torch          | resolved_away   | layout rewrite                                                                                         | refresh same-session baseline, add a packed-QKV or separated-Q/K/V trainer path, then compare against TK and run TinyStories |
| layernorm | backward_dinput         | `N=65536 C=768`                         | Torch native   | extra_benchmark | partial backward decomposition row; not the full trainer LayerNorm backward contract                   | partial backward decomposition row; not the full trainer LayerNorm backward contract                                         |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state` | Torch          | extra_benchmark | optimizer contract variant; current trainer objective row is no-master AdamW without this shape suffix | optimizer contract variant; current trainer objective row is no-master AdamW without this shape suffix                       |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`            | Torch          | extra_benchmark | non-equivalent BF16-state optimizer reference; current trainer objective uses FP32 moment buffers      | non-equivalent BF16-state optimizer reference; current trainer objective uses FP32 moment buffers                            |
| runtime   | cuda_copy_d2d           | `logits_elems=3296722944`               | Torch C++      | resolved_away   | profiler-only runtime row                                                                              | none; profiler-only runtime row with no current trainer call-site                                                            |
| runtime   | cuda_memset             | `grad_elems=124475904`                  | Torch C++      | resolved_away   | trainer route rejected                                                                                 | none; opt-in trainer route passed but x10 stability rejected promotion                                                       |
| runtime   | cuda_memset             | `hidden_elems=50331648`                 | Torch C++      | resolved_away   | trainer route rejected                                                                                 | none; opt-in trainer route passed but x10 stability rejected promotion                                                       |
| runtime   | cuda_memset             | `logits_elems=3296722944`               | Torch C++      | resolved_away   | profiler-only runtime row                                                                              | none; profiler-only runtime row with no current trainer call-site                                                            |

## Project-Wide Fastest Rows

| Suite     | Kernel                  | Shape                                        | Selected stack     | Time (us) | Scope                               | Call path                       |
| --------- | ----------------------- | -------------------------------------------- | ------------------ | --------- | ----------------------------------- | ------------------------------- |
| attention | backward                | `B=64 T=1024 C=768 NH=12 HS=64`              | Torch              | 2160.624  | python separated-Q/K/V              | operator_or_reference_prototype |
| attention | forward                 | `B=64 T=1024 C=768 NH=12 HS=64`              | Torch              | 556.565   | python separated-Q/K/V              | operator_or_reference_prototype |
| layernorm | backward                | `N=65536 C=3072`                             | CUDA               | 1095.961  | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | backward                | `N=65536 C=768`                              | CUDA               | 265.737   | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | backward_dinput         | `N=65536 C=3072`                             | Triton dInput-only | 799.040   | partial backward prototype          | operator_or_reference_prototype |
| layernorm | backward_dinput         | `N=65536 C=768`                              | Torch native       | 216.416   | partial backward prototype          | operator_or_reference_prototype |
| layernorm | forward                 | `N=65536 C=3072`                             | CUDA               | 537.587   | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | forward                 | `N=65536 C=768`                              | CUDA               | 135.130   | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | fused_residual_forward  | `N=65536 C=3072`                             | CUDA               | 1072.468  | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | fused_residual_forward  | `N=65536 C=768`                              | CUDA               | 271.036   | CUDA benchmark route                | trainer_or_cxx_route            |
| matmul    | dInp                    | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS             | 365.890   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp                    | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS             | 1328.430  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp                    | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS             | 1375.980  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp                    | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS             | 21018.670 | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp                    | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt           | 1011.900  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp+dGeLU              | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | TK                 | 1781.450  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS             | 326.890   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS             | 1309.130  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS             | 1309.540  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt           | 20689.360 | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS             | 993.410   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS             | 335.010   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS             | 1309.480  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS             | 1315.310  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt           | 20706.130 | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS             | 997.620   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd                     | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLASLt           | 369.450   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd                     | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLASLt           | 1343.640  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd                     | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt           | 22073.870 | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd                     | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt           | 1041.490  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd+gelu                | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLASLt           | 1471.710  | C++ benchmark route                 | trainer_or_cxx_route            |
| runtime   | adamw_update            | `params=124475904 no-master`                 | CUDA               | 1783.488  | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state`      | Torch              | 7284.800  | operator prototype                  | operator_or_reference_prototype |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`                 | Torch              | 1198.336  | non-equivalent BF16-state reference | operator_or_reference_prototype |
| runtime   | bias_add                | `BT=65536 OC=3072`                           | CUDA               | 528.467   | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | bias_add                | `BT=65536 OC=768`                            | CUDA               | 67.964    | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce        | `BT=65536 OC=2304`                           | CUDA               | 186.488   | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce        | `BT=65536 OC=3072`                           | CUDA               | 244.925   | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce        | `BT=65536 OC=768`                            | CUDA               | 24.514    | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | cuda_copy_d2d           | `hidden_elems=50331648`                      | CUDA runtime       | 131.588   | CUDA benchmark route                | profiler_runtime_benchmark_only |
| runtime   | cuda_copy_d2d           | `logits_elems=3296722944`                    | Torch C++          | 8633.024  | C++ API prototype                   | profiler_runtime_benchmark_only |
| runtime   | cuda_memset             | `grad_elems=124475904`                       | Torch C++          | 148.206   | C++ API prototype                   | libtorch_raw_pointer_prototype  |
| runtime   | cuda_memset             | `hidden_elems=50331648`                      | Torch C++          | 59.861    | C++ API prototype                   | libtorch_raw_pointer_prototype  |
| runtime   | cuda_memset             | `logits_elems=3296722944`                    | Torch C++          | 3911.808  | C++ API prototype                   | profiler_runtime_benchmark_only |
| runtime   | encoder_forward         | `B=64 T=1024 C=768`                          | CUDA               | 80.172    | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | fused_classifier        | `B=64 T=1024 V=50257 P=50304`                | CUDA               | 8749.869  | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | fused_classifier_loss   | `B=64 T=1024 V=50257 P=50304`                | CUDA               | 3893.421  | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | gelu_backward_inplace   | `BT=65536 C=3072`                            | CUDA               | 770.103   | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | gelu_forward            | `BT=65536 C=3072`                            | CUDA               | 527.468   | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | global_norm_squared     | `params=124475904`                           | CUDA               | 185.014   | CUDA benchmark route                | trainer_or_cxx_route            |

## Project-Wide Fastest Rows Used By Trainer

| Suite     | Kernel                 | Shape                                        | Selected stack | Time (us) | Scope                | Call path            |
| --------- | ---------------------- | -------------------------------------------- | -------------- | --------- | -------------------- | -------------------- |
| layernorm | backward               | `N=65536 C=768`                              | CUDA           | 265.737   | CUDA benchmark route | trainer_or_cxx_route |
| layernorm | forward                | `N=65536 C=768`                              | CUDA           | 135.130   | CUDA benchmark route | trainer_or_cxx_route |
| layernorm | fused_residual_forward | `N=65536 C=768`                              | CUDA           | 271.036   | CUDA benchmark route | trainer_or_cxx_route |
| matmul    | dInp                   | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1375.980  | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dInp                   | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS         | 21018.670 | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dInp                   | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt       | 1011.900  | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW                     | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS         | 326.890   | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW                     | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS         | 1309.130  | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW                     | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1309.540  | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW                     | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 20689.360 | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW                     | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS         | 993.410   | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW+accum               | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS         | 335.010   | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW+accum               | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS         | 1309.480  | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW+accum               | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1315.310  | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW+accum               | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 20706.130 | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | dW+accum               | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS         | 997.620   | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | fwd                    | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLASLt       | 369.450   | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | fwd                    | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLASLt       | 1343.640  | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | fwd                    | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 22073.870 | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | fwd                    | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt       | 1041.490  | C++ benchmark route  | trainer_or_cxx_route |
| matmul    | fwd+gelu               | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLASLt       | 1471.710  | C++ benchmark route  | trainer_or_cxx_route |
| runtime   | adamw_update           | `params=124475904 no-master`                 | CUDA           | 1783.488  | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | bias_add               | `BT=65536 OC=3072`                           | CUDA           | 528.467   | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | bias_add               | `BT=65536 OC=768`                            | CUDA           | 67.964    | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | bias_grad_reduce       | `BT=65536 OC=2304`                           | CUDA           | 186.488   | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | bias_grad_reduce       | `BT=65536 OC=3072`                           | CUDA           | 244.925   | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | bias_grad_reduce       | `BT=65536 OC=768`                            | CUDA           | 24.514    | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | encoder_forward        | `B=64 T=1024 C=768`                          | CUDA           | 80.172    | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | fused_classifier       | `B=64 T=1024 V=50257 P=50304`                | CUDA           | 8749.869  | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | fused_classifier_loss  | `B=64 T=1024 V=50257 P=50304`                | CUDA           | 3893.421  | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | gelu_backward_inplace  | `BT=65536 C=3072`                            | CUDA           | 770.103   | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | gelu_forward           | `BT=65536 C=3072`                            | CUDA           | 527.468   | CUDA benchmark route | trainer_or_cxx_route |
| runtime   | global_norm_squared    | `params=124475904`                           | CUDA           | 185.014   | CUDA benchmark route | trainer_or_cxx_route |

## Project-Wide Fastest Rows Resolved Away From Trainer

| Suite     | Kernel        | Shape                                       | Selected stack | Time (us) | Call path                       | Decision                   | Reason                                                                                                                                                                                                                                                                                                                                   | Evidence                                                                                                                                                                                             |
| --------- | ------------- | ------------------------------------------- | -------------- | --------- | ------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| attention | backward      | `B=64 T=1024 C=768 NH=12 HS=64`             | Torch          | 2160.624  | operator_or_reference_prototype | layout_rewrite_only        | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK.                                                                                                                                                                                                                           | 2026-05-20 target-context refresh: packed TK backward 2716.901 us; TorchPacked backward 4107.011 us; native Torch separated backward 2227.869 us (+1 more)                                           |
| attention | forward       | `B=64 T=1024 C=768 NH=12 HS=64`             | Torch          | 556.565   | operator_or_reference_prototype | layout_rewrite_only        | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK.                                                                                                                                                                                                                           | 2026-05-20 target-context refresh: packed TK forward 787.859 us; TorchPacked forward 1120.509 us; native Torch separated forward 570.848 us (+1 more)                                                |
| matmul    | dInp          | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS         | 365.890   | trainer_or_cxx_route            | rejected_x10_selector      | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection. Both the bundled attproj/MLP-up selector and the later attproj-only selector regressed in x10 TinyStories stability gates, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route.                                             | 2026-05-21 candidate selector round codex_sm120_round_cublas_dinp_attproj_fc_20260521 validated at avg_ms=2493.931 (+2 more)                                                                         |
| matmul    | dInp          | `fc M=65536 N=3072 K=768 bias=1 gelu=1`     | cuBLAS         | 1328.430  | trainer_or_cxx_route            | rejected_x10_selector      | Do not promote the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but both the broader attproj+MLP-up selector and the later FC-only selector regressed in x10 TinyStories stability gates, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. | 2026-05-21 optional stack refresh selected cuBLAS 1328.470 us versus cuBLASLt 1367.640 us for MLP-up dInput (+3 more)                                                                                |
| matmul    | dInp+dGeLU    | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK             | 1781.450  | trainer_or_cxx_route            | rejected_x10_selector      | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default.                                                                                                    | 2026-05-20 target-context test_matmul with LLMK_SM120_USE_TK_FUSED_DGELU_DINP failed only the GPT-2 fcproj fused dGELU dInput row with max abs diff 0.500000 against the strict <0.50 gate (+4 more) |
| runtime   | cuda_copy_d2d | `hidden_elems=50331648`                     | CUDA runtime   | 131.588   | profiler_runtime_benchmark_only | profiler_only_runtime_row  | Keep as benchmark evidence only. The refreshed optional round measured CUDA runtime as fastest for this profiler-only copy shape, but it is not a current trainer call path to promote.                                                                                                                                                  | 2026-05-21 refreshed optional runtime log: CUDA runtime 131.673 us, Torch C++ 131.781 us, CUDA kernel 141.964 us for hidden_elems=50331648 (+1 more)                                                 |
| runtime   | cuda_copy_d2d | `logits_elems=3296722944`                   | Torch C++      | 8633.024  | profiler_runtime_benchmark_only | profiler_only_runtime_row  | Keep as profiler/runtime evidence only. The refreshed LibTorch row is the fastest observed logits-copy row, but the current GPT-2 trainer has no logits-sized device-to-device copy call-site to promote.                                                                                                                                | 2026-05-21 raw-pointer LibTorch refresh: cached from_blob full-row parity PASS, CUDA runtime 8799.168 us, native CUDA kernel 8881.951 us, Torch C++ raw 8686.496 us (+2 more)                        |
| runtime   | cuda_memset   | `grad_elems=124475904`                      | Torch C++      | 148.206   | libtorch_raw_pointer_prototype  | rejected_x10_trainer_route | Do not promote the LibTorch gradients-zero trainer route by default. The opt-in C++ call-site now exists and passes correctness plus TinyStories smoke, but its x10 stability round regressed versus the current native x10 trainer selection.                                                                                           | 2026-05-21 refreshed native/optional runtime logs: CUDA runtime 148.470 us, Python Torch 148.104 us, Torch C++ 147.749 us for grad_elems=124475904 (+7 more)                                         |
| runtime   | cuda_memset   | `hidden_elems=50331648`                     | Torch C++      | 59.861    | libtorch_raw_pointer_prototype  | rejected_x10_trainer_route | Do not promote the LibTorch dresidual-zero trainer route by default. The C++ API feasibility row was tie-range, and the integrated trainer route regressed in the x10 TinyStories gate.                                                                                                                                                  | 2026-05-21 raw-pointer LibTorch refresh: cached from_blob full-row parity PASS, CUDA runtime 60.826 us, native CUDA kernel 60.056 us, Torch C++ raw 60.011 us (+4 more)                              |
| runtime   | cuda_memset   | `logits_elems=3296722944`                   | Torch C++      | 3911.808  | profiler_runtime_benchmark_only | profiler_only_runtime_row  | Keep as profiler/runtime evidence only. The current GPT-2 trainer does not issue a logits-sized memset; this row measures large-buffer runtime behavior rather than a promotable trainer call-site.                                                                                                                                      | 2026-05-21 raw-pointer LibTorch refresh: cached from_blob full-row parity PASS, CUDA runtime 3964.396 us, native CUDA kernel 4016.422 us, Torch C++ raw 3984.000 us (+2 more)                        |

## Extra Project-Wide Benchmark Rows

| Suite     | Kernel                  | Shape                                   | Selected stack     | Time (us) | Call path                       | Reason                                                                                                 |
| --------- | ----------------------- | --------------------------------------- | ------------------ | --------- | ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| layernorm | backward                | `N=65536 C=3072`                        | CUDA               | 1095.961  | trainer_or_cxx_route            | non-objective LayerNorm stress width; GPT-2 trainer LayerNorm uses C=768                               |
| layernorm | backward_dinput         | `N=65536 C=3072`                        | Triton dInput-only | 799.040   | operator_or_reference_prototype | non-objective LayerNorm stress width; GPT-2 trainer LayerNorm uses C=768                               |
| layernorm | backward_dinput         | `N=65536 C=768`                         | Torch native       | 216.416   | operator_or_reference_prototype | partial backward decomposition row; not the full trainer LayerNorm backward contract                   |
| layernorm | forward                 | `N=65536 C=3072`                        | CUDA               | 537.587   | trainer_or_cxx_route            | non-objective LayerNorm stress width; GPT-2 trainer LayerNorm uses C=768                               |
| layernorm | fused_residual_forward  | `N=65536 C=3072`                        | CUDA               | 1072.468  | trainer_or_cxx_route            | non-objective LayerNorm stress width; GPT-2 trainer LayerNorm uses C=768                               |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state` | Torch              | 7284.800  | operator_or_reference_prototype | optimizer contract variant; current trainer objective row is no-master AdamW without this shape suffix |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`            | Torch              | 1198.336  | operator_or_reference_prototype | non-equivalent BF16-state optimizer reference; current trainer objective uses FP32 moment buffers      |

## Resolved Optional-Stack Decisions

| Suite     | Kernel                  | Shape                                        | Selected stack     | Time (us) | Scope                               | Call path                       | Class                      | Gate                                                                                                                          | Decision                              |
| --------- | ----------------------- | -------------------------------------------- | ------------------ | --------- | ----------------------------------- | ------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| attention | backward                | `B=64 T=1024 C=768 NH=12 HS=64`              | Torch              | 2160.624  | python separated-Q/K/V              | operator_or_reference_prototype | layout rewrite             | refresh same-session baseline, add a packed-QKV or separated-Q/K/V trainer path, then compare against TK and run TinyStories  | layout_rewrite_only                   |
| attention | forward                 | `B=64 T=1024 C=768 NH=12 HS=64`              | Torch              | 556.565   | python separated-Q/K/V              | operator_or_reference_prototype | layout rewrite             | refresh same-session baseline, add a packed-QKV or separated-Q/K/V trainer path, then compare against TK and run TinyStories  | layout_rewrite_only                   |
| layernorm | backward_dinput         | `N=65536 C=768`                              | Torch native       | 216.416   | partial backward prototype          | operator_or_reference_prototype | reference/state gap        | add dweight/dbias accumulation before considering trainer promotion                                                           | partial_backward_only                 |
| matmul    | dInp                    | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS             | 365.890   | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | rejected_x10_selector                 |
| matmul    | dInp                    | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS             | 1328.430  | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | rejected_x10_selector                 |
| matmul    | dInp                    | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLASLt           | 1380.780  | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | benchmark_context_flip                |
| matmul    | dInp                    | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS             | 1012.560  | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | noise_floor_microbench_flip           |
| matmul    | dInp+dGeLU              | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | TK                 | 1781.450  | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | rejected_x10_selector                 |
| matmul    | fwd                     | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS             | 22140.630 | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | rejected_trainer_smoke                |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state`      | Torch              | 7284.800  | operator prototype                  | operator_or_reference_prototype | library integration        | refresh same-session baseline, add an explicit libtorch/C++ or equivalent native route, then run route parity and TinyStories | rejected_slower_than_trainer_baseline |
| runtime   | cuda_copy_d2d           | `hidden_elems=50331648`                      | CUDA runtime       | 131.588   | CUDA benchmark route                | profiler_runtime_benchmark_only |                            |                                                                                                                               | profiler_only_runtime_row             |
| runtime   | cuda_copy_d2d           | `logits_elems=3296722944`                    | Torch C++          | 8633.024  | C++ API prototype                   | profiler_runtime_benchmark_only | profiler-only runtime row  | none; profiler-only runtime row with no current trainer call-site                                                             | profiler_only_runtime_row             |
| runtime   | cuda_memset             | `grad_elems=124475904`                       | Torch C++          | 148.206   | C++ API prototype                   | libtorch_raw_pointer_prototype  | trainer route rejected     | none; opt-in trainer route passed but x10 stability rejected promotion                                                        | rejected_x10_trainer_route            |
| runtime   | cuda_memset             | `hidden_elems=50331648`                      | Torch C++          | 59.861    | C++ API prototype                   | libtorch_raw_pointer_prototype  | trainer route rejected     | none; opt-in trainer route passed but x10 stability rejected promotion                                                        | rejected_x10_trainer_route            |
| runtime   | cuda_memset             | `logits_elems=3296722944`                    | Torch C++          | 3911.808  | C++ API prototype                   | profiler_runtime_benchmark_only | profiler-only runtime row  | none; profiler-only runtime row with no current trainer call-site                                                             | profiler_only_runtime_row             |
| runtime   | gelu_backward_inplace   | `BT=65536 C=3072`                            | Triton             | 770.482   | operator prototype                  | operator_or_reference_prototype | native/codegen integration | refresh same-session baseline, add a trainer-callable Triton/C++ route, then run route parity and TinyStories                 | rejected_same_session_refresh         |
| layernorm | backward_dinput         | `N=65536 C=3072`                             | Triton dInput-only | 799.040   | partial backward prototype          | operator_or_reference_prototype | non-trainer shape          | non-trainer GPT-2 LayerNorm shape; keep as operator evidence unless the trainer adds this shape                               | non_trainer_shape                     |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`                 | Torch              | 1198.336  | non-equivalent BF16-state reference | operator_or_reference_prototype | contract mismatch          | match the trainer optimizer state contract before considering promotion                                                       | contract_mismatch                     |

## Inactive Native Microbench Selections

| Suite   | Kernel        | Shape                                        | Rejected stack | Current stack  | Decision                  |
| ------- | ------------- | -------------------------------------------- | -------------- | -------------- | ------------------------- |
| matmul  | dInp          | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS         | cuBLASLt       | rejected_x10_selector     |
| matmul  | dInp          | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS         | cuBLASLt       | rejected_x10_selector     |
| matmul  | dInp+dGeLU    | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | TK             | cuBLASLt fused | rejected_x10_selector     |
| matmul  | fwd           | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS         | cuBLASLt       | rejected_trainer_smoke    |
| runtime | cuda_copy_d2d | `hidden_elems=50331648`                      | CUDA runtime   | CUDA kernel    | profiler_only_runtime_row |
| runtime | cuda_memset   | `logits_elems=3296722944`                    | CUDA runtime   | CUDA kernel    | profiler_only_runtime_row |

## Attention Route Totals

| Source   | Stack                   | Shape                           | Scope                           | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note                                                                  |
| -------- | ----------------------- | ------------------------------- | ------------------------------- | -------------- | ------------ | ------------- | ---------- | -------- | --------------------------------------------------------------------- |
| native   | TK packed-QKV           | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 776.120      | 2702.327      | 3478.447   | True     |                                                                       |
| optional | TK packed-QKV           | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 774.667      | 2706.639      | 3481.306   | True     |                                                                       |
| optional | cuDNNPacked             | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 790.320      | 2765.123      | 3555.443   | True     |                                                                       |
| optional | TorchPacked             | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 1142.946     | 4002.704      | 5145.650   | True     |                                                                       |
| optional | TorchMaterializedPacked | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 1247.822     | 4149.318      | 5397.140   | True     |                                                                       |
| optional | TritonPacked            | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 2186.347     | -             | -          | False    | packed attention backward is not implemented in this Triton prototype |
| optional | Torch                   | `B=64 T=1024 C=768 NH=12 HS=64` | separated Q/K/V reference route | False          | 556.565      | 2160.624      | 2717.189   | True     |                                                                       |
| optional | cuDNN                   | `B=64 T=1024 C=768 NH=12 HS=64` | separated Q/K/V reference route | False          | 675.282      | 2342.368      | 3017.650   | True     |                                                                       |
| optional | Triton                  | `B=64 T=1024 C=768 NH=12 HS=64` | separated Q/K/V reference route | False          | 2079.432     | -             | -          | False    | attention backward is not implemented in this Triton prototype        |

## Extra Native Benchmark Selections

| Suite     | Kernel                 | Shape            | Selected stack | Time (us) | Call path            |
| --------- | ---------------------- | ---------------- | -------------- | --------- | -------------------- |
| layernorm | backward               | `N=65536 C=3072` | CUDA           | 1091.596  | trainer_or_cxx_route |
| layernorm | forward                | `N=65536 C=3072` | CUDA           | 538.402   | trainer_or_cxx_route |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA           | 1071.240  | trainer_or_cxx_route |

## Trainer Selection Rows

| Suite     | Kernel                 | Shape                                        | Current stack  | Time (us) | Call path                       |
| --------- | ---------------------- | -------------------------------------------- | -------------- | --------- | ------------------------------- |
| attention | backward               | `B=64 T=1024 C=768 NH=12 HS=64`              | TK packed-QKV  | 2702.327  | trainer_or_cxx_route            |
| attention | forward                | `B=64 T=1024 C=768 NH=12 HS=64`              | TK packed-QKV  | 776.120   | trainer_or_cxx_route            |
| layernorm | backward               | `N=65536 C=768`                              | CUDA           | 265.737   | trainer_or_cxx_route            |
| layernorm | forward                | `N=65536 C=768`                              | CUDA           | 135.136   | trainer_or_cxx_route            |
| layernorm | fused_residual_forward | `N=65536 C=768`                              | CUDA           | 274.999   | trainer_or_cxx_route            |
| matmul    | dInp                   | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLASLt       | 366.960   | trainer_or_cxx_route            |
| matmul    | dInp                   | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLASLt       | 1352.170  | trainer_or_cxx_route            |
| matmul    | dInp                   | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1375.980  | trainer_or_cxx_route            |
| matmul    | dInp                   | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS         | 21043.990 | trainer_or_cxx_route            |
| matmul    | dInp                   | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt       | 1011.900  | trainer_or_cxx_route            |
| matmul    | dInp+dGeLU             | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLASLt fused | 1809.010  | trainer_or_cxx_route            |
| matmul    | dW                     | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS         | 326.890   | trainer_or_cxx_route            |
| matmul    | dW                     | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS         | 1313.150  | trainer_or_cxx_route            |
| matmul    | dW                     | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1309.540  | trainer_or_cxx_route            |
| matmul    | dW                     | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 20850.930 | trainer_or_cxx_route            |
| matmul    | dW                     | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS         | 993.410   | trainer_or_cxx_route            |
| matmul    | dW+accum               | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS         | 335.010   | trainer_or_cxx_route            |
| matmul    | dW+accum               | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS         | 1313.410  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1319.490  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 20706.130 | trainer_or_cxx_route            |
| matmul    | dW+accum               | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS         | 997.620   | trainer_or_cxx_route            |
| matmul    | fwd                    | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLASLt       | 369.450   | trainer_or_cxx_route            |
| matmul    | fwd                    | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLASLt       | 1384.760  | trainer_or_cxx_route            |
| matmul    | fwd                    | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 22073.870 | trainer_or_cxx_route            |
| matmul    | fwd                    | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt       | 1041.490  | trainer_or_cxx_route            |
| matmul    | fwd+gelu               | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLASLt       | 1474.560  | trainer_or_cxx_route            |
| runtime   | adamw_update           | `params=124475904 no-master`                 | CUDA           | 1783.488  | trainer_or_cxx_route            |
| runtime   | bias_add               | `BT=65536 OC=3072`                           | CUDA           | 537.901   | trainer_or_cxx_route            |
| runtime   | bias_add               | `BT=65536 OC=768`                            | CUDA           | 91.817    | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce       | `BT=65536 OC=2304`                           | CUDA           | 186.765   | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce       | `BT=65536 OC=3072`                           | CUDA           | 245.147   | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce       | `BT=65536 OC=768`                            | CUDA           | 25.413    | trainer_or_cxx_route            |
| runtime   | cuda_copy_d2d          | `hidden_elems=50331648`                      | CUDA kernel    | 132.475   | profiler_runtime_benchmark_only |
| runtime   | cuda_copy_d2d          | `logits_elems=3296722944`                    | CUDA runtime   | 8698.899  | profiler_runtime_benchmark_only |
| runtime   | cuda_memset            | `grad_elems=124475904`                       | CUDA runtime   | 149.947   | trainer_or_cxx_route            |
| runtime   | cuda_memset            | `hidden_elems=50331648`                      | CUDA kernel    | 60.666    | trainer_or_cxx_route            |
| runtime   | cuda_memset            | `logits_elems=3296722944`                    | CUDA kernel    | 3929.050  | profiler_runtime_benchmark_only |
| runtime   | encoder_forward        | `B=64 T=1024 C=768`                          | CUDA           | 83.692    | trainer_or_cxx_route            |
| runtime   | fused_classifier       | `B=64 T=1024 V=50257 P=50304`                | CUDA           | 8749.869  | trainer_or_cxx_route            |
| runtime   | fused_classifier_loss  | `B=64 T=1024 V=50257 P=50304`                | CUDA           | 3893.421  | trainer_or_cxx_route            |
| runtime   | gelu_backward_inplace  | `BT=65536 C=3072`                            | CUDA           | 770.103   | trainer_or_cxx_route            |
| runtime   | gelu_forward           | `BT=65536 C=3072`                            | CUDA           | 527.468   | trainer_or_cxx_route            |
| runtime   | global_norm_squared    | `params=124475904`                           | CUDA           | 185.014   | trainer_or_cxx_route            |
