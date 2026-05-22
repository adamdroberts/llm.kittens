# Current SM120 Backend Selection

- native selection round: `scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521`
- optional-stack comparison round: `scratch/sm120_rounds/codex_sm120_round_optional_refresh_20260521`
- native training manifest: `scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/round-manifest.json`
- native training log: `scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_default_x10_20260521/train_gpt2cu.log`
- native selected rows: `42`
- extra native benchmark-only selections: `3`
- inactive native microbench selections: `5`
- optional non-trainer selected rows: `11`
- optional decision rows: `15`
- project-wide fastest rows: `49`
- project-wide Torch fastest rows: `8`
- project-wide Torch fastest rows partition: `0` trainer-used, `5` resolved, `3` extra
- project-wide Torch disposition rows with action/reason: `8`/`8`
- project-wide trainer-callable fastest rows: `38`
- project-wide fastest rows used by trainer: `31`
- project-wide fastest rows resolved away from trainer: `11`
- project-wide extra benchmark rows: `7`
- active promotion candidates: `0`

Use a native round with TinyStories training evidence as the current trainer backend mix. Use the optional-stack round as the project-wide fastest observed row set, including Torch wherever it wins an exact benchmark situation. Optional rows remain operator/reference evidence, or rejected trainer-callable microbench wins, unless a refreshed integration exposes a trainer call path and passes correctness plus TinyStories smoke gates. Every selected optional row without a trainer call path must have a matching inactive promotion decision before this artifact can be generated.

## Native Trainer Mix

| Stack          | Selected rows |
| -------------- | ------------- |
| CUDA           | 15            |
| CUDA runtime   | 4             |
| TK packed-QKV  | 2             |
| cuBLAS         | 10            |
| cuBLASLt       | 10            |
| cuBLASLt fused | 1             |

## Optional-Stack Decisions

| Status                                | Rows |
| ------------------------------------- | ---- |
| contract_mismatch                     | 1    |
| layout_rewrite_only                   | 2    |
| library_integration_not_justified     | 2    |
| native_replacement_rejected           | 2    |
| noise_floor_microbench_flip           | 1    |
| non_trainer_shape                     | 1    |
| partial_backward_only                 | 1    |
| rejected_same_session_refresh         | 1    |
| rejected_slower_than_trainer_baseline | 1    |
| rejected_x10_selector                 | 3    |

## Fastest Rows Not Used By Trainer

| Call path                       | Rows |
| ------------------------------- | ---- |
| libtorch_raw_pointer_prototype  | 1    |
| operator_or_reference_prototype | 5    |
| profiler_runtime_benchmark_only | 1    |
| trainer_or_cxx_route            | 4    |

### Decision Statuses

| Decision                          | Rows |
| --------------------------------- | ---- |
| layout_rewrite_only               | 2    |
| library_integration_not_justified | 2    |
| native_replacement_rejected       | 2    |
| noise_floor_microbench_flip       | 1    |
| rejected_same_session_refresh     | 1    |
| rejected_x10_selector             | 3    |

- trainer/C++ callable resolved rows with stability evidence: `4`/`4`
- resolved rows linked to decision table: `11`/`11`
- non-trainer resolved rows with action metadata: `7`/`7`

## Project-Wide Torch Fastest Rows

| Suite     | Kernel                  | Shape                                   | Selected stack | Time (us) | Scope                               | Call path                       |
| --------- | ----------------------- | --------------------------------------- | -------------- | --------- | ----------------------------------- | ------------------------------- |
| attention | backward                | `B=64 T=1024 C=768 NH=12 HS=64`         | Torch          | 2196.240  | python separated-Q/K/V              | operator_or_reference_prototype |
| attention | forward                 | `B=64 T=1024 C=768 NH=12 HS=64`         | Torch          | 556.573   | python separated-Q/K/V              | operator_or_reference_prototype |
| layernorm | backward_dinput         | `N=65536 C=768`                         | Torch native   | 214.592   | partial backward prototype          | operator_or_reference_prototype |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state` | Torch          | 7298.912  | operator prototype                  | operator_or_reference_prototype |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`            | Torch          | 1208.083  | non-equivalent BF16-state reference | operator_or_reference_prototype |
| runtime   | cuda_copy_d2d           | `logits_elems=3296722944`               | Torch          | 8662.848  | operator prototype                  | profiler_runtime_benchmark_only |
| runtime   | cuda_memset             | `hidden_elems=50331648`                 | Torch C++      | 59.875    | C++ API prototype                   | libtorch_raw_pointer_prototype  |
| runtime   | cuda_memset             | `logits_elems=3296722944`               | Torch          | 3943.104  | operator prototype                  | operator_or_reference_prototype |

## Project-Wide Torch Fastest Row Disposition

| Suite     | Kernel                  | Shape                                   | Selected stack | Disposition     | Class/Reason                                                                                           | Action/Gate                                                                                                                   |
| --------- | ----------------------- | --------------------------------------- | -------------- | --------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| attention | backward                | `B=64 T=1024 C=768 NH=12 HS=64`         | Torch          | resolved_away   | layout rewrite                                                                                         | refresh same-session baseline, add a packed-QKV or separated-Q/K/V trainer path, then compare against TK and run TinyStories  |
| attention | forward                 | `B=64 T=1024 C=768 NH=12 HS=64`         | Torch          | resolved_away   | layout rewrite                                                                                         | refresh same-session baseline, add a packed-QKV or separated-Q/K/V trainer path, then compare against TK and run TinyStories  |
| layernorm | backward_dinput         | `N=65536 C=768`                         | Torch native   | extra_benchmark | partial backward decomposition row; not the full trainer LayerNorm backward contract                   | partial backward decomposition row; not the full trainer LayerNorm backward contract                                          |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state` | Torch          | extra_benchmark | optimizer contract variant; current trainer objective row is no-master AdamW without this shape suffix | optimizer contract variant; current trainer objective row is no-master AdamW without this shape suffix                        |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`            | Torch          | extra_benchmark | non-equivalent BF16-state optimizer reference; current trainer objective uses FP32 moment buffers      | non-equivalent BF16-state optimizer reference; current trainer objective uses FP32 moment buffers                             |
| runtime   | cuda_copy_d2d           | `logits_elems=3296722944`               | Torch          | resolved_away   | library integration                                                                                    | refresh same-session baseline, add an explicit libtorch/C++ or equivalent native route, then run route parity and TinyStories |
| runtime   | cuda_memset             | `hidden_elems=50331648`                 | Torch C++      | resolved_away   | library integration                                                                                    | add an explicit libtorch trainer link gate, route parity, and TinyStories smoke before promotion                              |
| runtime   | cuda_memset             | `logits_elems=3296722944`               | Torch          | resolved_away   | library integration                                                                                    | refresh same-session baseline, add an explicit libtorch/C++ or equivalent native route, then run route parity and TinyStories |

## Project-Wide Fastest Rows

| Suite     | Kernel                  | Shape                                        | Selected stack     | Time (us) | Scope                               | Call path                       |
| --------- | ----------------------- | -------------------------------------------- | ------------------ | --------- | ----------------------------------- | ------------------------------- |
| attention | backward                | `B=64 T=1024 C=768 NH=12 HS=64`              | Torch              | 2196.240  | python separated-Q/K/V              | operator_or_reference_prototype |
| attention | forward                 | `B=64 T=1024 C=768 NH=12 HS=64`              | Torch              | 556.573   | python separated-Q/K/V              | operator_or_reference_prototype |
| layernorm | backward                | `N=65536 C=3072`                             | CUDA               | 1272.855  | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | backward                | `N=65536 C=768`                              | CUDA               | 288.147   | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | backward_dinput         | `N=65536 C=3072`                             | Triton dInput-only | 801.056   | partial backward prototype          | operator_or_reference_prototype |
| layernorm | backward_dinput         | `N=65536 C=768`                              | Torch native       | 214.592   | partial backward prototype          | operator_or_reference_prototype |
| layernorm | forward                 | `N=65536 C=3072`                             | CUDA               | 544.301   | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | forward                 | `N=65536 C=768`                              | CUDA               | 138.406   | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | fused_residual_forward  | `N=65536 C=3072`                             | CUDA               | 1082.597  | CUDA benchmark route                | trainer_or_cxx_route            |
| layernorm | fused_residual_forward  | `N=65536 C=768`                              | CUDA               | 275.342   | CUDA benchmark route                | trainer_or_cxx_route            |
| matmul    | dInp                    | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS             | 365.930   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp                    | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS             | 1364.900  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp                    | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS             | 1380.010  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp                    | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS             | 21272.170 | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp                    | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS             | 1012.440  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dInp+dGeLU              | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | TK                 | 1822.440  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS             | 328.780   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS             | 1317.020  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS             | 1311.440  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt           | 20899.630 | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW                      | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS             | 997.280   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS             | 332.650   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS             | 1315.250  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS             | 1318.570  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt           | 20986.710 | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | dW+accum                | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS             | 998.520   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd                     | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLASLt           | 371.910   | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd                     | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLASLt           | 1379.880  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd                     | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt           | 22396.620 | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd                     | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt           | 1041.660  | C++ benchmark route                 | trainer_or_cxx_route            |
| matmul    | fwd+gelu                | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLASLt           | 1471.130  | C++ benchmark route                 | trainer_or_cxx_route            |
| runtime   | adamw_update            | `params=124475904 no-master`                 | CUDA               | 1809.206  | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state`      | Torch              | 7298.912  | operator prototype                  | operator_or_reference_prototype |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`                 | Torch              | 1208.083  | non-equivalent BF16-state reference | operator_or_reference_prototype |
| runtime   | bias_add                | `BT=65536 OC=3072`                           | Triton             | 542.899   | operator prototype                  | operator_or_reference_prototype |
| runtime   | bias_add                | `BT=65536 OC=768`                            | CUDA               | 80.025    | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce        | `BT=65536 OC=2304`                           | CUDA               | 186.802   | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce        | `BT=65536 OC=3072`                           | CUDA               | 245.528   | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce        | `BT=65536 OC=768`                            | CUDA               | 24.630    | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | cuda_copy_d2d           | `hidden_elems=50331648`                      | CUDA runtime       | 131.615   | CUDA benchmark route                | profiler_runtime_benchmark_only |
| runtime   | cuda_copy_d2d           | `logits_elems=3296722944`                    | Torch              | 8662.848  | operator prototype                  | profiler_runtime_benchmark_only |
| runtime   | cuda_memset             | `hidden_elems=50331648`                      | Torch C++          | 59.875    | C++ API prototype                   | libtorch_raw_pointer_prototype  |
| runtime   | cuda_memset             | `logits_elems=3296722944`                    | Torch              | 3943.104  | operator prototype                  | operator_or_reference_prototype |
| runtime   | encoder_forward         | `B=64 T=1024 C=768`                          | CUDA               | 79.286    | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | fused_classifier        | `B=64 T=1024 V=50257 P=50304`                | CUDA               | 8942.509  | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | fused_classifier_loss   | `B=64 T=1024 V=50257 P=50304`                | CUDA               | 3953.133  | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | gelu_backward_inplace   | `BT=65536 C=3072`                            | Triton             | 781.888   | operator prototype                  | operator_or_reference_prototype |
| runtime   | gelu_forward            | `BT=65536 C=3072`                            | CUDA               | 528.059   | CUDA benchmark route                | trainer_or_cxx_route            |
| runtime   | global_norm_squared     | `params=124475904`                           | CUDA               | 185.069   | CUDA benchmark route                | trainer_or_cxx_route            |

## Project-Wide Fastest Rows Used By Trainer

| Suite     | Kernel                 | Shape                                        | Selected stack | Time (us) | Scope                | Call path                       |
| --------- | ---------------------- | -------------------------------------------- | -------------- | --------- | -------------------- | ------------------------------- |
| layernorm | backward               | `N=65536 C=768`                              | CUDA           | 288.147   | CUDA benchmark route | trainer_or_cxx_route            |
| layernorm | forward                | `N=65536 C=768`                              | CUDA           | 138.406   | CUDA benchmark route | trainer_or_cxx_route            |
| layernorm | fused_residual_forward | `N=65536 C=768`                              | CUDA           | 275.342   | CUDA benchmark route | trainer_or_cxx_route            |
| matmul    | dInp                   | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1380.010  | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dInp                   | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS         | 21272.170 | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW                     | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS         | 328.780   | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW                     | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS         | 1317.020  | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW                     | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1311.440  | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW                     | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 20899.630 | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW                     | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS         | 997.280   | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS         | 332.650   | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS         | 1315.250  | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1318.570  | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 20986.710 | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS         | 998.520   | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | fwd                    | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLASLt       | 371.910   | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | fwd                    | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLASLt       | 1379.880  | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | fwd                    | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 22396.620 | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | fwd                    | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt       | 1041.660  | C++ benchmark route  | trainer_or_cxx_route            |
| matmul    | fwd+gelu               | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLASLt       | 1471.130  | C++ benchmark route  | trainer_or_cxx_route            |
| runtime   | adamw_update           | `params=124475904 no-master`                 | CUDA           | 1809.206  | CUDA benchmark route | trainer_or_cxx_route            |
| runtime   | bias_add               | `BT=65536 OC=768`                            | CUDA           | 80.025    | CUDA benchmark route | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce       | `BT=65536 OC=2304`                           | CUDA           | 186.802   | CUDA benchmark route | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce       | `BT=65536 OC=3072`                           | CUDA           | 245.528   | CUDA benchmark route | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce       | `BT=65536 OC=768`                            | CUDA           | 24.630    | CUDA benchmark route | trainer_or_cxx_route            |
| runtime   | cuda_copy_d2d          | `hidden_elems=50331648`                      | CUDA runtime   | 131.615   | CUDA benchmark route | profiler_runtime_benchmark_only |
| runtime   | encoder_forward        | `B=64 T=1024 C=768`                          | CUDA           | 79.286    | CUDA benchmark route | trainer_or_cxx_route            |
| runtime   | fused_classifier       | `B=64 T=1024 V=50257 P=50304`                | CUDA           | 8942.509  | CUDA benchmark route | trainer_or_cxx_route            |
| runtime   | fused_classifier_loss  | `B=64 T=1024 V=50257 P=50304`                | CUDA           | 3953.133  | CUDA benchmark route | trainer_or_cxx_route            |
| runtime   | gelu_forward           | `BT=65536 C=3072`                            | CUDA           | 528.059   | CUDA benchmark route | trainer_or_cxx_route            |
| runtime   | global_norm_squared    | `params=124475904`                           | CUDA           | 185.069   | CUDA benchmark route | trainer_or_cxx_route            |

## Project-Wide Fastest Rows Resolved Away From Trainer

| Suite     | Kernel                | Shape                                       | Selected stack | Time (us) | Call path                       | Decision                          | Reason                                                                                                                                                                                                                                                                                                        | Evidence                                                                                                                                                                                             |
| --------- | --------------------- | ------------------------------------------- | -------------- | --------- | ------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| attention | backward              | `B=64 T=1024 C=768 NH=12 HS=64`             | Torch          | 2196.240  | operator_or_reference_prototype | layout_rewrite_only               | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK.                                                                                                                                                                                                | 2026-05-20 target-context refresh: packed TK backward 2716.901 us; TorchPacked backward 4107.011 us; native Torch separated backward 2227.869 us (+1 more)                                           |
| attention | forward               | `B=64 T=1024 C=768 NH=12 HS=64`             | Torch          | 556.573   | operator_or_reference_prototype | layout_rewrite_only               | Native Torch SDPA wins only for already-separated Q/K/V; trainer-shaped TorchPacked was slower than packed TK.                                                                                                                                                                                                | 2026-05-20 target-context refresh: packed TK forward 787.859 us; TorchPacked forward 1120.509 us; native Torch separated forward 570.848 us (+1 more)                                                |
| matmul    | dInp                  | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS         | 365.930   | trainer_or_cxx_route            | rejected_x10_selector             | Do not broaden the SM120 direct-cuBLAS dInput selector to attention projection; the x10 trainer stability round regressed, so the source default remains the huge-N LM-head-only direct-cuBLAS dInput route.                                                                                                  | 2026-05-21 candidate selector round codex_sm120_round_cublas_dinp_attproj_fc_20260521 validated at avg_ms=2493.931 (+1 more)                                                                         |
| matmul    | dInp                  | `fc M=65536 N=3072 K=768 bias=1 gelu=1`     | cuBLAS         | 1364.900  | trainer_or_cxx_route            | rejected_x10_selector             | Do not broaden the SM120 direct-cuBLAS dInput selector to the GPT-2 MLP-up row. The microbench row can favor cuBLAS, but the broader direct-cuBLAS dInput selector regressed in the x10 TinyStories stability gate, while the stream-sync default keeps cuBLASLt for this row and improves the trainer smoke. | 2026-05-21 optional stack refresh selected cuBLAS 1328.470 us versus cuBLASLt 1367.640 us for MLP-up dInput (+2 more)                                                                                |
| matmul    | dInp                  | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`    | cuBLAS         | 1012.440  | trainer_or_cxx_route            | noise_floor_microbench_flip       | Do not promote the qkv dInput cuBLAS microbench flip as a trainer default without a trainer smoke. The refreshed benchmark-only round picked cuBLAS by about 0.2%, while the stable x10 selection artifact has cuBLASLt ahead for the same row.                                                               | 2026-05-21 benchmark-only native refresh codex_sm120_round_native_attention_median_20260521: cuBLAS 1012.36 us versus cuBLASLt 1014.27 us (+2 more)                                                  |
| matmul    | dInp+dGeLU            | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK             | 1822.440  | trainer_or_cxx_route            | rejected_x10_selector             | Keep the cuBLASLt fused dGELU trainer route as the default. The opt-in TK exact-dGELU selector now passes correctness and has a focused row win, but its x10 TinyStories stability round regressed versus the current stable default.                                                                         | 2026-05-20 target-context test_matmul with LLMK_SM120_USE_TK_FUSED_DGELU_DINP failed only the GPT-2 fcproj fused dGELU dInput row with max abs diff 0.500000 against the strict <0.50 gate (+4 more) |
| runtime   | bias_add              | `BT=65536 OC=3072`                          | Triton         | 542.899   | operator_or_reference_prototype | library_integration_not_justified | Keep as operator evidence; the refreshed Triton edge is about 0.2% and not enough to justify a trainer-callable Triton route.                                                                                                                                                                                 | 2026-05-21 materialized attention stack round: Triton bias_add 527.930 us versus 3-sample CUDA 528.993 us (+2 more)                                                                                  |
| runtime   | cuda_copy_d2d         | `logits_elems=3296722944`                   | Torch          | 8662.848  | profiler_runtime_benchmark_only | native_replacement_rejected       | Keep as operator evidence; raw-pointer LibTorch preserves most of the copy edge, but Python Torch remains the fastest observed row and a linked trainer smoke is still required before promotion.                                                                                                             | 2026-05-20 target-context refresh: logits copy CUDA runtime 8876.186 us; CUDA kernel 9354.822 us (+1 more)                                                                                           |
| runtime   | cuda_memset           | `hidden_elems=50331648`                     | Torch C++      | 59.875    | libtorch_raw_pointer_prototype  | library_integration_not_justified | Keep as C++ API feasibility evidence; the LibTorch route is tie-range and does not justify a trainer dependency.                                                                                                                                                                                              | 2026-05-21 raw-pointer LibTorch refresh: cached from_blob full-row parity PASS, CUDA runtime 60.826 us, native CUDA kernel 60.056 us, Torch C++ raw 60.011 us (+1 more)                              |
| runtime   | cuda_memset           | `logits_elems=3296722944`                   | Torch          | 3943.104  | operator_or_reference_prototype | native_replacement_rejected       | Keep as operator evidence; native CUDA replacement was slower than CUDA runtime.                                                                                                                                                                                                                              | 2026-05-20 target-context refresh: logits memset CUDA runtime 4186.323 us; CUDA kernel 4374.765 us (+1 more)                                                                                         |
| runtime   | gelu_backward_inplace | `BT=65536 C=3072`                           | Triton         | 781.888   | operator_or_reference_prototype | rejected_same_session_refresh     | Rejected for trainer promotion; refreshed CUDA GELU backward was faster than Triton.                                                                                                                                                                                                                          | 2026-05-20 target-context refresh: CUDA 789.605 us; Triton 802.830 us                                                                                                                                |

## Extra Project-Wide Benchmark Rows

| Suite     | Kernel                  | Shape                                   | Selected stack     | Time (us) | Call path                       | Reason                                                                                                 |
| --------- | ----------------------- | --------------------------------------- | ------------------ | --------- | ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| layernorm | backward                | `N=65536 C=3072`                        | CUDA               | 1272.855  | trainer_or_cxx_route            | non-objective LayerNorm stress width; GPT-2 trainer LayerNorm uses C=768                               |
| layernorm | backward_dinput         | `N=65536 C=3072`                        | Triton dInput-only | 801.056   | operator_or_reference_prototype | non-objective LayerNorm stress width; GPT-2 trainer LayerNorm uses C=768                               |
| layernorm | backward_dinput         | `N=65536 C=768`                         | Torch native       | 214.592   | operator_or_reference_prototype | partial backward decomposition row; not the full trainer LayerNorm backward contract                   |
| layernorm | forward                 | `N=65536 C=3072`                        | CUDA               | 544.301   | trainer_or_cxx_route            | non-objective LayerNorm stress width; GPT-2 trainer LayerNorm uses C=768                               |
| layernorm | fused_residual_forward  | `N=65536 C=3072`                        | CUDA               | 1082.597  | trainer_or_cxx_route            | non-objective LayerNorm stress width; GPT-2 trainer LayerNorm uses C=768                               |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state` | Torch              | 7298.912  | operator_or_reference_prototype | optimizer contract variant; current trainer objective row is no-master AdamW without this shape suffix |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`            | Torch              | 1208.083  | operator_or_reference_prototype | non-equivalent BF16-state optimizer reference; current trainer objective uses FP32 moment buffers      |

## Resolved Optional-Stack Decisions

| Suite     | Kernel                  | Shape                                       | Selected stack     | Time (us) | Scope                               | Call path                       | Class                      | Gate                                                                                                                          | Decision                              |
| --------- | ----------------------- | ------------------------------------------- | ------------------ | --------- | ----------------------------------- | ------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| attention | backward                | `B=64 T=1024 C=768 NH=12 HS=64`             | Torch              | 2196.240  | python separated-Q/K/V              | operator_or_reference_prototype | layout rewrite             | refresh same-session baseline, add a packed-QKV or separated-Q/K/V trainer path, then compare against TK and run TinyStories  | layout_rewrite_only                   |
| attention | forward                 | `B=64 T=1024 C=768 NH=12 HS=64`             | Torch              | 556.573   | python separated-Q/K/V              | operator_or_reference_prototype | layout rewrite             | refresh same-session baseline, add a packed-QKV or separated-Q/K/V trainer path, then compare against TK and run TinyStories  | layout_rewrite_only                   |
| layernorm | backward_dinput         | `N=65536 C=768`                             | Torch native       | 214.592   | partial backward prototype          | operator_or_reference_prototype | reference/state gap        | add dweight/dbias accumulation before considering trainer promotion                                                           | partial_backward_only                 |
| matmul    | dInp                    | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS             | 365.930   | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | rejected_x10_selector                 |
| matmul    | dInp                    | `fc M=65536 N=3072 K=768 bias=1 gelu=1`     | cuBLAS             | 1364.900  | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | rejected_x10_selector                 |
| matmul    | dInp                    | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`    | cuBLAS             | 1012.440  | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | noise_floor_microbench_flip           |
| matmul    | dInp+dGeLU              | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK                 | 1822.440  | C++ benchmark route                 | trainer_or_cxx_route            |                            |                                                                                                                               | rejected_x10_selector                 |
| runtime   | adamw_update            | `params=124475904 no-master fp32-state`     | Torch              | 7298.912  | operator prototype                  | operator_or_reference_prototype | library integration        | refresh same-session baseline, add an explicit libtorch/C++ or equivalent native route, then run route parity and TinyStories | rejected_slower_than_trainer_baseline |
| runtime   | bias_add                | `BT=65536 OC=3072`                          | Triton             | 542.899   | operator prototype                  | operator_or_reference_prototype | native/codegen integration | refresh same-session baseline, add a trainer-callable Triton/C++ route, then run route parity and TinyStories                 | library_integration_not_justified     |
| runtime   | cuda_copy_d2d           | `logits_elems=3296722944`                   | Torch              | 8662.848  | operator prototype                  | profiler_runtime_benchmark_only | library integration        | refresh same-session baseline, add an explicit libtorch/C++ or equivalent native route, then run route parity and TinyStories | native_replacement_rejected           |
| runtime   | cuda_memset             | `hidden_elems=50331648`                     | Torch C++          | 59.875    | C++ API prototype                   | libtorch_raw_pointer_prototype  | library integration        | add an explicit libtorch trainer link gate, route parity, and TinyStories smoke before promotion                              | library_integration_not_justified     |
| runtime   | cuda_memset             | `logits_elems=3296722944`                   | Torch              | 3943.104  | operator prototype                  | operator_or_reference_prototype | library integration        | refresh same-session baseline, add an explicit libtorch/C++ or equivalent native route, then run route parity and TinyStories | native_replacement_rejected           |
| runtime   | gelu_backward_inplace   | `BT=65536 C=3072`                           | Triton             | 781.888   | operator prototype                  | operator_or_reference_prototype | native/codegen integration | refresh same-session baseline, add a trainer-callable Triton/C++ route, then run route parity and TinyStories                 | rejected_same_session_refresh         |
| layernorm | backward_dinput         | `N=65536 C=3072`                            | Triton dInput-only | 801.056   | partial backward prototype          | operator_or_reference_prototype | non-trainer shape          | non-trainer GPT-2 LayerNorm shape; keep as operator evidence unless the trainer adds this shape                               | non_trainer_shape                     |
| runtime   | adamw_update_bf16_state | `params=124475904 no-master`                | Torch              | 1208.083  | non-equivalent BF16-state reference | operator_or_reference_prototype | contract mismatch          | match the trainer optimizer state contract before considering promotion                                                       | contract_mismatch                     |

## Inactive Native Microbench Selections

| Suite  | Kernel     | Shape                                       | Rejected stack | Current stack  | Decision                    |
| ------ | ---------- | ------------------------------------------- | -------------- | -------------- | --------------------------- |
| matmul | dInp       | `attproj M=65536 N=768 K=768 bias=1 gelu=0` | cuBLAS         | cuBLASLt       | rejected_x10_selector       |
| matmul | dInp       | `fc M=65536 N=3072 K=768 bias=1 gelu=1`     | cuBLAS         | cuBLASLt       | rejected_x10_selector       |
| matmul | dInp       | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | cuBLASLt       | cuBLAS         | benchmark_context_flip      |
| matmul | dInp       | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`    | cuBLAS         | cuBLASLt       | noise_floor_microbench_flip |
| matmul | dInp+dGeLU | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0` | TK             | cuBLASLt fused | rejected_x10_selector       |

## Attention Route Totals

| Source   | Stack                   | Shape                           | Scope                           | Trainer-layout | Forward (us) | Backward (us) | Total (us) | Complete | Note                                                                  |
| -------- | ----------------------- | ------------------------------- | ------------------------------- | -------------- | ------------ | ------------- | ---------- | -------- | --------------------------------------------------------------------- |
| native   | TK packed-QKV           | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 784.691      | 2743.724      | 3528.415   | True     |                                                                       |
| optional | TK packed-QKV           | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 785.718      | 2743.201      | 3528.919   | True     |                                                                       |
| optional | cuDNNPacked             | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 805.997      | 2819.398      | 3625.395   | True     |                                                                       |
| optional | TorchPacked             | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 1149.664     | 4050.301      | 5199.965   | True     |                                                                       |
| optional | TorchMaterializedPacked | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 1262.984     | 4215.184      | 5478.168   | True     |                                                                       |
| optional | TritonPacked            | `B=64 T=1024 C=768 NH=12 HS=64` | packed trainer-layout route     | True           | 2205.301     | -             | -          | False    | packed attention backward is not implemented in this Triton prototype |
| optional | Torch                   | `B=64 T=1024 C=768 NH=12 HS=64` | separated Q/K/V reference route | False          | 556.573      | 2196.240      | 2752.813   | True     |                                                                       |
| optional | cuDNN                   | `B=64 T=1024 C=768 NH=12 HS=64` | separated Q/K/V reference route | False          | 675.635      | 2404.397      | 3080.032   | True     |                                                                       |
| optional | Triton                  | `B=64 T=1024 C=768 NH=12 HS=64` | separated Q/K/V reference route | False          | 2035.579     | -             | -          | False    | attention backward is not implemented in this Triton prototype        |

## Extra Native Benchmark Selections

| Suite     | Kernel                 | Shape            | Selected stack | Time (us) | Call path            |
| --------- | ---------------------- | ---------------- | -------------- | --------- | -------------------- |
| layernorm | backward               | `N=65536 C=3072` | CUDA           | 1273.941  | trainer_or_cxx_route |
| layernorm | forward                | `N=65536 C=3072` | CUDA           | 543.091   | trainer_or_cxx_route |
| layernorm | fused_residual_forward | `N=65536 C=3072` | CUDA           | 1082.067  | trainer_or_cxx_route |

## Trainer Selection Rows

| Suite     | Kernel                 | Shape                                        | Current stack  | Time (us) | Call path                       |
| --------- | ---------------------- | -------------------------------------------- | -------------- | --------- | ------------------------------- |
| attention | backward               | `B=64 T=1024 C=768 NH=12 HS=64`              | TK packed-QKV  | 2743.724  | trainer_or_cxx_route            |
| attention | forward                | `B=64 T=1024 C=768 NH=12 HS=64`              | TK packed-QKV  | 784.691   | trainer_or_cxx_route            |
| layernorm | backward               | `N=65536 C=768`                              | CUDA           | 282.924   | trainer_or_cxx_route            |
| layernorm | forward                | `N=65536 C=768`                              | CUDA           | 136.995   | trainer_or_cxx_route            |
| layernorm | fused_residual_forward | `N=65536 C=768`                              | CUDA           | 275.580   | trainer_or_cxx_route            |
| matmul    | dInp                   | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLASLt       | 369.020   | trainer_or_cxx_route            |
| matmul    | dInp                   | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLASLt       | 1357.230  | trainer_or_cxx_route            |
| matmul    | dInp                   | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1383.670  | trainer_or_cxx_route            |
| matmul    | dInp                   | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLAS         | 21261.820 | trainer_or_cxx_route            |
| matmul    | dInp                   | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt       | 1011.610  | trainer_or_cxx_route            |
| matmul    | dInp+dGeLU             | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLASLt fused | 1841.300  | trainer_or_cxx_route            |
| matmul    | dW                     | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS         | 328.190   | trainer_or_cxx_route            |
| matmul    | dW                     | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS         | 1309.120  | trainer_or_cxx_route            |
| matmul    | dW                     | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1308.930  | trainer_or_cxx_route            |
| matmul    | dW                     | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 20951.490 | trainer_or_cxx_route            |
| matmul    | dW                     | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS         | 1010.950  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLAS         | 332.000   | trainer_or_cxx_route            |
| matmul    | dW+accum               | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLAS         | 1333.200  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLAS         | 1353.250  | trainer_or_cxx_route            |
| matmul    | dW+accum               | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 20929.240 | trainer_or_cxx_route            |
| matmul    | dW+accum               | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLAS         | 999.970   | trainer_or_cxx_route            |
| matmul    | fwd                    | `attproj M=65536 N=768 K=768 bias=1 gelu=0`  | cuBLASLt       | 370.910   | trainer_or_cxx_route            |
| matmul    | fwd                    | `fcproj M=65536 N=768 K=3072 bias=1 gelu=0`  | cuBLASLt       | 1358.930  | trainer_or_cxx_route            |
| matmul    | fwd                    | `lmhead M=65536 N=50304 K=768 bias=0 gelu=0` | cuBLASLt       | 22341.620 | trainer_or_cxx_route            |
| matmul    | fwd                    | `qkv M=65536 N=2304 K=768 bias=1 gelu=0`     | cuBLASLt       | 1039.600  | trainer_or_cxx_route            |
| matmul    | fwd+gelu               | `fc M=65536 N=3072 K=768 bias=1 gelu=1`      | cuBLASLt       | 1497.480  | trainer_or_cxx_route            |
| runtime   | adamw_update           | `params=124475904 no-master`                 | CUDA           | 1807.082  | trainer_or_cxx_route            |
| runtime   | bias_add               | `BT=65536 OC=3072`                           | CUDA           | 536.362   | trainer_or_cxx_route            |
| runtime   | bias_add               | `BT=65536 OC=768`                            | CUDA           | 80.068    | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce       | `BT=65536 OC=2304`                           | CUDA           | 186.550   | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce       | `BT=65536 OC=3072`                           | CUDA           | 245.482   | trainer_or_cxx_route            |
| runtime   | bias_grad_reduce       | `BT=65536 OC=768`                            | CUDA           | 24.211    | trainer_or_cxx_route            |
| runtime   | cuda_copy_d2d          | `hidden_elems=50331648`                      | CUDA runtime   | 131.734   | profiler_runtime_benchmark_only |
| runtime   | cuda_copy_d2d          | `logits_elems=3296722944`                    | CUDA runtime   | 8774.458  | profiler_runtime_benchmark_only |
| runtime   | cuda_memset            | `hidden_elems=50331648`                      | CUDA runtime   | 59.800    | trainer_or_cxx_route            |
| runtime   | cuda_memset            | `logits_elems=3296722944`                    | CUDA runtime   | 3948.186  | trainer_or_cxx_route            |
| runtime   | encoder_forward        | `B=64 T=1024 C=768`                          | CUDA           | 84.260    | trainer_or_cxx_route            |
| runtime   | fused_classifier       | `B=64 T=1024 V=50257 P=50304`                | CUDA           | 8904.614  | trainer_or_cxx_route            |
| runtime   | fused_classifier_loss  | `B=64 T=1024 V=50257 P=50304`                | CUDA           | 3997.862  | trainer_or_cxx_route            |
| runtime   | gelu_backward_inplace  | `BT=65536 C=3072`                            | CUDA           | 780.873   | trainer_or_cxx_route            |
| runtime   | gelu_forward           | `BT=65536 C=3072`                            | CUDA           | 528.334   | trainer_or_cxx_route            |
| runtime   | global_norm_squared    | `params=124475904`                           | CUDA           | 185.112   | trainer_or_cxx_route            |
