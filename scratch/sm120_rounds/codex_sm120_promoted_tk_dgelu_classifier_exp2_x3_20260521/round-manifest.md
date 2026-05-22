# SM120 Round Manifest

- run label: `codex_sm120_promoted_tk_dgelu_classifier_exp2_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_tk_dgelu_classifier_exp2_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_tk_dgelu_classifier_exp2_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `614`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `518786594d4a7aebca4aeed79ad0bf86133b883b1d2ac2a5ab3b8a31a20ebfb2` |
| `test_attention` | `True` | `1800528` | `ebd081c3b31087fbeccb89224337841178968f17675ee8907385bf83798c3fe2` |
| `test_layernorm` | `True` | `1278296` | `e431b5eef37445109b6e8b670468c5c4db9f7164ce2f263e3e4a3ea1a5e4026d` |
| `test_bias` | `True` | `2052256` | `0f73e4ce2a24b38a6f139f535e913a0aa8515df12ab992ad07f3dd0aab648b56` |
| `test_gelu` | `True` | `1179912` | `6f83e86998c304f1877ef389ce1cdd2a36e72b912b1d798733afbe065e502748` |
| `test_fused_classifier` | `True` | `1208704` | `73153466eb217f882d24dee8b27eddfbb345e7905f4e63aa5871b8e6e7281aca` |
| `test_encoder` | `True` | `1210168` | `7f120996801c2380f5762b2b95716217e1a587034a10f531f0f363045dd3fb71` |
| `test_adamw` | `True` | `1183768` | `03f421b2cb8910952f2e43b5f1ee49683eb2325589dde6899ba1114628a5e371` |
| `test_global_norm` | `True` | `1179464` | `57025a4d3a280ec78b1d713d9095e18e626864803955ad99db983e8d5fab4ee7` |
| `bench_sm120_matmul` | `True` | `2369344` | `3205d4fe938905d52b1af09fd34beaf964010221ba3819843ad4ad9e7349c150` |
| `bench_sm120_attention` | `True` | `1768800` | `bc6c9c5a0ccfa7906b096186d3ed58bd78ab6105b84bbe07b5f356cac340357d` |
| `bench_sm120_layernorm` | `True` | `1274232` | `5e4fa8a0506f462cf8a27ef2a2ed8c80d4ef4156129d663d3e5d089c455df37f` |
| `bench_sm120_runtime` | `True` | `2234304` | `2f2e2e15aa4e2ccc573a89daa7de4bd2757a0fff624d5b177a48bb079eacd4fa` |
| `train_gpt2cu` | `True` | `3104616` | `0d54091aeaa825ef24a8112fa22c5a0bd0134307ca727d09d5ea5b62f8316c87` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
