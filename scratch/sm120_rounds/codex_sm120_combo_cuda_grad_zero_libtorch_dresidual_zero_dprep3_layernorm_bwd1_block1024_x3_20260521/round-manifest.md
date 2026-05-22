# SM120 Round Manifest

- run label: `codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_cuda_grad_zero_libtorch_dresidual_zero_dprep3_layernorm_bwd1_block1024_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `594`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `7e5ef8bda77eafc2b20ec04d0f81fdd65fb5008d99aff2e2b5d0147330ee30d0` |
| `test_attention` | `True` | `1800528` | `58670801740eddf26b4063164d71aceff81d59ec041a314845cee04f28ed6055` |
| `test_layernorm` | `True` | `1278296` | `65fe2946aa7545688b857cb24b75f9bdb52c3a63ba8cc32182180227d2c8f365` |
| `test_bias` | `True` | `2089120` | `1825b53bbd34b01d54472f52490403641122b143d54068f4212cc23a2a8d678c` |
| `test_gelu` | `True` | `1179912` | `694cb9b5f3444d7772a4379bcfb5c23146b54a364160b4315f4d0c83381a630a` |
| `test_fused_classifier` | `True` | `1208704` | `b2346a2835e3f5ad4838fe2055a1f6540fca4dfa33e0a761eebbe8c80a499c81` |
| `test_encoder` | `True` | `1210168` | `0d9a5419cb059e7cb5670082d39b72a6b57012e900873712e0ed95c5f9bed77a` |
| `test_adamw` | `True` | `1183768` | `a20eacea9032a6205a26e3d855dcb16f689cd43c4811336c361d6629facfb283` |
| `test_global_norm` | `True` | `1179464` | `310d308fc69444f0539734287093267848f5152c3a29bcc7798e9e1180f0e528` |
| `bench_sm120_matmul` | `True` | `2410304` | `d91f5a5703773f31cced02f42e8f7a94cc4d5cc56ba1a2a6098b83b3b7b40df0` |
| `bench_sm120_attention` | `True` | `1768800` | `3f29046883691414fd2042d70286fb3fcb53f498581f4f39e2af1c16dd92e594` |
| `bench_sm120_layernorm` | `True` | `1274232` | `5b089ce9fb4fc04e358d601ba799105dd4ed10175bbb7329176af932030df925` |
| `bench_sm120_runtime` | `True` | `2271168` | `83ece2e7bbf8710041deffada7745c13d044b24d9d2ddee922403bd1be4ce07c` |
| `train_gpt2cu` | `True` | `3105552` | `767ce9a93a08717d36a880954381d8ae05e2ea550559669b4971cb22400d23b3` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
