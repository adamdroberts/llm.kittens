# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_dresidual_zero_dprep3_layernorm_bwd1_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `592`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `89dafe4a26014c3a82659cca9d7f751f5c0adcecf2aa8906b0b60ba7dc0b3119` |
| `test_attention` | `True` | `1800528` | `2a7d01ddd45c5aa38b64f74dc96941c5947b2b689f3867b5cea8186e7cfbaa28` |
| `test_layernorm` | `True` | `1278296` | `5f167da3d530ecad16ca5076b8becf8732f973dad890694920440b28b6aa350f` |
| `test_bias` | `True` | `2089120` | `b23ad332d17a90c4edbfa7a4b34405c3e1e6a1bd26cb27411d12a87710e3bc1f` |
| `test_gelu` | `True` | `1179912` | `0d8e561ee4c001ba4972b3ad906d9c37385670ca63dab58d6b88702a40b0b082` |
| `test_fused_classifier` | `True` | `1208704` | `759038654d138138ae4ba3293270eddcbdb8bc0548a3132248a72c31a84d5e53` |
| `test_encoder` | `True` | `1210168` | `11f4db2ed0f8709ccd5c01113085a412595a41832c03534012f19dd24bd17993` |
| `test_adamw` | `True` | `1183768` | `968b8e9ab6bee2cc9c023dfc1ba8d159837a3ff698bf677aac65fb2752b24b05` |
| `test_global_norm` | `True` | `1179464` | `e4201130e83704e206b44a14ff1510d197316f5574cc9a7e57bc9f7615939a3f` |
| `bench_sm120_matmul` | `True` | `2410304` | `010b9dc919e55bed4989051f3fc7a3997b0a9016c114cc37ce79747146015207` |
| `bench_sm120_attention` | `True` | `1768800` | `c42c500919de04c4415ece77b47cb76b36ab32da5d46b4355ef4815610ec9766` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0db83791a3739fbaa6e2e85ef30ee2791477bdf9989fc66423164641397185cf` |
| `bench_sm120_runtime` | `True` | `2271168` | `6cf59362e1ead59d240ac730fe41767f4949c4ed043c2fd2f0086d75e83947e1` |
| `train_gpt2cu` | `True` | `3105552` | `434166f391dd27f3722041592dca4874b562360feb29d52278881a95d209deae` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
