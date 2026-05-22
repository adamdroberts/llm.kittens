# SM120 Round Manifest

- run label: `codex_sm120_restore_promoted_default_after_disable_cublas_bwd_tk_dgelu_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_restore_promoted_default_after_disable_cublas_bwd_tk_dgelu_20260522`
- train output dir: `log124M/5090_S_codex_sm120_restore_promoted_default_after_disable_cublas_bwd_tk_dgelu_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `646`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `fd176e3a5b301d755a18741c29d66b75f90d7219c8fe69bfc0dcd9858b6cdf87` |
| `test_attention` | `True` | `1800528` | `3a3fcc8d4251c373e0cbdecfeb635c89cf65455182aaa7ea366a68e40baf4049` |
| `test_layernorm` | `True` | `1278296` | `9181a5d2d6bd15f3b736d07401f71011f601dedcf615811ddc1b72ddea1deb06` |
| `test_bias` | `True` | `2089120` | `f4e34663ed1278b5ad826f2d89da3866f11d5276958805ed4f18838596026e3e` |
| `test_gelu` | `True` | `1179912` | `f9a461e14755f5849f00dfcbb2384bf9bdaecff5c622ad22bfd47a4b11b91fce` |
| `test_fused_classifier` | `True` | `1208704` | `f1bac3559141100abaf040b94d251c1ef557d179b679d550229d56fee9090c04` |
| `test_encoder` | `True` | `1210168` | `b34d9b76278e42a6e0357fb04c2af44f22074c05ef9b5faa438c0aab95f0de7d` |
| `test_adamw` | `True` | `1183768` | `012b29c2641f4d44f0b1fe2f30674c37e7e97d831f1734fe47224cbf5cd2ecbd` |
| `test_global_norm` | `True` | `1179464` | `d8c690709f945b98df12fa64ba9de9e49363e24e156d169a6af1a8931b839f8f` |
| `bench_sm120_matmul` | `True` | `2410304` | `322427cdad91683c4edfd01e4484269ea051ce7322d5c82d78adf4b18184a1c3` |
| `bench_sm120_attention` | `True` | `1768800` | `2bbfd33f08c287e1411579e7200e3ef98f8e0cbd31da9b04438eca8a0794eaaf` |
| `bench_sm120_layernorm` | `True` | `1274232` | `4c574061497e8e19680cd1c6ef4ec196ec49dd1a9a612254fd8e854ba0a7dd96` |
| `bench_sm120_runtime` | `True` | `2271168` | `543cd14fbe856d49186c501d3603017999064dd1fa97678e5ad39649efd903cd` |
| `train_gpt2cu` | `True` | `3105552` | `1d89cc669368b2d163f3f0edf994ead1aeab075b833b056d4f99de4b18a33d37` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
