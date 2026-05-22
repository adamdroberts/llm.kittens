# SM120 Round Manifest

- run label: `codex_sm120_promoted_libtorch_grad_zero_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_libtorch_grad_zero_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_promoted_libtorch_grad_zero_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `616`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `5dfba28585d9417a35b427e7664d1004acc7682ee1a69859ababa4b160b74a7e` |
| `test_attention` | `True` | `1800528` | `17e847a2b35228b0d2710fd82b85ff339c70cd9667252ef23f86be372c4ce3f1` |
| `test_layernorm` | `True` | `1278296` | `f418713d783c68dca69f815352977065967f4a2e75403a3db0fb9254ff2bbe18` |
| `test_bias` | `True` | `2089120` | `8aa597f99fd12225eb37cb127c93b0837cda4084276862126080d975ba979126` |
| `test_gelu` | `True` | `1179912` | `5df6889c8277c09134129614d1f89a39912c30d730a2d3983858f664bc7e751f` |
| `test_fused_classifier` | `True` | `1208704` | `ecb02abb4be87dc4d0ded78991c41b8cf731fad50c3e45ec69a308aff5254de0` |
| `test_encoder` | `True` | `1210168` | `b1f6ae1352671bfa3a30cda48c59bc87bb88b7469b1a94756c8d1d7080f1082d` |
| `test_adamw` | `True` | `1183768` | `7f4abfa41e3668f70aacac9e05634d4487aa260756cc90f705c71a191efb30f8` |
| `test_global_norm` | `True` | `1179464` | `04110b5d24099862316f502817e19aa043a3b81fb64ab430b2431e848ab6efc4` |
| `bench_sm120_matmul` | `True` | `2410304` | `e69323be7332c18ca9547465890d26c15599270b45754baeb2d1e858730ce2b8` |
| `bench_sm120_attention` | `True` | `1768800` | `8eb4c7114f35c170e29c5225502f26e8a478e849b5665bb5ff77a14e1ae0804f` |
| `bench_sm120_layernorm` | `True` | `1274232` | `0745e468ffff2313e05d6bd0e74a80691e5cef29a497493a09d7155b76f8ee1c` |
| `bench_sm120_runtime` | `True` | `2271168` | `a935a79eb4e171c156cea863e32377326c1de180a4d6c600f99b1ccb450e6d02` |
| `train_gpt2cu` | `True` | `3105632` | `f8d8d975ffa331f5f29fb67101887ea5968e3861f8ba75139f22303c83d37471` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
