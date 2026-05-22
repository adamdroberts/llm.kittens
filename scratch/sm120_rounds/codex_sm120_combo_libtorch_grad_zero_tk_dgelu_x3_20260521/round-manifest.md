# SM120 Round Manifest

- run label: `codex_sm120_combo_libtorch_grad_zero_tk_dgelu_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_libtorch_grad_zero_tk_dgelu_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_libtorch_grad_zero_tk_dgelu_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `1`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `580`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2190792` | `42ea07953375330c87659adb7a9c31c85fcdca3269ef6ebf09458ff0c051175f` |
| `test_attention` | `True` | `1800528` | `0755f282abfafaea817691aa20f9617524e2a0a12b8c9ea0beaf1cf3b00d7fa8` |
| `test_layernorm` | `True` | `1278296` | `167099bd2cac6168d5a25a6d3a100efcfe5b088a646067ad2e0aad6ad5c17276` |
| `test_bias` | `True` | `2052256` | `8c8c293cfd46e7e4605dbef87b4bf657155f3370322130a90483722c2e9e221c` |
| `test_gelu` | `True` | `1179912` | `632bd1659ae46d75dbfb3e3c068bcb81b8b769c2383428233029bd7aa73ce8a3` |
| `test_fused_classifier` | `True` | `1208704` | `955d8a7f0b3d84a69c42a6cd78edb1820de7a24ca94b94ee1ee53096bf475b4d` |
| `test_encoder` | `True` | `1210168` | `9da4005003580dfa04f4a3306d93d851c20de279c4a110381ab8100730bb0c26` |
| `test_adamw` | `True` | `1183768` | `74a73dc038d536fc9c1b7e0200d390128a1173a93c10f069520f1ec8f380f925` |
| `test_global_norm` | `True` | `1179464` | `efac1dd308546533416b26e915cf84aaa43d1f6ec2213bb4de15c15f12ce1a37` |
| `bench_sm120_matmul` | `True` | `2369344` | `d2db9a08e58670c83035963fe9295a8822c68e209f83c5333d1dbb26547a57ed` |
| `bench_sm120_attention` | `True` | `1768800` | `0d6b94d48d9c414e127c393e510cc79b2af686d41403ef9be7eb87ad226ef91a` |
| `bench_sm120_layernorm` | `True` | `1274232` | `c3bdf4670281524b88d5616ae57a5444dd242d255328614177d52e50f925fef6` |
| `bench_sm120_runtime` | `True` | `2234304` | `d1e724be5a70b0495b202373b30a37a2728f4da0833ace968d97d22186766b44` |
| `train_gpt2cu` | `True` | `3104616` | `d4f047c48030d443c2de418a79d0f8a3087d3b6a653b33225f8976eabd64f5c9` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
