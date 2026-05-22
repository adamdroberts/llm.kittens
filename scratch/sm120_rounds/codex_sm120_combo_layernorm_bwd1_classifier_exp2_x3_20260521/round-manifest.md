# SM120 Round Manifest

- run label: `codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521`
- train output dir: `log124M/5090_S_codex_sm120_combo_layernorm_bwd1_classifier_exp2_x3_20260521`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `0`
- git commit: `0f21747`
- changed paths: `515`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `09786903690dfc9092fb24486907e807dec44d5838136693f3075cce60dde741` |
| `test_attention` | `True` | `1760032` | `c175c76f022a25d93600021a82650e88c73783024cce3f37788ce1f4b41756f1` |
| `test_layernorm` | `True` | `1237784` | `bdf06a201be0db42cd0261a58067f73920f2f64bf24cb28c23f5bb04f2ee210d` |
| `test_bias` | `True` | `2048616` | `169383b4b043ba8152750cf990cb226b7817909fae3a057eb30ea97accfc80df` |
| `test_gelu` | `True` | `1139336` | `312ed639d6adf3c1180e35e018ecbfa39a0f24bf262af63d9707fac0b820f1e9` |
| `test_fused_classifier` | `True` | `1164032` | `96db1242e7ea6576d7d0023597deead139bba3e78f45a81620dbd8822baf6226` |
| `test_encoder` | `True` | `1165512` | `304fa53a5f9e7af7198f8745148d99f9ce9992f7cf52e27143a4116e0b61265c` |
| `test_adamw` | `True` | `1138408` | `31a45cb4255bb981eff72b6a31470d0a7dea40c4549fde851e1e0a51b5001973` |
| `test_global_norm` | `True` | `1138880` | `2ff8b421ce99dc3f1ade98b6d6d9200c1e0f8a90ade0264966712890523014c9` |
| `bench_sm120_matmul` | `True` | `2373912` | `2c1fe938ce1302f6b1890263908f6c35b977d5deb7d50dc16fe40dc4e030044d` |
| `bench_sm120_attention` | `True` | `1728312` | `ab7e60433c34cd88e0dc224beb497096096adf527d2395e90cdbade3c5095cc9` |
| `bench_sm120_layernorm` | `True` | `1233728` | `2768672c15bde549cdc564010c16695a97058d7d0ba000f31d224744c1ddb762` |
| `bench_sm120_runtime` | `True` | `2221864` | `ccaed5ffe3dbd94064a8da6a8220d58f5eb68b072ea47ac0ca208019c57e32ce` |
| `train_gpt2cu` | `True` | `3060032` | `a0022697fd7ee8073c7b6a71152d1ad86326ab336361550b30ffaa23b97fbbb2` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
