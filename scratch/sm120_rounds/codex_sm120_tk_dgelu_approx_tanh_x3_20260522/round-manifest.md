# SM120 Round Manifest

- run label: `codex_sm120_tk_dgelu_approx_tanh_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_tk_dgelu_approx_tanh_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_tk_dgelu_approx_tanh_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- train zero stage: `1`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `680`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2231752` | `5c498a16f65007cfcd316823924a18fd17d4f4d4538d9c1298141a79a5f220d6` |
| `test_attention` | `True` | `1800528` | `46dabb7d778bedd2ae1976f272f842b798ac43e8f5b9c55288c057f51d2a45fd` |
| `test_layernorm` | `True` | `1278296` | `4407727e859231a08bfeecb437168426b41b09ce5208948806b686736dbeb3f6` |
| `test_bias` | `True` | `2089120` | `cb7f579e442c36e101b79f7c4040b0ef9bf437879d08ccdb6747f583038cf09d` |
| `test_gelu` | `True` | `1179912` | `c1938649728bd8e78a817f49b4e6e1c96c397c2ec3f8856639906816b21fd086` |
| `test_fused_classifier` | `True` | `1208704` | `4165483ae68b0d3e9a300cf48681ab2cf4b09f04af68c45956217ef93a9e0326` |
| `test_encoder` | `True` | `1210168` | `9f931966974a2ccdbdb468673315387b4066810f952ce35daff53345df79748c` |
| `test_adamw` | `True` | `1183768` | `4e8ea0568710de8d58fbbe8a3ce7e8430af0ba4d77566bb4233f657e525ad7db` |
| `test_global_norm` | `True` | `1179464` | `45d9c2e4919c7d81d6b80042e817c4c8df0b008302b5fe32ba3ea65d3fd2d712` |
| `bench_sm120_matmul` | `True` | `2410304` | `1d7e957aee0e829756a8306ed4178733aa6523b3c43de263c04b58d2610544bf` |
| `bench_sm120_attention` | `True` | `1768800` | `a62f267433633f3bf4ce5b3a44fb945b8bb37986988e19c67afef2c201262028` |
| `bench_sm120_layernorm` | `True` | `1274232` | `7f889739f846c9382de3ad94d266857f55873df1770f767e1f5a11c01ddf1b02` |
| `bench_sm120_runtime` | `True` | `2271168` | `f4b18b7ad3ae2a2ddd6b82dc20a6c5da696ee4c3aae6c1a9a0d2570607b55d4c` |
| `train_gpt2cu` | `True` | `3141480` | `549736f9558be4bee7a6f28fee305f21491c5a3a51ba22b7c6e17a8d9b2ae084` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
