# SM120 Round Manifest

- run label: `codex_sm120_promoted_default_after_epilogue_probe_x3_20260522`
- artifact dir: `scratch/sm120_rounds/codex_sm120_promoted_default_after_epilogue_probe_x3_20260522`
- train output dir: `log124M/5090_S_codex_sm120_promoted_default_after_epilogue_probe_x3_20260522`
- device arch: `SM120`
- max steps: `3`
- SM120 LibTorch grad-zero route: `0`
- SM120 LibTorch dresidual-zero route: `1`
- git commit: `0f21747`
- changed paths: `640`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2173624` | `035ba5a0402e4f81d13bef9f8a0f5acc609b2a33f4b544532503f83ed4215f77` |
| `test_attention` | `True` | `1800528` | `2dacbe7ba7635e60ae5cc178c39001cc885258c1770e8975374f253088535d89` |
| `test_layernorm` | `True` | `1278296` | `6a963098c6a8c4b47917760070e7256a23285e636d28199e4fec1db8674d7476` |
| `test_bias` | `True` | `2089120` | `7bdcbb01ad64553a904adb0139581ed62d5a71cff32fa264c39bb682627a108f` |
| `test_gelu` | `True` | `1179912` | `a5ca422ed30ba976717cbfd8ef16c9bd02e0f0f861f323d79b5c00472e0cf47a` |
| `test_fused_classifier` | `True` | `1208704` | `dc8a41eb4cad21f4434f4def376993eba0d5fd2fdcc835c2ef1c0e6d27a5c616` |
| `test_encoder` | `True` | `1210168` | `afa2d2100f13de12fece6cc41694568da35d857e4bf5cae3789b8ecb306ea64f` |
| `test_adamw` | `True` | `1183768` | `db115e3277f7f35cbc51041fa5180adf478127789b75f8d96f9026107be5bc9e` |
| `test_global_norm` | `True` | `1179464` | `a8fecd6c90596167b5f09c2c1820983b5606fc743d914c0a2925f913b944274d` |
| `bench_sm120_matmul` | `True` | `2410304` | `fc8155706bb94e58a9d549b3e081b7c866543c35c3989ae15356b83422f4fc6a` |
| `bench_sm120_attention` | `True` | `1768800` | `cb65023edb5ed46b89f7c05a842078a46cb14415d29740e5f713b21667ad6ce2` |
| `bench_sm120_layernorm` | `True` | `1274232` | `cb7f0ccb586ecf627d4a2f3346e0a461c0515e8cb7ec5d6da31027f544480223` |
| `bench_sm120_runtime` | `True` | `2271168` | `5c72e2a3404bb58b243255d253db00b97a1914916ea692d2f639425a480d1956` |
| `train_gpt2cu` | `True` | `3105552` | `304550070058a3abe5d455d248688a586d923321115fcadeb10be072b2e9cf04` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
