# SM120 Round Manifest

- run label: `codex_sm120_round_tk_fused_dgelu_exact_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_tk_fused_dgelu_exact_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_tk_fused_dgelu_exact_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `486`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2154392` | `4bb5696372867525ca449cf3439746d07f74747f79a874617d82aee31c20b148` |
| `test_attention` | `True` | `1760032` | `66aad9dfc815964e0fc49f7587804651816253b6d8a06a26e344f100dfed094d` |
| `test_layernorm` | `True` | `1237784` | `b2e9750a150a3516fe2192b1bc8b2ec348d00b2b346df684df6cdec344711258` |
| `test_bias` | `True` | `2007656` | `90824cbe539e91f9211e0d233ff1e887f5e8eaf9908cd1e0db644520bea49d6f` |
| `test_gelu` | `True` | `1139336` | `680d0a45a966fbf8abe6d95fadb8f3024336962ad25c4ef32852a777eb433370` |
| `test_fused_classifier` | `True` | `1164032` | `212e3352ff12c5a7e67dd91821942485a9ba0fbe030d8cf7c2fc2d8d8142e02a` |
| `test_encoder` | `True` | `1165512` | `4c464725702f8ece388163a1bda0943d26cced426372abb47ea9a206fa47acfc` |
| `test_adamw` | `True` | `1138408` | `c92e479abb1e1725bced7c3cbfc317005f1320aa285533f2e0ab275228a48075` |
| `test_global_norm` | `True` | `1138880` | `d3595121fe3771e620fc7c0f48a732f8c7ba8f4568d9db33edfa4f6d3b0bd58f` |
| `bench_sm120_matmul` | `True` | `2332952` | `6a395d31054738348383dd57836e979570c784a42e4fcf7557df60ee8d43fde2` |
| `bench_sm120_attention` | `True` | `1731352` | `3ea32ca5fcbbe8c5c2a73d23d4733b94c546c052e3ae898a445b9d60033d3c71` |
| `bench_sm120_layernorm` | `True` | `1233728` | `5701f9cf2c82e7fbe01b77f5ca91e4f74ff88dc1100d99b059b0c50836aed029` |
| `bench_sm120_runtime` | `True` | `2180712` | `296586bbce5cd9490699a0decce68f44f384ad1ca965e9ef4bcf129df91cff2e` |
| `train_gpt2cu` | `True` | `3040952` | `65a41c7508ba587f2b118eac40ecfd7eccd483f98b468a6c5294503c240806c1` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
