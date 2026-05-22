# SM120 Round Manifest

- run label: `codex_sm120_round_backward_stream_sync_x10_20260521`
- artifact dir: `scratch/sm120_rounds/codex_sm120_round_backward_stream_sync_x10_20260521`
- train output dir: `log124M/5090_S_codex_sm120_round_backward_stream_sync_x10_20260521`
- device arch: `SM120`
- max steps: `10`
- git commit: `0f21747`
- changed paths: `490`

## Binaries

| Path | Exists | Size bytes | SHA256 |
|---|---:|---:|---|
| `test_matmul` | `True` | `2133120` | `15aec4eec151b9b0afee09e49ff6544cb0f56ca8a2157e00700234309a69b1e2` |
| `test_attention` | `True` | `1760032` | `e6de0193c5d494d841d91774c318de53b7ef02f088817d9155274cc901cbdd26` |
| `test_layernorm` | `True` | `1237784` | `4357f6ecbe89d5b0e8e30237953f2a85e4619dffedffe257a9357283fe087adc` |
| `test_bias` | `True` | `2048616` | `72715d551a1d1b88c14ce6a116617d4f15a9f01e623cee25a5d671d048c63046` |
| `test_gelu` | `True` | `1139336` | `b2139a480e910971dd281d385415bbc0205e617ba13fa2082ea7222c55ec12c7` |
| `test_fused_classifier` | `True` | `1164032` | `e2d10581ef8627f4eed3e1ffb49ce98d9860dd04ec0e24331b82fbaf833f979e` |
| `test_encoder` | `True` | `1165512` | `0ac2303b0159fb682a0b4470aedcc6086a5be758c14d62a51e8f456ea9d955a2` |
| `test_adamw` | `True` | `1138408` | `e9eaf01247c366e53ad442db9a04577099a5fdde1b54e00880c0bdc6ad963681` |
| `test_global_norm` | `True` | `1138880` | `1261fe5a29c85220dc54de875833412e2cd247f55a365cd9a1f9dfd05fe02057` |
| `bench_sm120_matmul` | `True` | `2373912` | `4460698daddda4dfc9334336fa25dc54db52e1ce25e2481889f24606c1e6289a` |
| `bench_sm120_attention` | `True` | `1728312` | `18830455b84c7c9aeeb8db589507f9562079a2c098c6ffd83479208ec3689202` |
| `bench_sm120_layernorm` | `True` | `1233728` | `3c546dd93e7172cf6ce0d5b42e898aa00baf7c847b8d7f28b1cb6edfb0c1e2b1` |
| `bench_sm120_runtime` | `True` | `2217576` | `c2dc84ca879a63fb2d36e0ba3190ba37e51dec002b9b77903f380e945a919716` |
| `train_gpt2cu` | `True` | `3045944` | `64f424e174d3dae5caaa8b48e531e7d4ea458c0b1c6d3783a44a7e888fc20ad3` |

## Toolchain

- nvcc returncode: `0`
```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```
