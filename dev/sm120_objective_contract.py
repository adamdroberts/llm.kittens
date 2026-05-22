"""Shared SM120 optimization objective contract.

These lists are consumed by the backend-stack probe, round manifest writer, and
artifact validator. Keeping them in one module avoids accepting a round where
one tool has silently drifted from the objective enforced by another.
"""

CURRENT_NATIVE_SELECTION_ROUND = "scratch/sm120_rounds/codex_sm120_runtime_grad_zero_default_audit_x10_20260522"
CURRENT_OPTIONAL_STACK_ROUND = "scratch/sm120_rounds/codex_sm120_optional_refresh_current2_20260522"

OBJECTIVE_STACKS = (
    "ThunderKittens 2.0",
    "cuBLAS",
    "cuBLASLt",
    "cuDNN",
    "Triton",
    "Torch",
    "CuTeDSL",
    "Plain CUDA",
)

ENVIRONMENT_STACKS = ("GPU runtime",)

OBJECTIVE_FAMILIES = (
    "gemm_forward",
    "gemm_forward_fused_gelu",
    "gemm_backward_dinput",
    "gemm_backward_dinput_fused_dgelu",
    "gemm_backward_dweight",
    "gemm_backward_dweight_accum",
    "bias_add",
    "bias_gradient_reduce",
    "gelu_forward",
    "gelu_backward",
    "attention_forward",
    "attention_backward",
    "layernorm_forward",
    "layernorm_fused_residual_forward",
    "layernorm_backward",
    "classifier_softmax_cross_entropy_dlogits",
    "adamw",
    "global_norm",
    "encoder_forward",
    "cuda_memset",
    "cuda_copy_d2d",
)

MATMUL_SHAPE_REQUIREMENTS = (
    ("fwd", ("qkv", "attproj", "fcproj", "lmhead")),
    ("fwd+gelu", ("fc",)),
    ("dInp", ("qkv", "attproj", "fc", "fcproj", "lmhead")),
    ("dInp+dGeLU", ("fcproj",)),
    ("dW", ("qkv", "attproj", "fc", "fcproj", "lmhead")),
    ("dW+accum", ("qkv", "attproj", "fc", "fcproj", "lmhead")),
)

MATMUL_SELECTION_SHAPES = {
    "qkv": "qkv M=65536 N=2304 K=768 bias=1 gelu=0",
    "attproj": "attproj M=65536 N=768 K=768 bias=1 gelu=0",
    "fc": "fc M=65536 N=3072 K=768 bias=1 gelu=1",
    "fcproj": "fcproj M=65536 N=768 K=3072 bias=1 gelu=0",
    "lmhead": "lmhead M=65536 N=50304 K=768 bias=0 gelu=0",
}

ATTENTION_SELECTION_SHAPE = "B=64 T=1024 C=768 NH=12 HS=64"
LAYERNORM_SELECTION_SHAPE = "N=65536 C=768"
CLASSIFIER_SELECTION_SHAPE = "B=64 T=1024 V=50257 P=50304"

EXPECTED_RUNTIME_KERNELS = {
    "bias_add",
    "gelu_forward",
    "gelu_backward_inplace",
    "bias_grad_reduce",
    "fused_classifier",
    "global_norm_squared",
    "adamw_update",
    "encoder_forward",
    "cuda_memset",
    "cuda_copy_d2d",
}

RUNTIME_SHAPE_REQUIREMENTS = (
    ("bias_add", ("BT=65536 OC=768", "BT=65536 OC=3072")),
    ("bias_grad_reduce", ("BT=65536 OC=768", "BT=65536 OC=2304", "BT=65536 OC=3072")),
    ("cuda_memset", ("hidden_elems=50331648", "grad_elems=124475904", "logits_elems=3296722944")),
    ("cuda_copy_d2d", ("hidden_elems=50331648", "logits_elems=3296722944")),
)

LIBTORCH_RUNTIME_SHAPE_REQUIREMENTS = (
    ("cuda_memset", ("hidden_elems=50331648", "grad_elems=124475904", "logits_elems=3296722944")),
    ("cuda_copy_d2d", ("hidden_elems=50331648", "logits_elems=3296722944")),
)

LIBTORCH_RUNTIME_SUPPLEMENTAL_SHAPE_REQUIREMENTS = (
    ("gelu_forward", ("BT=65536 C=3072",)),
)

RUNTIME_SELECTION_SHAPES = {
    "gelu_forward": ("BT=65536 C=3072",),
    "gelu_backward_inplace": ("BT=65536 C=3072",),
    "fused_classifier": (CLASSIFIER_SELECTION_SHAPE,),
    "fused_classifier_loss": (CLASSIFIER_SELECTION_SHAPE,),
    "adamw_update": ("params=124475904 no-master",),
    "global_norm_squared": ("params=124475904",),
    "encoder_forward": ("B=64 T=1024 C=768",),
    **{kernel: shapes for kernel, shapes in RUNTIME_SHAPE_REQUIREMENTS},
}


def expected_trainer_selection_keys() -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for kernel, shape_names in MATMUL_SHAPE_REQUIREMENTS:
        for shape_name in shape_names:
            keys.add(("matmul", kernel, MATMUL_SELECTION_SHAPES[shape_name]))
    for kernel in ("forward", "backward"):
        keys.add(("attention", kernel, ATTENTION_SELECTION_SHAPE))
    for kernel in ("forward", "fused_residual_forward", "backward"):
        keys.add(("layernorm", kernel, LAYERNORM_SELECTION_SHAPE))
    for kernel, shapes in RUNTIME_SELECTION_SHAPES.items():
        for shape in shapes:
            keys.add(("runtime", kernel, shape))
    return keys

CORRECTNESS_TESTS = (
    "test_matmul",
    "test_attention",
    "test_layernorm",
    "test_bias",
    "test_gelu",
    "test_fused_classifier",
    "test_encoder",
    "test_adamw",
    "test_global_norm",
)

BENCHMARK_TARGETS = (
    "bench_sm120_matmul",
    "bench_sm120_attention",
    "bench_sm120_layernorm",
    "bench_sm120_runtime",
)

PYTHON_STACK_BENCHMARK_LOGS = (
    "bench_sm120_torch_matmul.log",
    "bench_sm120_cutedsl_matmul.log",
    "bench_sm120_triton_matmul.log",
    "bench_sm120_torch_attention.log",
    "bench_sm120_cudnn_attention.log",
    "bench_sm120_triton_attention.log",
    "bench_sm120_torch_classifier.log",
    "bench_sm120_triton_classifier.log",
    "bench_sm120_layernorm_python_stacks.log",
    "bench_sm120_triton_runtime.log",
    "bench_sm120_torch_runtime.log",
    "bench_sm120_libtorch_runtime.log",
)

LIBTORCH_TRAINER_LINK_LOG = "validate_libtorch_trainer_link.log"

TRAINING_TARGETS = ("train_gpt2cu",)

EXPECTED_MANIFEST_BINARIES = CORRECTNESS_TESTS + BENCHMARK_TARGETS + TRAINING_TARGETS
