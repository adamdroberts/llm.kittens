#!/usr/bin/env python3
"""Source-level guards for the opt-in GPT-2 bias+GELU GEMM epilogue."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TK_GEMM = ROOT / "llmc" / "tk" / "gemm_h100.cuh"
MATMUL = ROOT / "llmc" / "matmul.cuh"
GPT2 = ROOT / "train_gpt2.cu"
SMOKE = ROOT / "dev" / "cuda" / "test_matmul.cu"
HARNESS = ROOT / "scripts" / "validate_goal_h100.sh"
PROFILE_SCRIPT = ROOT / "profile_gpt2cu.py"
PROFILE_BINARY = ROOT / "profile_gpt2.cu"

LARGER_GPT_SCRIPTS = [
    ROOT / "scripts" / "run_gpt2_350M.sh",
    ROOT / "scripts" / "run_gpt2_774M.sh",
    ROOT / "scripts" / "run_gpt2_1558M.sh",
    ROOT / "scripts" / "run_gpt3_125M.sh",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def extract_function(text: str, marker: str) -> str:
    start = text.find(marker)
    if start == -1:
        raise AssertionError(f"missing function marker {marker!r}")
    body_start = text.find("{", start)
    if body_start == -1:
        raise AssertionError(f"missing function body for {marker!r}")
    depth = 1
    i = body_start + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise AssertionError(f"unterminated function {marker!r}")
    return text[body_start + 1 : i - 1]


def require_contains(text: str, needle: str, context: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"{context} missing {needle!r}")


def require_absent(text: str, needle: str, context: str, failures: list[str]) -> None:
    if needle in text:
        failures.append(f"{context} unexpectedly contains {needle!r}")


def require_all(text: str, needles: list[str], context: str, failures: list[str]) -> None:
    for needle in needles:
        require_contains(text, needle, context, failures)


def require_order(text: str, needles: list[str], context: str, failures: list[str]) -> None:
    cursor = -1
    for needle in needles:
        offset = text.find(needle)
        if offset == -1:
            failures.append(f"{context} missing {needle!r}")
            continue
        if offset <= cursor:
            failures.append(f"{context} order regression at {needle!r}")
        cursor = offset


def validate_tk_gemm(tk_gemm: str, failures: list[str]) -> None:
    require_all(
        tk_gemm,
        [
            "__device__ static inline void apply_gelu",
            "float cube = 0.044715f * v * v * v;",
            "if constexpr (APPLY_BIAS)",
            "if constexpr (STORE_PRE_GELU)",
            "tma::store_async(args.globals.P",
            "if constexpr (APPLY_GELU)",
            "apply_gelu<wide_tile>(c_wide);",
            "using matmul_default_nt_bias_gelu = matmul_template<2, 4, 8, false, true, true, true, true>;",
            "using matmul_small_n_nt_bias_gelu = matmul_template<2, 2, 8, false, true, true, true, true>;",
            "cudaStream_t stream = 0, bf16* d_pre_gelu = nullptr,",
            "global_layout Pg{ d_pre_gelu == nullptr ? d_C : d_pre_gelu, nullptr, nullptr, M_, N_ };",
            "globals G{ Ag, Bg, Cg, Pg, d_bias };",
        ],
        rel(TK_GEMM),
        failures,
    )


def validate_matmul(matmul: str, failures: list[str]) -> None:
    default_forward = extract_function(matmul, "inline void matmul_forward(")
    fused_forward = extract_function(matmul, "inline void matmul_forward_gelu(")
    support_gate = extract_function(matmul, "inline bool matmul_forward_gelu_supported")

    require_all(
        support_gate,
        ["return matmul_tk_shape_ok(B * T, OC, C);"],
        f"{rel(MATMUL)} matmul_forward_gelu_supported",
        failures,
    )
    require_all(
        default_forward,
        [
            "llmk::gemm::matmul_default_nt",
            "llmk::gemm::matmul_small_n_nt",
            "add_bias(out, bias, M, OC, stream);",
        ],
        f"{rel(MATMUL)} default forward",
        failures,
    )
    require_all(
        fused_forward,
        [
            "assert(out != nullptr);",
            "assert(pre_gelu != nullptr);",
            'assert(bias != nullptr && "matmul_forward_gelu: fused path expects a bias vector");',
            "assert(matmul_forward_gelu_supported(B, T, C, OC));",
            "auto* P_= llmk::to_bf16(pre_gelu);",
            "auto* bias_ = llmk::to_bf16(const_cast<floatX*>(bias));",
            "llmk::gemm::launch<llmk::gemm::matmul_default_nt_bias_gelu>",
            "llmk::gemm::launch<llmk::gemm::matmul_small_n_nt_bias_gelu>",
            "A, B_, C_, M, N, K, stream, P_, bias_",
        ],
        f"{rel(MATMUL)} fused forward",
        failures,
    )


def validate_gpt2(gpt2: str, failures: list[str]) -> None:
    forward = extract_function(gpt2, "void gpt2_forward(")
    backward = extract_function(gpt2, "void gpt2_backward_and_reduce(")
    dry_run = extract_function(gpt2, "void gpt2_dry_run_validate(")

    require_all(
        gpt2,
        [
            "int gelu_fusion; // TK GEMM epilogue GELU fusion (0=none, 1=forward, 2=reserved)",
            "model->gelu_fusion = 0; // default: off until the fused TK epilogue has H100 numerical validation",
            'fprintf(stderr, "  -ge <int>   gelu fusion: 0=none, 1=TK forward epilogue, 2=reserved (default: 0)\\n");',
            "int gelu_fusion = -1; // 0 = none, 1 = TK forward epilogue, 2 = reserved (-1 => per-GPU default)",
            "else if (argv[i][1] == 'g' && argv[i][2] == 'e') { gelu_fusion = atoi(argv[i+1]); }",
            "if (gelu_fusion == -1) { gelu_fusion = 0; }",
            'printf0("| gelu_fusion           | %-50d |\\n", gelu_fusion);',
            "model.gelu_fusion = gelu_fusion;",
        ],
        rel(GPT2),
        failures,
    )
    require_all(
        dry_run,
        [
            "model.gelu_fusion = gelu_fusion;",
            "gpt2_validate_zero_tensor_sharding",
        ],
        f"{rel(GPT2)} dry-run descriptor",
        failures,
    )
    require_order(
        forward,
        [
            "if (model->gelu_fusion >= 1 && matmul_forward_gelu_supported(B, T, C, 4 * C))",
            "matmul_forward_gelu(l_fch_gelu, l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C, main_stream);",
            "} else {",
            "matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, main_stream);",
            "gelu_forward(l_fch_gelu, l_fch, B*T*4*C, main_stream);",
            "matmul_forward(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, main_stream);",
        ],
        f"{rel(GPT2)} forward epilogue routing",
        failures,
    )
    require_order(
        backward,
        [
            "floatX* l_fch_pre_gelu = acts.fch + l * B * T * 4*C;",
            "if(model->recompute >= 1)",
            "gelu_forward(l_fch_gelu, l_fch_pre_gelu, B*T*4*C, main_stream);",
            "matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw",
            "gelu_backward_inplace(dl_bt4c, l_fch_pre_gelu, B*T*4*C, main_stream);",
        ],
        f"{rel(GPT2)} backward pre-GELU use",
        failures,
    )


def validate_smoke(smoke: str, failures: list[str]) -> None:
    require_all(
        smoke,
        [
            "naive_gemm_bias_gelu_ref",
            'Shape s = {1024, 4096, 1024, "forward bias+GELU epilogue (default)"};',
            "matmul_forward_gelu(dGelu_tk, dPre_tk, dA, dW, dBias,",
            "naive_gemm_bias_gelu_ref<<<grid, block>>>(dA, dW, dBias, dPre_ref, dGelu_ref",
            "double pre_diff = max_abs_diff(hPre_tk, hPre_ref);",
            "double gelu_diff = max_abs_diff(hGelu_tk, hGelu_ref);",
            'printf("test_matmul smoke OK\\n");',
        ],
        rel(SMOKE),
        failures,
    )


def validate_scripts(failures: list[str]) -> None:
    for script in LARGER_GPT_SCRIPTS:
        require_contains(script.read_text(), "-ge 1", rel(script), failures)
    require_absent(
        (ROOT / "scripts" / "run_gpt2_124M.sh").read_text(),
        "-ge 1",
        "scripts/run_gpt2_124M.sh conservative default",
        failures,
    )


def validate_harness(harness: str, failures: list[str]) -> None:
    require_all(
        harness,
        [
            "test_matmul test_attention",
            'run_contains "$bin smoke OK" "./$bin"',
            "dev/validate_epilogue_source.py",
            'run_contains "GELU epilogue source guards OK" python3 dev/validate_epilogue_source.py',
        ],
        rel(HARNESS),
        failures,
    )


def validate_profile_path(profile_script: str, profile_binary: str, failures: list[str]) -> None:
    require_all(
        profile_script,
        [
            'parser.add_argument("--gelu-fusion", type=int, choices=(0, 1), default=0',
            '"--gelu-fusion", str(args.gelu_fusion)',
        ],
        rel(PROFILE_SCRIPT),
        failures,
    )
    require_all(
        profile_binary,
        [
            'strcmp(argv[i], "--gelu-fusion") == 0 || strcmp(argv[i], "-ge") == 0',
            "model.gelu_fusion = gelu_fusion;",
            'printf("gelu fusion: %d\\n", gelu_fusion);',
        ],
        rel(PROFILE_BINARY),
        failures,
    )


def main() -> None:
    failures: list[str] = []
    validate_tk_gemm(TK_GEMM.read_text(), failures)
    validate_matmul(MATMUL.read_text(), failures)
    validate_gpt2(GPT2.read_text(), failures)
    validate_smoke(SMOKE.read_text(), failures)
    validate_scripts(failures)
    validate_harness(HARNESS.read_text(), failures)
    validate_profile_path(PROFILE_SCRIPT.read_text(), PROFILE_BINARY.read_text(), failures)
    if failures:
        raise AssertionError("\n".join(failures))
    print("GELU epilogue source guards OK")


if __name__ == "__main__":
    main()
