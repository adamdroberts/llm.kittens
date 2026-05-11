import pytest


@pytest.mark.kernel
def test_attention_gqa(kernel_runner):
    r = kernel_runner.run("test_attention_gqa")
    assert r.passed, f"exit={r.exit_code}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
