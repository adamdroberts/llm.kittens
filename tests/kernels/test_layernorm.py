import pytest


@pytest.mark.kernel
def test_layernorm(kernel_runner):
    r = kernel_runner.run("test_layernorm")
    assert r.passed, f"exit={r.exit_code}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
