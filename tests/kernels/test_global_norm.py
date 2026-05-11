import pytest


@pytest.mark.kernel
def test_global_norm(kernel_runner):
    r = kernel_runner.run("test_global_norm")
    assert r.passed, f"exit={r.exit_code}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
