import pytest


@pytest.mark.kernel
def test_adamw(kernel_runner):
    r = kernel_runner.run("test_adamw")
    assert r.passed, f"exit={r.exit_code}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
