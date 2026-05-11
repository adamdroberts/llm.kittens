import pytest


@pytest.mark.kernel
def test_rope(kernel_runner):
    r = kernel_runner.run("test_rope")
    assert r.passed, f"exit={r.exit_code}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
