import pytest


@pytest.mark.kernel
def test_fused_classifier(kernel_runner):
    r = kernel_runner.run("test_fused_classifier")
    assert r.passed, f"exit={r.exit_code}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
