"""
Regression test: importing autourgos_openaichat must not mutate process-wide
gRPC/TensorFlow/glog environment variables. configure_runtime_environment()
used to set GRPC_VERBOSITY/GLOG_minloglevel/TF_CPP_MIN_LOG_LEVEL and filter
gRPC UserWarnings globally, unconditionally, merely as an import-time side
effect -- irrelevant to a pure openai/httpx-based HTTP client with no gRPC/
TensorFlow/glog dependency anywhere. Run in a subprocess with a clean
environment since the current test process may already have these set from
an earlier (pre-fix) import in this session.
"""

import subprocess
import sys


def test_importing_package_does_not_set_grpc_tensorflow_env_vars():
    code = (
        "import os\n"
        "import autourgos_openaichat\n"
        "for var in ('GRPC_VERBOSITY', 'GLOG_minloglevel', 'TF_CPP_MIN_LOG_LEVEL'):\n"
        "    assert var not in os.environ, f'{var} was set: {os.environ[var]!r}'\n"
        "print('OK')\n"
    )
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("GRPC_VERBOSITY", "GLOG_minloglevel", "TF_CPP_MIN_LOG_LEVEL")}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_configure_runtime_environment_is_a_harmless_noop():
    """The function stays importable/callable (backward compat) but does nothing."""
    import os

    from autourgos_openaichat import configure_runtime_environment

    for var in ("GRPC_VERBOSITY", "GLOG_minloglevel", "TF_CPP_MIN_LOG_LEVEL"):
        os.environ.pop(var, None)

    configure_runtime_environment()

    for var in ("GRPC_VERBOSITY", "GLOG_minloglevel", "TF_CPP_MIN_LOG_LEVEL"):
        assert var not in os.environ
