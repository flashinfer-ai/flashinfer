from types import SimpleNamespace

import pytest

from tests import conftest
from flashinfer.jit import JitSpecCuteDsl, MissingJITCacheError


@pytest.mark.cpu_only
def test_missing_cute_dsl_cache_is_reported_and_skipped(monkeypatch):
    monkeypatch.delenv("FLASHINFER_STRICT_MOE_EP_TESTS", raising=False)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    spec = JitSpecCuteDsl("test_module", "test_kernel", lambda: None, "source-hash")
    error = MissingJITCacheError("missing CuTe-DSL artifact", spec=spec)
    item = SimpleNamespace(nodeid="tests/test_example.py::test_kernel")
    previous_missing = conftest._MISSING_JIT_CACHE_MODULES.copy()

    hook = conftest.pytest_runtest_call(item)
    next(hook)
    try:
        with pytest.raises(pytest.skip.Exception, match=spec.name):
            hook.throw(error)
        assert (
            item.nodeid,
            spec.name,
            str(
                {
                    "name": spec.name,
                    "artifact_path": str(spec.get_library_path()),
                }
            ),
        ) in conftest._MISSING_JIT_CACHE_MODULES
    finally:
        conftest._MISSING_JIT_CACHE_MODULES.clear()
        conftest._MISSING_JIT_CACHE_MODULES.update(previous_missing)
