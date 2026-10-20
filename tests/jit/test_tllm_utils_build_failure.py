import threading
import time

import pytest

import flashinfer.tllm_utils as tllm_utils


class _FailingSpec:
    builds = 0
    build_seconds = 0.0

    def build_and_load(self):
        type(self).builds += 1
        time.sleep(type(self).build_seconds)
        raise RuntimeError("Ninja build failed. Ninja output:\nfatal error: x.h")


@pytest.fixture
def failing_trtllm_utils_build(monkeypatch):
    _FailingSpec.builds = 0
    _FailingSpec.build_seconds = 0.0
    warnings = []
    monkeypatch.setattr(tllm_utils, "gen_trtllm_utils_module", _FailingSpec)
    monkeypatch.setattr(
        tllm_utils.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    tllm_utils._load_trtllm_utils_module.cache_clear()
    yield warnings
    tllm_utils._load_trtllm_utils_module.cache_clear()


def test_failed_trtllm_utils_build_is_attempted_and_reported_once(
    failing_trtllm_utils_build,
):
    for _ in range(3):
        with pytest.raises(RuntimeError, match="failed to build") as excinfo:
            tllm_utils.delay_kernel(10)
        assert "fatal error: x.h" in str(excinfo.value.__cause__)

    assert _FailingSpec.builds == 1
    assert len(failing_trtllm_utils_build) == 1
    assert "trtllm_utils" in failing_trtllm_utils_build[0]
    assert "fatal error: x.h" in failing_trtllm_utils_build[0]


def test_concurrent_first_calls_share_one_failed_build(failing_trtllm_utils_build):
    _FailingSpec.build_seconds = 0.05
    start = threading.Barrier(4)
    errors = []

    def call():
        start.wait()
        try:
            tllm_utils.delay_kernel(10)
        except RuntimeError as e:
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 4
    assert _FailingSpec.builds == 1
    assert len(failing_trtllm_utils_build) == 1


def test_trtllm_utils_module_is_built_once_on_success(monkeypatch):
    builds = []
    delays = []

    class _Module:
        def delay_kernel(self, us):
            delays.append(us)

    class _Spec:
        def build_and_load(self):
            builds.append(1)
            return _Module()

    monkeypatch.setattr(tllm_utils, "gen_trtllm_utils_module", _Spec)
    tllm_utils._load_trtllm_utils_module.cache_clear()
    try:
        tllm_utils.delay_kernel(5)
        tllm_utils.delay_kernel(7)
    finally:
        tllm_utils._load_trtllm_utils_module.cache_clear()

    assert builds == [1]
    assert delays == [5, 7]
