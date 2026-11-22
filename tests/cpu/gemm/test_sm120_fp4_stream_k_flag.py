"""CPU checks for FLASHINFER_FP4_GEMM_DISABLE_STREAM_K on the SM120 FP4 runner."""

from types import SimpleNamespace

import pytest

import flashinfer.gemm.gemm_base as gemm_base


def _runner(monkeypatch, with_export, calls=None):
    module = SimpleNamespace(fp4_gemm=None, fp4_gemm_tactic_num=lambda: 8)
    if with_export:

        def is_stream_k(tactic):
            if calls is not None:
                calls.append(tactic)
            # Among tactics 0 to 7, CutlassFp4GemmRunner::getConfigs() marks 2, 3, 6
            # and 7 as stream-K. This stand-in marks 2 and 5, so the tests fail if the
            # runner applies a tactic % 4 >= 2 rule instead of asking the module.
            return tactic in (2, 5)

        module.fp4_gemm_tactic_is_stream_k = is_stream_k
    spec = SimpleNamespace(build_and_load=lambda: module)
    monkeypatch.setattr(gemm_base, "gen_gemm_sm120_module_cutlass_fp4", lambda: spec)
    gemm_base.get_gemm_sm120_module_cutlass_fp4.cache_clear()
    runner = gemm_base.get_gemm_sm120_module_cutlass_fp4().cutlass_fp4_gemm_runner()
    gemm_base.get_gemm_sm120_module_cutlass_fp4.cache_clear()
    return runner


@pytest.mark.cpu_only
@pytest.mark.parametrize("with_export", [True, False])
def test_flag_unset_keeps_every_tactic(monkeypatch, with_export):
    monkeypatch.delenv("FLASHINFER_FP4_GEMM_DISABLE_STREAM_K", raising=False)
    calls = []
    runner = _runner(monkeypatch, with_export, calls)
    assert runner.get_valid_tactics([], None) == list(range(8))
    assert runner.validate_tactic([], 2)
    # With the flag unset, fp4_gemm_tactic_is_stream_k is never called.
    assert calls == []


@pytest.mark.cpu_only
def test_flag_set_uses_the_exported_classifier(monkeypatch):
    monkeypatch.setenv("FLASHINFER_FP4_GEMM_DISABLE_STREAM_K", "1")
    runner = _runner(monkeypatch, with_export=True)
    runner_hash = hash(runner)
    assert runner.get_valid_tactics([], None) == [0, 1, 3, 4, 6, 7]
    # Building the stream-K set must not change hash(runner), which keys the
    # in-process autotune cache.
    assert hash(runner) == runner_hash
    assert not runner.validate_tactic([], 5)
    assert runner.validate_tactic([], 3)
    assert runner.validate_tactic([], -1)


@pytest.mark.cpu_only
def test_flag_set_on_a_module_without_the_export_raises(monkeypatch):
    monkeypatch.setenv("FLASHINFER_FP4_GEMM_DISABLE_STREAM_K", "1")
    runner = _runner(monkeypatch, with_export=False)
    with pytest.raises(RuntimeError, match="fp4_gemm_tactic_is_stream_k"):
        runner.get_valid_tactics([], None)
    # A tactic read back from the cache goes through validate_tactic, which also
    # raises.
    with pytest.raises(RuntimeError, match="fp4_gemm_tactic_is_stream_k"):
        runner.validate_tactic([], 3)


@pytest.mark.cpu_only
def test_flag_set_without_a_classifier_keeps_every_tactic(monkeypatch):
    # Like the SM100 and SM103 module getters, this passes no is_stream_k_tactic, so
    # the flag does not apply.
    monkeypatch.setenv("FLASHINFER_FP4_GEMM_DISABLE_STREAM_K", "1")
    module = SimpleNamespace(fp4_gemm=None, fp4_gemm_tactic_num=lambda: 8)
    runner = gemm_base._create_cutlass_fp4_gemm_module(
        module, "test::fp4_gemm", "test_fp4_gemm"
    ).cutlass_fp4_gemm_runner()
    assert runner.get_valid_tactics([], None) == list(range(8))
    assert runner.validate_tactic([], 2)
