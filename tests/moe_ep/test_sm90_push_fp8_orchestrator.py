"""CPU-only state-machine checks for the SM90 push runner."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest


_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "flashinfer"
    / "moe_ep"
    / "kernel_src"
    / "sm90_push_megamoe"
)


class _Viewable:
    def view(self, dtype):
        return self


class _FakeStream:
    def __init__(self, stream_id, log):
        self.cuda_stream = stream_id
        self._log = log

    def wait_event(self, _event):
        self._log.append(f"wait_event:{self.cuda_stream}")


class _FakeEvent:
    def __init__(self, log, *, complete=True):
        self._log = log
        self._complete = complete

    def query(self):
        return self._complete

    def record(self, stream):
        self._log.append(f"event_record:{stream.cuda_stream}")


class _FakeAbortStream:
    def __init__(self, log):
        self.log = log

    def synchronize(self):
        self.log.append("abort_sync")


class _FakeAbortModule:
    def __init__(self, log, *, fail=False):
        self.log = log
        self.fail = fail

    def sm90_push_publish_abort(self, *args):
        self.log.append(("publish_abort", args))
        if self.fail:
            raise RuntimeError("sticky CUDA error")


class _FakeGemmRunner:
    def __init__(self, log):
        self._log = log
        self._calls = 0

    def moe_gemm(self, *args):
        self._calls += 1
        self._log.append("fc1" if self._calls == 1 else "fc2")


class _FakePipe:
    def __init__(self, log):
        self.log = log
        self.H = 128
        self.K = 2
        self.token_capacity = 64
        self.out_dtype = object()
        self.device = object()
        self.config = SimpleNamespace(fuse_fc1_epilogue=False, fuse_act=True)
        self._offsets = object()
        self._pad_base = object()
        self._m_dev = object()
        self._p_dev = object()
        self.module = SimpleNamespace()

    def proto_begin_round(self):
        self.log.append("begin")

    def proto_dispatch(self, *args):
        self.log.append("dispatch")

    def proto_wait_prefix(self):
        self.log.append("wait_prefix")

    def proto_compact(self, *args):
        self.log.append("compact")

    def proto_silu_mul_quant(self, *args):
        self.log.append("act")

    def proto_combine(self, *args):
        self.log.append("combine")

    def proto_wait_combine(self):
        self.log.append("wait_combine")

    def proto_reduce(self, output, num_tokens):
        self.log.append("reduce")
        self.reduced_output = output
        self.reduced_tokens = num_tokens
        return output

    def proto_ack(self):
        self.log.append("ack")

    def proto_abort(self):
        self.log.append("abort")

    def destroy(self):
        self.log.append("destroy")


def _runner():
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import runner as module

    log = []
    pipe = _FakePipe(log)
    runner = object.__new__(module.Sm90PushMoERunner)
    runner.pipe = pipe
    runner._state = module._RunnerState.IDLE
    runner._staged_tokens = None
    runner._staged_stream = None
    runner._staged_stream_capturing = False
    runner._staged_output = None
    runner._caller_ready_event = _FakeEvent(log)
    runner._round_event = _FakeEvent(log)
    runner._round_stream_id = None
    runner.record_stages = False
    runner._test_current_stream = _FakeStream(17, log)
    runner._current_stream = lambda: (runner._test_current_stream, True)

    def stream_context(stream):
        log.append(f"stream_context:{stream.cuda_stream}")
        return contextlib.nullcontext()

    runner._stream_context = stream_context
    runner._validate_round_inputs = lambda x, ids, weights: 3
    runner._validate_output = lambda output, num_tokens: None
    runner.runner = _FakeGemmRunner(log)
    runner.I = 128
    runner.a1 = _Viewable()
    runner.sfa1 = object()
    runner.meta = object()
    runner.row_expert = object()
    runner.h = object()
    runner.a2 = _Viewable()
    runner.sfa2 = object()
    runner.y = object()
    runner.w13_fp8 = object()
    runner.w13_sf = object()
    runner.w2_fp8 = object()
    runner.w2_sf = object()
    runner._workspace = object()
    return runner, pipe, log


@pytest.mark.parametrize(
    ("rank", "prepare_phase"),
    (
        (0, "gemm-jit-cache-load"),
        (1, "gemm-jit-cache-warm"),
        (2, "gemm-jit-cache-warm"),
        (3, "gemm-jit-cache-warm"),
    ),
)
def test_gemm_jit_collective_elects_one_leader_per_cache_and_tactic_identity(
    monkeypatch, rank, prepare_phase
):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import runner as module

    cache_identities = [
        ("host-a", "cache-a", 9, 0, 132, True, 128, 4, 256, 256, False, False),
        ("host-a", "cache-a", 9, 0, 132, True, 128, 4, 256, 256, False, True),
        ("host-b", "cache-a", 9, 0, 132, True, 128, 4, 256, 256, False, False),
        ("host-a", "cache-a", 9, 0, 132, True, 128, 4, 256, 512, False, False),
    ]
    instance = object.__new__(module.Sm90PushMoERunner)
    instance.pipe = SimpleNamespace(rank=rank, _comm=object())
    instance._gemm_jit_cache_identity = lambda: cache_identities[rank]
    calls = []
    phase = [None]
    instance._prepare_gemm_jit = lambda: calls.append(phase[0])

    def guarded(_comm, _rank, name, fn):
        phase[0] = name
        payload = fn()
        if name == "gemm-jit-cache-layout":
            return cache_identities
        return [payload] * len(cache_identities)

    monkeypatch.setattr(module, "_run_guarded_phase", guarded)
    instance._prepare_gemm_jit_collective()

    assert calls == [prepare_phase]


def test_gemm_jit_cache_identity_includes_tactic_inputs(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import runner as module

    instance = object.__new__(module.Sm90PushMoERunner)
    instance.pipe = SimpleNamespace(
        device=object(),
        token_capacity=64,
        K=2,
        E=4,
        H=256,
        config=SimpleNamespace(fuse_fc1_epilogue=False),
    )
    instance.I = 256
    instance.runner = SimpleNamespace(
        get_deepgemm_cache_dir=lambda: str(tmp_path / "nested" / ".."),
        get_deepgemm_nvcc_compiler=lambda: "/missing/nvcc",
        is_deepgemm_jit_enabled=lambda: True,
    )
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(major=9, minor=0, multi_processor_count=78),
    )
    monkeypatch.setattr(instance, "_nvcc_available", lambda _command: False)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "host-a")

    assert instance._gemm_jit_cache_identity() == (
        "host-a",
        module.os.path.normcase(module.os.path.realpath(str(tmp_path))),
        9,
        0,
        78,
        True,
        128,
        4,
        256,
        256,
        False,
        False,
    )


def test_gemm_jit_allows_warm_cache_without_nvcc(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import runner as module

    instance = object.__new__(module.Sm90PushMoERunner)
    instance.pipe = SimpleNamespace(
        H=256,
        config=SimpleNamespace(fuse_fc1_epilogue=True),
    )
    instance.I = 256
    instance.runner = SimpleNamespace(
        is_deepgemm_jit_enabled=lambda: True,
        is_moe_gemm_fc1_fused_jit_cache_ready=lambda _n, _k: True,
        is_moe_gemm_jit_cache_ready=lambda _n, _k: True,
        get_deepgemm_nvcc_compiler=lambda: "/missing/nvcc",
        get_deepgemm_cache_dir=lambda: "/warm/cache",
    )
    monkeypatch.delenv("TRTLLM_DG_ENABLED", raising=False)
    monkeypatch.setattr(instance, "_nvcc_available", lambda _command: False)

    instance._require_gemm_jit_runtime()


def test_gemm_jit_rejects_cold_cache_without_nvcc(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import runner as module

    instance = object.__new__(module.Sm90PushMoERunner)
    instance.pipe = SimpleNamespace(
        H=256,
        config=SimpleNamespace(fuse_fc1_epilogue=False),
    )
    instance.I = 256
    instance.runner = SimpleNamespace(
        is_deepgemm_jit_enabled=lambda: True,
        is_moe_gemm_jit_cache_ready=lambda _n, _k: False,
        get_deepgemm_nvcc_compiler=lambda: "/missing/nvcc",
        get_deepgemm_cache_dir=lambda: "/cold/cache",
    )
    monkeypatch.delenv("TRTLLM_DG_ENABLED", raising=False)
    monkeypatch.setattr(instance, "_nvcc_available", lambda _command: False)

    with pytest.raises(
        RuntimeError,
        match=r"cold DeepGEMM cache miss.*TRTLLM_DG_ENABLED=0",
    ):
        instance._require_gemm_jit_runtime()


def test_fused_fc1_rejects_disabled_deepgemm(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import runner as module

    instance = object.__new__(module.Sm90PushMoERunner)
    instance.pipe = SimpleNamespace(
        config=SimpleNamespace(fuse_fc1_epilogue=True),
    )
    instance.runner = SimpleNamespace(is_deepgemm_jit_enabled=lambda: False)
    monkeypatch.setenv("TRTLLM_DG_ENABLED", "0")

    with pytest.raises(RuntimeError, match="requires DeepGEMM JIT"):
        instance._require_gemm_jit_runtime()


def test_runner_preserves_protocol_and_compute_order():
    runner, pipe, log = _runner()
    output = object()

    runner.stage_inputs(object(), object(), object(), output=output)
    result = runner.compute(output=output)

    assert result is output
    assert pipe.reduced_output is output
    assert pipe.reduced_tokens == 3
    assert log == [
        "begin",
        "dispatch",
        "wait_prefix",
        "compact",
        "fc1",
        "act",
        "fc2",
        "combine",
        "wait_combine",
        "reduce",
        "ack",
    ]
    assert runner.state == "idle"
    assert runner._staged_output is None


def test_runner_rejects_compute_before_stage_and_double_stage():
    runner, _, _ = _runner()
    output = object()
    with pytest.raises(RuntimeError, match="preceding stage_inputs"):
        runner.compute(output=output)

    runner.stage_inputs(object(), object(), object(), output=output)
    with pytest.raises(RuntimeError, match="called twice"):
        runner.stage_inputs(object(), object(), object(), output=object())
    assert runner.state == "staged"


def test_runner_rejects_cross_stream_round_while_previous_event_is_incomplete():
    runner, _, log = _runner()
    runner._round_event = _FakeEvent(log, complete=False)
    runner._round_stream_id = 17
    runner._test_current_stream = _FakeStream(23, log)
    runner._current_stream = lambda: (runner._test_current_stream, False)

    with pytest.raises(RuntimeError, match="previous round is still executing"):
        runner.stage_inputs(object(), object(), object(), output=object())

    assert runner.state == "idle"
    assert log == []


def test_runner_rejects_invalid_output_before_dispatch():
    runner, _, log = _runner()
    invalid_output = object()

    def reject_output(output, num_tokens):
        assert output is invalid_output
        assert num_tokens == 3
        raise ValueError("invalid output")

    runner._validate_output = reject_output
    with pytest.raises(ValueError, match="invalid output"):
        runner.stage_inputs(object(), object(), object(), output=invalid_output)

    assert runner.state == "idle"
    assert runner._staged_output is None
    assert log == []


def test_runner_finishes_round_before_rejecting_a_different_output():
    runner, _, log = _runner()
    staged_output = object()

    runner.stage_inputs(object(), object(), object(), output=staged_output)
    with pytest.raises(RuntimeError, match="same output tensor"):
        runner.compute(output=object())

    assert runner.state == "idle"
    assert runner._staged_output is None
    assert log == [
        "begin",
        "dispatch",
        "wait_prefix",
        "compact",
        "fc1",
        "act",
        "fc2",
        "combine",
        "wait_combine",
        "reduce",
        "ack",
    ]


def test_runner_poisoned_after_mid_round_failure():
    runner, pipe, log = _runner()
    output = object()

    def fail_wait_prefix():
        log.append("wait_prefix")
        raise RuntimeError("injected failure")

    pipe.proto_wait_prefix = fail_wait_prefix
    runner.stage_inputs(object(), object(), object(), output=output)
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.compute(output=output)

    assert runner.state == "poisoned"
    assert log == ["begin", "dispatch", "wait_prefix", "abort"]
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.stage_inputs(object(), object(), object(), output=output)


def test_runner_completes_on_staged_stream_before_joining_caller_stream():
    runner, _, log = _runner()
    output = object()
    runner.stage_inputs(object(), object(), object(), output=output)
    runner._test_current_stream = _FakeStream(23, log)

    assert runner.compute(output=output) is output

    assert runner.state == "idle"
    assert runner._staged_stream is None
    assert log == [
        "begin",
        "dispatch",
        "event_record:23",
        "wait_event:17",
        "stream_context:17",
        "wait_prefix",
        "compact",
        "fc1",
        "act",
        "fc2",
        "combine",
        "wait_combine",
        "reduce",
        "ack",
        "event_record:17",
        "wait_event:23",
    ]


def test_runner_destroy_is_idempotent():
    runner, _, log = _runner()
    runner._staged_stream = _FakeStream(17, log)
    runner._staged_stream_capturing = True
    runner._staged_output = object()
    runner.destroy()
    runner.destroy()
    assert runner.state == "destroyed"
    assert runner._staged_stream is None
    assert not runner._staged_stream_capturing
    assert runner._staged_output is None
    assert log == ["destroy"]


@pytest.mark.parametrize("round_open", [True, False])
def test_pipe_abort_uses_control_stream_and_device_anchor(monkeypatch, round_open):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import protocol

    log = []
    pipe = object.__new__(protocol.Sm90PushPipe)
    pipe._poisoned = False
    pipe._round_open = round_open
    pipe._abort_stream = _FakeAbortStream(log)
    pipe._round = object()
    pipe.module = _FakeAbortModule(log)
    pipe._layout_args = lambda: ("layout",)

    def use_stream(stream):
        assert stream is pipe._abort_stream
        log.append("abort_stream")
        return contextlib.nullcontext()

    monkeypatch.setattr(protocol.torch.cuda, "stream", use_stream)
    pipe.proto_abort()
    pipe.proto_abort()

    assert pipe._poisoned is True
    assert pipe._round_open is False
    assert log == [
        "abort_stream",
        ("publish_abort", ("layout", pipe._round)),
        "abort_sync",
    ]


def test_pipe_abort_keeps_local_poison_when_cuda_is_sticky(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import protocol

    log = []
    pipe = object.__new__(protocol.Sm90PushPipe)
    pipe._poisoned = False
    pipe._round_open = True
    pipe._abort_stream = _FakeAbortStream(log)
    pipe._round = object()
    pipe.module = _FakeAbortModule(log, fail=True)
    pipe._layout_args = lambda: ("layout",)
    monkeypatch.setattr(
        protocol.torch.cuda,
        "stream",
        lambda stream: contextlib.nullcontext(),
    )

    pipe.proto_abort()

    assert pipe._poisoned is True
    assert pipe._round_open is False
    assert log == [("publish_abort", ("layout", pipe._round))]


def test_abort_cuda_contract_is_wired_through_waits_and_traps():
    header = (_PACKAGE_ROOT / "src" / "a2a" / "sm90_push_a2a.cuh").read_text(
        encoding="utf-8"
    )
    ops = (_PACKAGE_ROOT / "src" / "a2a" / "sm90_push_a2a_ops.cu").read_text(
        encoding="utf-8"
    )
    protocol = (_PACKAGE_ROOT / "shim" / "protocol.py").read_text(encoding="utf-8")

    assert "pool_head_offset + kPoolHeadStorageBytes" in header
    assert "kPoolHeadStorageBytes + kAbortCellStorageBytes <= 128" in header
    assert "abort_cells_offset" not in header + ops + protocol
    assert "pool_head_padding_end = _align(off + 8)" in protocol
    assert "torch.cuda.Stream(device=dv, priority=-1)" in protocol
    assert "with torch.cuda.stream(self._abort_stream)" in protocol
    assert "_host_round" not in protocol
    assert "sm90_push_publish_abort" in protocol
    assert "publish_abort_kernel<<<1, kMaxEpSize" in ops
    assert "pack_count_tag(L.rank + 1, 0)" in ops
    assert "publish_abort_kernel(PushLayout L," not in ops
    assert (
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(sm90_push_publish_abort, "
        "sm90_push_publish_abort);"
    ) in ops
    assert "abort_rank_plus_one != 0" in header
    assert "mov.u64 %0, %%globaltimer;" in header
    assert "300ull * 1000ull * 1000ull * 1000ull" in header
    assert "clock64()" not in header
    assert "uint32_t tag, uint32_t abort_tag" not in header
    assert "wait_tag_u64(L, L.ack_cell(L.rank, d), want)" in ops
    assert "wait_tag_u64(L, cells + i, tag)" in ops
    assert "wait_tag_u64(L, L.cdone_cell(L.rank, s), tag)" in ops
    assert "acc > static_cast<int64_t>(m_cap)" in ops
    assert "accumulated row count %lld exceeds m_cap %d" in ops
    assert "count_kernel<K>" in ops
    assert ">(L, ids, E_total, lcp, loffp, rc, lt);" in ops

    for source in (header, ops):
        lines = source.splitlines()
        trap_lines = [
            i for i, line in enumerate(lines) if 'asm volatile("trap;")' in line
        ]
        assert trap_lines
        for line_number in trap_lines:
            assert any(
                "publish_abort_all" in line
                for line in lines[max(0, line_number - 8) : line_number]
            )
