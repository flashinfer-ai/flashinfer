from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep import Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_runner import (
    Sm90PushNvFp4MoERunner,
)
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.runner import (
    Sm90PushMoERunner,
)


class _FakeStream:
    cuda_stream = 1


def test_nvfp4_rs_runner_rejects_fused_activation(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import nvfp4_runner
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_weights import (
        Sm90PushNvFp4Weights,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.protocol import (
        Sm90PushCombine,
    )

    weights = object.__new__(Sm90PushNvFp4Weights)
    object.__setattr__(weights, "nvfp4_mode", "w4a16_rs")
    object.__setattr__(weights, "w13", object())
    object.__setattr__(weights, "w2", object())
    pipe = SimpleNamespace(
        config=SimpleNamespace(
            fuse_fc1_epilogue=False,
            combine_dtype=Sm90PushCombine.BF16,
            fuse_act=True,
        ),
        _comm=object(),
        rank=0,
    )
    monkeypatch.setattr(
        nvfp4_runner,
        "_run_guarded_phase",
        lambda _comm, _rank, _name, callback: callback(),
    )

    with pytest.raises(ValueError, match="W4A16-RS requires fuse_act=False"):
        Sm90PushNvFp4MoERunner(pipe, weights)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rs_n_tactic", 128),
        ("rs_stages", 2),
        ("rs_stage_k", 128),
    ],
)
def test_nvfp4_rs_runner_rejects_unsupported_tactic(monkeypatch, field, value):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import nvfp4_runner
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_weights import (
        Sm90PushNvFp4Weights,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.protocol import (
        Sm90PushCombine,
    )

    weights = object.__new__(Sm90PushNvFp4Weights)
    object.__setattr__(weights, "nvfp4_mode", "w4a16_rs")
    object.__setattr__(weights, "w13", object())
    object.__setattr__(weights, "w2", object())
    pipe = SimpleNamespace(
        config=SimpleNamespace(
            fuse_fc1_epilogue=False,
            combine_dtype=Sm90PushCombine.BF16,
            fuse_act=False,
        ),
        _comm=object(),
        rank=0,
    )
    monkeypatch.setattr(
        nvfp4_runner,
        "_run_guarded_phase",
        lambda _comm, _rank, _name, callback: callback(),
    )

    kwargs = {field: value}
    with pytest.raises(ValueError, match="supports only the N64/S3/K64 tactic"):
        Sm90PushNvFp4MoERunner(pipe, weights, **kwargs)


class _FakePipe:
    H = 8
    K = 2
    token_capacity = 4
    device = torch.device("cpu")
    out_dtype = torch.bfloat16

    def __init__(self):
        self.calls = []

    def proto_begin_round(self):
        self.calls.append("begin_round")

    def proto_dispatch(self, _x, _topk_ids, _topk_weights):
        self.calls.append("dispatch")

    def proto_wait_prefix(self):
        self.calls.append("wait_prefix")

    def proto_combine(self, _y, _meta):
        self.calls.append("combine")

    def proto_wait_combine(self):
        self.calls.append("wait_combine")

    def proto_reduce(self, _output, _num_tokens):
        self.calls.append("reduce")

    def proto_ack(self):
        self.calls.append("ack")

    def proto_abort(self):
        self.calls.append("abort")


class _FakeRunner(Sm90PushMoERunner):
    def __init__(self, pipe, *, fail_at=None, activation_stage="activation"):
        self._init_round_state(pipe)
        self.fail_at = fail_at
        self.activation_stage = activation_stage
        self.y = object()
        self.meta = object()

    def _current_stream(self):
        return _FakeStream(), True

    def _run_hook(self, name):
        self.pipe.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} failed")

    def _round_compact(self):
        self._run_hook("compact")

    def _round_fc1(self):
        self._run_hook("fc1")

    def _round_activation(self):
        self._run_hook("activation")

    def _round_activation_stage(self):
        return self.activation_stage

    def _round_fc2(self):
        self._run_hook("fc2")


def _inputs():
    return (
        torch.empty(2, 8, dtype=torch.bfloat16),
        torch.empty(2, 2, dtype=torch.int32),
        torch.empty(2, 2, dtype=torch.float32),
    )


def _run(runner):
    x, topk_ids, topk_weights = _inputs()
    output = torch.empty(2, 8, dtype=torch.bfloat16)
    runner.stage_inputs(x, topk_ids, topk_weights)
    return runner.compute(output=output)


def test_nvfp4_static_tactic_defaults():
    config = Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig(intermediate_size=128, top_k=2)
    assert config.nvfp4_mode == "w4a8"
    assert (config.rs_n_tactic, config.rs_stages, config.rs_stage_k) == (64, 3, 64)


def test_nvfp4_backend_freezes_rs_experiment_knobs(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import nvfp4_runner

    ffi_runner = SimpleNamespace(
        get_workspace_size=lambda *_args: 0,
        configure_workspace=lambda _workspace: None,
    )
    calls = []

    def _create(*args, **kwargs):
        calls.append((args, kwargs))
        return ffi_runner

    monkeypatch.setattr(
        nvfp4_runner,
        "create_sm90_push_nvfp4_rs_gemm_runner",
        _create,
    )
    runner = object.__new__(Sm90PushNvFp4MoERunner)
    runner._rs_n_tactic = 64
    runner._rs_stages = 3
    runner._rs_stage_k = 64
    runner._padded_max_rows = 128
    runner.pipe = SimpleNamespace(E=2, device=torch.device("cpu"))

    runner._new_rs_runner(128, 128)

    assert calls == [(("rs_wgmma", 64, 3, 64), {"use_environment": False})]


def test_nvfp4_runner_reuses_the_fp8_transaction_state_machine():
    assert Sm90PushNvFp4MoERunner.forward is Sm90PushMoERunner.forward
    assert Sm90PushNvFp4MoERunner.stage_inputs is Sm90PushMoERunner.stage_inputs
    assert Sm90PushNvFp4MoERunner.compute is Sm90PushMoERunner.compute


def test_round_order():
    pipe = _FakePipe()
    runner = _FakeRunner(pipe)
    output = _run(runner)
    assert pipe.calls == [
        "begin_round",
        "dispatch",
        "wait_prefix",
        "compact",
        "fc1",
        "activation",
        "fc2",
        "combine",
        "wait_combine",
        "reduce",
        "ack",
    ]
    assert output.shape == (2, 8)


def test_optional_activation_stage_is_omitted():
    pipe = _FakePipe()
    runner = _FakeRunner(pipe, activation_stage=None)
    _run(runner)
    assert "activation" not in pipe.calls


def test_mid_round_failure_aborts_and_poisons():
    pipe = _FakePipe()
    runner = _FakeRunner(pipe, fail_at="fc1")
    with pytest.raises(RuntimeError, match="fc1 failed"):
        _run(runner)
    assert pipe.calls[-1] == "abort"
    with pytest.raises(RuntimeError, match="poisoned"):
        _run(runner)


def test_validation_precedes_protocol_submission():
    pipe = _FakePipe()
    runner = _FakeRunner(pipe)
    x, topk_ids, topk_weights = _inputs()
    with pytest.raises(ValueError, match="x must be"):
        runner.stage_inputs(x.float(), topk_ids, topk_weights)
    assert pipe.calls == []
