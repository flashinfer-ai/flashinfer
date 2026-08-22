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
    assert config.payload_layout == 4
    assert config.tma_cache_capacity == 128


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
