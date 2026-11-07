# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only regressions for MX storage, repeated routes and compiler devices."""

from contextlib import nullcontext
from dataclasses import dataclass, replace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner


@dataclass
class _TensorMetadata:
    shape: tuple
    dtype: torch.dtype
    device: torch.device = torch.device("cuda:1")
    pointer: int = 256
    is_cuda: bool = True

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self.pointer

    def view(self, dtype):
        return replace(self, dtype=dtype)

    def to(self, dtype):
        return replace(self, dtype=dtype)


def _runner(monkeypatch, weight_format, experts=4, top_k=2):
    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda _: (10, 0))
    monkeypatch.setattr(
        "flashinfer.fused_moe.backends.frost_low_latency.runner.version",
        lambda _: "4.8.0",
    )
    config = MoEConfig(
        RoutingConfig(experts, top_k),
        QuantConfig(weight=weight_format, activation=QuantFormat.MXFP8),
        ExpertConfig(128),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    return FrostLowLatencyRunner(config, torch.device("cuda:1"))


def _packs(monkeypatch, runner):
    e, k = runner.config.routing.num_experts, runner.config.routing.top_k
    fp4 = runner.config.quant.weight is QuantFormat.MXFP4
    dtype, pack = (torch.uint8, 2) if fp4 else (torch.float8_e4m3fn, 1)
    values = {
        "w1_weight": _TensorMetadata((e, 256, 256 // pack), dtype),
        "w2_weight": _TensorMetadata((e, 256, 128 // pack), dtype),
        "w1_weight_sf": _TensorMetadata((e, 2, 2, 32, 4, 4), torch.uint8),
        "w2_weight_sf": _TensorMetadata((e, 2, 1, 32, 4, 4), torch.uint8),
    }
    act = MoEActivationPack(
        _TensorMetadata((2, 256), torch.float8_e4m3fn),
        _TensorMetadata((2, 8), torch.uint8),
        _TensorMetadata((2, k), torch.int32),
        _TensorMetadata((2, k), torch.float32),
    )
    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, dtype, device: _TensorMetadata(
            (shape,) if isinstance(shape, int) else shape, dtype, device
        ),
    )
    return act, MoEWeightPack({"frost_low_latency": values}), values


@pytest.mark.parametrize("fmt", [QuantFormat.MXFP4, QuantFormat.MXFP8])
@pytest.mark.parametrize(
    "name", ["w1_weight", "w2_weight", "w1_weight_sf", "w2_weight_sf"]
)
@pytest.mark.parametrize("offset", [0, 8, 16])
def test_mx_alignment_matches_tensor_copy_format(monkeypatch, fmt, name, offset):
    runner = _runner(monkeypatch, fmt)
    runner.check_support()
    runner.build()
    act, pack, values = _packs(monkeypatch, runner)
    values[name].pointer += offset
    alignment = 32 if fmt is QuantFormat.MXFP4 and name.endswith("weight") else 16
    if offset % alignment:
        with pytest.raises(
            ValueError, match=f"{name} must be {alignment}-byte aligned"
        ):
            runner.pack_inputs(act, pack)
    else:
        assert len(runner.pack_inputs(act, pack)) == 15


@pytest.mark.parametrize("fmt", [QuantFormat.MXFP4, QuantFormat.MXFP8])
@pytest.mark.parametrize("experts,top_k", [(4, 2), (4, 6), (4, 32), (4, 33), (40, 40)])
def test_mx_repeated_routes_have_an_executable_default(
    monkeypatch, fmt, experts, top_k
):
    runner = _runner(monkeypatch, fmt, experts, top_k)
    if top_k > experts and top_k > 32:
        with pytest.raises(NotImplementedError, match="top_k.*32"):
            runner.check_support()
        return
    runner.check_support()
    runner.build()
    inputs = [_TensorMetadata((2, 256), torch.float8_e4m3fn)]
    from flashinfer.fused_moe.backends.frost_low_latency import runtime

    monkeypatch.setattr(runtime, "execute", lambda inputs, config, tactic: tactic)
    choices = runner.get_valid_tactics(inputs, None)
    if top_k > experts:
        assert choices == [("ffma", 1)]
        assert runner.forward(inputs) == ("ffma", 1)
        with pytest.raises(ValueError, match="top_k <= num_experts"):
            runner.forward(inputs, tactic=128)
    else:
        assert 128 in choices and 256 in choices
        assert runner.forward(inputs) == 128


@pytest.mark.parametrize("fmt", [QuantFormat.MXFP4, QuantFormat.MXFP8])
def test_mx_compilation_targets_input_device_and_rebinds_cached_calls(monkeypatch, fmt):
    pytest.importorskip("cutlass", minversion="4.8")
    from flashinfer.fused_moe.backends.frost_low_latency import runtime

    runner = _runner(monkeypatch, fmt)
    runner.check_support()
    runner.build()
    act, pack, _ = _packs(monkeypatch, runner)
    inputs = runner.pack_inputs(act, pack)
    monkeypatch.setattr(runtime, "_COMPILED", {})
    monkeypatch.setattr(runtime, "from_dlpack", lambda x: x)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    compiled, launches = [], []

    def compile_stub(*args, **kwargs):
        compiled.append(kwargs["options"])
        return lambda *live: launches.append(live)

    monkeypatch.setattr(runtime.cute, "compile", compile_stub)
    monkeypatch.setattr(runtime.cute.runtime, "make_fake_stream", lambda **_: None)
    requested_devices = []
    _capability = (10, 0)

    def device_capability(device):
        requested_devices.append(device)
        return _capability

    monkeypatch.setattr("flashinfer.utils.get_compute_capability", device_capability)
    for _capability in ((10, 0), (10, 3), (10, 3), (10, 7)):
        inputs = [replace(v) if isinstance(v, _TensorMetadata) else v for v in inputs]
        assert runtime.execute(inputs, runner.config, 128) is inputs[13]
        assert launches[-1][0] is inputs[0]
    assert requested_devices == [torch.device("cuda:1")] * 4
    assert compiled == [
        f"--gpu-arch sm_{major}{minor}a --enable-tvm-ffi"
        for major, minor in ((10, 0), (10, 3), (10, 7))
    ]
