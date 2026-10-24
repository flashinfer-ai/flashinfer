# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4 import frontend
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.frontend import (
    FrontendConfig,
    KernelLaunchSpec,
    Mxfp8Mxfp4Frontend,
    launch_cache_key,
)


class _FakeTensor:
    def __init__(
        self,
        pointer: int,
        shape: tuple[int, ...],
        dtype=torch.uint8,
        *,
        strides: tuple[int, ...] | None = None,
    ):
        self._pointer = pointer
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device("cuda:0")
        self.is_cuda = True
        self._strides = strides

    def data_ptr(self):
        return self._pointer

    def stride(self):
        if self._strides is not None:
            return self._strides
        stride = 1
        result = []
        for dim in reversed(self.shape):
            result.append(stride)
            stride *= dim
        return tuple(reversed(result))

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        expected = 1
        for size, stride in zip(
            reversed(self.shape), reversed(self.stride()), strict=True
        ):
            if size > 1 and stride != expected:
                return False
            expected *= size
        return True

    def storage_offset(self):
        return 0

    def __getitem__(self, key):
        del key
        return self


def _config(**updates):
    values = dict(
        hidden=128,
        intermediate=256,
        topk=2,
        num_experts=8,
        max_tokens=16,
        tactic=(256, 256, 128),
    )
    values.update(updates)
    return FrontendConfig(**values)


def _spec(pointer_offset=0, stream=17):
    ptr = iter(range(1000 + pointer_offset, 1020 + pointer_offset))
    return KernelLaunchSpec(
        activation=_FakeTensor(next(ptr), (16, 128), torch.float8_e4m3fn),
        activation_scales=_FakeTensor(next(ptr), (16, 4), torch.float8_e8m0fnu),
        fc1_weight=_FakeTensor(next(ptr), (8, 512, 64)),
        fc1_weight_scales=_FakeTensor(next(ptr), (8, 512, 4)),
        fc2_weight=_FakeTensor(next(ptr), (8, 128, 128)),
        fc2_weight_scales=_FakeTensor(next(ptr), (8, 128, 8)),
        route_ids=_FakeTensor(next(ptr), (16, 2), torch.int64),
        route_scores=_FakeTensor(next(ptr), (16, 2), torch.float32),
        output=_FakeTensor(next(ptr), (16, 128), torch.bfloat16),
        workspace_regions={"scheduler": _FakeTensor(next(ptr), (1024,))},
        num_tokens=7,
        stream=stream,
        static_tactic=(256, 256, 128),
    )


def test_frontend_config_requires_k32_dimensions():
    with pytest.raises(ValueError, match="multiple of 32"):
        _config(hidden=96 + 1)


def test_launch_key_tracks_pointer_shape_stride_stream_and_tactic():
    spec = _spec()
    assert launch_cache_key(spec) == launch_cache_key(spec)
    assert launch_cache_key(spec) != launch_cache_key(_spec(pointer_offset=100))
    assert launch_cache_key(spec) != launch_cache_key(replace(spec, stream=18))
    assert launch_cache_key(spec) != launch_cache_key(
        replace(spec, static_tactic=(128, 256, 128))
    )


def test_frontend_accepts_packed_transpose_but_weight_adapter_rejects_bad_stride():
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.packed_weight_views import (
        PackedWeightViews,
    )

    config = _config()
    spec = _spec()
    fc1 = _FakeTensor(
        4096,
        (8, 64, 512),
        strides=(512 * 64, 1, 64),
    )
    fc2 = _FakeTensor(
        8192,
        (8, 128, 128),
        strides=(128 * 128, 1, 128),
    )
    valid = replace(spec, fc1_weight=fc1, fc2_weight=fc2)

    frontend.validate_launch_spec(config, valid)
    PackedWeightViews(
        fc1,
        fc2,
        hidden_size=config.hidden,
        intermediate_size=config.intermediate,
    )

    malformed_fc1 = _FakeTensor(
        4096,
        (8, 64, 512),
        strides=(512 * 64, 512, 1),
    )
    malformed = replace(valid, fc1_weight=malformed_fc1)
    # The frontend deliberately leaves packed-weight layout validation to the
    # pointer adapter, which owns the exact preprocessing ABI.
    frontend.validate_launch_spec(config, malformed)
    with pytest.raises(ValueError, match="packed-K-major strides"):
        PackedWeightViews(
            malformed_fc1,
            fc2,
            hidden_size=config.hidden,
            intermediate_size=config.intermediate,
        )


def test_frontend_compiles_once_and_rebinds_on_runtime_pointer_change(monkeypatch):
    monkeypatch.setattr(frontend, "validate_launch_spec", lambda config, spec: None)
    monkeypatch.setattr(frontend, "_ensure_not_capturing", lambda what: None)

    class Adapter:
        compile_count = 0
        bind_count = 0
        launch_count = 0

        def compile(self, config, spec):
            del config, spec
            self.compile_count += 1
            return object()

        def bind(self, compiled, config, spec):
            del compiled, config, spec
            self.bind_count += 1

            def launch():
                self.launch_count += 1

            return launch

    adapter = Adapter()
    runner = Mxfp8Mxfp4Frontend(_config(), adapter)
    first = _spec()
    runner.warmup(first)
    runner.run(first)
    runner.run(_spec(pointer_offset=100))

    assert adapter.compile_count == 1
    assert adapter.bind_count == 2
    assert adapter.launch_count == 2


def test_capture_miss_fails_before_validation_or_compile(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    class Adapter:
        def compile(self, config, spec):  # pragma: no cover - must not execute
            raise AssertionError

        def bind(self, compiled, config, spec):  # pragma: no cover
            raise AssertionError

    with pytest.raises(RuntimeError, match="warmup"):
        Mxfp8Mxfp4Frontend(_config(), Adapter()).run(_spec())


def test_capture_stream_only_rebind_reuses_warmed_arguments(monkeypatch):
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    launches = []

    class Adapter:
        def compile(self, config, spec):
            assert not capturing
            return object()

        def bind(self, compiled, config, spec):
            assert not capturing
            return lambda: launches.append(spec.stream)

        def rebind_stream(self, launch, stream):
            assert callable(launch)
            return lambda: launches.append(stream)

    runner = Mxfp8Mxfp4Frontend(_config(), Adapter())
    spec = _spec()
    runner.warmup(spec)
    capturing = True
    runner.run(replace(spec, stream=18))
    runner.run(replace(spec, stream=19))
    runner.run(spec)
    assert launches == [18, 19, 17]

    # Stream rebinding cannot bypass any buffer, layout, or token-count guard.
    for changed in (
        replace(spec, output=_FakeTensor(4096, (16, 128), torch.bfloat16)),
        replace(spec, activation=_FakeTensor(4096, (16, 128), torch.float8_e4m3fn)),
        replace(spec, workspace_regions={"scheduler": _FakeTensor(4096, (1024,))}),
        replace(spec, num_tokens=8),
        replace(spec, static_tactic=(128, 128, 128)),
    ):
        with pytest.raises(RuntimeError, match="warmup"):
            runner.run(replace(changed, stream=18))
    assert launches == [18, 19, 17]


def test_capture_stream_rebind_requires_adapter_opt_in(monkeypatch):
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    class Adapter:
        def compile(self, config, spec):
            assert not capturing
            return object()

        def bind(self, compiled, config, spec):
            assert not capturing
            return lambda: None

    runner = Mxfp8Mxfp4Frontend(_config(), Adapter())
    runner.warmup(_spec())
    capturing = True
    with pytest.raises(RuntimeError, match="warmup"):
        runner.run(_spec(stream=18))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prequantized_staging_is_bit_exact_and_masks_tail(monkeypatch):
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4 import staging

    monkeypatch.setattr(
        staging,
        "_shim_helpers",
        lambda: (None, None, None, lambda output, count: None),
    )
    data = (
        torch.arange(256, device="cuda", dtype=torch.int32)
        .to(torch.uint8)
        .view(torch.float8_e4m3fn)
        .reshape(2, 128)
    )
    scales = (
        torch.arange(8, device="cuda", dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .reshape(2, 4)
    )
    ids = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    scores = torch.rand((2, 2), device="cuda", dtype=torch.float32)
    data_out = torch.empty((4, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scales_out = torch.empty((4, 4), device="cuda", dtype=torch.float8_e8m0fnu)
    ids_out = torch.zeros((4, 2), device="cuda", dtype=torch.int64)
    scores_out = torch.empty((4, 2), device="cuda", dtype=torch.float32)

    staged = staging.stage_inputs(
        data,
        scales,
        ids,
        scores,
        data_out,
        scales_out,
        ids_out,
        scores_out,
        quantize_input=False,
    )

    assert staged.num_tokens == 2
    assert torch.equal(data_out[:2].view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(scales_out[:2].view(torch.uint8), scales.view(torch.uint8))
    assert torch.equal(ids_out[:2], ids.to(torch.int64))
    assert torch.equal(scores_out[:2], scores)
    assert (ids_out[2:] == -1).all()


def test_compile_cache_distinguishes_persistent_clusters() -> None:
    config, spec = _config(), _spec()
    keys = [
        frontend.compile_cache_key(
            replace(config, tactic=(("persistent_clusters", clusters),)),
            replace(spec, static_tactic=(("persistent_clusters", clusters),)),
        )
        for clusters in (None, 64, 72)
    ]
    assert len(set(keys)) == 3


def test_output_shape_validation():
    config = _config()
    spec = _spec()
    frontend.validate_launch_spec(config, spec)
    with pytest.raises(ValueError, match="output must be BF16"):
        frontend.validate_launch_spec(
            config,
            replace(spec, output=_FakeTensor(2000, (16, 2, 128), torch.bfloat16)),
        )
