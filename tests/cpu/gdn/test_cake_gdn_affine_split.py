# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Host-only tests of the opt-in Cake GDN affine-split prefill
(``use_affine_split=True``): argument rejections and their order, the
dispatch (the split route never reaches the CP bands or the unsplit
kernel), the planner constants and grids, and the generated program
table against its sources and the host's argument order."""

from __future__ import annotations

import re

import pytest
import torch

from flashinfer import gdn_prefill
from flashinfer.gdn_kernels.blackwell import cake_gdn_affine_split as split
from flashinfer.jit import cake_gdn_affine_split as programs
from flashinfer.jit import core as jit_core
from flashinfer.jit.cake_program import GRID_ARGS, TARGET_FLAGS, CakeProgram

pytestmark = pytest.mark.cpu_only


def _inputs(hq=2, hv=8, t=2048, dtype=torch.float16):
    q = torch.zeros((t, hq, 128), dtype=dtype)
    v = torch.zeros((t, hv, 128), dtype=dtype)
    return q, torch.zeros_like(q), v, torch.tensor([0, t], dtype=torch.int32)


def _device(monkeypatch, capability=(10, 3), sm_count=152):
    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _d: sm_count)
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _d: capability)
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _d: "test GPU")
    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")


# --- rejections before any work -------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(use_affine_split=1), "use_affine_split must be a bool, got int"),
        (
            dict(_affine_split_count=2),
            "_affine_split_count requires use_affine_split=True",
        ),
        (
            dict(use_affine_split=True, _affine_split_count=0),
            "_affine_split_count must be a positive int, got 0",
        ),
        (
            dict(use_affine_split=True, _affine_split_count=True),
            "_affine_split_count must be a positive int, got True",
        ),
        (
            dict(use_affine_split=True, backend="auto"),
            "use_affine_split=True requires backend=\"cake_gdn\", got backend='auto'",
        ),
        (
            dict(use_affine_split=True, backend="sm120_cute"),
            "use_affine_split=True requires backend=\"cake_gdn\", got backend='sm120_cute'",
        ),
        (
            dict(use_affine_split=True, use_cp=True),
            "use_affine_split=True cannot be combined with use_cp=True or _cp_chunk_len",
        ),
        (
            dict(use_affine_split=True, _cp_chunk_len=128),
            "use_affine_split=True cannot be combined with use_cp=True or _cp_chunk_len",
        ),
    ],
)
def test_argument_rejections(kwargs, message):
    q, k, v, cu = _inputs()
    kwargs.setdefault("backend", "cake_gdn")
    with pytest.raises(ValueError, match=re.escape(message)):
        gdn_prefill.chunk_gated_delta_rule(q, k, v, cu_seqlens=cu, **kwargs)


def test_missing_backend_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(
        gdn_prefill, "_chunk_gated_delta_rule_gdn_affine_split_sm100", None
    )
    q, k, v, cu = _inputs()
    with pytest.raises(
        RuntimeError, match="the Cake GDN affine-split backend is not installed"
    ):
        gdn_prefill.chunk_gated_delta_rule(
            q, k, v, cu_seqlens=cu, backend="cake_gdn", use_affine_split=True
        )


def test_unsupported_features_are_named_in_order():
    q, k, v, cu = _inputs()
    t, hv = q.shape[0], v.shape[1]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "use_affine_split=True does not support checkpoint_every_n_tokens, "
            "state_checkpoints, checkpoint_cu_starts, state_indices"
        ),
    ):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            k,
            v,
            cu_seqlens=cu,
            backend="cake_gdn",
            use_affine_split=True,
            checkpoint_every_n_tokens=64,
            state_checkpoints=torch.zeros(t // 64, hv, 128, 128),
            checkpoint_cu_starts=torch.tensor([0, t // 64], dtype=torch.int32),
            state_indices=torch.zeros(1, dtype=torch.int32),
        )
    with pytest.raises(
        ValueError,
        match=re.escape("use_affine_split=True does not support prefix_lens"),
    ):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            k,
            v,
            cu_seqlens=cu,
            backend="cake_gdn",
            use_affine_split=True,
            prefix_lens=torch.zeros(1, dtype=torch.int32),
        )


@pytest.mark.parametrize(
    "capability,message",
    [
        ((10, 0), "use_affine_split=True: sm_100a is not yet supported"),
        ((10, 7), "use_affine_split=True: sm_107a is not supported (sm_103a only)"),
        ((9, 0), "use_affine_split=True: sm_90a is not supported (sm_103a only)"),
        ((12, 0), "use_affine_split=True: sm_120a is not supported (sm_103a only)"),
        ((8, 0), "use_affine_split=True: sm_80 is not supported (sm_103a only)"),
    ],
)
def test_other_architectures_are_rejected(monkeypatch, capability, message):
    _device(monkeypatch, capability)
    q, k, v, cu = _inputs()
    with pytest.raises(ValueError, match=re.escape(message)):
        gdn_prefill.chunk_gated_delta_rule(
            q, k, v, cu_seqlens=cu, backend="cake_gdn", use_affine_split=True
        )


# --- dispatch ------------------------------------------------------------------


@pytest.mark.parametrize("capability,sm_count", [((10, 0), 148), ((10, 3), 152)])
@pytest.mark.parametrize("use_cp", ["auto", False])
def test_split_route_skips_the_cp_bands_and_the_unsplit_kernel(
    monkeypatch, capability, sm_count, use_cp
):
    """A shape inside the SM100/SM103 ``use_cp="auto"`` band: with the
    option the split route alone runs; without it the call routes as before."""

    calls: list[str] = []
    _device(monkeypatch, capability, sm_count)
    monkeypatch.setattr(gdn_prefill, "should_use_cp_host", lambda *_a, **_k: True)
    monkeypatch.setattr(
        gdn_prefill,
        "_chunk_gated_delta_rule_gdn_cp_sm100",
        lambda *_a, **_k: calls.append("gdn_cp"),
    )
    monkeypatch.setattr(
        gdn_prefill,
        "cp_delta_rule_dsl_sm100",
        lambda *_a, **_k: calls.append("cute_cp"),
    )

    def cake_non_cp(**kwargs):
        calls.append("cake_non_cp")
        return kwargs["output"]

    def affine_split(**kwargs):
        calls.append("affine_split")
        return kwargs["output"]

    monkeypatch.setattr(gdn_prefill, "_run_cake_gdn_prefill", cake_non_cp)
    monkeypatch.setattr(gdn_prefill, "_run_cake_gdn_affine_split", affine_split)
    q, k, v, cu = _inputs()
    out = gdn_prefill.chunk_gated_delta_rule(
        q, k, v, cu_seqlens=cu, backend="cake_gdn", use_cp=use_cp, use_affine_split=True
    )
    assert out.shape == (2048, 8, 128) and calls == ["affine_split"]
    calls.clear()
    gdn_prefill.chunk_gated_delta_rule(
        q, k, v, cu_seqlens=cu, backend="cake_gdn", use_cp=use_cp
    )
    expected = "cake_non_cp"
    assert calls == [expected]


def test_split_route_receives_the_resolved_arguments(monkeypatch):
    seen = {}
    _device(monkeypatch)
    monkeypatch.setattr(
        gdn_prefill,
        "_run_cake_gdn_affine_split",
        lambda **kwargs: seen.update(kwargs) or kwargs["output"],
    )
    q, k, v, cu = _inputs()
    gdn_prefill.chunk_gated_delta_rule(
        q,
        k,
        v,
        cu_seqlens=cu,
        backend="cake_gdn",
        use_affine_split=True,
        _affine_split_count=3,
        scale=0.0,
    )
    assert seen["split_count"] == 3 and seen["max_seqlen"] == 2048
    assert seen["scale"] == pytest.approx(128**-0.5)
    assert seen["device_capability"] == (10, 3) and seen["cuda_major"] == 13
    assert seen["output"].shape == (2048, 8, 128)


# --- planner -------------------------------------------------------------------


def test_plan_constants_on_152_sms():
    assert [split.tile_cap(h, 152) for h in (4, 8, 16, 32)] == [10, 5, 2, 1]
    assert [split.plan_constants(h, 152) for h in (4, 8, 16, 32)] == [
        (8, 8),
        (5, 8),
        (2, 16),
        (1, 16),
    ]
    assert split.plan_constants(4, 152, split_count=3) == (3, 1)
    assert split.plan_constants(8, 152, split_count=8) == (5, 1)
    assert split.plan_constants(32, 152, split_count=4) == (1, 1)


@pytest.mark.parametrize(
    "heads,tokens,expected",
    [
        (4, 512, 1),
        (4, 1024, 1),
        (4, 1025, 2),
        (4, 1152, 2),
        (4, 2048, 4),
        (4, 4096, 8),
        (4, 8192, 8),
        (8, 512, 1),
        (8, 1024, 1),
        (8, 1025, 2),
        (8, 4096, 5),
        (16, 1024, 1),
        (16, 1984, 1),
        (16, 2048, 2),
        (16, 8192, 2),
        (32, 8192, 1),
    ],
)
def test_segment_count_of_the_planner(heads, tokens, expected):
    plan = split.plan_split(num_heads=heads, max_seqlen=tokens, sm_count=152)
    assert plan.p_bound == expected


def test_forced_split_count_ignores_the_minimum_length():
    assert split.SPLIT_MIN_TOKENS == 1024
    assert split.plan_split(
        num_heads=4, max_seqlen=1000, sm_count=152, split_count=3
    ) == (3, 1, 3)


def test_segment_count_edges():
    assert split.segment_count(0, 8, 1) == 1
    assert split.segment_count(1, 8, 1) == 1
    assert split.segment_count(65, 8, 1) == 2
    assert split.segment_count(1000, 8, 1) == 8
    assert split.segment_count(10**6, 3, 8) == 3


def test_grids():
    assert split.pass1_grid(1, 8, 5, 152) == (144, 144)
    assert split.pass1_grid(1, 4, 8, 152) == (120, 120)
    assert split.pass1_grid(1, 16, 2, 152) == (96, 96)
    compose = 8 * 128 // 8
    assert split.pass2_grid(1, 8, 4096, 5, 8, 152) == compose + 2 * 152 - compose
    assert split.pass2_grid(1, 4, 256, 2, 4, 152) == 128 + 2 * 4 * (2 + 2)


# --- the generated program table -------------------------------------------------


def test_program_table_matches_the_sources_and_the_host(monkeypatch):
    # The spec checks the build's target architectures; a CPU runner detects
    # none, so pin the one these programs ship for.
    monkeypatch.setattr(
        jit_core.current_compilation_context, "TARGET_CUDA_ARCHS", {(10, "3a")}
    )
    assert programs.PROGRAMS == (
        programs.PASS1,
        programs.PASS2_W4,
        programs.PASS2_W8,
        programs.PASS2_W16,
    )
    assert tuple(programs.PASS1.args) == split._PASS1_ARGS
    for program in programs.PROGRAMS[1:]:
        assert tuple(program.args) == split._PASS2_ARGS
        assert program.block == (256, 1, 1)
    assert programs.PASS1.block == (384, 1, 1)
    assert {split._pass2_program(h) for h in (4, 8, 16, 32)} == set(
        programs.PROGRAMS[1:]
    )
    for program in programs.PROGRAMS:
        assert isinstance(program, CakeProgram)
        assert program.targets == ("sm_103a",)
        assert program.kernel == "kernel_" + program.name
        # Pass 2 runs in stream order after pass 1: no launch is a dependent launch.
        assert program.use_pdl is False and program.cooperative is False
        assert program.cluster == (1, 1, 1)
        assert tuple(program.args[-3:]) == GRID_ARGS
        assert set(program.tensors) <= set(program.args)
        kernel, binding = (path.read_text() for path in program.source_paths())
        assert program.kernel in kernel and program.kernel in binding
        assert f"namespace cake_binding_{program.kernel}" in binding
        assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run," in binding
        assert '#include "cake_ffi_checks.h"' in binding
        assert "CAKE_KERNEL_SYMBOL" not in kernel + binding
        spec = program.spec("sm_103a")
        options = program.options_for("sm_103a")
        flags = [str(flag) for flag in spec.extra_cuda_cflags]
        tail = [*TARGET_FLAGS["sm_103a"], "--fmad=false", *options]
        assert flags[-len(tail) :] == tail
        assert program.module_name("sm_103a").startswith(f"{program.name}_sm_103a_")
