# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the Python SDPA launch binder's paged-pool contract, metadata only (the launch functions are recorders, no device).

Each paged pool binds its OWN declared in-page layout -- ``paged_hnd`` for K, ``paged_v_hnd`` for V, independently, as cuDNN
Frontend's native binders do -- so a plan whose K pool is HND and V pool NHD (or the reverse) binds each pool by its own
strides, and a runtime pool whose layout differs from its declaration is refused by name before anything launches.  Both
directions, for the dense (decode) frame and the packed (extend) frame, plus the capability flags the vendored shim spells for
these classes (kept in step with the binder by the exporter's ``PYTHON_BINDER_FLAGS``)."""

import ast
import pathlib

import pytest

pytest.importorskip("cutlass")  # the vendored tree imports the CuTe DSL at module level

from flashinfer.experimental.qwen_3_5_block import _sdpa_binder as binder  # noqa: E402
from flashinfer.experimental.qwen_3_5_block._cudnn_frost import (  # noqa: E402
    _compiled_module as shim,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import (  # noqa: E402
    _binding_reference as ref,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import (  # noqa: E402
    prepared as prep,
)

D, KH, PAGE, N_PAGES, B, MAX_PAGES, DT = 256, 2, 16, 32, 4, 8, "bfloat16"
LAYOUTS = [
    (False, False),
    (True, True),
    (False, True),
    (True, False),
]  # (K pool HND, V pool HND)


def _pool_facts(ptr: int, hnd: bool, n_pages: int = N_PAGES):
    """A compact ``[n_pages, H_kv, page_size, D]`` pool in the kernel's view: HND (rows inside a head) or NHD (heads inside a row)."""
    shape = (n_pages, KH, PAGE, D)
    strides = (PAGE * KH * D, PAGE * D, D, 1) if hnd else (PAGE * KH * D, D, KH * D, 1)
    span = sum((n - 1) * st for n, st in zip(shape, strides, strict=True)) + 1
    return prep.BufferFacts(ptr, DT, (2, 0), span, shape, strides)


def _kernel_pool_strides(hnd: bool):
    """The ``(page, row, head)`` stride triple the frame carries for a compact pool of this layout."""
    return (PAGE * KH * D, D, PAGE * D) if hnd else (PAGE * KH * D, KH * D, D)


def _dense_spec(k_hnd: bool, v_hnd: bool):
    """The dense (decode) launch spec of a paged bf16 d256 plan, its operands' facts and the frames its launch records."""
    s = prep.DenseLaunchSpec()
    s.b, s.qh, s.kh, s.d_qk, s.d_v = B, 8, KH, D, D
    s.s_q_max, s.s_k_max, s.page_size, s.tile_n = 1, 128, PAGE, 128
    s.paged, s.paged_hnd, s.paged_v_hnd = True, k_hnd, v_hnd
    s.has_lse, s.has_sink, s.combine_q_lens = True, False, False
    s.seq_kv_present, s.seq_q_present = True, True
    s.device_index, s.split, s.window_right = 0, 1, 0
    s.ragged = s.fp32_partial = s.shape_fixed = s.lpt_grid_fixed = False
    s.kv_tail_native = s.causal = s.causal_bottom_right = False
    s.gate_expect = s.quant = s.combine = None
    s.expect = dict.fromkeys(("q", "k", "v", "o"), DT)
    # the host slots of the Rubin d256 decode tile the block runs (its runtime arguments, Constexpr ones excluded)
    host_path = (
        pathlib.Path(prep.__file__).parent / "kernels" / "sm107" / "decode_d256_f16.py"
    )
    host = next(
        n
        for n in ast.parse(host_path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_host"
    )
    s.order = [
        a.arg for a in host.args.args if "Constexpr" not in ast.unparse(a.annotation)
    ]
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = [None] * len(s.order)
    s.template[s.index["seq_q_lens_addr"]] = 0
    s.owner = object()
    frames = []
    s.fn = lambda *frame: frames.append(frame)
    facts = {}
    for i, role in enumerate(("q", "o")):
        shape, strides = (B, 8, 1, D), (8 * D, D, 8 * D, 1)
        span = sum((n - 1) * st for n, st in zip(shape, strides, strict=True)) + 1
        facts[role] = prep.BufferFacts(
            0x1000 * (i + 1), DT, (2, 0), span, shape, strides
        )
    facts["k"], facts["v"] = _pool_facts(0x3000, k_hnd), _pool_facts(0x4000, v_hnd)
    for i, role in enumerate(("seq_q_lens", "seq_kv_lens")):
        facts[role] = prep.BufferFacts(
            0x10000 + i * 0x1000, "int32", (2, 0), B, (B,), (1,)
        )
    facts["lse"] = prep.BufferFacts(
        0x20000, "float32", (2, 0), B * 8, (B, 8, 1), (8, 1, 8)
    )
    for i, role in enumerate(("block_table", "block_table_v")):
        facts[role] = prep.BufferFacts(
            0x30000 + i * 0x1000,
            "int32",
            (2, 0),
            B * MAX_PAGES,
            (B, MAX_PAGES),
            (MAX_PAGES, 1),
        )
    return s, facts, frames


def _thd_spec(k_hnd: bool, v_hnd: bool):
    """The packed (extend) launch spec of a paged bf16 d256 plan, its operands' facts and the frames its launch records."""
    s = ref.ReferenceThdLaunchSpec()
    s.b, s.qh, s.kh, s.d_qk, s.d_v = B, 8, KH, D, D
    s.cga_tile_m = 512
    s.paged, s.page_size, s.paged_hnd, s.paged_v_hnd = True, PAGE, k_hnd, v_hnd
    s.has_sink, s.lse_padded = False, False
    s.has_lse, s.lse_head_major = True, False
    s.lse_head_stride, s.lse_stride = 0, None
    s.device_index, s.lens_form = (
        0,
        1,
    )  # cumulative Q offsets, direct per-sequence KV lengths
    s.total_q, s.total_kv, s.off_o_desc = 16, 512, 4096
    s.gate_expect = s.quant = None
    s.expect = dict.fromkeys(("q", "k", "v", "o"), DT)
    s.decl = {
        name: (h, D, h * D, D, 1, h * D)
        for name, h in (("q", 8), ("k", KH), ("v", KH), ("o", 8))
    }
    s.order = sorted(
        (prep._FILLED_AT_BUILD | prep._FILLED_PER_CALL) - {"table_v_strides"}
    )
    s.index = {name: i for i, name in enumerate(s.order)}
    s.template = [None] * len(s.order)
    s.template[s.index["n_thd_units"]] = 32
    s.template[s.index["lse_ext"]] = s.lse_head_stride
    s.template[s.index["scale_softmax_log2"]] = 0.125
    s._geometry_cache, s.native = None, None
    s.owner = object()
    frames = []
    s.fn = lambda *frame: frames.append(frame)
    facts = {}
    for i, (role, h, sq) in enumerate((("q", 8, 4), ("o", 8, 4))):
        shape, strides = (B, h, sq, D), (sq * h * D, D, h * D, 1)
        facts[role] = prep.BufferFacts(
            0x1000 * (i + 1), DT, (2, 0), B * h * sq * D, shape, strides
        )
    facts["k"], facts["v"] = (
        _pool_facts(0x3000, k_hnd, n_pages=8),
        _pool_facts(0x4000, v_hnd, n_pages=8),
    )
    facts["q_lens"] = prep.BufferFacts(0x10000, "int32", (2, 0), B + 1, (B + 1,), (1,))
    facts["kv_lens"] = prep.BufferFacts(0x11000, "int32", (2, 0), B, (B,), (1,))
    facts["lse"] = prep.BufferFacts(
        0x20000, "float32", (2, 0), 128, (B, 8, 4, 1), (32, 1, 8, 1)
    )
    for i, role in enumerate(("block_table", "block_table_v")):
        facts[role] = prep.BufferFacts(
            0x40000 + i * 0x1000, "int32", (2, 0), 16, (B, 1, 4, 1), (4, 4, 1, 1)
        )
    return s, facts, frames


def _pack(facts, roles):
    return binder.native_pack_from_facts([facts.get(r) for r in roles], roles), tuple(
        range(len(roles))
    )


def _run_dense(s, facts, stream=17):
    pack, indices = _pack(facts, prep._NATIVE_DENSE_ROLES)
    return binder.SdpaDenseBinder(s).execute(pack, indices, stream)


def _run_thd(s, facts, workspace=0x30000, stream=17):
    pack, indices = _pack(facts, prep._NATIVE_THD_ROLES)
    return binder.SdpaThdBinder(s).execute(pack, indices, workspace, stream)


@pytest.mark.parametrize("k_hnd,v_hnd", LAYOUTS)
def test_dense_binds_each_pool_by_its_own_layout_and_rebinds(k_hnd, v_hnd):
    s, facts, frames = _dense_spec(k_hnd, v_hnd)
    assert _run_dense(s, facts) is True
    (frame,) = frames
    assert (
        frame[s.index["k_ptr"]] == facts["k"].ptr
        and frame[s.index["v_ptr"]] == facts["v"].ptr
    )
    assert frame[s.index["k_strides"]] == _kernel_pool_strides(k_hnd)
    assert frame[s.index["v_strides"]] == _kernel_pool_strides(v_hnd)
    assert frame[s.index["n_pages"]] == N_PAGES and frame[s.index["stream"]] == 17
    # a later step rebinds new buffers of the same layouts: the plan's declaration is immutable, the frame follows the operands
    moved = {name: f._replace(ptr=f.ptr + 0x100000) for name, f in facts.items()}
    assert _run_dense(s, moved, stream=23) is True
    assert (
        frames[1][s.index["k_ptr"]] == moved["k"].ptr
        and frames[1][s.index["stream"]] == 23
    )
    assert frames[1][s.index["v_strides"]] == _kernel_pool_strides(v_hnd)


@pytest.mark.parametrize("k_hnd,v_hnd", LAYOUTS)
@pytest.mark.parametrize("flipped", ["k", "v"])
def test_dense_refuses_a_pool_in_the_other_layout_before_launch(k_hnd, v_hnd, flipped):
    s, facts, frames = _dense_spec(k_hnd, v_hnd)
    assert (
        _run_dense(s, facts) is True
    )  # the refusal must hold after a valid warm binding too
    declared = k_hnd if flipped == "k" else v_hnd
    wrong = dict(facts, **{flipped: _pool_facts(facts[flipped].ptr, not declared)})
    with pytest.raises(
        ValueError,
        match=f"{flipped}: this artifact was compiled for {'HND' if declared else 'NHD'}",
    ):
        _run_dense(s, wrong)
    assert len(frames) == 1  # nothing launched for the refused call


@pytest.mark.parametrize("k_hnd,v_hnd", LAYOUTS)
def test_thd_binds_each_pool_by_its_own_layout(k_hnd, v_hnd):
    s, facts, frames = _thd_spec(k_hnd, v_hnd)
    assert _run_thd(s, facts) is True
    (frame,) = frames
    assert frame[s.index["k_strides"]] == _kernel_pool_strides(k_hnd)
    assert frame[s.index["v_strides"]] == _kernel_pool_strides(v_hnd)
    assert (
        frame[s.index["problem_size"]][4] == 4 * PAGE
    )  # max_pages x page_size: the KV extent the tables address
    assert frame[s.index["block_table_ptr"]] == facts["block_table"].ptr
    assert frame[s.index["meta_ptr"]] == 0x30000 and frame[s.index["stream"]] == 17


@pytest.mark.parametrize("k_hnd,v_hnd", LAYOUTS)
@pytest.mark.parametrize("flipped", ["k", "v"])
def test_thd_refuses_a_pool_in_the_other_layout_before_launch(k_hnd, v_hnd, flipped):
    s, facts, frames = _thd_spec(k_hnd, v_hnd)
    assert _run_thd(s, facts) is True
    declared = k_hnd if flipped == "k" else v_hnd
    wrong = dict(
        facts, **{flipped: _pool_facts(facts[flipped].ptr, not declared, n_pages=8)}
    )
    with pytest.raises(
        ValueError,
        match=f"{flipped}: this artifact was compiled for {'HND' if declared else 'NHD'}",
    ):
        _run_thd(s, wrong)
    assert len(frames) == 1


def test_shim_capability_flags_match_the_python_binder():
    """The vendored shim answers ``check_support`` for the binder that serves the launch: every ``supports_*`` flag it spells
    equals the Python binder class's (the dense ``supports_combine_q_lens`` pin included)."""
    for shim_cls, py_cls in (
        (shim._SdpaDenseBinder, binder.SdpaDenseBinder),
        (shim._SdpaThdBinder, binder.SdpaThdBinder),
    ):
        flags = {a for a in dir(shim_cls) if a.startswith("supports_")}
        assert flags == {a for a in dir(py_cls) if a.startswith("supports_")}, (
            shim_cls,
            py_cls,
        )
        for a in flags:
            assert getattr(shim_cls, a) == getattr(py_cls, a), a
    assert shim._SdpaDenseBinder.supports_combine_q_lens is False
