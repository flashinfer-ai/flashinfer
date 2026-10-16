# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The DECODE form against the fp32 oracle: the two Qwen3.5 TP-1 shard geometries, q_len 1 and 4, a short / a long / a padded
sequence, both FlashInfer cache layouts, the functional entry point and the wrapper; the routed q_len 8 / 32 cells, the
``(k, v)`` pair at page 64 / 128, the explicit split and the unsplit tile (the gate as its own launch), the separate cache-write
launch, the run-time typed declines, a cached step under ``set_sync_debug_mode("error")`` and the compiled-plan cache's reload
path."""

import pytest
import torch

from tests.test_helpers.qwen_3_5_block_cases import PagedCase

pytestmark = [pytest.mark.arch_rubin, pytest.mark.usefixtures("rubin")]

GEOMETRIES = {
    "24_2": dict(d_model=2560, num_qo_heads=24, num_kv_heads=2),
    "32_2": dict(d_model=4096, num_qo_heads=32, num_kv_heads=2),
}
LENS = (
    100,
    500,
    0,
)  # one KV tile with a tail, four tiles with a tail, a PADDED entry (block 0, slot -1, seq_len 0)


def _geometry(name):
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry

    d = GEOMETRIES[name]
    return Qwen35BlockGeometry(
        d_model=d["d_model"],
        num_qo_heads=d["num_qo_heads"],
        num_kv_heads=d["num_kv_heads"],
    )


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
@pytest.mark.parametrize("q_len", [1, 4])
def test_decode_matches_the_oracle(name, q_len):
    from flashinfer.qwen_3_5_block import qwen_3_5_block_bf16_decode

    g = _geometry(name)
    case = PagedCase(
        g, LENS, [q_len if L else 0 for L in LENS], page_size=16, layout="NHD", seed=1
    )
    # a padded entry's rows: present in the batch (uniform q_len), zero-length in the case -- hand it finite rows
    b = len(LENS)
    h = torch.zeros(b * q_len, g.d_model, dtype=torch.bfloat16, device="cuda")
    cos = torch.ones(b * q_len, g.rope_dim, dtype=torch.bfloat16, device="cuda")
    sin = torch.zeros_like(cos)
    slots = torch.full((b * q_len,), -1, dtype=torch.int32, device="cuda")
    live = [i for i, L in enumerate(LENS) if L]
    for j, i in enumerate(live):
        h[i * q_len : (i + 1) * q_len] = case.hidden_states[j * q_len : (j + 1) * q_len]
        cos[i * q_len : (i + 1) * q_len] = case.rope_cos[j * q_len : (j + 1) * q_len]
        sin[i * q_len : (i + 1) * q_len] = case.rope_sin[j * q_len : (j + 1) * q_len]
        slots[i * q_len : (i + 1) * q_len] = case.slot_mapping[
            j * q_len : (j + 1) * q_len
        ]
    out = qwen_3_5_block_bf16_decode(
        h,
        cos,
        sin,
        case.weights,
        case.w_o,
        geometry=g,
        kv_cache=case.kv_cache,
        slot_mapping=slots,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
        q_len_per_req=q_len,
    )
    torch.cuda.synchronize()
    assert out.shape == (b * q_len, g.d_model) and out.dtype == torch.bfloat16
    padded = [i for i, L in enumerate(LENS) if L == 0]
    for i in padded:
        assert (out[i * q_len : (i + 1) * q_len] == 0).all(), (
            "a padded entry (seq_len 0) must yield out rows exactly 0"
        )
    rows = torch.cat([out[i * q_len : (i + 1) * q_len] for i in live])
    case.check_out(rows, label=f"decode {name} q{q_len}")
    case.check_written_rows(label=f"decode {name} q{q_len}")
    assert (case.kv_cache[0] == 0).all(), (
        "the NULL page (block 0) must never be written"
    )


def test_decode_layouts_are_bitwise():
    """The same step through an NHD and an HND cache binds without a repack, two launches are bitwise, and the two layouts give
    bitwise the same output."""
    from flashinfer.qwen_3_5_block import qwen_3_5_block_bf16_decode

    g = _geometry("24_2")
    outs = {}
    for layout in ("NHD", "HND"):
        case = PagedCase(g, (300, 77), (2, 2), page_size=16, layout=layout, seed=2)
        args = dict(geometry=g, kv_layout=layout, q_len_per_req=2, **case.decode_args())
        out = qwen_3_5_block_bf16_decode(
            case.hidden_states,
            case.rope_cos,
            case.rope_sin,
            case.weights,
            case.w_o,
            **args,
        )
        torch.cuda.synchronize()
        case.check_out(out, label=f"decode {layout}")
        again = qwen_3_5_block_bf16_decode(
            case.hidden_states,
            case.rope_cos,
            case.rope_sin,
            case.weights,
            case.w_o,
            **args,
        )
        torch.cuda.synchronize()
        assert torch.equal(out, again), "two launches of the same step differ"
        outs[layout] = out.clone()
    assert torch.equal(outs["NHD"], outs["HND"]), (
        "NHD and HND pools holding the same values must give bitwise the same output"
    )


def test_wrapper_plan_run_and_the_declaration_facts():
    from flashinfer.qwen_3_5_block import Qwen35BlockWrapper

    g = _geometry("24_2")
    case = PagedCase(
        g, (257, 16), (1, 1), page_size=32, layout="NHD", seed=3, table_width=64
    )
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = Qwen35BlockWrapper(ws, "NHD", geometry=g)
    w.plan(
        "decode",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=32,
        batch_size=2,
        q_len_per_req=1,
        block_table=case.block_table,
    )
    plan = w.plan_info
    assert (
        plan.form == "decode"
        and plan.page_size == 32
        and plan.table_row_stride == 64
        and plan.max_pages == 64
    )
    assert (
        plan.workspace_size <= ws.numel()
        and plan.declaration["decode_route"]["tile"] == "decode_d256_f16"
    )
    assert (
        plan.declaration["gate_fused"]
        and plan.declaration["cache_write_fused"]
        and plan.declaration["decode_split_kv"] >= 2
    )
    out = w.run(
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.weights,
        case.w_o,
        **case.decode_args(),
    )
    torch.cuda.synchronize()
    case.check_out(out, label="wrapper decode p32")
    assert plan.supports(
        hidden_states=case.hidden_states,
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        **case.decode_args(),
    )
    assert not plan.supports(
        hidden_states=case.hidden_states[:1],
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        **case.decode_args(),
    )
    # a workspace too small for the plan is a typed ValueError naming the size
    with pytest.raises(ValueError, match="workspace_size"):
        Qwen35BlockWrapper(
            torch.empty(16, dtype=torch.uint8, device="cuda"), "NHD", geometry=g
        ).plan(
            "decode",
            weights=case.weights,
            w_o=case.w_o,
            kv_cache=case.kv_cache,
            page_size=32,
            batch_size=2,
            q_len_per_req=1,
            block_table=case.block_table,
        )


def test_declines_at_plan_time_are_typed_and_named():
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        prepare_qwen_3_5_block_bf16,
    )

    g = _geometry("24_2")
    case = PagedCase(g, (64,), (1,), page_size=16, layout="NHD", seed=4)
    common = dict(
        geometry=g,
        device="cuda",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=16,
        batch_size=1,
        block_table=case.block_table,
    )
    with pytest.raises(NotImplementedError, match="decode envelope"):
        prepare_qwen_3_5_block_bf16("decode", q_len_per_req=33, **common)
    with pytest.raises(ValueError, match="page_size"):
        prepare_qwen_3_5_block_bf16("decode", **dict(common, page_size=8))
    with pytest.raises(
        ValueError, match="kv_cache"
    ):  # the cache's page size disagrees with the plan's
        prepare_qwen_3_5_block_bf16("decode", **dict(common, page_size=32))
    with pytest.raises(NotImplementedError, match="head_dim=256"):
        prepare_qwen_3_5_block_bf16(
            "decode",
            **dict(
                common,
                geometry=Qwen35BlockGeometry(
                    d_model=2560, num_qo_heads=24, num_kv_heads=2, head_dim=128
                ),
            ),
        )
    with pytest.raises(NotImplementedError, match="bfloat16"):
        _fp16_entry(common)


def _fp16_entry(common):
    """The bf16 entry refuses f16 weights by name (the dtype is the entry's, not an argument)."""
    from flashinfer.experimental.qwen_3_5_block import api as _api

    _api.prepare("decode", dtype=torch.float16, **common)


def _plan_decode(case, g, **kw):
    from flashinfer.qwen_3_5_block import prepare_qwen_3_5_block_bf16

    return prepare_qwen_3_5_block_bf16(
        "decode",
        geometry=g,
        device="cuda",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=case.page_size,
        batch_size=case.b,
        block_table=case.block_table,
        **kw,
    )


def _run(plan, case, **kw):
    out = plan.run(
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.weights,
        case.w_o,
        **case.decode_args(),
        **kw,
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("q_len", [8, 32])
def test_decode_past_the_packed_envelope_is_routed_on_the_unpacked_tile(q_len):
    """q_len 5..32: routed on the unpacked decode tile (one Q head per CTA, ``claimed`` False) and declared; the oracle holds."""
    g = _geometry("24_2")
    case = PagedCase(g, (300, 77), (q_len, q_len), page_size=16, layout="NHD", seed=21)
    plan = _plan_decode(case, g, q_len_per_req=q_len)
    route = plan.declaration["decode_route"]
    assert route["tile"] == "decode_d256_f16" and route["s_q"] == q_len
    assert (
        route["pack_gqa"] is False
        and route["claimed"] is False
        and route["n_q"] >= q_len
    )
    out = _run(plan, case)
    case.check_out(out, label=f"decode q{q_len} routed")
    case.check_written_rows(label=f"decode q{q_len} routed")


def test_decode_kv_pair_spelling_at_page_64_hnd():
    """A ``(k_cache, v_cache)`` pair of 4-D HND tensors at page 64: the functional entry reads the page size off the pair."""
    from flashinfer.qwen_3_5_block import qwen_3_5_block_bf16_decode

    g = _geometry("24_2")
    case = PagedCase(g, (200, 1000), (1, 1), page_size=64, layout="HND", seed=22)
    kv = (case.kv_cache[:, 0], case.kv_cache[:, 1])
    out = qwen_3_5_block_bf16_decode(
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.weights,
        case.w_o,
        geometry=g,
        kv_layout="HND",
        kv_cache=kv,
        slot_mapping=case.slot_mapping,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
    )
    torch.cuda.synchronize()
    case.check_out(out, label="decode (k, v) p64 HND")
    case.check_written_rows(label="decode (k, v) p64 HND")


def test_decode_page_128_max_seq_len_and_the_explicit_split():
    """Page 128 with a logical KV bound; the split model's plan and an explicit ``decode_split_kv=2`` agree with the oracle."""
    g = _geometry("24_2")
    case = PagedCase(
        g, (500, 130), (1, 1), page_size=128, layout="NHD", seed=23, table_width=8
    )
    auto = _plan_decode(case, g, max_seq_len=1024)
    assert auto.max_seq_len == 1024 and auto.declaration["decode_split_kv"] >= 2
    case.check_out(_run(auto, case), label="decode p128 split model")
    explicit = _plan_decode(case, g, max_seq_len=1024, decode_split_kv=2)
    d = explicit.declaration
    assert (
        d["decode_split_kv"] == 2
        and d["gate_fused"] is True
        and "_SigmoidGate" not in d["stages"]
    )
    case.check_out(_run(explicit, case), label="decode p128 split 2")


def test_decode_unsplit_tile_runs_the_gate_as_its_own_launch():
    """``decode_split_kv=1``: the decode tile runs unsplit, so the sigmoid gate cannot ride the split combine and runs as the
    separate elementwise launch (the vendored ``kernels/elementwise.py`` arm); the same function as the split plan."""
    g = _geometry("24_2")
    case = PagedCase(
        g, (500, 130), (1, 1), page_size=16, layout="NHD", seed=31, table_width=40
    )
    unsplit = _plan_decode(case, g, decode_split_kv=1)
    d = unsplit.declaration
    assert (
        d["decode_split_kv"] == 1
        and d["gate_fused"] is False
        and "_SigmoidGate" in d["stages"]
    ), d
    out = _run(unsplit, case)
    case.check_out(out, label="decode split 1 (gate launch)")
    case.check_written_rows(label="decode split 1 (gate launch)")
    split = _plan_decode(case, g, decode_split_kv=2)
    assert split.declaration["gate_fused"] is True
    from tests.test_helpers.qwen_3_5_block_reference import cosine

    assert cosine(out, _run(split, case)) > 0.9999, (
        "the unsplit and the split plans compute the same function"
    )


def test_decode_separate_cache_write_launch():
    """The write-through as its own launch (``fuse_cache_write=False`` on the vendored declaration, through the adapter's test
    hook): the vendored ``kernels/cache_write.py`` arm writes the pools the oracle predicts."""
    from flashinfer.experimental.qwen_3_5_block import api as _api

    g = _geometry("24_2")
    case = PagedCase(g, (300, 77), (2, 2), page_size=16, layout="NHD", seed=33)
    plan = _api.prepare(
        "decode",
        geometry=g,
        device="cuda",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=16,
        batch_size=2,
        q_len_per_req=2,
        block_table=case.block_table,
        _declaration_overrides={"fuse_cache_write": False},
    )
    d = plan.declaration
    assert d["cache_write_fused"] is False and "_CacheWrite" in d["stages"], d
    out = _run(plan, case)
    case.check_out(out, label="decode separate cache write")
    case.check_written_rows(label="decode separate cache write")
    assert (case.kv_cache[0] == 0).all(), "the NULL page must never be written"


def test_run_time_declines_are_typed_and_named():
    """Every malformed operand of a run is a ``ValueError`` naming it, host-side, before any launch."""
    g = _geometry("24_2")
    case = PagedCase(g, (300, 77), (1, 1), page_size=16, layout="NHD", seed=29)
    plan = _plan_decode(case, g)
    a = case.decode_args()
    w = (case.hidden_states, case.rope_cos, case.rope_sin, case.weights, case.w_o)
    with pytest.raises(ValueError, match="seq_lens must be torch.int32"):
        plan.run(*w, **dict(a, seq_lens=a["seq_lens"].to(torch.int64)))
    with pytest.raises(ValueError, match="block_table must be"):
        plan.run(*w, **dict(a, block_table=a["block_table"][:1]))
    with pytest.raises(ValueError, match="kv_cache does not match the plan's pools"):
        plan.run(*w, **dict(a, kv_cache=a["kv_cache"][:-1]))
    with pytest.raises(ValueError, match="hidden_states must be torch.bfloat16"):
        plan.run(case.hidden_states.float(), *w[1:], **a)
    with pytest.raises(ValueError, match="w_o must be"):
        plan.run(*w[:4], case.w_o.t().contiguous(), **a)
    with pytest.raises(ValueError, match="workspace_size"):
        plan.run(
            *w, workspace_buffer=torch.empty(8, dtype=torch.uint8, device="cuda"), **a
        )
    with pytest.raises(ValueError, match="slot_mapping must be a 1-D tensor"):
        plan.run(*w, **dict(a, slot_mapping=a["slot_mapping"][:-1]))
    with pytest.raises(ValueError, match="kv_cache \\(K\\) must be torch.bfloat16"):
        plan.run(*w, **dict(a, kv_cache=a["kv_cache"].to(torch.float16)))
    assert not plan.supports(
        hidden_states=case.hidden_states.float(),
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        **a,
    )


def test_a_cached_decode_step_runs_without_a_host_sync():
    """The functional entry point over a cached plan under ``set_sync_debug_mode("error")``: no host read of device data."""
    from flashinfer.qwen_3_5_block import qwen_3_5_block_bf16_decode

    g = _geometry("24_2")
    case = PagedCase(g, (300, 77), (2, 2), page_size=16, layout="NHD", seed=28)
    args = dict(geometry=g, q_len_per_req=2, **case.decode_args())
    w = (case.hidden_states, case.rope_cos, case.rope_sin, case.weights, case.w_o)
    first = qwen_3_5_block_bf16_decode(
        *w, **args
    )  # the plan compiles here (a compile may synchronize)
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        again = qwen_3_5_block_bf16_decode(*w, **args)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    assert torch.equal(first, again)


def test_compiled_plan_cache_lives_under_the_flashinfer_root_and_reloads():
    """The vendored tree's compiled-plan cache and rendered GEMM sources sit under FlashInfer's cache root (unless the caller chose
    another through the tree's own variables), and a second plan of the same bucket reloads the exported kernels from the files
    once the in-process memo is cleared -- the reload path a serving process takes at its next start-up."""
    import pathlib

    from flashinfer.jit.env import FLASHINFER_CACHE_DIR
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import compiled_cache

    g = _geometry("24_2")
    case = PagedCase(
        g, (257, 16), (1, 1), page_size=32, layout="NHD", seed=3, table_width=64
    )
    plan = _plan_decode(case, g)
    caches = plan.declaration["caches"]
    import os

    root = str(FLASHINFER_CACHE_DIR / "qwen_3_5_block")
    if not os.environ.get("CUDNN_FRONTEND_COMPILED_CACHE"):
        assert caches["compiled_plans"] == str(pathlib.Path(root) / "compiled_plans")
    if not os.environ.get("CUDNN_FRONTEND_GEMM_KERNEL_CACHE"):
        assert caches["gemm_sources"] == str(pathlib.Path(root) / "gemm_sources")
    assert any(
        pathlib.Path(caches["gemm_sources"]).glob("gen_*/generated_kernel.py")
    ), "the rendered GEMM sources"
    out = _run(plan, case)
    case.check_out(out, label="decode before the reload")
    if not caches["compiled_plans_enabled"]:
        pytest.skip(
            "the compiled-plan cache is disabled (CUDNN_FRONTEND_DISABLE_COMPILED_CACHE)"
        )
    assert any(pathlib.Path(caches["compiled_plans"]).rglob("kernel.o")), (
        "the exported kernel objects"
    )
    compiled_cache.clear_memo()
    compiled_cache.reset_stats()
    again = _plan_decode(case, g)
    stats = compiled_cache.stats()
    assert stats["hits"] >= 1, stats
    assert torch.equal(_run(again, case), out), (
        "a plan reloaded from the files differs from the compiled one"
    )
