"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests of the experimental NVFP4 attention backend package (SM100, SM103,
SM120): BF16 -> BF16 correctness against an FP32 reference over the operand
regimes the kernels were validated on (unit normal, V x 8, one-hot rows,
outlier channels of Q and K or of Q alone), tails (``S % 128 != 0``),
``S < 128``, ``B > 1``, ``H`` in {5, 12, 40} and both storage orders of the
logical ``[B, H, S, 128]`` view; repeatability and value freedom between
launches; CUDA-graph replay; the workspace contract (on SM100 / SM103 the
split-tile partials sized by the device's SM count); the host plan of the
registry records, the SM100 / SM103 split plan and the ``build.json`` contract.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.prefill`` public entries are covered by
``test_cake_nvfp4_attention_api.py``.
"""

import re
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("nvfp4_attention")
backend = cake_package("nvfp4_attention", "cake_backend")
loader = cake_package("nvfp4_attention", "cake_jit")
cake_build = cake_package("nvfp4_attention", "cake_build")

HEAD_DIM = 128
ATOL, RTOL = 1.0, 0.1
OUTLIER_CHANNELS = (3, 40, 77, 120)
REGIMES = ("randn", "v_x8", "onehot", "outlier32", "outlier32_q")
# Entries of ``build.json`` carry the compile options the program's kernel was validated with: fast math on the
# SM100 / SM103 table (``delta``) and attention programs, nvcc's defaults alone on the statistics and quantization
# programs and on every SM120 program (``_expected_nvcc_flags``).
NVCC_FLAGS_BY_ROLE = {
    "stats": [],
    "quantize": [],
    "delta": ["--use_fast_math"],
    "attention": ["--use_fast_math"],
}


def _expected_nvcc_flags(record, arch):
    """The ``nvcc_flags`` of the ``build.json`` entry of ``record`` built for ``arch``: the SM120 programs spell
    no compile options; the others carry their role's."""
    return [] if arch == "sm_120a" else NVCC_FLAGS_BY_ROLE[record["role"]]


def _capability():
    return (
        tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    )


def _arch():
    capability = _capability()
    return loader.arch_for_capability(capability) if capability is not None else None


def _skip_unless_supported():
    if _arch() is None:
        pytest.skip("an SM100, SM103 or SM120 GPU is required")


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries when they have no prebuilt file before the first test, through the
    # spec objects the package's loader itself builds on first use; prebuilt or cached libraries load without
    # compiling.
    if _arch() is not None:
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


# --- Inputs and the reference ---------------------------------------------------


def make_inputs(
    batch, heads, seqlen, *, operands="randn", layout="HND", seed=0, device="cuda"
):
    """BF16 ``q``, ``k``, ``v`` of one regime as logical ``[B, H, S, 128]`` views: ``HND`` storage is the view
    itself, ``NHD`` storage is a ``[B, S, H, 128]`` buffer viewed through ``transpose(1, 2)``."""
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seqlen, HEAD_DIM)
    q, k, v = (torch.randn(shape, generator=generator) for _ in range(3))
    if operands == "v_x8":
        v *= 8.0
    elif operands == "onehot":
        for b in range(batch):
            for h in range(heads):
                keys = torch.randperm(seqlen, generator=generator)
                q[b, h] = k[b, h][keys] * 8.0
    elif operands == "outlier32":
        q[..., OUTLIER_CHANNELS] *= 32.0
        k[..., OUTLIER_CHANNELS] *= 32.0
    elif operands == "outlier32_q":
        q[..., OUTLIER_CHANNELS] *= 32.0
    elif operands != "randn":
        raise ValueError(f"unknown operand regime {operands!r}")
    tensors = []
    for x in (q, k, v):
        x = x.to(dtype=torch.bfloat16, device=device)
        if layout == "NHD":
            x = x.transpose(1, 2).contiguous().transpose(1, 2)
            assert not x.is_contiguous()
        tensors.append(x)
    return tuple(tensors)


def _empty_like_layout(x, layout):
    if layout == "NHD":
        return torch.empty_like(x.transpose(1, 2)).transpose(1, 2)
    return torch.empty_like(x)


def reference(q, k, v, softmax_scale=None, chunk=2048):
    """FP32 attention of the BF16 inputs, one head and ``chunk`` query rows at a time (the full score matrix of a
    long row would not fit)."""
    scale = HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            kf, vf = k[b, h].float(), v[b, h].float()
            for start in range(0, q.shape[2], chunk):
                qf = q[b, h, start : start + chunk].float()
                scores = (qf @ kf.T) * scale
                out[b, h, start : start + chunk] = torch.softmax(scores, dim=-1) @ vf
    return out


def _check(out, expected):
    assert torch.isfinite(out).all(), "the output has NaN or Inf"
    actual = out.float()
    rel_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    try:
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    except AssertionError as exc:
        raise AssertionError(f"relative L2 error {float(rel_l2):.4g}; {exc}") from None
    return float(rel_l2)


# --- Correctness over the validated regimes -------------------------------------

SHAPES = [
    # (batch, heads, seqlen, operands, layout): tails, S < 128, S = 1, B > 1, H in {5, 12, 40}, both storage orders.
    (1, 5, 1, "randn", "HND"),
    (1, 5, 100, "randn", "NHD"),
    (2, 12, 257, "randn", "HND"),
    (1, 12, 1100, "v_x8", "NHD"),
    (1, 40, 1100, "onehot", "HND"),
    (2, 5, 1100, "outlier32", "NHD"),
    (1, 12, 1100, "outlier32_q", "HND"),
    (1, 40, 4800, "randn", "NHD"),
    (1, 5, 4800, "outlier32_q", "HND"),
    (1, 8, 32760, "randn", "HND"),
]


@pytest.mark.parametrize("batch,heads,seqlen,operands,layout", SHAPES)
def test_nvfp4_attention(batch, heads, seqlen, operands, layout):
    _skip_unless_supported()
    q, k, v = make_inputs(
        batch,
        heads,
        seqlen,
        operands=operands,
        layout=layout,
        seed=2026 + seqlen + heads,
    )
    out = _empty_like_layout(q, layout)
    attention = backend.prepare_nvfp4_attention(q, k, v, out, backend="cake")
    assert attention() is out
    if operands == "outlier32" or (operands == "outlier32_q" and _arch() == "sm_120a"):
        # Q and K outlier channels (x32) give raw scores of several hundred whose top-2 gaps are below the NVFP4
        # recipe's score error, so whole softmax rows legitimately follow the quantized operands rather than the
        # BF16 inputs.  The acceptance oracle of this regime is the dequantized-operand reference of the kernel's
        # own quantized Q, K and V (plus the recipe's emulation band); this public test cannot rebuild it (the two
        # E2M1 levels, their tiled scales, the transposed V order and the table program's compensation), so the
        # package's kernel-test parity record applies that oracle, and this row checks finiteness and the launch
        # contract (bitwise replay below) only.
        # On SM120 the Q-only outlier regime (``outlier32_q``) belongs here too: the INT8 form scales each 128-row
        # Q block with one scale, which the x32 outlier channels dominate, so the other channels of the block
        # quantize coarsely and the softmax rows follow the quantized Q (relative L2 ~0.15 against the BF16
        # inputs).  The kernel-test oracle and the emulation band accept these shapes (the package's parity
        # record carries them as correctness rows); on SM100 / SM103 the two-level NVFP4 form keeps the BF16
        # comparison below.
        assert torch.isfinite(out).all(), "the output has NaN or Inf"
    else:
        _check(out, reference(q, k, v))
    # Replays are bitwise repeatable.
    snapshot = out.clone()
    out.zero_()
    assert attention.launch() is out
    torch.testing.assert_close(out, snapshot, atol=0, rtol=0)


def test_nvfp4_attention_softmax_scale_keyword():
    _skip_unless_supported()
    q, k, v = make_inputs(1, 5, 1100, seed=11)
    out = torch.empty_like(q)
    backend.prepare_nvfp4_attention(q, k, v, out, backend="cake", softmax_scale=0.05)()
    _check(out, reference(q, k, v, softmax_scale=0.05))
    with pytest.raises(ValueError, match="softmax_scale"):
        backend.prepare_nvfp4_attention(q, k, v, out, backend="cake", softmax_scale=0.0)
    with pytest.raises(ValueError, match="softmax_scale"):
        backend.prepare_nvfp4_attention(
            q, k, v, out, backend="cake", softmax_scale=-1.0
        )


def test_nvfp4_attention_strided_views_match_contiguous_storage():
    """An ``[B, S, H, 128]`` buffer viewed as ``[B, H, S, 128]`` is read in place and written in place, and gives the
    contiguous copies' output bit for bit (the same quantized operands, the same reduction orders)."""
    _skip_unless_supported()
    q, k, v = make_inputs(2, 5, 1100, seed=3)
    expected = torch.empty_like(q)
    backend.prepare_nvfp4_attention(q, k, v, expected, backend="cake")()
    views = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v)]
    out = _empty_like_layout(q, "NHD")
    attention = backend.prepare_nvfp4_attention(*views, out, backend="cake")
    assert attention() is out
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert torch.equal(out.transpose(1, 2).contiguous().transpose(1, 2), out)


def test_nvfp4_attention_values_are_free_between_launches():
    """Quantization is part of every launch: new values in the bound tensors reach the next launch."""
    _skip_unless_supported()
    q, k, v = make_inputs(1, 5, 257, seed=5)
    out = torch.empty_like(q)
    attention = backend.prepare_nvfp4_attention(q, k, v, out, backend="cake")
    attention()
    q2, k2, v2 = make_inputs(1, 5, 257, seed=6)
    q.copy_(q2), k.copy_(k2), v.copy_(v2)
    attention()
    fresh = torch.empty_like(q)
    backend.prepare_nvfp4_attention(q2, k2, v2, fresh, backend="cake")()
    torch.testing.assert_close(out, fresh, atol=0, rtol=0)


def test_nvfp4_attention_graph_replay():
    """The runner is captured by the caller; replays reproduce the eager launch bit for bit and leave the counters
    at zero."""
    _skip_unless_supported()
    q, k, v = make_inputs(2, 5, 1100, seed=9)
    out = torch.empty_like(q)
    attention = backend.prepare_nvfp4_attention(q, k, v, out, backend="cake")
    expected = attention().clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        attention()  # warm the stream before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        attention()
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    layout = backend.workspace_layout(
        _arch(), batch=2, heads=5, seqlen=1100, sm_count=sm_count
    )
    assert int(layout.view(attention.workspace, "counters").sum()) == 0
    if "split_counters" in layout.regions:
        assert int(layout.view(attention.workspace, "split_counters").sum()) == 0


def test_nvfp4_attention_workspace_contract():
    _skip_unless_supported()
    q, k, v = make_inputs(1, 12, 1100, seed=2)
    out = torch.empty_like(q)
    needed = backend.nvfp4_attention_workspace_bytes(1, 12, 1100, q.device)
    assert needed > 0 and needed % backend.ALIGN == 0
    attention = backend.prepare_nvfp4_attention(q, k, v, out, backend="cake")
    assert attention.workspace_bytes == needed == attention.workspace.numel()
    expected = attention().clone()
    pool = torch.empty(needed + 4096, dtype=torch.uint8, device=q.device)
    pooled = backend.prepare_nvfp4_attention(
        q, k, v, out, backend="cake", workspace=pool
    )
    assert pooled.workspace is pool and pooled.workspace_bytes == needed
    torch.testing.assert_close(pooled(), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="workspace holds"):
        backend.prepare_nvfp4_attention(
            q, k, v, out, backend="cake", workspace=pool[: needed - 1]
        )
    with pytest.raises(ValueError, match="uint8"):
        backend.prepare_nvfp4_attention(
            q, k, v, out, backend="cake", workspace=torch.empty(needed, device=q.device)
        )


def test_nvfp4_attention_route_metadata_and_stages():
    _skip_unless_supported()
    q, k, v = make_inputs(1, 5, 257, seed=4)
    out = torch.empty_like(q)
    attention = backend.prepare_nvfp4_attention(q, k, v, out, backend="cake")
    arch = _arch()
    chain = loader.records_for(arch)
    # The chain's stages in launch order: four on SM100 / SM103 (``delta`` between ``quantize`` and
    # ``attention``), three on SM120.
    assert [stage for stage, _, _ in attention.stages] == list(chain)
    assert len(attention.stages) == (3 if arch == "sm_120a" else 4)
    metadata = attention.route_metadata
    assert metadata["arch"] == arch and metadata["records"] == chain
    assert set(metadata["grids"]) == set(chain)
    assert metadata["recipe"] == backend.RECIPE_ID
    # The QK operand format is the attention build's: the two-level residual form on SM100 / SM103, the
    # development placeholder of a build that spells none on SM120.
    assert metadata["qk_format"] == ("int8_blk" if arch == "sm_120a" else "nvfp4_res2d")
    assert metadata["tile_order"] in ("asc", "desc") and metadata["key_groups"] in (
        "consecutive",
        "interleaved32",
    )
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    for stage, (record_stage, _entry, arguments) in zip(
        chain, attention.stages, strict=True
    ):
        record = loader.MODULES[chain[stage]]
        grid = loader.launch_grid(
            record, batch=1, heads=5, seqlen=257, sm_count=sm_count
        )
        assert record_stage == stage and metadata["grids"][stage] == grid
        assert (
            len(arguments) == len(record["arg_plan"]) and tuple(arguments[-3:]) == grid
        )
    attention_record = loader.MODULES[chain["attention"]]
    if attention_record["grid_rule"]["kind"] == "split_persistent_tiles":
        plan = loader.split_plan(
            attention_record, batch=1, heads=5, seqlen=257, sm_count=sm_count
        )
        assert metadata["split_plan"] == {
            "n_full_tiles": plan["n_full_tiles"],
            "n_chunks": plan["n_chunks"],
            "n_items": plan["n_items"],
            "grid": plan["grid"],
            "sm_count": sm_count,
        }
        assert metadata["grids"]["attention"] == (plan["grid"], 1, 1)
    else:
        assert "split_plan" not in metadata
    # A stage alone launches (the chain's order is the caller's to respect); ``delta`` is a stage of the SM100 /
    # SM103 chain only.
    attention.launch_stage("stats")
    if arch == "sm_120a":
        with pytest.raises(KeyError):
            attention.launch_stage("delta")
    else:
        attention.launch_stage("delta")
    with pytest.raises(KeyError):
        attention.launch_stage("merge")


def test_nvfp4_attention_prepare_validation():
    _skip_unless_supported()
    q, k, v = make_inputs(1, 5, 257, seed=1)
    out = torch.empty_like(q)
    with pytest.raises(ValueError, match="backend"):
        backend.prepare_nvfp4_attention(q, k, v, out, backend="unknown")
    with pytest.raises(TypeError, match="BF16"):
        backend.prepare_nvfp4_attention(q.float(), k, v, out, backend="cake")
    with pytest.raises(ValueError, match="shape"):
        backend.prepare_nvfp4_attention(q, k[:, :4], v, out, backend="cake")
    with pytest.raises(ValueError, match=r"\[B, H, S, 128\]"):
        backend.prepare_nvfp4_attention(
            q[..., :64], k[..., :64], v[..., :64], out[..., :64], backend="cake"
        )
    # A shape-valid [B, H, S, 128] view whose head dimension is strided (every other element of a 256-wide buffer).
    strided = torch.empty(
        (1, 5, 257, 2 * HEAD_DIM), dtype=torch.bfloat16, device=q.device
    )[..., ::2]
    with pytest.raises(ValueError, match="stride 1"):
        backend.prepare_nvfp4_attention(strided, k, v, out, backend="cake")
    padded = torch.empty(
        (1, 5, 257, HEAD_DIM + 8), dtype=torch.bfloat16, device=q.device
    )[..., :HEAD_DIM]
    with pytest.raises(ValueError, match="not a multiple of 16"):
        backend.prepare_nvfp4_attention(padded, k, v, out, backend="cake")
    with pytest.raises(ValueError, match="CUDA"):
        backend.prepare_nvfp4_attention(
            q.cpu(), k.cpu(), v.cpu(), out.cpu(), backend="cake"
        )


def test_nvfp4_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        backend.prepare_nvfp4_attention(None, None, None, None, backend="unknown")


# --- The host plan (no GPU) ------------------------------------------------------


def test_host_plan_of_the_records():
    """The registry is a role table: one record per stage per architecture, every kernel parameter named in the
    package vocabulary, the grid rules evaluated by ``launch_grid`` and the arguments bound by ``launch_args``."""
    assert loader.STAGES == ("stats", "quantize", "delta", "attention")
    assert loader.OPTIONAL_STAGES == ("delta",)
    assert set(loader.LOGICAL_TENSORS).isdisjoint(loader.LOGICAL_SCALARS)
    # Eight records: SM100 and SM103 share the statistics, quantization and table (``delta``) programs and each
    # has its own attention build; SM120 has its three (no ``delta`` stage: its attention compensates in-kernel).
    assert len(loader.MODULES) == 8
    assert loader.records_for("sm_100a") == {
        "stats": "stats_sm100",
        "quantize": "quantize_sm100",
        "delta": "delta_sm100",
        "attention": "attention_sm100",
    }
    assert loader.records_for("sm_103a") == {
        "stats": "stats_sm100",
        "quantize": "quantize_sm100",
        "delta": "delta_sm100",
        "attention": "attention_sm103",
    }
    assert loader.records_for("sm_120a") == {
        "stats": "stats_sm120",
        "quantize": "quantize_sm120",
        "attention": "attention_sm120",
    }
    for arch in loader.ARCHES:
        chain = loader.records_for(arch)
        # Launch order is ``STAGES`` order; only an optional stage may be missing.
        assert list(chain) == [stage for stage in loader.STAGES if stage in chain]
        assert set(loader.STAGES) - set(chain) <= set(loader.OPTIONAL_STAGES)
        assert ("delta" in chain) is (arch != "sm_120a")
        for stage, name in chain.items():
            record = loader.MODULES[name]
            assert record["role"] == stage and record["stage"] == loader.STAGES.index(
                stage
            )
            assert arch in record["arches"]
            for kind, key in record["arg_plan"]:
                if kind == "grid":
                    continue
                logical = record["logical"][key]
                assert logical in (
                    loader.LOGICAL_TENSORS
                    if kind == "buffer"
                    else loader.LOGICAL_SCALARS
                )
                if kind == "buffer":
                    assert record["dtypes"][key] in backend._TORCH_DTYPES, (name, key)
            assert set(record["dtypes"]) == {
                key for kind, key in record["arg_plan"] if kind == "buffer"
            }
            rule = record["workspace_rule"]
            if (
                stage == "stats"
                or record["grid_rule"]["kind"] == "split_persistent_tiles"
            ):
                # The statistics counters; on SM100 / SM103 also the attention's split partials and counters.
                assert (
                    rule["kind"] == "prepare_carved"
                    and rule["counters"] in rule["regions"]
                )
            else:
                assert rule is None
            assert (
                record["launch"]["use_pdl"] is (stage != "stats")
                and record["launch"]["cooperative"] is False
            )
    with pytest.raises(RuntimeError, match="delivers 0 'stats' programs for sm_90a"):
        loader.records_for("sm_90a")
    # Grid rules.
    sm100 = loader.records_for("sm_100a")
    assert loader.launch_grid(
        loader.MODULES[sm100["attention"]], batch=1, heads=40, seqlen=4800, sm_count=148
    ) == (148, 1, 1)
    assert loader.launch_grid(
        loader.MODULES[sm100["attention"]], batch=1, heads=1, seqlen=1, sm_count=3
    ) == (2, 1, 1)
    assert loader.launch_grid(
        loader.MODULES[sm100["stats"]], batch=2, heads=5, seqlen=1000, sm_count=148
    ) == (20, 1, 1)
    assert loader.launch_grid(
        loader.MODULES[sm100["quantize"]], batch=2, heads=5, seqlen=1000, sm_count=148
    ) == (80, 1, 1)
    # The table program: one CTA per (batch, head, 128-block tile of the 512-padded Q blocks, 128-key tile), at
    # most one per SM: items = B * H * ceil(ceil(S / 512) * 4 / 128) * ceil(S / 128).
    delta = loader.MODULES[sm100["delta"]]
    assert delta["grid_rule"]["kind"] == "capped_table_tiles"
    assert loader.launch_grid(delta, batch=1, heads=40, seqlen=4800, sm_count=148) == (
        148,
        1,
        1,
    )  # 40 * 1 * 38 = 1520 items
    assert loader.launch_grid(delta, batch=1, heads=7, seqlen=133, sm_count=148) == (
        14,
        1,
        1,
    )  # 7 * 1 * 2 items
    assert loader.launch_grid(delta, batch=1, heads=1, seqlen=1, sm_count=148) == (
        1,
        1,
        1,
    )
    assert loader.launch_grid(delta, batch=1, heads=2, seqlen=32760, sm_count=148) == (
        148,
        1,
        1,
    )  # 2 * 2 * 256 = 1024 items
    sm120 = loader.records_for("sm_120a")
    assert loader.launch_grid(
        loader.MODULES[sm120["attention"]], batch=1, heads=40, seqlen=4800, sm_count=170
    ) == (170, 1, 1)
    assert loader.launch_grid(
        loader.MODULES[sm120["stats"]], batch=1, heads=1, seqlen=257, sm_count=170
    ) == (3, 1, 1)
    assert loader.launch_grid(
        loader.MODULES[sm120["quantize"]], batch=1, heads=40, seqlen=32760, sm_count=170
    ) == (1360, 1, 1)
    with pytest.raises(ValueError, match="at least 2 SMs"):
        loader.launch_grid(
            loader.MODULES[sm100["attention"]], batch=1, heads=1, seqlen=512, sm_count=1
        )
    with pytest.raises(ValueError, match="positive extents"):
        loader.launch_grid(
            loader.MODULES[sm100["attention"]],
            batch=0,
            heads=1,
            seqlen=512,
            sm_count=148,
        )
    with pytest.raises(ValueError, match="unknown Cake NVFP4 attention grid rule"):
        loader.launch_grid(
            {"grid_rule": {"kind": "linear"}},
            batch=1,
            heads=1,
            seqlen=512,
            sm_count=148,
        )
    # The argument plan: values by logical name, then the grid axes.
    record = loader.MODULES[sm100["attention"]]
    values = {
        record["logical"][key]: object()
        for kind, key in record["arg_plan"]
        if kind != "grid"
    }
    args = loader.launch_args(record, values, (24, 1, 1))
    assert args[-3:] == (24, 1, 1)
    assert args[:-3] == tuple(
        values[record["logical"][key]]
        for kind, key in record["arg_plan"]
        if kind != "grid"
    )
    with pytest.raises(ValueError, match="needs logical argument 'out'"):
        loader.launch_args(
            record, {k: v for k, v in values.items() if k != "out"}, (2, 1, 1)
        )


def _views(batch, heads, seqlen, layout):
    """CPU logical ``[B, H, S, 128]`` views of ``layout`` buffers (the kernels never run here)."""
    tensors = []
    for _ in range(4):
        if layout == "NHD":
            x = torch.empty(
                (batch, seqlen, heads, 128), dtype=torch.bfloat16
            ).transpose(1, 2)
        else:
            x = torch.empty((batch, heads, seqlen, 128), dtype=torch.bfloat16)
        tensors.append(x)
    return tensors


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_values_bind_the_kernels_views_of_the_workspace(layout):
    """Host plan on the CPU: every record's buffers are the carved regions viewed as the record's dtypes (the SM100 /
    SM103 quantizer takes the FP32 rows ``qm32`` / ``kc32`` as bytes, the table program ``delta`` as FP32 and
    writes the BF16 image ``tab`` the attention seeds its scores from), the caller's q / k / v the contiguous span
    their view covers (same first element), out the view itself; the scalars are the kernels' block counts; the
    sm120 unit table spells (batch, head << 16 | q_tile, begin, end)."""
    batch, heads, seqlen = 2, 3, 1100
    q, k, v, out = _views(batch, heads, seqlen, layout)
    common = backend._common_scalars("sm_120a", q, k, v, out, 128**-0.5)
    assert (
        common["num_qblocks"] == 9
        and common["num_kblocks"] == 18
        and common["num_units"] == 54
    )
    assert common["q_rows_pad"] == 1152 and common["kv_rows_pad"] == 1152
    sm100 = backend._common_scalars("sm_100a", q, k, v, out, 128**-0.5)
    assert (
        sm100["q_rows_pad"] == 1536
        and sm100["num_qblocks_pad"] == 12
        and sm100["kv_rows_pad"] == 1152
    )
    for arch in loader.ARCHES:
        layout_ = backend.workspace_layout(
            arch, batch=batch, heads=heads, seqlen=seqlen, sm_count=148
        )
        workspace = torch.zeros(layout_.total_bytes, dtype=torch.uint8)
        tensors = {name: layout_.view(workspace, name) for name in layout_.regions}
        tensors.update(q=q, k=k, v=v, out=out)
        common = backend._common_scalars(arch, q, k, v, out, 128**-0.5)
        chain = loader.records_for(arch)
        assert len(chain) == (3 if arch == "sm_120a" else 4)
        for stage, name in chain.items():
            record = loader.MODULES[name]
            values = backend._values_for(
                record, tensors=tensors, common=common, sm_count=148
            )
            for kind, key in record["arg_plan"]:
                if kind != "buffer":
                    continue
                logical = record["logical"][key]
                value = values[logical]
                assert value.dtype == backend._TORCH_DTYPES[record["dtypes"][key]], (
                    arch,
                    stage,
                    key,
                )
                source = tensors[logical]
                assert value.data_ptr() == source.data_ptr(), (arch, stage, key)
                if logical in ("q", "k", "v"):
                    assert value.ndim == 1 and value.is_contiguous()
                    assert value.numel() == 1 + sum(
                        (d - 1) * st
                        for d, st in zip(source.shape, source.stride(), strict=True)
                    )
                elif logical == "out":
                    assert value is source
                else:
                    assert (
                        value.numel() * value.element_size()
                        == source.numel() * source.element_size()
                    )
            # The values hold what the record binds; the record's tile counts are the kernels' work-item counts.
            scalars = backend._record_scalars(record, common, sm_count=148)
            if stage == "stats":
                assert scalars["num_stats_tiles"] == (3 if arch != "sm_120a" else 9)
                assert scalars["num_items"] == (18 if arch != "sm_120a" else 54)
            if stage == "quantize":
                assert scalars["num_items"] == (72 if arch != "sm_120a" else 216)
                if arch != "sm_120a":
                    # The FP32 rows of the table program's operands, taken as byte pointers (``qm4`` -> ``qm32``,
                    # ``sfqm`` -> ``kc32``), alongside the second E2M1 level.
                    assert record["logical"]["qm4"] == "qm32"
                    assert record["logical"]["sfqm"] == "kc32"
                    assert values["qm32"].dtype == torch.uint8
                    assert values["kc32"].dtype == torch.uint8
                    assert values["qm32"].shape == (6 * 12, 512)
                    assert values["kc32"].shape == (6 * 1152, 512)
                    for logical in ("q4l", "q_sf_l", "k4l", "k_sf_l"):
                        assert values[logical].dtype == torch.uint8, logical
            if stage == "delta":
                # The table program: FP32 operands, the BF16 (hi, lo) seed image, the padded row counts.
                assert arch != "sm_120a"
                assert values["qm32"].dtype == torch.float32
                assert values["kc32"].dtype == torch.float32
                assert values["tab"].dtype == torch.bfloat16
                assert values["qm32"] is tensors["qm32"]
                assert values["kc32"] is tensors["kc32"]
                assert values["tab"] is tensors["tab"]
                assert values["kv_rows_pad"] == 1152
                assert values["num_qblocks_pad"] == 12
                assert values["total_bh"] == 6
                assert set(values) == {
                    "qm32",
                    "kc32",
                    "tab",
                    "kv_rows_pad",
                    "num_qblocks_pad",
                    "total_bh",
                }
            if stage == "attention" and arch != "sm_120a":
                # 18 tiles on 74 clusters: every tile is a last-wave tile, split into 4 of its 9 key blocks' chunks.
                assert scalars["split_full_tiles"] == 0 and scalars["split_chunks"] == 4
                assert values["split_full_tiles"] == 0 and values["split_chunks"] == 4
                # The seed image rides in the ``QM`` slot as BF16; the two FP32 scale slots both read ``mean_k``;
                # the second E2M1 level follows the first.
                assert record["logical"]["QM"] == "tab"
                assert record["dtypes"]["QM"] == "bf16"
                assert values["tab"].dtype == torch.bfloat16
                assert values["tab"] is tensors["tab"]
                assert record["logical"]["q_scale"] == "mean_k"
                assert record["logical"]["k_scale"] == "mean_k"
                assert values["mean_k"].dtype == torch.float32
                assert values["mean_k"] is tensors["mean_k"]
                assert record["logical"]["QL"] == "q4l"
                assert record["logical"]["KL"] == "k4l"
                assert (
                    record["logical"]["SFQL"] == "q_sf_l"
                    and record["logical"]["SFKL"] == "k_sf_l"
                )
                assert "qm32" not in values and "kc32" not in values
            assert set(values) == {
                record["logical"][key]
                for kind, key in record["arg_plan"]
                if kind != "grid"
            }
        if arch == "sm_120a":
            # Tensor-map ranks of the attention: token-major INT8 code rows, [rows, 16] V scale tiles, transposed
            # V; the INT8 offset rows and amax statistics per (batch, Q block, head) / (head, key block).
            assert tensors["q8"].shape == (2 * 1152, 3, 128) and tensors[
                "k8"
            ].shape == (2 * 1152, 3, 128)
            assert tensors["v_sf"].shape == (3 * 2 * 9 * 64, 16) and tensors[
                "vt4"
            ].shape == (3, 128, 1152)
            assert tensors["qm8"].shape == (2 * 9, 3, 128) and tensors[
                "k_amax"
            ].shape == (3, 2 * 9)
            assert tensors["k_minmax"].shape == (2 * 9, 3, 2, 128)
            backend._unit_table(layout_, workspace)
            table = tensors["unit_table"]
            assert table.shape == (54, 4)
            assert table[0].tolist() == [0, 0, 0, 1100]
            assert table[9].tolist() == [0, 1 << 16, 0, 1100]
            assert table[27].tolist() == [1, 0, 1152, 1152 + 1100]
            assert table[-1].tolist() == [1, (2 << 16) | 8, 1152, 1152 + 1100]
        else:
            assert tensors["q4"].shape == (6 * 1536, 64) and tensors["q_sf"].shape == (
                6 * 1536 // 4,
                32,
            )
            # The second E2M1 level mirrors the first.
            assert tensors["q4l"].shape == tensors["q4"].shape
            assert tensors["q_sf_l"].shape == tensors["q_sf"].shape
            assert tensors["k4"].shape == (6 * 1152, 64) and tensors["k_sf"].shape == (
                6 * 1152 // 4,
                32,
            )
            assert tensors["k4l"].shape == tensors["k4"].shape
            assert tensors["k_sf_l"].shape == tensors["k_sf"].shape
            assert tensors["vt4"].shape == (6 * 128, 576) and tensors[
                "v_sf_lo"
            ].shape == (6 * 1152 // 8, 32)
            # The table program's FP32 operands (one row per padded Q block, one per padded key) and its BF16
            # image: 16 BF16 per row, 64 rows per (group of 8 Q blocks, 64-key tile); 12 blocks -> 2 groups,
            # 1152 keys -> 18 tiles.
            assert tensors["qm32"].dtype == torch.float32
            assert tensors["qm32"].shape == (6 * 12, 128)
            assert tensors["kc32"].dtype == torch.float32
            assert tensors["kc32"].shape == (6 * 1152, 128)
            assert tensors["tab"].dtype == torch.bfloat16
            assert tensors["tab"].shape == (6 * 2 * 18 * 64, 16)
            assert tensors["partials"].shape == (6 * 3, 128)
            # The split partials of the 18 x 4 chunk items and two counters per split tile.
            assert tensors["part_o"].shape == (72, 512, 128)
            assert tensors["part_ml"].shape == (72, 512, 2)
            assert tensors["split_counters"].shape == (36,)
    # A record binding a dtype the host does not view as is refused before anything is loaded.
    record = dict(loader.MODULES[loader.records_for("sm_120a")["quantize"]])
    record["dtypes"] = {**record["dtypes"], "q8": "f16"}
    layout_ = backend.workspace_layout(
        "sm_120a", batch=batch, heads=heads, seqlen=seqlen, sm_count=148
    )
    workspace = torch.zeros(layout_.total_bytes, dtype=torch.uint8)
    tensors = {name: layout_.view(workspace, name) for name in layout_.regions}
    tensors.update(q=q, k=k, v=v, out=out)
    with pytest.raises(RuntimeError, match="states dtype 'f16' for 'q8'"):
        backend._values_for(record, tensors=tensors, common=common, sm_count=148)


def test_workspace_layout_is_the_vocabulary_on_aligned_slots():
    for arch in loader.ARCHES:
        layout = backend.workspace_layout(
            arch, batch=2, heads=5, seqlen=1100, sm_count=148
        )
        offsets = [region.offset for region in layout.regions.values()]
        assert offsets == sorted(offsets) and all(
            offset % backend.ALIGN == 0 for offset in offsets
        )
        assert set(layout.regions) <= set(loader.LOGICAL_TENSORS)
        # SM100 / SM103: the two E2M1 levels of Q and K with their scales (8), the transposed V and its two scale
        # tiles (3), the table program's operands and image (3), the statistics (3) and the split regions (3);
        # SM120 (int8_blk): the statistics (8, incl. the INT8 amax rows), the INT8 Q / K codes and offset rows (3),
        # the transposed V and its scales (2) and the unit table.
        assert len(layout.regions) == (14 if arch == "sm_120a" else 20), arch
        assert layout.total_bytes % backend.ALIGN == 0 and layout.total_bytes >= sum(
            r.nbytes for r in layout.regions.values()
        )
        chain = loader.records_for(arch)
        for name in chain.values():
            record = loader.MODULES[name]
            for kind, key in record["arg_plan"]:
                if kind == "buffer" and record["logical"][key] not in (
                    "q",
                    "k",
                    "v",
                    "out",
                ):
                    assert record["logical"][key] in layout.regions, (arch, name, key)
            if record["workspace_rule"] is not None:
                assert set(record["workspace_rule"]["regions"]) <= set(layout.regions)
    bytes_small = backend.workspace_layout(
        "sm_100a", batch=1, heads=1, seqlen=1, sm_count=148
    ).total_bytes
    bytes_large = backend.workspace_layout(
        "sm_100a", batch=1, heads=1, seqlen=4800, sm_count=148
    ).total_bytes
    assert 0 < bytes_small < bytes_large


# (batch, heads, seqlen, sm_count) -> the SM100 / SM103 attention's last-wave split plan, by hand: n_clusters =
# sm_count // 2 persistent clusters over total_tiles = B * H * ceil(S / 512) tiles; the n_split = total_tiles %
# n_clusters tiles of the last wave are split into min(4, ceil(S / 128), n_clusters // n_split) key chunks each,
# none when that is below 2; grid = 2 * min(n_clusters, n_items).
SPLIT_PLANS = [
    (
        # 2560 tiles on 74 clusters leave 44 for the last wave: 74 // 44 = 1 chunk, so no split.
        (1, 40, 32760, 148),
        {
            "total_tiles": 2560,
            "key_blocks": 256,
            "n_split": 0,
            "n_chunks": 1,
            "n_full_tiles": 2560,
            "n_items": 2560,
            "grid": 148,
        },
    ),
    (
        # 320 tiles on 74 clusters: 24 last-wave tiles x 3 chunks after 296 whole tiles.
        (1, 5, 32760, 148),
        {
            "total_tiles": 320,
            "key_blocks": 256,
            "n_split": 24,
            "n_chunks": 3,
            "n_full_tiles": 296,
            "n_items": 368,
            "grid": 148,
        },
    ),
    (
        # Two tiles of 8 key blocks: both split into 4 chunks; 8 items on 8 clusters.
        (1, 1, 1000, 148),
        {
            "total_tiles": 2,
            "key_blocks": 8,
            "n_split": 2,
            "n_chunks": 4,
            "n_full_tiles": 0,
            "n_items": 8,
            "grid": 16,
        },
    ),
    (
        # 2560 tiles on 80 clusters: whole waves, no split.
        (1, 40, 32760, 160),
        {
            "total_tiles": 2560,
            "key_blocks": 256,
            "n_split": 0,
            "n_chunks": 1,
            "n_full_tiles": 2560,
            "n_items": 2560,
            "grid": 160,
        },
    ),
]


@pytest.mark.parametrize("problem,expected", SPLIT_PLANS)
def test_split_plan_of_the_sm100_attention(problem, expected):
    """``split_plan`` is the kernel's schedule as a pure function of (B * H * ceil(S / 512), ceil(S / 128),
    sm_count), the same on SM100 and SM103; ``launch_grid`` is its grid; the SM120 record has no split plan."""
    batch, heads, seqlen, sm_count = problem
    for arch in ("sm_100a", "sm_103a"):
        record = loader.MODULES[loader.records_for(arch)["attention"]]
        assert record["grid_rule"]["kind"] == "split_persistent_tiles"
        plan = loader.split_plan(
            record, batch=batch, heads=heads, seqlen=seqlen, sm_count=sm_count
        )
        assert plan == {"n_clusters": sm_count // 2, **expected}
        assert loader.launch_grid(
            record, batch=batch, heads=heads, seqlen=seqlen, sm_count=sm_count
        ) == (expected["grid"], 1, 1)
    with pytest.raises(ValueError, match="split_persistent_tiles"):
        loader.split_plan(
            loader.MODULES[loader.records_for("sm_120a")["attention"]],
            batch=batch,
            heads=heads,
            seqlen=seqlen,
            sm_count=sm_count,
        )


def _slot(nbytes):
    """``nbytes`` rounded up to the next 256-byte workspace slot."""
    return -(-nbytes // backend.ALIGN) * backend.ALIGN


def _sm100_regions_bytes(batch, heads, seqlen, *, rows, n_split):
    """The SM100 / SM103 regions by hand, in carve order with their unpadded byte counts: per (batch, head) the two
    E2M1 levels of Q (``64 Sq`` codes and ``8 Sq`` scales each) and of K (``64 Skv`` / ``8 Skv`` each), the
    transposed V (``64 Skv``) with its two scale tiles (``4 Skv`` each), the table program's FP32 operands
    (``qm32``: ``4 Sq``, one 512-byte row per 128-row Q block; ``kc32``: ``512 Skv``) and its BF16 image (``tab``:
    ``32 ceil(Sq / 1024) Skv`` -- 64 rows of 16 BF16 per (group of 8 Q blocks, 64-key tile)), ``mean_k`` (512),
    ``partials`` (``512 ceil(S / 512)``) and one u32 counter; then ``rows`` split-partial rows (256 KiB of ``part_o``
    and 4 KiB of ``part_ml`` each) and ``max(1, 2 n_split)`` split counters."""
    bh = batch * heads
    sq, skv, t512 = -(-seqlen // 512) * 512, -(-seqlen // 128) * 128, -(-seqlen // 512)
    groups, key_tiles = -(-(sq // 128) // 8), skv // 64
    return [
        ("q4", bh * sq * 64),
        ("q_sf", bh * sq * 8),
        ("q4l", bh * sq * 64),
        ("q_sf_l", bh * sq * 8),
        ("k4", bh * skv * 64),
        ("k_sf", bh * skv * 8),
        ("k4l", bh * skv * 64),
        ("k_sf_l", bh * skv * 8),
        ("vt4", bh * skv * 64),
        ("v_sf_lo", bh * skv * 4),
        ("v_sf_hi", bh * skv * 4),
        ("qm32", bh * (sq // 128) * 128 * 4),
        ("kc32", bh * skv * 128 * 4),
        ("tab", bh * groups * key_tiles * 64 * 16 * 2),
        ("mean_k", bh * 128 * 4),
        ("partials", bh * t512 * 128 * 4),
        ("counters", bh * 4),
        ("part_o", rows * 512 * 128 * 4),
        ("part_ml", rows * 512 * 2 * 4),
        ("split_counters", 4 * max(1, 2 * n_split)),
    ]


def _sm100_workspace_bytes(batch, heads, seqlen, *, rows, n_split):
    """The SM100 / SM103 byte count: every region of :func:`_sm100_regions_bytes` rounded up to its own 256-byte
    slot, as ``workspace_layout`` carves them (each region starts on the boundary after the previous one)."""
    return sum(
        _slot(nbytes)
        for _name, nbytes in _sm100_regions_bytes(
            batch, heads, seqlen, rows=rows, n_split=n_split
        )
    )


def test_sm100_workspace_is_sized_for_the_split_plan_of_the_device():
    """The SM100 / SM103 workspace carries the split partials of the device's plan: one row and one counter when
    nothing splits, ``n_split * n_chunks`` rows and ``2 * n_split`` counters otherwise; the SM120 workspace does not
    read the SM count."""
    for arch in ("sm_100a", "sm_103a"):
        for sm_count in (148, 160):  # (1, 40, 32760) splits nothing on either device
            layout = backend.workspace_layout(
                arch, batch=1, heads=40, seqlen=32760, sm_count=sm_count
            )
            assert layout.total_bytes == _sm100_workspace_bytes(
                1, 40, 32760, rows=1, n_split=0
            )
            # Region by region, in carve order, each on its own slot.
            expected = _sm100_regions_bytes(1, 40, 32760, rows=1, n_split=0)
            assert [name for name, _ in expected] == list(layout.regions)
            offset = 0
            for name, nbytes in expected:
                region = layout.regions[name]
                assert (region.offset, region.nbytes) == (offset, nbytes), name
                offset += _slot(nbytes)
            assert offset == layout.total_bytes
            # Sq = Skv = 32768: the table program's image and operands dominate -- tab = 40 * 32 groups * 512
            # key tiles * 64 rows * 32 bytes, kc32 = 40 * 32768 keys * 512 bytes, qm32 = 40 * 256 blocks * 512.
            assert layout.regions["tab"].nbytes == 1_342_177_280
            assert layout.regions["kc32"].nbytes == 671_088_640
            assert layout.regions["qm32"].nbytes == 5_242_880
            part_o, part_ml = layout.regions["part_o"], layout.regions["part_ml"]
            assert part_o.shape == (1, 512, 128) and part_o.nbytes == 262144
            assert part_ml.shape == (1, 512, 2) and part_ml.nbytes == 4096
            counters = layout.regions["split_counters"]
            assert counters.shape == (1,) and counters.nbytes == 4
        # (1, 5, 32760) on 148 SMs: 24 split tiles x 3 chunks = 72 rows, 48 counters.
        layout = backend.workspace_layout(
            arch, batch=1, heads=5, seqlen=32760, sm_count=148
        )
        assert layout.regions["part_o"].shape == (72, 512, 128)
        assert layout.regions["part_o"].nbytes == 72 * 262144
        assert layout.regions["part_ml"].shape == (72, 512, 2)
        assert layout.regions["split_counters"].shape == (48,)
        assert layout.total_bytes == _sm100_workspace_bytes(
            1, 5, 32760, rows=72, n_split=24
        )
        # (1, 1, 1000) on 148 SMs: two tiles x 4 chunks.
        layout = backend.workspace_layout(
            arch, batch=1, heads=1, seqlen=1000, sm_count=148
        )
        assert layout.regions["part_o"].shape == (8, 512, 128)
        assert layout.regions["split_counters"].shape == (4,)
        assert layout.total_bytes == _sm100_workspace_bytes(
            1, 1, 1000, rows=8, n_split=2
        )
        assert list(layout.regions)[-3:] == ["part_o", "part_ml", "split_counters"]
        assert layout.sm_count == 148
    small = backend.workspace_layout(
        "sm_120a", batch=1, heads=5, seqlen=1000, sm_count=148
    )
    large = backend.workspace_layout(
        "sm_120a", batch=1, heads=5, seqlen=1000, sm_count=170
    )
    assert small.regions == large.regions and small.total_bytes == large.total_bytes
    assert "part_o" not in small.regions
    with pytest.raises(ValueError, match="at least 2 SMs"):
        backend.workspace_layout("sm_100a", batch=1, heads=1, seqlen=1, sm_count=1)
    with pytest.raises(ValueError, match="positive extents"):
        backend.workspace_layout("sm_100a", batch=1, heads=1, seqlen=1, sm_count=0)


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library, architecture) the registry
    references, each with the kernel and binding sources under ``csrc/`` of this package, their digests, a closure
    digest and the compile options the program's kernel was validated with; the registry itself records the host
    plan only.
    """
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(loader.ARCHES)
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"]
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    # Eight records, eleven (library, architecture) entries: the SM100 / SM103 stats, quantize and delta programs
    # are each built twice, every other record once.
    assert len(loader.MODULES) == 8
    assert sum(len(record["arches"]) for record in loader.MODULES.values()) == 11
    for name, record in loader.MODULES.items():
        library = record["library"]
        assert record["kernel"] == library
        assert re.fullmatch(r"cake_nvfp4_attention_[0-9a-f]{20}", library)
        assert record["kernel_symbol"] == f"kernel_{library}"
        assert record["ffi_entry"] == "run"
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert record["arches"] and set(record["arches"]) <= set(loader.ARCHES), name
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["dynamic_smem_bytes"] >= 0
        assert len(launch["block"]) == 3 and len(launch["cluster"]) == 3
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ]
        for arch in record["arches"]:
            entry_name = loader.library_name(name, arch)
            referenced.add(entry_name)
            entry = libraries[entry_name]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{library}_kernel.cu",
                f"csrc/{library}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert entry["nvcc_flags"] == _expected_nvcc_flags(record, arch), entry_name
            assert re.fullmatch(r"\d+\.\d+", entry["min_cuda"]), entry_name
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
