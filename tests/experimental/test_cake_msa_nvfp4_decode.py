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
"""

import pytest
import torch

from flashinfer.experimental.msa_nvfp4_decode import cake_jit
from flashinfer.experimental.msa_nvfp4_decode.cake_backend import (
    DATA_DIM,
    HEAD_DIM,
    MAIN_KWARGS,
    PAGE_SIZE,
    SCALE_DIM,
    SPLIT_FACTORS,
    STATS_PER_SLOT,
    SUPPORTED_COMPUTE_CAPABILITIES,
    WORKSPACE_ALIGN,
    generated_program_available,
    msa_nvfp4_decode_workspace_size,
    persistent_cta_capacity,
    prepare_msa_nvfp4_sparse_decode as prepare_backend,
    split_factor,
    validate_msa_nvfp4_decode_inputs,
    workspace_layout,
)
from flashinfer.msa_ops import _nvfp4_decode_sm100 as upstream
from flashinfer.msa_ops import (
    msa_sparse_decode_attention,
    prepare_msa_nvfp4_sparse_decode,
)
from tests.test_helpers.cake_msa_nvfp4_inputs import (
    build_decode_inputs,
    page_layout,
    upstream_route_kwargs,
)

ATOL = RTOL = 1e-2
# FlashInfer's acceptance for its own NVFP4 route
# (tests/msa_ops/test_msa_nvfp4_decode_sm100.py ``_assert_peer``): the route
# accumulates at lower precision than the FP32 oracle on some head geometries,
# so it is compared on whole-output cosine / relative Frobenius error rather
# than element-wise tolerance.  The generated program is held to ATOL/RTOL.
PEER_MIN_COSINE = 0.99
PEER_MAX_REL_FRO = 0.06


def _assert_peer(
    actual, expected, *, min_cosine=PEER_MIN_COSINE, max_rel_fro=PEER_MAX_REL_FRO
):
    a = actual.float().reshape(1, -1)
    b = expected.float().reshape(1, -1)
    cosine = torch.nn.functional.cosine_similarity(a, b).item()
    rel = ((a - b).norm() / b.norm().clamp(min=1e-12)).item()
    assert cosine >= min_cosine, f"cosine {cosine}"
    assert rel <= max_rel_fro, f"rel_fro {rel}"


def _supported_device() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program():
    if not _supported_device():
        pytest.skip("requires a compute capability 10.0/10.3 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated NVFP4 MSA decode program not registered for this device")


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_page_layout_matches_the_route_contract(num_kv_heads):
    ours = page_layout(num_kv_heads)
    theirs = upstream.page_layout(num_kv_heads)
    assert ours["k_scale"] == theirs["k_scale_byte_offset"]
    assert ours["v_data"] == theirs["v_data_byte_offset"]
    assert ours["v_scale"] == theirs["v_scale_byte_offset"]
    assert ours["page_bytes"] == theirs["page_bytes"]
    assert ours["page_bytes"] == 2 * num_kv_heads * PAGE_SIZE * (DATA_DIM + SCALE_DIM)


@pytest.mark.parametrize(
    "items,capacity,max_pages,expected",
    [
        (512, 148, 64, 1),  # tp1 batch 128: the items alone fill the machine
        (128, 148, 512, 1),  # tp1 batch 32 at 64K: 256 CTAs would exceed 148
        (64, 148, 512, 2),  # tp1 batch 16 at 64K
        (64, 160, 512, 2),
        (32, 148, 782, 4),  # tp1 batch 8 at 100K
        (32, 160, 782, 4),
        (4, 148, 1563, 8),  # tp1 batch 1 at 200K
        (1, 160, 8192, 8),  # tp4 batch 1 at 1M
        (8, 148, 3, 1),  # 257-token tail: two page pairs, never split
        (8, 148, 6, 1),  # three pairs: still below the four-pair threshold
        (8, 148, 7, 4),  # four pairs admit splitting, at most one pair per split
        (8, 148, 8, 4),
        (8, 148, 16, 8),  # eight pairs: eight-way
        (20, 148, 16, 4),  # 20 items: 40 and 80 CTAs fit, 160 do not
        (18, 148, 16, 8),  # 18 items: 36, 72 and 144 CTAs fit
    ],
)
def test_split_factor_rule(items, capacity, max_pages, expected):
    assert split_factor(items, capacity, max_pages) == expected


def test_workspace_layout_sizes_and_alignment():
    assert workspace_layout(37, 1) == {"total": 0}
    items, splits = 32, 4
    layout = workspace_layout(items, splits)
    slots = items * splits
    assert layout["partial_o"] == (0, slots * STATS_PER_SLOT * HEAD_DIM * 4)
    assert layout["partial_m"][1] == slots * STATS_PER_SLOT * 4
    assert layout["partial_d"][1] == slots * STATS_PER_SLOT * 4
    assert layout["split_completion"][1] == items * 4
    offsets = [
        layout[k][0]
        for k in ("partial_o", "partial_m", "partial_d", "split_completion")
    ]
    assert offsets == sorted(offsets)
    assert all(o % WORKSPACE_ALIGN == 0 for o in offsets)
    assert layout["total"] % WORKSPACE_ALIGN == 0
    assert layout["total"] >= sum(
        layout[k][1]
        for k in ("partial_o", "partial_m", "partial_d", "split_completion")
    )
    with pytest.raises(ValueError, match="split factor"):
        workspace_layout(4, 3)


def _cpu_inputs(seq_lens=(300, 257), num_kv_heads=4, group_size=16, seqlen_q=1):
    return build_decode_inputs(
        seq_lens,
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        seqlen_q=seqlen_q,
        device="cpu",
        seed=3,
    )


def _validate(inputs, **overrides):
    args = dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        q2k_indices=inputs["q2k_indices"],
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        seqlen_q=inputs["seqlen_q"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
        out=None,
        lse=None,
    )
    args.update(overrides)
    return validate_msa_nvfp4_decode_inputs(
        args.pop("q"),
        args.pop("k"),
        args.pop("v"),
        args.pop("q2k_indices"),
        args.pop("k_scale"),
        args.pop("v_scale"),
        args.pop("page_table"),
        args.pop("seqused_k"),
        **args,
    )


def test_validate_accepts_contract_inputs():
    inputs = _cpu_inputs((300, 257, 5000), num_kv_heads=2, group_size=16, seqlen_q=2)
    assert _validate(inputs) == (3, 32, 2, inputs["num_pages"], inputs["max_pages"])
    # E4M3-typed scale views are the same bytes.
    assert (
        _validate(
            inputs,
            k_scale=inputs["k_scale"].view(torch.float8_e4m3fn),
            v_scale=inputs["v_scale"].view(torch.float8_e4m3fn),
        )[0]
        == 3
    )


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda i: dict(q=i["q"].to(torch.float16)), "bfloat16"),
        (lambda i: dict(q=i["q"][:, :, :64]), "bfloat16"),
        (lambda i: dict(seqlen_q=33), "seqlen_q"),
        (lambda i: dict(seqlen_q=2), "batch"),
        (lambda i: dict(q2k_indices=i["q2k_indices"][:, :, :8].contiguous()), "topk"),
        (
            lambda i: dict(q2k_indices=i["q2k_indices"].transpose(0, 1).contiguous()),
            "q2k_indices",
        ),
        (lambda i: dict(q2k_indices=i["q2k_indices"].to(torch.int64)), "q2k_indices"),
        (lambda i: dict(k_scale=i["k_scale"][..., :4]), "k_scale"),
        (lambda i: dict(v_scale=i["v_scale"].to(torch.int8)), "v_scale"),
        (
            # A token stride of two rows: the head tile is no longer contiguous.
            lambda i: dict(
                k=torch.as_strided(
                    i["pool"],
                    tuple(i["k"].shape),
                    (i["k"].stride(0), i["k"].stride(1), 2 * DATA_DIM, 1),
                )
            ),
            "contiguously",
        ),
        (lambda i: dict(page_table=i["page_table"].to(torch.int64)), "page_table"),
        (lambda i: dict(page_table=i["page_table"][:1]), "page_table"),
        (lambda i: dict(seqused_k=i["seqused_k"][:1]), "seqused_k"),
        (lambda i: dict(k_global_scale=0.0), "positive"),
        (lambda i: dict(out=torch.empty_like(i["q"], dtype=torch.float16)), "out"),
        (lambda i: dict(lse=torch.empty(i["q"].shape[0], 3)), "lse"),
    ],
)
def test_validate_rejects(mutate, message):
    inputs = _cpu_inputs()
    with pytest.raises(ValueError, match=message):
        _validate(inputs, **mutate(inputs))


def test_validate_rejects_a_gqa_group_above_sixteen():
    inputs = _cpu_inputs(num_kv_heads=1, group_size=16)
    with pytest.raises(ValueError, match="GQA group"):
        _validate(inputs, q=torch.cat([inputs["q"], inputs["q"]], dim=1))


def test_prepare_requires_cuda_tensors():
    inputs = _cpu_inputs()
    with pytest.raises(ValueError, match="CUDA"):
        prepare_backend(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            k_scale=inputs["k_scale"],
            v_scale=inputs["v_scale"],
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            k_global_scale=inputs["k_global_scale"],
            v_global_scale=inputs["v_global_scale"],
        )


def test_public_entry_routes_only_to_cake():
    inputs = _cpu_inputs()
    with pytest.raises(ValueError, match="backend"):
        prepare_msa_nvfp4_sparse_decode(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            k_scale=inputs["k_scale"],
            v_scale=inputs["v_scale"],
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            k_global_scale=1.0,
            v_global_scale=1.0,
            backend="cute",
        )


EXPECTED_ARG_PLAN = (
    [["tma_buffer", n] for n in MAIN_KWARGS[:5]]
    + [["buffer", n] for n in MAIN_KWARGS[5:16]]
    + [["parameter", n] for n in MAIN_KWARGS[16:23]]
    + [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
)


def test_registry_records_match_the_host_binding():
    if not cake_jit.MODULES:
        pytest.skip("no generated program registered in this checkout")
    root = cake_jit.Path(cake_jit.__file__).resolve().parent / "csrc"
    seen = set()
    per_arch_ctas = {}
    for name, record in cake_jit.MODULES.items():
        assert record["arch"] in SUPPORTED_COMPUTE_CAPABILITIES.values()
        assert int(record["splits"]) in SPLIT_FACTORS
        assert int(record["ctas_per_sm"]) >= 1
        per_arch_ctas.setdefault(record["arch"], set()).add(int(record["ctas_per_sm"]))
        key = (record["arch"], int(record["splits"]))
        assert key not in seen, f"duplicate registration for {key}"
        seen.add(key)
        main = record["main"]
        assert main["ffi_entry"] == "run"
        assert [list(item) for item in main["arg_plan"]] == EXPECTED_ARG_PLAN
        assert record["closure_sha256"] == main["closure_sha256"]
        assert len(main["sources"]) == 2
        for relative in main["sources"]:
            assert (root / relative).is_file(), relative
            assert relative.startswith(f"cake_msa_nvfp4_decode/{record['arch']}/")
        assert cake_jit.select_module(record["arch"], int(record["splits"])) == name
    assert all(len(v) == 1 for v in per_arch_ctas.values())
    for arch in per_arch_ctas:
        assert cake_jit.registered_split_factors(arch) == SPLIT_FACTORS
    with pytest.raises(NotImplementedError):
        cake_jit.select_module("sm_90a", 1)


# ---------------------------------------------------------------------------
# Device (SM100 / SM103 with the generated program registered)
# ---------------------------------------------------------------------------


def _oracle(inputs):
    return upstream.reference(
        q=inputs["q"],
        k_data=inputs["k"],
        v_data=inputs["v"],
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        q2k_indices=inputs["q2k_indices"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        softmax_scale=inputs["softmax_scale"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
        out=torch.empty_like(inputs["q"]),
        seqlen_q=inputs["seqlen_q"],
        causal=True,
    )


def _prepare(inputs, *, workspace=None, out=None, lse=None):
    device = inputs["q"].device
    batch = int(inputs["seqused_k"].numel())
    num_kv_heads = int(inputs["k"].shape[1])
    if workspace is None:
        workspace = torch.empty(
            msa_nvfp4_decode_workspace_size(
                batch, num_kv_heads, device, seqlen_q=inputs["seqlen_q"]
            ),
            dtype=torch.uint8,
            device=device,
        )
    runner = prepare_msa_nvfp4_sparse_decode(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q2k_indices"],
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
        workspace_buffer=workspace,
        seqlen_q=inputs["seqlen_q"],
        softmax_scale=inputs["softmax_scale"],
        out=out,
        lse=lse,
    )
    return runner, workspace


def _run_and_check(seq_lens, num_kv_heads, *, group_size=16, seqlen_q=1, seed):
    inputs = build_decode_inputs(
        seq_lens,
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        seqlen_q=seqlen_q,
        device="cuda",
        seed=seed,
    )
    runner, workspace = _prepare(inputs)
    items = len(seq_lens) * seqlen_q * num_kv_heads
    assert runner.splits == split_factor(
        items, persistent_cta_capacity(inputs["q"].device), inputs["max_pages"]
    )
    runner.out.fill_(float("nan"))
    out = runner()
    torch.cuda.synchronize()
    expected = _oracle(inputs)
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
    assert torch.isfinite(runner.lse).all()
    assert runner.lse.shape == (inputs["q"].shape[0], inputs["q"].shape[1])
    return inputs, runner, workspace


@pytest.mark.parametrize(
    "seq_lens,num_kv_heads,group_size,seqlen_q",
    [
        ([257, 300], 4, 16, 1),  # 257-token tail: two page pairs, unsplit
        ([8192] * 4, 4, 16, 1),  # 16 items: eight-way split
        ([65536] * 16, 4, 16, 1),  # 64 items: two-way split
        ([100_000] * 8, 4, 16, 1),  # 32 items: four-way split
        ([1, 129, 8193, 300, 5000, 777, 65536, 4096], 1, 16, 1),  # ragged tp4 geometry
        ([4096] * 3, 1, 8, 1),  # tp8 geometry: eight query heads per KV head
        ([5000, 130], 4, 16, 2),  # two query tokens per request
        ([8192] * 32, 4, 16, 4),  # 512 items: unsplit with four tokens
        ([2048] * 4, 2, 16, 8),  # eight tokens per request, tp2 geometry
    ],
)
def test_decode_matches_the_fp32_oracle(seq_lens, num_kv_heads, group_size, seqlen_q):
    _require_program()
    _run_and_check(
        seq_lens,
        num_kv_heads,
        group_size=group_size,
        seqlen_q=seqlen_q,
        seed=len(seq_lens) + seqlen_q,
    )


@pytest.mark.parametrize(
    "seq_lens,num_kv_heads",
    [([8192] * 8, 4), ([300, 65536, 1029], 1)],
)
def test_matches_the_existing_nvfp4_route(seq_lens, num_kv_heads):
    """Both readers of the layout contract agree with the oracle on the same pages.

    The generated program is checked element-wise in ``_run_and_check``; the
    existing route is checked at its own peer tolerance.
    """
    _require_program()
    inputs, runner, _ = _run_and_check(seq_lens, num_kv_heads, seed=41)
    route_out = torch.empty_like(inputs["q"])
    msa_sparse_decode_attention(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q2k_indices"],
        out=route_out,
        **upstream_route_kwargs(inputs),
    )
    torch.cuda.synchronize()
    expected = _oracle(inputs)
    _assert_peer(route_out, expected)
    _assert_peer(runner.out, route_out)


def test_serves_a_geometry_the_existing_route_declines():
    """Eight query heads per KV head is outside the route's allowlist; the
    generated program serves it from the same pages."""
    _require_program()
    inputs, _, _ = _run_and_check([4096] * 3, 1, group_size=8, seed=43)
    with pytest.raises(NotImplementedError):
        msa_sparse_decode_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            out=torch.empty_like(inputs["q"]),
            **upstream_route_kwargs(inputs),
        )


def test_graph_replay_follows_new_queries_and_selections():
    """Capture once; replay after new queries and a new top-k selection are written in place."""
    _require_program()
    inputs, runner, _ = _run_and_check(
        [8192] * 4, 4, seed=11
    )  # eight-way split: exercises the workspace
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    gen = torch.Generator(device="cuda").manual_seed(2026)
    for step in range(3):
        inputs["q"].copy_(
            torch.randn(inputs["q"].shape, generator=gen, device="cuda").to(
                torch.bfloat16
            )
        )
        if step:
            fresh = build_decode_inputs(
                [8192] * 4, num_kv_heads=4, device="cuda", seed=100 + step
            )
            inputs["q2k_indices"].copy_(fresh["q2k_indices"])
            del fresh
        runner.out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = _oracle(inputs)
        torch.testing.assert_close(runner.out, expected, atol=ATOL, rtol=RTOL)


def test_launch_makes_no_allocation():
    _require_program()
    _, runner, _ = _run_and_check(
        [65536] * 2, 4, seed=5
    )  # split path binds the workspace
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0


def test_split_batches_require_a_sized_workspace():
    _require_program()
    inputs = build_decode_inputs([65536] * 2, num_kv_heads=4, device="cuda", seed=9)
    with pytest.raises(ValueError, match="workspace_buffer"):
        _prepare(inputs, workspace=torch.empty(0, dtype=torch.uint8, device="cuda"))
    kwargs = dict(
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
    )
    with pytest.raises(ValueError, match="workspace_buffer"):
        prepare_msa_nvfp4_sparse_decode(
            inputs["q"], inputs["k"], inputs["v"], inputs["q2k_indices"], **kwargs
        )
    # An unsplit batch needs no workspace at all.
    big = build_decode_inputs([4096] * 40, num_kv_heads=4, device="cuda", seed=10)
    runner = prepare_msa_nvfp4_sparse_decode(
        big["q"],
        big["k"],
        big["v"],
        big["q2k_indices"],
        k_scale=big["k_scale"],
        v_scale=big["v_scale"],
        page_table=big["page_table"],
        seqused_k=big["seqused_k"],
        k_global_scale=big["k_global_scale"],
        v_global_scale=big["v_global_scale"],
    )
    assert runner.splits == 1
    runner()
    torch.cuda.synchronize()
    torch.testing.assert_close(runner.out, _oracle(big), atol=ATOL, rtol=RTOL)
