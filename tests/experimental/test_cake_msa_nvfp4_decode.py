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
    arch_for,
    DATA_DIM,
    generated_program_available,
    PAGE_SIZE,
    persistent_cta_capacity,
    prepare_msa_nvfp4_sparse_decode as prepare_backend,
    SCALE_DIM,
    short_grid,
    short_program_record,
    short_route_applies,
    split_factor,
    SPLIT_FACTORS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    tail_plan,
    TOPK,
    validate_msa_nvfp4_decode_inputs,
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
        pytest.skip("requires a compute capability 10.0/10.3/10.7 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated NVFP4 MSA decode program not registered for this device")


def _require_existing_route():
    """The peer checks below need the existing ``msa_ops`` NVFP4 route on this
    device. That route is gated by its own capability map, separately from the
    generated programs of the ``cake`` backend, so a device the backend serves
    may still be one the route declines."""
    if upstream._target_for(torch.device("cuda")) is None:
        pytest.skip("the existing NVFP4 route is not enabled on this device")


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


def test_toolchain_guard_declines_programs_the_toolkit_cannot_compile(monkeypatch):
    """A checkout may register sm_107a programs on a toolkit without compute_107a:
    the JIT declines them up front and the backend reports no program for 10.7."""
    import flashinfer.compilation_context as compilation_context

    monkeypatch.setattr(compilation_context, "_nvcc_supports_sm107", lambda: False)
    cake_jit.toolchain_supports.cache_clear()
    try:
        assert cake_jit.toolchain_supports("sm_100a")
        assert cake_jit.toolchain_supports("sm_103a")
        assert not cake_jit.toolchain_supports("sm_107a")
        assert not cake_jit.toolchain_supports("sm_120a")
        for name in cake_jit.programs_for("sm_107a"):
            cake_jit.gen_program_spec.cache_clear()
            with pytest.raises(RuntimeError, match="compute_107a"):
                cake_jit.gen_program_spec(name, "sm_107a")
            break
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (10, 7):
            assert not generated_program_available(torch.device("cuda"))
    finally:
        cake_jit.toolchain_supports.cache_clear()
        cake_jit.gen_program_spec.cache_clear()


@pytest.mark.parametrize("arch", sorted(SUPPORTED_COMPUTE_CAPABILITIES.values()))
def test_every_architecture_registers_the_split_ladder(arch):
    """Behaviour of the registry: each supported architecture resolves a plain
    persistent program for every split factor and (optionally) one short-item
    program, each with a source pair the JIT can hand to nvcc."""
    if not cake_jit.programs_for(arch):
        pytest.skip(f"no generated program registered for {arch}")
    assert cake_jit.registered_split_factors(arch) == SPLIT_FACTORS
    names = {cake_jit.select_program(arch, splits=s) for s in SPLIT_FACTORS}
    assert len(names) == len(SPLIT_FACTORS)  # one distinct program per factor
    for _name, record in cake_jit.tail_programs(arch):
        assert int(record["splits"]) > 1
        assert all(
            int(sms) > 0 and int(ctas) >= int(record["splits"])
            for sms, ctas in record["cluster_capacity"].items()
        )
        names.add(cake_jit.select_program(arch, splits=record["splits"], tail=True))
    short = cake_jit.short_program(arch)
    if short is not None:
        record = cake_jit.PROGRAMS[short]
        assert int(record["max_pages"]) >= 1 and int(record["cluster"]) >= 2
        assert int(record["max_clusters"]) >= 1
        names.add(short)
    root = cake_jit.Path(cake_jit.__file__).resolve().parent / "csrc"
    for name in names:
        record = cake_jit.PROGRAMS[name]
        assert arch in record["arches"]
        assert len(record["sources"]) == 2
        assert all((root / relative).is_file() for relative in record["sources"])
        if cake_jit.toolchain_supports(arch):
            spec = cake_jit.gen_program_spec(name, arch)
            assert spec.name == f"{name}_{arch}"
    for route in ("persistent", "short"):
        plan = cake_jit.ARG_PLANS.get(route)
        if plan:
            assert [n for k, n in plan if k == "grid"] == ["grid_x", "grid_y", "grid_z"]
    with pytest.raises(NotImplementedError):
        cake_jit.select_program("sm_90a", splits=1)
    assert cake_jit.short_program("sm_90a") is None
    assert cake_jit.tail_programs("sm_90a") == []


def test_tail_plan_rule():
    if not cake_jit.tail_programs("sm_107a"):
        pytest.skip("no last-round-split program registered for sm_107a")
    # 212 resident CTAs: 256 items are 1.21 -> 2 rounds plain, 1.5 with a two-way last round on a
    # 212-CTA grid; 512 items 2.42 -> 3 plain, 2.5 split.
    assert tail_plan("sm_107a", 256, 212, TOPK) == (2, 212)
    assert tail_plan("sm_107a", 512, 212, TOPK) == (2, 212)
    # One round already; 176 remainder items x 2 do not fit the grid; fewer than four page pairs
    # never split; an unknown part has no capacity entry; Blackwell registers no tail program.
    assert tail_plan("sm_107a", 128, 212, TOPK) is None
    assert tail_plan("sm_107a", 1024, 212, TOPK) is None
    assert tail_plan("sm_107a", 256, 212, 6) is None
    assert tail_plan("sm_107a", 256, 200, TOPK) is None
    assert tail_plan("sm_100a", 512, 148, TOPK) is None


# ---------------------------------------------------------------------------
# Device (SM100 / SM103 / SM107 with the generated program registered)
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


def _prepare(inputs, *, out=None, lse=None):
    return prepare_msa_nvfp4_sparse_decode(
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
        seqlen_q=inputs["seqlen_q"],
        softmax_scale=inputs["softmax_scale"],
        out=out,
        lse=lse,
    )


def _run_and_check(seq_lens, num_kv_heads, *, group_size=16, seqlen_q=1, seed):
    inputs = build_decode_inputs(
        seq_lens,
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        seqlen_q=seqlen_q,
        device="cuda",
        seed=seed,
    )
    runner = _prepare(inputs)
    items = len(seq_lens) * seqlen_q * num_kv_heads
    device = inputs["q"].device
    arch = arch_for(device)
    assert runner.arch == arch
    if short_route_applies(arch, inputs["max_pages"]):
        # Requests of at most the short program's page budget take the
        # eight-CTA-cluster program: one cluster per work item, no split.
        assert runner.route == "short"
        assert runner.splits == 1
        record = short_program_record(arch)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        assert runner.num_ctas == short_grid(items, sms, record)[0]
    else:
        assert runner.route == "persistent"
        capacity = persistent_cta_capacity(device)
        expected_splits = split_factor(items, capacity, inputs["max_pages"])
        # A part that registers a last-round-split program (sm_107a) turns a
        # plain persistent launch whose last round is short into S-way cluster
        # units for the remainder items; the plan is the production rule.
        plan = (
            tail_plan(arch, items, capacity, inputs["max_pages"])
            if expected_splits == 1
            else None
        )
        if plan is None:
            assert runner.splits == expected_splits
            assert runner.tail is False
        else:
            assert runner.tail is True
            assert (runner.splits, runner.num_ctas) == plan
        assert runner.program == cake_jit.select_program(
            arch, splits=runner.splits, tail=runner.tail
        )
    runner.out.fill_(float("nan"))
    out = runner()
    torch.cuda.synchronize()
    expected = _oracle(inputs)
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
    assert torch.isfinite(runner.lse).all()
    assert runner.lse.shape == (inputs["q"].shape[0], inputs["q"].shape[1])
    return inputs, runner


@pytest.mark.parametrize(
    "seq_lens,num_kv_heads,group_size,seqlen_q",
    [
        (
            [257, 300],
            4,
            16,
            1,
        ),  # 257-token tail: three pages, short program when registered
        (
            [385, 300],
            1,
            8,
            1,
        ),  # eight query heads per KV head through the short program
        ([513, 385], 4, 16, 1),  # five pages: just beyond the short program's budget
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


def _batch_requesting(splits: int, capacity: int, num_kv_heads: int) -> int:
    """Smallest batch of 64K-token requests whose ``batch * num_kv_heads`` items
    make ``split_factor`` pick ``splits`` on a ``capacity``-CTA part, or 0 when
    the part cannot request that factor with whole requests (checked, not
    assumed: the rule is re-evaluated on the chosen item count)."""
    pages = (65536 + PAGE_SIZE - 1) // PAGE_SIZE
    if splits == 1:
        # one CTA per item: the items alone fill more than half of the part
        items = (capacity // 2 // num_kv_heads + 1) * num_kv_heads
    else:
        # the rule doubles while every (item, split) CTA fits one wave, so the
        # largest item count with items * splits <= capacity selects ``splits``
        items = capacity // splits // num_kv_heads * num_kv_heads
    if items < num_kv_heads or items > capacity:
        return 0
    return (
        items // num_kv_heads if split_factor(items, capacity, pages) == splits else 0
    )


def test_every_split_factor_runs_its_program():
    """Each registered split factor of the attached part prepares, launches and
    matches the oracle.  The batch for each factor is derived from the part's
    resident-CTA capacity (so every factor is actually requested on a 148-,
    160- or 212-CTA part instead of assuming a fixed batch list), and the
    selection is asserted, not assumed."""
    _require_program()
    device = torch.device("cuda")
    capacity = persistent_cta_capacity(device)
    requested, seen = {}, {}
    for splits in SPLIT_FACTORS:
        batch = _batch_requesting(splits, capacity, 4)
        if batch == 0:
            continue
        requested[splits] = batch
        _inputs, runner = _run_and_check([65536] * batch, 4, seed=50 + batch)
        assert runner.route == "persistent" and not runner.tail, (
            splits,
            batch,
            capacity,
            runner.route,
            runner.tail,
        )
        assert runner.splits == splits, (splits, batch, capacity, runner.splits)
        seen[splits] = runner.splits
    assert set(seen) == set(requested) and len(requested) >= 2, (
        capacity,
        requested,
        seen,
    )
    assert set(requested) == set(SPLIT_FACTORS), (
        capacity,
        requested,
    )  # every registered factor is reachable on a supported part


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
    _require_existing_route()
    inputs, runner = _run_and_check(seq_lens, num_kv_heads, seed=41)
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
    _require_existing_route()
    inputs, _ = _run_and_check([4096] * 3, 1, group_size=8, seed=43)
    with pytest.raises(NotImplementedError):
        msa_sparse_decode_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            out=torch.empty_like(inputs["q"]),
            **upstream_route_kwargs(inputs),
        )


def _require_short_program():
    _require_program()
    if short_program_record(arch_for(torch.device("cuda"))) is None:
        pytest.skip(
            "short-item NVFP4 MSA decode program not registered for this device"
        )


def test_short_items_take_the_cluster_program():
    """Requests within the short program's page budget run one eight-CTA cluster
    per work item from the same pages, with no allocation at launch."""
    _require_short_program()
    inputs, runner = _run_and_check([257, 300, 1, 512], 4, seed=17)
    assert runner.route == "short"
    # A KV head serving fewer than sixteen query heads (tp8 geometry) is served
    # from the same program; rows beyond the group are never read or written.
    _run_and_check([385, 300, 129], 1, group_size=8, seed=29)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0
    torch.testing.assert_close(runner.out, _oracle(inputs), atol=ATOL, rtol=RTOL)


def test_short_program_graph_replay_follows_new_selections():
    _require_short_program()
    inputs, runner = _run_and_check([300] * 8, 4, seed=19)
    assert runner.route == "short"
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    fresh = build_decode_inputs([300] * 8, num_kv_heads=4, device="cuda", seed=23)
    inputs["q"].copy_(fresh["q"])
    inputs["q2k_indices"].copy_(fresh["q2k_indices"])
    del fresh
    runner.out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(runner.out, _oracle(inputs), atol=ATOL, rtol=RTOL)


def test_graph_replay_follows_new_queries_and_selections():
    """Capture once; replay after new queries and a new top-k selection are written in place."""
    _require_program()
    inputs, runner = _run_and_check([8192] * 4, 4, seed=11)  # eight-way split
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
    _, runner = _run_and_check([65536] * 2, 4, seed=5)  # split path
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0


def test_multi_round_batches_take_the_last_round_split():
    _require_program()
    device = torch.device("cuda")
    arch = arch_for(device)
    if not cake_jit.tail_programs(arch):
        pytest.skip(f"no last-round-split program registered for {arch}")
    capacity = persistent_cta_capacity(device)
    # More items than resident CTAs (256 on the 212-SM part) with eight page pairs each.
    inputs = build_decode_inputs([1024] * 64, num_kv_heads=4, device="cuda", seed=12)
    items = 64 * 4
    plan = tail_plan(arch, items, capacity, inputs["max_pages"])
    assert plan is not None and items > capacity
    runner = _prepare(inputs)
    assert runner.tail and runner.route == "persistent"
    assert (runner.splits, runner.num_ctas) == plan
    assert cake_jit.PROGRAMS[runner.program]["tail"] is True
    runner()
    torch.cuda.synchronize()
    torch.testing.assert_close(runner.out, _oracle(inputs), atol=ATOL, rtol=RTOL)
    assert torch.isfinite(runner.lse).all()
