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
    UNION_MAX_SEQLEN_Q,
    UNION_MIN_SEQLEN_Q,
    UNION_Q_VALUES,
    UNION_SPLIT_FACTORS,
    union_plan,
    union_program_available,
    union_q_for,
    union_route_applies,
    union_split_factor,
    validate_msa_nvfp4_decode_inputs,
    validate_use_cache_hit,
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
# Part B gate of the union program: per-row relative L2 error against the FP32 oracle.
UNION_MAX_ROW_REL_L2 = 0.004
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


@pytest.mark.parametrize("value", [False, True, "auto"])
def test_use_cache_hit_accepts_exactly_its_three_values(value):
    assert (
        validate_use_cache_hit(value) is value or validate_use_cache_hit(value) == value
    )


@pytest.mark.parametrize(
    "value", [0, 1, "True", "false", "Auto", "AUTO", None, 1.0, "", b"auto"]
)
def test_use_cache_hit_rejects_everything_else(value):
    with pytest.raises(ValueError, match="use_cache_hit"):
        validate_use_cache_hit(value)


def test_use_cache_hit_domain_is_a_uniform_seqlen_q_in_two_to_four():
    assert (UNION_MIN_SEQLEN_Q, UNION_MAX_SEQLEN_Q) == (2, 4)
    assert [union_route_applies(q) for q in (1, 2, 3, 4, 5, 8, 32)] == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert [union_q_for(q) for q in (1, 2, 3, 4)] == [2, 2, 4, 4]


def test_use_cache_hit_true_refuses_a_seqlen_q_outside_the_domain():
    """The refusal is a host decision made before any device check: seqlen_q 8 is
    inside the API's [1, 32] but outside the union program's [2, 4]."""
    inputs = _cpu_inputs((2000, 1500), num_kv_heads=2, group_size=16, seqlen_q=8)
    with pytest.raises(ValueError, match="use_cache_hit=True requires"):
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
            seqlen_q=8,
            use_cache_hit=True,
        )
    # ``1 == True`` is not an accepted spelling.
    with pytest.raises(ValueError, match="use_cache_hit must be"):
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
            seqlen_q=8,
            use_cache_hit=1,
        )


@pytest.mark.parametrize(
    "units,num_sms,pairs_max,expected",
    [
        # 148- and 152-SM parts agree on every row of the export denominator.
        # S doubles while ``units x 2S`` CTAs fit one wave (and 2S <= pairs, S < 8).
        (2, 148, 32, 8),  # tp2 batch 1 (Q4, 150K context): the eight-way cap
        (2, 160, 32, 8),
        (16, 148, 32, 8),  # tp2 batch 8: 16 x 8 = 128 CTAs fit one wave on both parts
        (16, 160, 32, 8),
        (32, 148, 32, 4),  # tp1 batch 8: 32 x 4 = 128 fits, 32 x 8 = 256 does not
        (32, 160, 32, 4),
        (64, 148, 32, 2),  # tp2 batch 32: 64 x 2 = 128 fits, 64 x 4 = 256 does not
        (64, 160, 32, 2),
        (
            128,
            148,
            32,
            1,
        ),  # tp4 batch 128 / tp2 batch 64: the units alone exceed half the part
        (128, 160, 32, 1),
        (256, 148, 32, 1),
        (8, 148, 4, 4),  # kv900 (8 pages, 4 pairs): at most one pair per CTA
        (8, 148, 3, 1),  # fewer than four pairs never split
        (6, 148, 9, 8),  # kv2200 correctness rows (18 pages, 9 pairs)
        (32, 148, 16, 4),  # Q2 coverage: 16 Q = 32 ids cap, 8K context
        (32, 160, 16, 4),
        (64, 148, 16, 2),
        (64, 160, 16, 2),
        (128, 148, 16, 1),
        (128, 160, 16, 1),
    ],
)
def test_union_split_factor_rule(units, num_sms, pairs_max, expected):
    assert union_split_factor(units, num_sms, pairs_max) == expected


def test_union_plan_rule():
    # 8 requests x 2 KV heads at seqlen_q 4 and 1200 pages (64 ids cap -> 32 pairs): 16 units,
    # eight-way clusters (16 x 8 = 128 CTAs fit one wave), 128 CTAs.
    assert union_plan(
        total_q=32, seqlen_q=4, num_kv_heads=2, max_pages=1200, num_sms=148
    ) == (
        4,
        8,
        16,
        (128, 1, 1),
    )
    # seqlen_q 3 runs the width-4 program; the unsplit grid is persistent over min(units, SMs).
    assert union_plan(
        total_q=384, seqlen_q=3, num_kv_heads=2, max_pages=64, num_sms=148
    ) == (
        4,
        1,
        256,
        (148, 1, 1),
    )
    assert union_plan(
        total_q=384, seqlen_q=3, num_kv_heads=2, max_pages=64, num_sms=160
    ) == (
        4,
        1,
        256,
        (160, 1, 1),
    )
    # seqlen_q 2: the width-2 program, 32 ids cap -> 16 pairs; 64 units: 64 x 2 = 128 CTAs
    # fit one wave, 64 x 4 = 256 do not -> two-way clusters, 128 CTAs.
    assert union_plan(
        total_q=64, seqlen_q=2, num_kv_heads=2, max_pages=64, num_sms=148
    ) == (
        2,
        2,
        64,
        (128, 1, 1),
    )
    with pytest.raises(ValueError, match="unit"):
        union_plan(total_q=0, seqlen_q=2, num_kv_heads=2, max_pages=64, num_sms=148)


@pytest.mark.parametrize("arch", sorted(SUPPORTED_COMPUTE_CAPABILITIES.values()))
def test_architectures_with_union_programs_register_the_whole_ladder(arch):
    """An architecture that registers union programs registers one per
    (query-group width, split factor) pair, each with a source pair the JIT
    can hand to nvcc; the union argument plan carries the selection strides."""
    pairs = cake_jit.registered_union_pairs(arch)
    if not pairs:
        with pytest.raises(NotImplementedError):
            cake_jit.select_union_program(arch, union_q=2, splits=1)
        pytest.skip(f"no union program registered for {arch}")
    assert pairs == tuple((q, s) for q in UNION_Q_VALUES for s in UNION_SPLIT_FACTORS)
    root = cake_jit.Path(cake_jit.__file__).resolve().parent / "csrc"
    names = set()
    for union_q, splits in pairs:
        name = cake_jit.select_union_program(arch, union_q=union_q, splits=splits)
        record = cake_jit.PROGRAMS[name]
        assert record["route"] == "union"
        assert record["cluster"] is (splits > 1)
        assert arch in record["arches"] and len(record["sources"]) == 2
        assert all((root / relative).is_file() for relative in record["sources"])
        names.add(name)
    assert len(names) == len(pairs)  # one distinct program per pair
    plan = cake_jit.ARG_PLANS["union"]
    plan_names = [n for _k, n in plan]
    assert {"q2k_token_stride", "q2k_head_stride"} <= set(plan_names)
    assert plan_names[-3:] == ["grid_x", "grid_y", "grid_z"]
    with pytest.raises(NotImplementedError):
        cake_jit.select_union_program("sm_90a", union_q=2, splits=1)


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
    for route in ("persistent", "short", "union"):
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
    """FP32 oracle (Part B contract): ``upstream.reference`` rounds to ``out.dtype``,
    so the buffer is FP32, and its einsums run with cuBLAS TF32 switched off
    (the NGC images export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 as the default)."""
    q = inputs["q"]
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        reference = upstream.reference(
            q=q,
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
            out=torch.empty(q.shape, dtype=torch.float32, device=q.device),
            seqlen_q=inputs["seqlen_q"],
            causal=True,
        )
        torch.cuda.synchronize()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    return reference


def _assert_close(actual, expected):
    """BF16 output vs the FP32 oracle: element-wise ``|a - e| <= ATOL + RTOL * |e|``."""
    torch.testing.assert_close(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)


def _prepare(inputs, *, out=None, lse=None, use_cache_hit="auto"):
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
        use_cache_hit=use_cache_hit,
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
    _assert_close(out, expected)
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
    _assert_close(runner.out, _oracle(inputs))


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
    _assert_close(runner.out, _oracle(inputs))


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
        _assert_close(runner.out, expected)


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
    _assert_close(runner.out, _oracle(inputs))
    assert torch.isfinite(runner.lse).all()


# ---------------------------------------------------------------------------
# Device: the request-packed union program (``use_cache_hit=True``)
# ---------------------------------------------------------------------------


def _require_union_program():
    _require_program()
    arch = arch_for(torch.device("cuda"))
    if not union_program_available(arch):
        pytest.skip(f"no request-packed NVFP4 MSA decode program registered for {arch}")


def _device_sms() -> int:
    return torch.cuda.get_device_properties(torch.device("cuda")).multi_processor_count


def _assert_row_rel_l2(actual, expected, *, limit=UNION_MAX_ROW_REL_L2):
    a = actual.float().reshape(-1, actual.shape[-1])
    b = expected.float().reshape(-1, expected.shape[-1])
    rel = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp(min=1e-6)
    assert float(rel.max()) <= limit, f"max row rel-L2 {float(rel.max())} > {limit}"


def _run_union(
    seq_lens,
    num_kv_heads,
    *,
    group_size=16,
    seqlen_q,
    overlap,
    seed,
    page_table_order="identity",
    k_global_scale=None,
    v_global_scale=None,
):
    """Prepare with ``use_cache_hit=True``, assert the production plan, launch and check the oracle."""
    extra = {}
    if k_global_scale is not None:
        extra["k_global_scale"] = k_global_scale
    if v_global_scale is not None:
        extra["v_global_scale"] = v_global_scale
    inputs = build_decode_inputs(
        seq_lens,
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        seqlen_q=seqlen_q,
        device="cuda",
        seed=seed,
        q_overlap_fraction=overlap,
        page_table_order=page_table_order,
        **extra,
    )
    runner = _prepare(inputs, use_cache_hit=True)
    assert runner.route == "union"
    assert runner.tail is False
    union_q, splits, _units, grid = union_plan(
        total_q=int(inputs["q"].shape[0]),
        seqlen_q=seqlen_q,
        num_kv_heads=num_kv_heads,
        max_pages=inputs["max_pages"],
        num_sms=_device_sms(),
    )
    assert (runner.union_q, runner.splits, runner.num_ctas) == (
        union_q,
        splits,
        grid[0],
    )
    assert runner.program == cake_jit.select_union_program(
        runner.arch, union_q=runner.union_q, splits=runner.splits
    )
    assert cake_jit.PROGRAMS[runner.program]["cluster"] is (runner.splits > 1)
    runner.out.fill_(float("nan"))
    runner.lse.fill_(float("nan"))
    out = runner()
    torch.cuda.synchronize()
    expected = _oracle(inputs)
    _assert_close(out, expected)
    _assert_row_rel_l2(out, expected)
    assert torch.isfinite(runner.lse).all()
    assert runner.lse.shape == (inputs["q"].shape[0], inputs["q"].shape[1])
    return inputs, runner


@pytest.mark.parametrize(
    "seq_lens,num_kv_heads,seqlen_q,overlap,page_table_order,scales",
    [
        (
            [8192] * 8,
            2,
            4,
            0.5,
            "identity",
            None,
        ),  # tp2 batch 8: 16 units, eight-way clusters
        (
            [8192] * 8,
            2,
            4,
            0.0,
            "identity",
            None,
        ),  # disjoint selections: |U| = 64 per unit
        ([8192] * 8, 2, 4, 1.0, "identity", None),  # identical history: |U| = 16 + 3
        (
            [8192] * 8,
            2,
            3,
            0.75,
            "identity",
            None,
        ),  # seqlen_q 3: width-4 program, phantom token
        (
            [2200] * 3,
            2,
            2,
            0.5,
            "shuffled",
            None,
        ),  # width-2 program, 18 pages, 24-token page tail
        ([2200] * 3, 1, 4, 0.5, "shared_prefix", None),  # tp4, shared-prefix page table
        (
            [900] * 4,
            2,
            4,
            0.5,
            "identity",
            None,
        ),  # fewer than sixteen visible pages, -1 tails
        (
            [153600] * 2,
            2,
            4,
            0.75,
            "identity",
            None,
        ),  # 1200 pages per request, eight-way clusters
        (
            [8192] * 64,
            2,
            4,
            0.0,
            "identity",
            None,
        ),  # 128 units: the unsplit persistent program
        ([8192] * 8, 4, 4, 1.0, "identity", None),  # tp1 geometry (64 query heads)
        ([8192] * 8, 1, 4, 0.0, "identity", None),  # tp4 geometry (16 query heads)
        (
            [5000, 130, 300, 2049],
            2,
            4,
            0.5,
            "shuffled",
            None,
        ),  # ragged, requests within one page
        ([8192] * 8, 2, 4, 0.5, "identity", (1.75, 0.6)),  # non-unit global scales
    ],
)
def test_union_decode_matches_the_fp32_oracle(
    seq_lens, num_kv_heads, seqlen_q, overlap, page_table_order, scales
):
    _require_union_program()
    k_scale, v_scale = scales if scales is not None else (None, None)
    _run_union(
        seq_lens,
        num_kv_heads,
        seqlen_q=seqlen_q,
        overlap=overlap,
        seed=len(seq_lens) * 7 + seqlen_q + int(overlap * 100),
        page_table_order=page_table_order,
        k_global_scale=k_scale,
        v_global_scale=v_scale,
    )


def _union_batch_requesting(
    splits: int, num_sms: int, num_kv_heads: int, union_q: int
) -> int:
    """Smallest batch of 8K-token requests whose ``batch * num_kv_heads`` units make
    ``union_split_factor`` pick ``splits`` on a ``num_sms`` part, or 0 when the part
    cannot request that factor with whole requests (checked, not assumed)."""
    pairs = (min(TOPK * union_q, (8192 + PAGE_SIZE - 1) // PAGE_SIZE) + 1) // 2
    if splits == 1:
        units = (num_sms // 2 // num_kv_heads + 1) * num_kv_heads
    else:
        units = num_sms // splits // num_kv_heads * num_kv_heads
    if units < num_kv_heads:
        return 0
    return (
        units // num_kv_heads
        if union_split_factor(units, num_sms, pairs) == splits
        else 0
    )


def test_every_union_program_runs_on_the_attached_part():
    """Each registered (query-group width, split factor) program prepares, launches and
    matches the oracle; the batch for each factor is derived from the part's SM count and
    the selection is asserted, not assumed."""
    _require_union_program()
    arch = arch_for(torch.device("cuda"))
    num_sms = _device_sms()
    seen = set()
    for union_q, splits in cake_jit.registered_union_pairs(arch):
        batch = _union_batch_requesting(splits, num_sms, 2, union_q)
        if batch == 0:
            continue
        _inputs, runner = _run_union(
            [8192] * batch, 2, seqlen_q=union_q, overlap=0.5, seed=200 + batch + union_q
        )
        assert (runner.union_q, runner.splits) == (union_q, splits), (
            batch,
            num_sms,
            runner,
        )
        seen.add((union_q, splits))
    assert seen == set(cake_jit.registered_union_pairs(arch)), (num_sms, seen)


def test_use_cache_hit_true_is_a_no_op_at_seqlen_q_one():
    _require_program()
    inputs = build_decode_inputs([8192] * 4, num_kv_heads=4, device="cuda", seed=61)
    plain = _prepare(inputs, use_cache_hit=False)
    switched = _prepare(inputs, use_cache_hit=True)
    assert switched.route == plain.route != "union"
    assert switched.program == plain.program
    plain.out.fill_(float("nan"))
    reference = plain().clone()
    torch.cuda.synchronize()
    switched.out.fill_(float("nan"))
    out = switched()
    torch.cuda.synchronize()
    assert torch.equal(out, reference)


def test_auto_resolves_to_the_per_token_programs():
    """``"auto"`` is the default and currently selects exactly what ``False`` selects;
    only an explicit ``True`` reaches the union program."""
    _require_program()
    inputs = build_decode_inputs(
        [8192] * 8,
        num_kv_heads=2,
        seqlen_q=4,
        device="cuda",
        seed=63,
        q_overlap_fraction=0.5,
    )
    auto = _prepare(inputs)
    plain = _prepare(inputs, use_cache_hit=False)
    assert auto.route == plain.route != "union"
    assert (auto.program, auto.splits, auto.tail, auto.grid) == (
        plain.program,
        plain.splits,
        plain.tail,
        plain.grid,
    )
    plain.out.fill_(float("nan"))
    reference = plain().clone()
    torch.cuda.synchronize()
    auto.out.fill_(float("nan"))
    out = auto()
    torch.cuda.synchronize()
    assert torch.equal(out, reference)
    _assert_close(out, _oracle(inputs))
    if union_program_available(arch_for(inputs["q"].device)):
        assert _prepare(inputs, use_cache_hit=True).route == "union"


def test_union_output_is_independent_of_sibling_selections():
    """A token's output must not depend on which pages the other tokens of its request
    select: the newest token of every request keeps its row while its siblings are
    re-drawn with a different overlap pattern, and its output and LSE are bitwise
    unchanged (masked-out union pages leave no trace)."""
    _require_union_program()
    seq_lens, num_kv_heads, seqlen_q = [8192] * 8, 2, 4
    inputs, runner = _run_union(
        seq_lens, num_kv_heads, seqlen_q=seqlen_q, overlap=0.5, seed=71
    )
    out_a = runner.out.clone()
    lse_a = runner.lse.clone()
    probe = torch.tensor(
        [b * seqlen_q + (seqlen_q - 1) for b in range(len(seq_lens))], device="cuda"
    )
    others = torch.tensor(
        [t for t in range(len(seq_lens) * seqlen_q) if t % seqlen_q != seqlen_q - 1],
        device="cuda",
    )
    fresh = build_decode_inputs(
        seq_lens,
        num_kv_heads=num_kv_heads,
        seqlen_q=seqlen_q,
        device="cuda",
        seed=72,
        q_overlap_fraction=0.0,
    )
    mixed = fresh["q2k_indices"].clone()
    mixed[:, probe] = inputs["q2k_indices"][:, probe]
    assert not torch.equal(mixed[:, others], inputs["q2k_indices"][:, others])
    del fresh
    inputs["q2k_indices"].copy_(mixed)
    runner.out.fill_(float("nan"))
    runner.lse.fill_(float("nan"))
    out_b = runner()
    torch.cuda.synchronize()
    assert torch.equal(out_b[probe], out_a[probe])
    assert torch.equal(runner.lse[probe], lse_a[probe])
    expected = _oracle(inputs)
    _assert_close(out_b, expected)
    _assert_row_rel_l2(out_b, expected)


def test_union_graph_replay_follows_new_queries_and_selections():
    """Capture the union program once; replay after new queries and a new selection
    are written in place."""
    _require_union_program()
    inputs, runner = _run_union([8192] * 8, 2, seqlen_q=4, overlap=0.5, seed=81)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    gen = torch.Generator(device="cuda").manual_seed(2027)
    for step in range(3):
        inputs["q"].copy_(
            torch.randn(inputs["q"].shape, generator=gen, device="cuda").to(
                torch.bfloat16
            )
        )
        if step:
            fresh = build_decode_inputs(
                [8192] * 8,
                num_kv_heads=2,
                seqlen_q=4,
                device="cuda",
                seed=90 + step,
                q_overlap_fraction=0.25 * step,
            )
            inputs["q2k_indices"].copy_(fresh["q2k_indices"])
            del fresh
        runner.out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = _oracle(inputs)
        _assert_close(runner.out, expected)
        _assert_row_rel_l2(runner.out, expected)


def test_union_launch_makes_no_allocation():
    _require_union_program()
    _, runner = _run_union([8192] * 8, 2, seqlen_q=4, overlap=0.5, seed=83)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0


def test_use_cache_hit_true_under_capture_requires_a_built_program():
    """A first-time JIT build cannot happen inside a CUDA-graph capture: under a real
    capture the union route binds a program built before capture and refuses one that
    was not, before any build could start."""
    _require_union_program()
    inputs, runner = _run_union([8192] * 8, 2, seqlen_q=4, overlap=0.5, seed=85)
    key = (runner.program, runner.arch)
    assert cake_jit.program_loaded(*key)
    # The per-token program is built eagerly too, so binding it under capture below
    # cannot start a build either.
    per_token = _prepare(inputs, use_cache_hit=False)
    assert per_token.route != "union"
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            # Built by the eager preparation above: binding under capture is allowed and
            # the captured launch is the union program.
            captured = _prepare(
                inputs, out=runner.out, lse=runner.lse, use_cache_hit=True
            )
            assert (captured.route, captured.program) == ("union", runner.program)
            captured()
            # Forget the built program for the rest of the capture (real registry state,
            # restored below): the same preparation now refuses before reaching the JIT.
            module = cake_jit._LOADED.pop(key)
            try:
                with pytest.raises(ValueError, match="before capture"):
                    _prepare(inputs, out=runner.out, lse=runner.lse, use_cache_hit=True)
            finally:
                cake_jit._LOADED[key] = module
            # The per-token routes are not subject to the union capture rule.
            assert (
                _prepare(
                    inputs, out=runner.out, lse=runner.lse, use_cache_hit=False
                ).route
                == per_token.route
            )
    torch.cuda.synchronize()
    # The launch bound under capture replays correctly.
    runner.out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    expected = _oracle(inputs)
    _assert_close(runner.out, expected)
    _assert_row_rel_l2(runner.out, expected)


def _shared_launcher_programs():
    """``PROGRAMS`` records whose host launcher is one shared per-route source."""
    shared = {}
    suffix = "_binding.cu"
    for name, record in cake_jit.PROGRAMS.items():
        stem = cake_jit.Path(record["sources"][1]).name[: -len(suffix)]
        if stem != name:
            shared[name] = record
    return shared


def _msa_nvfp4_decode_csrc():
    return cake_jit.Path(cake_jit.__file__).resolve().parent / "csrc"


def test_shared_launchers_take_each_program_from_its_compile_flags():
    """Every ``#ifndef CAKE_* / #error`` selector of a shared launcher source is set exactly
    once by the record's compile flags; the kernel symbol and the host shim match the record and
    the launcher stem is the record's route."""
    import re

    csrc = _msa_nvfp4_decode_csrc()
    shared = _shared_launcher_programs()
    assert shared, "no shared launchers registered"
    for name, record in shared.items():
        stem = cake_jit.Path(record["sources"][1]).name[: -len("_binding.cu")]
        assert stem == f"cake_msa_nvfp4_decode_{record['route']}", (name, stem)
        text = (csrc / record["sources"][1]).read_text()
        guards = re.findall(r"#ifndef (CAKE_\w+)\n#error", text)
        assert "CAKE_PROGRAM_KERNEL" in guards, (name, guards)
        assert "CAKE_PROGRAM_SHIM" in guards, (name, guards)
        assert "CAKE_BIND_SMEM" in guards, (name, guards)
        assert f"-DCAKE_PROGRAM_KERNEL=kernel_{name}" in record["compile_flags"], name
        shim = [
            f for f in record["compile_flags"] if f.startswith("-DCAKE_PROGRAM_SHIM=")
        ]
        assert len(shim) == 1 and re.fullmatch(
            r"-DCAKE_PROGRAM_SHIM=cake_host_shim_[0-9a-f]{16}", shim[0]
        ), (name, shim)
        for macro in guards:
            hits = [f for f in record["compile_flags"] if f.startswith(f"-D{macro}=")]
            assert len(hits) == 1, (name, macro, hits)


def test_shared_launchers_have_the_generated_form():
    """Each shared launcher source carries the generated -D-selected form and no digest."""
    import re

    csrc = _msa_nvfp4_decode_csrc()
    launchers = {r["sources"][1] for r in _shared_launcher_programs().values()}
    assert launchers, "no shared launchers registered"
    for launcher in sorted(launchers):
        text = (csrc / launcher).read_text()
        assert 'extern "C" __global__ void CAKE_PROGRAM_KERNEL(' in text, launcher
        assert "namespace CAKE_PROGRAM_SHIM {" in text, launcher
        assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_PROGRAM_SHIM::Run);" in text, (
            launcher
        )
        assert "config.dynamicSmemBytes = CAKE_PASTE(CAKE_BIND_SMEM, u);" in text, (
            launcher
        )
        assert not re.search(r"[0-9a-f]{20}", text), launcher


def test_shared_launchers_serve_one_route_each():
    """Records sharing a launcher source share one route and no two carry the same full
    compile-flag set."""
    by_stem = {}
    for record in _shared_launcher_programs().values():
        stem = cake_jit.Path(record["sources"][1]).name[: -len("_binding.cu")]
        by_stem.setdefault(stem, []).append(record)
    assert by_stem, "no shared launchers registered"
    for stem, records in by_stem.items():
        assert len({r["route"] for r in records}) == 1, stem
        tuples = {tuple(r["compile_flags"]) for r in records}
        assert len(tuples) == len(records), (stem, len(tuples), len(records))
