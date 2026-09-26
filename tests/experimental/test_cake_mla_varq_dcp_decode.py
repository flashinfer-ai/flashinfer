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

# Correctness tests for the experimental Cake compact variable-query MLA decode
# with static decode context parallelism (DCP).  Fixtures, cyclic rank packing,
# the global-coordinate reference, the natural-log cross-rank merge and the
# tolerances mirror tests/attention/test_cute_dsl_mla_dcp.py.

import math

import pytest
import torch

from flashinfer.experimental.cake_mla_varq_dcp_decode import cake_backend, cake_jit
from flashinfer.experimental.cake_mla_varq_dcp_decode.cake_backend import (
    CLUSTER_SIZE,
    HEAD_DIM_QK,
    HEAD_DIM_V,
    ITEM_GROUP_VARIANTS,
    MAX_ITEMS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    SUPPORTED_PAGE_SIZES,
    TILE_KV,
    TILE_Q,
    cake_mla_varq_dcp_decode_workspace_size,
    item_groups_for,
    launches_merge,
    max_cake_mla_varq_dcp_decode_workspace_size,
    plan_varq_dcp_decode,
    prepare_cake_mla_varq_dcp_decode,
    validate_cake_mla_varq_dcp_decode_inputs,
    workspace_layout,
)

_LATENT_DIM = HEAD_DIM_V
_SOFTMAX_SCALE = 1.0 / math.sqrt(_LATENT_DIM)
_PAGE_PERMUTATION_RANK_STRIDE = 7919


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _local_length(global_length: int, cp_world: int, cp_rank: int) -> int:
    return max(_ceil_div(global_length - cp_rank, cp_world), 0)


# ---------------------------------------------------------------------------
# CPU tests: host plan, workspace layout, validation
# ---------------------------------------------------------------------------


def test_item_groups_for():
    assert item_groups_for(1) == ITEM_GROUP_VARIANTS[0]
    assert item_groups_for(128) == ITEM_GROUP_VARIANTS[0]
    assert item_groups_for(129) == ITEM_GROUP_VARIANTS[1]
    assert item_groups_for(MAX_ITEMS) == ITEM_GROUP_VARIANTS[1]
    with pytest.raises(ValueError, match="exceed the scheduler capacity"):
        item_groups_for(MAX_ITEMS + 1)


@pytest.mark.parametrize("num_sms", [132, 148, 160])
def test_plan_invariants(num_sms):
    resident = num_sms // CLUSTER_SIZE
    for batch_size in (1, 4, 16, 64, 128):
        for max_q_len, num_heads in ((1, 128), (3, 96), (4, 64), (2, 12), (8, 24)):
            if batch_size * _ceil_div(max_q_len * num_heads, TILE_Q) > MAX_ITEMS:
                continue
            for max_seq_len in (1, 130, 1024, 4101, 8192, 32768, 131072):
                plan = plan_varq_dcp_decode(
                    batch_size=batch_size,
                    max_q_len=max_q_len,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    num_sms=num_sms,
                )
                tiles_max = _ceil_div(max_q_len * num_heads, TILE_Q)
                assert plan["tiles_max"] == tiles_max
                assert plan["items"] == batch_size * tiles_max
                assert plan["max_local_tiles"] == max(
                    1, _ceil_div(max_seq_len, TILE_KV)
                )
                assert 1 <= plan["grid_clusters"] <= resident
                assert plan["partial_rows"] % TILE_Q == 0 and plan["partial_rows"] > 0
                assert 1 <= plan["merge_grid"] <= num_sms
                if plan["partition_mode"]:
                    assert plan["items"] <= 128
                    assert plan["grid_clusters"] == resident
                    assert plan["max_units"] == 1
                    assert plan["partial_rows"] == resident * TILE_Q
                else:
                    assert plan["max_units"] >= 2
                # A static-only plan never splits, so it never launches the merge.
                assert not (plan["static_only"] and launches_merge(plan))
                assert plan["can_split"] == (plan["max_local_tiles"] > plan["unit_min"])
                layout = workspace_layout(plan)
                offset = 0
                for name in (
                    "partial_o",
                    "partial_lse",
                    "sched_counters",
                    "unit_flags",
                    "split_meta",
                    "merge_ctl",
                    "dbg",
                ):
                    start, nbytes = layout[name]
                    assert start == offset and start % 256 == 0
                    offset += -(-nbytes // 256) * 256
                assert layout["total"] == offset
                if plan["can_split"]:
                    assert (
                        layout["partial_o"][1] == plan["partial_rows"] * HEAD_DIM_V * 2
                    )
                else:
                    assert layout["partial_o"][1] == 0 and layout["partial_lse"][1] == 0
                assert cake_mla_varq_dcp_decode_workspace_size(
                    batch_size=batch_size,
                    max_q_len=max_q_len,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    num_sms=num_sms,
                ) <= max_cake_mla_varq_dcp_decode_workspace_size(
                    batch_size=batch_size,
                    max_q_len=max_q_len,
                    num_heads=num_heads,
                    num_sms=num_sms,
                )


def test_plan_rejects_out_of_domain():
    with pytest.raises(ValueError, match="num_heads"):
        plan_varq_dcp_decode(
            batch_size=1, max_q_len=1, num_heads=129, max_seq_len=64, num_sms=148
        )
    with pytest.raises(ValueError, match="scheduler capacity"):
        plan_varq_dcp_decode(
            batch_size=513, max_q_len=1, num_heads=128, max_seq_len=64, num_sms=148
        )
    with pytest.raises(ValueError, match="partition_mode 1"):
        plan_varq_dcp_decode(
            batch_size=256,
            max_q_len=1,
            num_heads=64,
            max_seq_len=64,
            num_sms=148,
            partition_mode=1,
        )


def _host_inputs(dtype=torch.bfloat16, page_size=64, num_heads=24, batch_size=4):
    total_q = 8
    return dict(
        query=torch.empty((total_q, num_heads, HEAD_DIM_QK), dtype=dtype),
        kv_cache=torch.empty((6, page_size, HEAD_DIM_QK), dtype=dtype),
        page_table=torch.zeros((batch_size, 3), dtype=torch.int32),
        seq_lens=torch.ones((batch_size,), dtype=torch.int32),
        cum_seq_lens_q=torch.tensor([0, 4, 5, 5, 8], dtype=torch.int32),
        max_q_len=8,
        max_seq_len=130,
        cp_world=4,
        cp_rank=0,
        causal_seqlens_kv_global=torch.ones((batch_size,), dtype=torch.int32),
    )


def test_validate_accepts_host_tensors():
    meta = validate_cake_mla_varq_dcp_decode_inputs(**_host_inputs())
    assert meta == dict(
        dtype="bf16",
        batch_size=4,
        num_heads=24,
        total_q=8,
        page_size=64,
        num_pages=6,
        max_pages=3,
    )
    assert (
        validate_cake_mla_varq_dcp_decode_inputs(
            **_host_inputs(dtype=torch.float8_e4m3fn, page_size=128)
        )["dtype"]
        == "fp8"
    )


def test_validate_rejects_unsupported_inputs():
    def rejects(match, **overrides):
        inputs = _host_inputs()
        inputs.update(overrides)
        with pytest.raises((ValueError, TypeError), match=match):
            validate_cake_mla_varq_dcp_decode_inputs(**inputs)

    rejects(
        "unsupported query / KV dtype",
        query=torch.empty((8, 24, HEAD_DIM_QK), dtype=torch.float16),
    )
    rejects(
        "share one dtype",
        kv_cache=torch.empty((6, 64, HEAD_DIM_QK), dtype=torch.float8_e4m3fn),
    )
    rejects(
        "query must have shape",
        query=torch.empty((8, HEAD_DIM_QK), dtype=torch.bfloat16),
    )
    rejects(
        "num_heads must be in",
        query=torch.empty((8, 129, HEAD_DIM_QK), dtype=torch.bfloat16),
    )
    rejects(
        "page_size must be one of",
        kv_cache=torch.empty((6, 16, HEAD_DIM_QK), dtype=torch.bfloat16),
    )
    rejects("576 wide", kv_cache=torch.empty((6, 64, 512), dtype=torch.bfloat16))
    rejects(
        "page_table must be an int32",
        page_table=torch.zeros((4, 3), dtype=torch.float32),
    )
    rejects(
        "seq_lens must have 4 entries", seq_lens=torch.ones((3,), dtype=torch.int32)
    )
    rejects(
        "cum_seq_lens_q must have 5 entries",
        cum_seq_lens_q=torch.zeros((4,), dtype=torch.int32),
    )
    rejects("0 <= cp_rank < cp_world", cp_rank=4)
    rejects("causal_seqlens_kv_global is required", causal_seqlens_kv_global=None)
    rejects("must be a torch.Tensor", causal_seqlens_kv_global=[1, 1, 1, 1])
    rejects("max_q_len must be a positive int", max_q_len=0)
    rejects("batch_size \\* max_q_len", max_q_len=1)
    rejects("exceeds the page table capacity", max_seq_len=193)
    rejects(
        "out must be a bfloat16",
        out=torch.empty((8, 24, HEAD_DIM_V), dtype=torch.float32),
    )
    rejects("lse must be a float32", lse=torch.empty((8, 24), dtype=torch.bfloat16))


def test_route_name_format():
    assert (
        cake_jit.route_name("sm_100a", "bf16", 64, 4, 1) == "bf16_p64_g4_pm1__sm_100a"
    )
    assert (
        cake_jit.route_name("sm_103a", "fp8", 128, 16, 0) == "fp8_p128_g16_pm0__sm_103a"
    )
    for module, record in cake_jit.MODULES.items():
        assert record["arch"] in cake_jit.ARCH_NVCC_FLAGS
        assert record["tma_workspace_bytes"] == 0, module
    for name, record in cake_jit.ROUTES.items():
        assert name.endswith("__" + record["arch"])
        assert record["main"] in cake_jit.MODULES
    for arch, module in cake_jit.MERGE_MODULES.items():
        assert cake_jit.MODULES[module]["arch"] == arch


# ---------------------------------------------------------------------------
# GPU fixtures (upstream test_cute_dsl_mla_dcp semantics)
# ---------------------------------------------------------------------------


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("SM100 or SM103 required")
    return device


def _skip_unless_registered(
    device, *, dtype, page_size, batch_size, max_q_len, num_heads, max_seq_len
):
    """Skip when this checkout does not register the variant the plan selects."""
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    plan = plan_varq_dcp_decode(
        batch_size=batch_size,
        max_q_len=max_q_len,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        num_sms=num_sms,
    )
    kind = cake_backend.KV_DTYPES[dtype]
    groups = item_groups_for(plan["items"])
    if not cake_backend.generated_program_available(
        device,
        dtype=kind,
        page_size=page_size,
        item_groups=groups,
        partition_mode=plan["partition_mode"],
    ):
        pytest.skip(
            f"generated program {kind}_p{page_size}_g{groups}_pm{plan['partition_mode']} "
            f"not registered in this checkout ({cake_jit.TRACKING_ISSUE})"
        )
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    if launches_merge(plan) and arch not in cake_jit.MERGE_MODULES:
        pytest.skip(
            f"merge kernel for {arch} not registered ({cake_jit.TRACKING_ISSUE})"
        )
    return plan


def _make_batched_inputs(
    global_lengths, q_lens, num_heads, dtype, *, seed, max_q_len=None
):
    """Quantized-once compact Q and one padded global-coordinate KV pool."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    storage_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
    batch_size = len(global_lengths)
    max_q_len = max(q_lens) if max_q_len is None else max_q_len
    dense = (
        torch.randn(
            batch_size,
            max_q_len,
            num_heads,
            HEAD_DIM_QK,
            device=device,
            dtype=storage_dtype,
        )
        * 0.1
    ).to(dtype)
    query = torch.cat([dense[b, :q] for b, q in enumerate(q_lens)])
    global_kv = (
        torch.randn(
            batch_size,
            max(global_lengths),
            HEAD_DIM_QK,
            device=device,
            dtype=storage_dtype,
        )
        * 0.1
    ).to(dtype)
    cum_seq_lens_q = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    global_lens = torch.tensor(global_lengths, dtype=torch.int32, device=device)
    return query, cum_seq_lens_q, max_q_len, global_kv, global_lens


def _pack_cyclic_rank_pages(
    global_kv, global_lens_host, cp_world, cp_rank, *, page_size, permute, seed
):
    """Pack ``g % cp_world == cp_rank`` tokens into a rank-local paged cache."""
    device = global_kv.device
    batch_size = len(global_lens_host)
    local_lens = [_local_length(g, cp_world, cp_rank) for g in global_lens_host]
    pages_per = [max(1, _ceil_div(n, page_size)) for n in local_lens]
    total_pages = sum(pages_per)
    if permute:
        gen = torch.Generator(device="cpu").manual_seed(
            seed + _PAGE_PERMUTATION_RANK_STRIDE * cp_rank
        )
        page_ids = torch.randperm(total_pages, generator=gen)
    else:
        page_ids = torch.arange(total_pages)
    page_ids = page_ids.to(device)
    cache = torch.zeros(
        total_pages, page_size, HEAD_DIM_QK, dtype=global_kv.dtype, device=device
    )
    page_table = torch.zeros(
        batch_size, max(pages_per), dtype=torch.int32, device=device
    )
    offset = 0
    for b, (g, n, count) in enumerate(
        zip(global_lens_host, local_lens, pages_per, strict=True)
    ):
        ids = page_ids[offset : offset + count]
        page_table[b, :count] = ids.to(torch.int32)
        if n:
            padded = torch.zeros(
                count * page_size, HEAD_DIM_QK, dtype=global_kv.dtype, device=device
            )
            padded[:n] = global_kv[b, cp_rank:g:cp_world]
            cache[ids] = padded.view(count, page_size, HEAD_DIM_QK)
        offset += count
    seq_lens = torch.tensor(local_lens, dtype=torch.int32, device=device)
    return cache, page_table, seq_lens, max(1, max(local_lens))


def _reference_variable_q(
    query, cum_seq_lens_q, global_kv, global_lens_host, *, cp_world, cp_rank
):
    """FP32 reference in global coordinates (compact O, natural-log LSE, -inf rows)."""
    total_q, num_heads, _ = query.shape
    device = query.device
    out = torch.zeros(
        total_q, num_heads, _LATENT_DIM, dtype=torch.float32, device=device
    )
    lse = torch.full(
        (total_q, num_heads), -math.inf, dtype=torch.float32, device=device
    )
    offsets = cum_seq_lens_q.tolist()
    for b, (q_begin, q_end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        q_len = q_end - q_begin
        g = int(global_lens_host[b])
        if q_len == 0 or g <= cp_rank:
            continue
        keys = global_kv[b, cp_rank:g:cp_world].float()
        positions = torch.arange(cp_rank, g, cp_world, device=device)
        bounds = g - q_len + torch.arange(q_len, device=device)
        visible = positions.unsqueeze(0) <= bounds.unsqueeze(1)  # [q_len, K_local]
        scores = (
            torch.einsum("qhd,kd->qhk", query[q_begin:q_end].float(), keys)
            * _SOFTMAX_SCALE
        )
        scores = scores.masked_fill(~visible.unsqueeze(1), -math.inf)
        row_lse = torch.logsumexp(scores, dim=-1)
        has_keys = torch.isfinite(row_lse)
        shift = torch.where(has_keys, row_lse, torch.zeros_like(row_lse))
        probs = torch.exp(scores - shift.unsqueeze(-1))
        probs = torch.where(visible.unsqueeze(1), probs, torch.zeros_like(probs))
        row_out = torch.einsum("qhk,kd->qhd", probs, keys[:, :_LATENT_DIM])
        out[q_begin:q_end] = torch.where(
            has_keys.unsqueeze(-1), row_out, torch.zeros_like(row_out)
        )
        lse[q_begin:q_end] = torch.where(
            has_keys, row_lse, torch.full_like(row_lse, -math.inf)
        )
    return out, lse


def _merge_rank_outputs_natural_log(rank_outputs, rank_lses):
    outputs = torch.stack([o.float() for o in rank_outputs])
    lses = torch.stack([l.float() for l in rank_lses])
    max_lse = lses.max(dim=0).values
    has_keys = torch.isfinite(max_lse)
    safe_max = torch.where(has_keys, max_lse, torch.zeros_like(max_lse))
    weights = torch.where(
        torch.isfinite(lses),
        torch.exp(lses - safe_max.unsqueeze(0)),
        torch.zeros_like(lses),
    )
    weight_sum = weights.sum(dim=0)
    safe_sum = torch.where(has_keys, weight_sum, torch.ones_like(weight_sum))
    merged_out = (outputs * weights.unsqueeze(-1)).sum(dim=0) / safe_sum.unsqueeze(-1)
    merged_out = torch.where(
        has_keys.unsqueeze(-1), merged_out, torch.zeros_like(merged_out)
    )
    merged_lse = torch.where(
        has_keys, safe_max + torch.log(safe_sum), torch.full_like(max_lse, -math.inf)
    )
    return merged_out, merged_lse


def _assert_close_to_reference(out, lse, ref_out, ref_lse, dtype):
    if dtype == torch.float8_e4m3fn:
        out_atol, out_rtol, lse_atol, lse_rtol = 0.1, 0.1, 0.2, 0.1
    else:
        out_atol = out_rtol = lse_atol = lse_rtol = 1e-2
    assert not torch.isnan(out.float()).any()
    torch.testing.assert_close(out.float(), ref_out, atol=out_atol, rtol=out_rtol)
    # Exact -inf positions, then the finite rows.
    ref_neg_inf = torch.isneginf(ref_lse)
    assert torch.equal(torch.isneginf(lse.float()), ref_neg_inf)
    finite = ~ref_neg_inf
    torch.testing.assert_close(
        lse.float()[finite], ref_lse[finite], atol=lse_atol, rtol=lse_rtol
    )
    assert torch.equal(
        out.float()[ref_neg_inf], torch.zeros_like(out.float()[ref_neg_inf])
    )


def _assert_replay_close(out, lse, other_out, other_lse):
    """Two launches on identical inputs agree within the documented replay spread.

    When a request's local KV range is split into units on different clusters
    the kernel merges the partials in unit-completion order from BF16-staged
    partials (README, "Numerics and reproducibility"): consecutive launches can
    differ by a few BF16 ulps of ``out`` and about 3e-3 of ``lse``; unsplit
    rows are bitwise reproducible.  The -inf positions of ``lse`` (rows with
    no keys) and the zero rows of ``out`` are always exact.
    """
    assert not torch.isnan(out.float()).any()
    neg_inf = torch.isneginf(other_lse.float())
    assert torch.equal(torch.isneginf(lse.float()), neg_inf)
    torch.testing.assert_close(out.float(), other_out.float(), atol=1e-3, rtol=2.0**-7)
    finite = ~neg_inf
    torch.testing.assert_close(
        lse.float()[finite], other_lse.float()[finite], atol=4e-3, rtol=0
    )
    assert torch.equal(
        out.float()[neg_inf], torch.zeros_like(out.float()[neg_inf])
    )


def _launch_rank(
    query,
    cum_seq_lens_q,
    max_q_len,
    global_kv,
    global_lens,
    *,
    cp_world,
    cp_rank,
    page_size=64,
    permute=False,
    seed=0,
    enable_dcp=True,
):
    """Prepare and launch one rank through the prepared runner; returns (out, lse, runner)."""
    device = query.device
    global_lens_host = global_lens.tolist()
    kv_cache, page_table, seq_lens, max_local_len = _pack_cyclic_rank_pages(
        global_kv,
        global_lens_host,
        cp_world,
        cp_rank,
        page_size=page_size,
        permute=permute,
        seed=seed,
    )
    batch_size = len(global_lens_host)
    num_heads = int(query.shape[1])
    _skip_unless_registered(
        device,
        dtype=query.dtype,
        page_size=page_size,
        batch_size=batch_size,
        max_q_len=max_q_len,
        num_heads=num_heads,
        max_seq_len=max_local_len,
    )
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.empty(
        cake_mla_varq_dcp_decode_workspace_size(
            batch_size=batch_size,
            max_q_len=max_q_len,
            num_heads=num_heads,
            max_seq_len=max_local_len,
            num_sms=num_sms,
        ),
        dtype=torch.uint8,
        device=device,
    )
    out = torch.full(
        (int(query.shape[0]), num_heads, HEAD_DIM_V),
        math.nan,
        dtype=torch.bfloat16,
        device=device,
    )
    lse = torch.full(
        (int(query.shape[0]), num_heads), math.nan, dtype=torch.float32, device=device
    )
    runner = prepare_cake_mla_varq_dcp_decode(
        query,
        kv_cache,
        page_table,
        seq_lens,
        cum_seq_lens_q,
        max_q_len,
        max_seq_len=max_local_len,
        softmax_scale=_SOFTMAX_SCALE,
        workspace_buffer=workspace,
        causal_seqlens_kv_global=global_lens if enable_dcp else None,
        cp_world=cp_world if enable_dcp else 1,
        cp_rank=cp_rank if enable_dcp else 0,
        out=out,
        lse=lse,
    )
    got_out, got_lse = runner.launch()
    assert got_out is out and got_lse is lse
    torch.cuda.synchronize()
    return out, lse, runner


def _assert_variable_q_dcp_rank_merge(
    query,
    cum_seq_lens_q,
    max_q_len,
    global_kv,
    global_lens,
    *,
    cp_world,
    dtype,
    page_size=64,
    permute=False,
    seed=0,
):
    """Every rank-local state against the reference, then the merged full context."""
    global_lens_host = global_lens.tolist()
    rank_outputs, rank_lses, merges = [], [], []
    for cp_rank in range(cp_world):
        out, lse, runner = _launch_rank(
            query,
            cum_seq_lens_q,
            max_q_len,
            global_kv,
            global_lens,
            cp_world=cp_world,
            cp_rank=cp_rank,
            page_size=page_size,
            permute=permute,
            seed=seed,
        )
        ref_out, ref_lse = _reference_variable_q(
            query,
            cum_seq_lens_q,
            global_kv,
            global_lens_host,
            cp_world=cp_world,
            cp_rank=cp_rank,
        )
        _assert_close_to_reference(out, lse, ref_out, ref_lse, dtype)
        rank_outputs.append(out.clone())
        rank_lses.append(lse.clone())
        merges.append(runner.launches_merge)
    merged_out, merged_lse = _merge_rank_outputs_natural_log(rank_outputs, rank_lses)
    ref_out, ref_lse = _reference_variable_q(
        query, cum_seq_lens_q, global_kv, global_lens_host, cp_world=1, cp_rank=0
    )
    _assert_close_to_reference(merged_out, merged_lse, ref_out, ref_lse, dtype)
    return merges


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_variable_q_dcp_rank_merge(dtype):
    """Compact ragged Q with DCP, an empty request, an empty rank and split items."""
    _device_or_skip()
    query, cum_q, max_q_len, global_kv, global_lens = _make_batched_inputs(
        (1, 130, 3, 4101), (4, 1, 0, 3), 24, dtype, seed=70, max_q_len=8
    )
    _assert_variable_q_dcp_rank_merge(
        query, cum_q, max_q_len, global_kv, global_lens, cp_world=4, dtype=dtype
    )


def test_variable_q_dcp_fp8_direct_empty_rank():
    """FP8 whole-item epilogue with short requests and empty local ranks."""
    _device_or_skip()
    query, cum_q, max_q_len, global_kv, global_lens = _make_batched_inputs(
        (1, 130, 3, 129), (4, 1, 0, 3), 24, torch.float8_e4m3fn, seed=72, max_q_len=8
    )
    merges = _assert_variable_q_dcp_rank_merge(
        query,
        cum_q,
        max_q_len,
        global_kv,
        global_lens,
        cp_world=4,
        dtype=torch.float8_e4m3fn,
    )
    assert merges == [False] * 4


def test_public_api_h96_w2():
    """Route compact DCP through the public experimental entry point."""
    device = _device_or_skip()
    from flashinfer.mla import cake_mla_varq_dcp_decode

    query, cum_q, max_q_len, global_kv, global_lens = _make_batched_inputs(
        (256, 130), (4, 1), 96, torch.bfloat16, seed=71
    )
    global_lens_host = global_lens.tolist()
    kv_cache, page_table, seq_lens, max_local_len = _pack_cyclic_rank_pages(
        global_kv, global_lens_host, 2, 0, page_size=64, permute=False, seed=0
    )
    _skip_unless_registered(
        device,
        dtype=torch.bfloat16,
        page_size=64,
        batch_size=2,
        max_q_len=max_q_len,
        num_heads=96,
        max_seq_len=max_local_len,
    )
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.empty(
        max_cake_mla_varq_dcp_decode_workspace_size(
            batch_size=2, max_q_len=max_q_len, num_heads=96, num_sms=num_sms
        ),
        dtype=torch.uint8,
        device=device,
    )
    out, lse = cake_mla_varq_dcp_decode(
        query,
        kv_cache,
        workspace,
        page_table,
        seq_lens,
        max_local_len,
        _SOFTMAX_SCALE,
        cum_seq_lens_q=cum_q,
        max_q_len=max_q_len,
        enable_dcp=True,
        cp_world=2,
        cp_rank=0,
        causal_seqlens_kv_global=global_lens,
    )
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference_variable_q(
        query, cum_q, global_kv, global_lens_host, cp_world=2, cp_rank=0
    )
    _assert_close_to_reference(out, lse, ref_out, ref_lse, torch.bfloat16)
    with pytest.raises(ValueError, match="require enable_dcp=True"):
        cake_mla_varq_dcp_decode(
            query,
            kv_cache,
            workspace,
            page_table,
            seq_lens,
            max_local_len,
            _SOFTMAX_SCALE,
            cum_seq_lens_q=cum_q,
            max_q_len=max_q_len,
            cp_world=2,
        )
    with pytest.raises(ValueError, match="causal_seqlens_kv_global is required"):
        cake_mla_varq_dcp_decode(
            query,
            kv_cache,
            workspace,
            page_table,
            seq_lens,
            max_local_len,
            _SOFTMAX_SCALE,
            cum_seq_lens_q=cum_q,
            max_q_len=max_q_len,
            enable_dcp=True,
            cp_world=2,
            cp_rank=0,
        )


@pytest.mark.parametrize("num_heads", [12, 24, 48])
def test_packed_head_counts(num_heads):
    """Head counts that pack several tokens into one 128-row M tile."""
    _device_or_skip()
    query, cum_q, max_q_len, global_kv, global_lens = _make_batched_inputs(
        (300, 5, 1000, 64),
        (2, 1, 2, 1),
        num_heads,
        torch.bfloat16,
        seed=73,
        max_q_len=2,
    )
    _assert_variable_q_dcp_rank_merge(
        query,
        cum_q,
        max_q_len,
        global_kv,
        global_lens,
        cp_world=2,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    "dtype,page_size,num_heads,cp_world",
    [
        (torch.bfloat16, 32, 96, 8),
        (torch.float8_e4m3fn, 128, 64, 4),
        (torch.bfloat16, 128, 64, 4),
        (torch.float8_e4m3fn, 32, 96, 8),
    ],
    ids=["bf16_p32", "fp8_p128", "bf16_p128", "fp8_p32"],
)
def test_page_sizes(dtype, page_size, num_heads, cp_world):
    """32- and 128-token pages with permuted page ids (skips unregistered variants)."""
    _device_or_skip()
    assert page_size in SUPPORTED_PAGE_SIZES
    query, cum_q, max_q_len, global_kv, global_lens = _make_batched_inputs(
        (2048, 700, 33, 1500), (3, 0, 2, 1), num_heads, dtype, seed=74, max_q_len=4
    )
    _assert_variable_q_dcp_rank_merge(
        query,
        cum_q,
        max_q_len,
        global_kv,
        global_lens,
        cp_world=cp_world,
        dtype=dtype,
        page_size=page_size,
        permute=True,
        seed=74,
    )


def test_world1_var_q_matches_disabled_dcp():
    """World-one DCP (causal bound = local length) equals the request-local causal decode."""
    _device_or_skip()
    query, cum_q, max_q_len, global_kv, global_lens = _make_batched_inputs(
        (1024, 3, 129, 640), (2, 1, 2, 1), 12, torch.bfloat16, seed=75, max_q_len=2
    )
    dcp_out, dcp_lse, _ = _launch_rank(
        query, cum_q, max_q_len, global_kv, global_lens, cp_world=1, cp_rank=0
    )
    base_out, base_lse, _ = _launch_rank(
        query,
        cum_q,
        max_q_len,
        global_kv,
        global_lens,
        cp_world=1,
        cp_rank=0,
        enable_dcp=False,
    )
    # Both launches split the 1024-token request across clusters; they agree
    # within the documented merge-order spread, not bitwise.
    _assert_replay_close(dcp_out, dcp_lse, base_out, base_lse)
    ref_out, ref_lse = _reference_variable_q(
        query, cum_q, global_kv, global_lens.tolist(), cp_world=1, cp_rank=0
    )
    _assert_close_to_reference(dcp_out, dcp_lse, ref_out, ref_lse, torch.bfloat16)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_split_items_merge_kernel(dtype):
    """Long ragged requests whose plan splits items and launches the merge kernel."""
    _device_or_skip()
    torch.manual_seed(76)
    gen = torch.Generator(device="cpu").manual_seed(76)
    global_lengths = tuple(
        int(torch.randint(1, 8193, (1,), generator=gen).item()) for _ in range(63)
    ) + (8192,)
    q_lens = tuple(
        int(torch.randint(1, 4, (1,), generator=gen).item()) for _ in range(64)
    )
    query, cum_q, max_q_len, global_kv, global_lens = _make_batched_inputs(
        global_lengths, q_lens, 16, dtype, seed=76, max_q_len=3
    )
    out, lse, runner = _launch_rank(
        query,
        cum_q,
        max_q_len,
        global_kv,
        global_lens,
        cp_world=1,
        cp_rank=0,
        permute=True,
        seed=76,
    )
    assert runner.plan["partition_mode"] == 0 and runner.launches_merge
    ref_out, ref_lse = _reference_variable_q(
        query, cum_q, global_kv, global_lens.tolist(), cp_world=1, cp_rank=0
    )
    _assert_close_to_reference(out, lse, ref_out, ref_lse, dtype)


def test_prepared_runner_replays_without_allocation():
    """Repeated launches reuse the self-resetting scheduler state and allocate nothing."""
    _device_or_skip()
    query, cum_q, max_q_len, global_kv, global_lens = _make_batched_inputs(
        (1, 130, 3, 4101), (4, 1, 0, 3), 24, torch.bfloat16, seed=70, max_q_len=8
    )
    out, lse, runner = _launch_rank(
        query, cum_q, max_q_len, global_kv, global_lens, cp_world=4, cp_rank=3
    )
    first = (out.clone(), lse.clone())
    for _ in range(3):
        out.fill_(math.nan)
        lse.fill_(math.nan)
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        runner()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
        assert after["allocation.all.freed"] == before["allocation.all.freed"]
        # The 4101-token request splits into units; replays agree within the
        # documented merge-order spread (bitwise only for unsplit rows).
        _assert_replay_close(out, lse, first[0], first[1])


def test_rejects_unsupported_compute_capability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("device is supported; nothing to reject")
    inputs = {
        k: (v.cuda() if isinstance(v, torch.Tensor) else v)
        for k, v in _host_inputs().items()
    }
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="compute capability"):
        prepare_cake_mla_varq_dcp_decode(
            inputs["query"],
            inputs["kv_cache"],
            inputs["page_table"],
            inputs["seq_lens"],
            inputs["cum_seq_lens_q"],
            inputs["max_q_len"],
            max_seq_len=inputs["max_seq_len"],
            softmax_scale=_SOFTMAX_SCALE,
            workspace_buffer=workspace,
            causal_seqlens_kv_global=inputs["causal_seqlens_kv_global"],
            cp_world=inputs["cp_world"],
            cp_rank=inputs["cp_rank"],
        )


def test_matches_cute_dsl_mla_decode():
    """Same rank-local inputs through the upstream CuTe-DSL DCP path agree within tolerance."""
    device = _device_or_skip()
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")
    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        _get_split_kv_and_workspace_size,
        cute_dsl_mla_decode,
    )
    from flashinfer.cute_dsl.utils import get_num_sm

    query, cum_q, max_q_len, global_kv, global_lens = _make_batched_inputs(
        (4096, 130, 3, 2049), (3, 1, 0, 2), 128, torch.bfloat16, seed=77, max_q_len=3
    )
    out, lse, _ = _launch_rank(
        query, cum_q, max_q_len, global_kv, global_lens, cp_world=8, cp_rank=1
    )
    kv_cache, page_table, seq_lens, max_local_len = _pack_cyclic_rank_pages(
        global_kv, global_lens.tolist(), 8, 1, page_size=64, permute=False, seed=0
    )
    _, workspace_size = _get_split_kv_and_workspace_size(
        4, max_q_len, 128, _LATENT_DIM, get_num_sm(device), max_local_len
    )
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.int8, device=device)
    peer_out, peer_lse = cute_dsl_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=64,
        block_tables=page_table,
        seq_lens=seq_lens,
        max_seq_len=max_local_len,
        softmax_scale=_SOFTMAX_SCALE,
        is_var_seq=True,
        return_lse=True,
        cum_seq_lens_q=cum_q,
        max_q_len=max_q_len,
        enable_dcp=True,
        cp_world=8,
        cp_rank=1,
        causal_seqlens_kv_global=global_lens,
    )
    torch.cuda.synchronize()
    assert torch.equal(torch.isneginf(lse), torch.isneginf(peer_lse.float()))
    finite = ~torch.isneginf(lse)
    torch.testing.assert_close(out.float(), peer_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        lse[finite], peer_lse.float()[finite], atol=1e-2, rtol=1e-2
    )
