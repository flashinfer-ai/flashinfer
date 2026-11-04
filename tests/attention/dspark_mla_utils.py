"""DSpark-shaped sparse MLA inputs and direct-fusion correctness helpers.

Each request owns disjoint physical cache slots. Verification rows share a
compressed history pool, not a top-k list, and their SWA windows shift by one.
Fixture packing, index selection and FP64 checks are never part of timings.
"""

import math

import torch

from tests.attention.mega_mla_utils import fixture, unpack_output, rope_reference


def _round_up(n, alignment):
    return ((n + alignment - 1) // alignment) * alignment


def make_wrapper(cache_format):
    from flashinfer.mla import SparseMLASm120Wrapper

    if cache_format in ("v41_fp8", "v41_mixed"):
        return SparseMLASm120Wrapper(
            backend="sparse",
            kv_scale_format="ue8m0_g32_inline",
            compute_precision="fp8",
            extra_kv_fp4=cache_format == "v41_mixed",
        )
    return SparseMLASm120Wrapper(backend="sparse", kv_cache_format=cache_format)


def dspark_fixture(
    requests=1,
    phase="verify",
    cache_format="v41_mixed",
    *,
    heads=64,
    history_tokens=32768,
    compression_ratio=2,
    prefill_tokens=4096,
    prefill_cp_size=1,
    prefill_cp_rank=0,
    input_scale=1.0,
    with_sink=True,
    seed=42,
    keep_decoded=True,
):
    """Build actual local shapes from docs/dspark_speculative_decoding_and_sparse_attn.md.

    verify: 7 rows/request, shifting 128-entry SWA + independent top-512.
    swa:    7 rows/request, shifting 128-entry SWA only.
    draft:  6 rows/request, the supplied 134-entry SWA selection padded to 192.
    prefill: full request batch, Q sequence sharded by interleaved CP.

    prefill_tokens is local Q rows per request, never KV length. The global
    chunk has prefill_tokens * prefill_cp_size tokens per request; every rank
    sees its full gathered cache. CP=1 preserves historical fixture callers.

    The draft workload models the documented supplied selection, not a generic
    causal-mask rule: all six rows consume the same 134 live SWA entries.
    """
    if phase not in ("verify", "swa", "draft", "prefill"):
        raise ValueError(f"unknown phase: {phase}")
    if requests < 1 or history_tokens < 0 or compression_ratio not in (1, 2):
        raise ValueError("invalid request count, history length or compression ratio")
    if prefill_cp_size < 1 or not 0 <= prefill_cp_rank < prefill_cp_size:
        raise ValueError("invalid prefill CP size/rank")
    rows_per_request = (
        prefill_tokens if phase == "prefill" else (6 if phase == "draft" else 7)
    )
    cp_size = prefill_cp_size if phase == "prefill" else 1
    cp_rank = prefill_cp_rank if phase == "prefill" else 0
    global_chunk_tokens = rows_per_request * cp_size
    if rows_per_request < 1 or history_tokens + global_chunk_tokens > 100001:
        raise ValueError("query positions must fit the fixture's 100001-row RoPE table")
    tokens = requests * rows_per_request
    primary_width = 192 if phase == "draft" else 128
    extra_width = 512 if phase in ("verify", "prefill") else 0
    primary_origin = max(0, history_tokens - 127)
    primary_live = (
        134
        if phase == "draft"
        else history_tokens + global_chunk_tokens - primary_origin
    )
    primary_stride = _round_up(primary_live, 64)
    extra_stride = _round_up(
        max(1, (history_tokens + global_chunk_tokens) // compression_ratio), 64
    )
    f = fixture(
        tokens,
        heads,
        cache_format,
        pool_tokens=requests * primary_stride,
        topk=primary_width,
        extra_topk=extra_width,
        extra_pool_tokens=requests * extra_stride,
        input_scale=input_scale,
        with_sink=with_sink,
        seed=seed,
        keep_decoded=keep_decoded,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    primary = torch.full((tokens, primary_width), -1, dtype=torch.int32)
    primary_lengths = torch.empty(tokens, dtype=torch.int32)
    extra = (
        torch.full((tokens, extra_width), -1, dtype=torch.int32)
        if extra_width
        else None
    )
    extra_lengths = torch.empty(tokens, dtype=torch.int32) if extra_width else None
    positions = torch.empty(tokens, dtype=torch.int32)
    for request in range(requests):
        for row in range(rows_per_request):
            t = request * rows_per_request + row
            position = history_tokens + cp_rank + row * cp_size
            positions[t] = position
            if phase == "draft":
                local = torch.arange(134, dtype=torch.int32)
            else:
                start = max(0, position - 127)
                local = (
                    torch.arange(start, position + 1, dtype=torch.int32)
                    - primary_origin
                )
            primary[t, : local.numel()] = request * primary_stride + local
            primary_lengths[t] = local.numel()
            if extra_width:
                # Only completed compression groups are visible at this query.
                available = (position + 1) // compression_ratio
                count = min(extra_width, available)
                selected = torch.randperm(available, generator=generator)[:count]
                extra[t, :count] = request * extra_stride + selected.to(torch.int32)
                extra_lengths[t] = count
    f["positions"].copy_(positions)
    f["indices"][0].copy_(primary)
    f["lengths"][0].copy_(primary_lengths)
    if extra_width:
        f["indices"][1].copy_(extra)
        f["lengths"][1].copy_(extra_lengths)
    f["workload"] = dict(
        requests=requests,
        phase=phase,
        tokens=tokens,
        heads=heads,
        rows_per_request=rows_per_request,
        local_q_tokens_per_request=rows_per_request,
        global_chunk_tokens_per_request=global_chunk_tokens,
        logical_kv_tokens_per_request=history_tokens + global_chunk_tokens,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_layout="interleave",
        batch_sharded=False,
        heads_sharded=False,
        history_tokens=history_tokens,
        compression_ratio=compression_ratio,
        primary_width=primary_width,
        extra_width=extra_width,
        primary_slots_per_request=primary_stride,
        extra_slots_per_request=extra_stride if extra_width else 0,
        input_scale=input_scale,
        with_sink=with_sink,
        seed=seed,
        selection="shifted SWA and independent causal top-k per query; disjoint request pools",
    )
    f["direct_wrapper"] = make_wrapper(cache_format)
    attach_direct_calls(f)
    return f


def attach_direct_calls(f):
    """Keep direct and automatic preparation caches separate for honest comparisons."""
    if "direct_wrapper" not in f:
        f["direct_wrapper"] = make_wrapper(f["cache_format"])
    wrapper = f["direct_wrapper"]

    def kwargs():
        return dict(
            topk_length=f["lengths"][0],
            attn_sink=f["sink"],
            extra_kv_cache=f["caches"][1],
            extra_indices=f["indices"][1],
            extra_topk_length=f["lengths"][1],
            return_lse=True,
            lse_scale=math.log(2),
            execution="direct",
        )

    def direct_core():
        return wrapper.run(
            f["rotated"],
            f["caches"][0],
            f["indices"][0],
            f["context"],
            512**-0.5,
            **kwargs(),
        )

    def direct_baseline():
        f["prologue"]()
        lse = direct_core()
        f["post"]()
        return f["out"], f["scales"], lse

    def direct_mega():
        return wrapper.run_mega(
            f["q"],
            f["caches"][0],
            f["indices"][0],
            f["out"],
            512**-0.5,
            positions=f["positions"],
            cos_sin=f["cs"],
            output_scales=f["scales"],
            **kwargs(),
        )

    f.update(
        direct_core=direct_core,
        direct_baseline=direct_baseline,
        direct_mega=direct_mega,
    )
    return f


def independent_reference(f, rows=None):
    """FP64 attention over independently decoded cache; robust to empty selections."""
    if rows is None:
        rows = slice(None)
    q = rope_reference(f["q"][rows], f["positions"][rows], f["cs"]).bfloat16().double()
    selected, valid = [], []
    for decoded, indices, lengths in zip(
        f["decoded"], f["indices"], f["lengths"], strict=True
    ):
        if indices is None:
            continue
        if decoded is None:
            raise ValueError("independent reference requires keep_decoded=True")
        idx, lens = indices[rows], lengths[rows]
        selected.append(decoded[idx.clamp_min(0).long()].double())
        valid.append(
            (idx >= 0)
            & (torch.arange(idx.shape[1], device=idx.device)[None] < lens[:, None])
        )
    kv, mask = torch.cat(selected, dim=1), torch.cat(valid, dim=1)
    scores = (q @ kv.transpose(1, 2)) * 512**-0.5
    scores.masked_fill_(~mask[:, None, :], -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    if f["sink"] is not None:
        lse = torch.logaddexp(lse, f["sink"].double()[None])
    probabilities = torch.exp(scores - lse[..., None])
    probabilities = torch.where(torch.isfinite(lse[..., None]), probabilities, 0)
    context = probabilities @ kv
    return rope_reference(context, f["positions"][rows], f["cs"], inverse=True), lse


def assert_direct_fusion(f):
    """Exact same-route fusion gate, independent of the default split schedule."""
    out, scales, lse = f["direct_baseline"]()
    expected = out.float().clone(), scales.clone(), lse.clone()
    f["context"].fill_(123)
    q_before = f["q"].clone()
    actual_lse = f["direct_mega"]()
    for actual, target in zip(
        (f["out"].float(), f["scales"], actual_lse), expected, strict=True
    ):
        torch.testing.assert_close(actual, target, atol=0, rtol=0)
    torch.testing.assert_close(f["q"], q_before, atol=0, rtol=0)
    assert (f["context"] == 123).all()
    return actual_lse


def numerical_metrics(f, lse, rows=None):
    expected, expected_lse = independent_reference(f, rows)
    actual = unpack_output(f["out"], f["scales"])
    if rows is not None:
        actual, lse = actual[rows], lse[rows]
    relative_l2 = (
        (
            (actual.double() - expected).square().sum()
            / expected.square().sum().clamp_min(1e-30)
        )
        .sqrt()
        .item()
    )
    finite = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(lse), finite)
    return dict(
        relative_l2=relative_l2,
        lse_max_abs=(lse.double()[finite] - expected_lse[finite]).abs().max().item()
        if finite.any()
        else 0.0,
        finite_output=bool(torch.isfinite(actual).all()),
    )


def assert_original_one_split(f, original_module):
    """Compare new direct output to an untouched build with the same one-split route.

    The original FP8 decode still launches its merge and retains scratch, so
    this checks the removed boundary rather than trusting the new fused and
    unfused kernels to share the same implementation bug.
    """
    from flashinfer.mla._sparse_mla_sm120._prepared import (
        validate_metadata,
        device_caps,
    )

    f["direct_baseline"]()
    direct = next(iter(f["direct_wrapper"]._prepared_calls.values())).plan.inspect()
    nvfp4 = f["cache_format"] == "nvfp4"
    model = 6 if f["cache_format"] in ("v41_fp8", "v41_mixed") else 1
    m = validate_metadata(
        f["q"],
        f["caches"][0],
        f["indices"][0],
        f["context"],
        f["lengths"][0],
        f["sink"],
        f["caches"][1],
        f["indices"][1],
        f["lengths"][1],
        None,
        None,
        None,
        model=model,
        is_dsv4_nvfp4=nvfp4,
        extra_fp4=f["cache_format"] == "v41_mixed",
        value_dim=512,
    )._replace(variant=int(direct["variant"]))
    chunks = (m.topk + 63) // 64 + (m.extra_topk + 63) // 64
    sm, shared = device_caps(f["q"].device)
    if nvfp4:
        plan = original_module.dsv4_nvfp4_resolve_attention(
            m, 3, chunks, sm, shared, False
        )
        execute = original_module.dsv4_nvfp4_execute_attention
    else:
        numeric = 1 if str(direct["numeric_route"]) == "hybrid" else 0
        plan = original_module.resolve_attention(m, numeric, chunks, sm, shared)
        execute = original_module.execute_attention
    assert plan.inspect()["active_splits"] == 1
    workspace = [
        torch.empty(
            tuple(shape), dtype=getattr(torch, str(dtype)), device=f["q"].device
        )
        for shape, dtype, _, _ in plan.workspace()
    ]
    mid, mlse, lse = workspace
    f["prologue"]()
    execute(
        plan,
        f["rotated"],
        f["caches"][0],
        f["indices"][0],
        mid,
        mlse,
        f["context"],
        lse,
        512**-0.5,
        f["lengths"][0],
        f["sink"],
        f["caches"][1],
        f["indices"][1],
        f["lengths"][1],
        math.log(2),
    )
    f["post"]()
    expected = f["out"].float().clone(), f["scales"].clone(), lse.clone()
    actual_lse = f["direct_mega"]()
    for actual, target in zip(
        (f["out"].float(), f["scales"], actual_lse), expected, strict=True
    ):
        torch.testing.assert_close(actual, target, atol=0, rtol=0)
    return dict(plan.inspect())


def attach_fused_calls(f):
    """Single-launch last-CTA finalization, keeping the original local split plan."""
    if "fused_wrapper" not in f:
        f["fused_wrapper"] = make_wrapper(f["cache_format"])

    def fused_mega():
        return f["fused_wrapper"].run_mega(
            f["q"],
            f["caches"][0],
            f["indices"][0],
            f["out"],
            512**-0.5,
            positions=f["positions"],
            cos_sin=f["cs"],
            output_scales=f["scales"],
            topk_length=f["lengths"][0],
            attn_sink=f["sink"],
            extra_kv_cache=f["caches"][1],
            extra_indices=f["indices"][1],
            extra_topk_length=f["lengths"][1],
            return_lse=True,
            lse_scale=math.log(2),
            execution="fused",
        )

    f["fused_mega"] = fused_mega
    return f


def assert_fused_fusion(f):
    """Bitwise numerical match to the original full pipeline and split schedule."""
    attach_fused_calls(f)
    out, scales, lse = f["baseline"]()
    expected = out.float().clone(), scales.clone(), lse.clone()
    f["context"].fill_(123)
    q_before = f["q"].clone()
    actual_lse = f["fused_mega"]()
    for actual, target in zip(
        (f["out"].float(), f["scales"], actual_lse), expected, strict=True
    ):
        torch.testing.assert_close(actual, target, atol=0, rtol=0)
    torch.testing.assert_close(f["q"], q_before, atol=0, rtol=0)
    assert (f["context"] == 123).all()
    return actual_lse


def native_grouped32_calls(f, cpb):
    """Prepare matching original-width, half-width, and fused native plans."""
    from flashinfer.mla._sparse_mla_sm120._execution import get_sparse_mla_sm120_module
    from flashinfer.mla._sparse_mla_sm120._prepared import (
        validate_metadata,
        device_caps,
    )

    module = get_sparse_mla_sm120_module()
    m = validate_metadata(
        f["q"],
        f["caches"][0],
        f["indices"][0],
        f["context"],
        f["lengths"][0],
        f["sink"],
        f["caches"][1],
        f["indices"][1],
        f["lengths"][1],
        None,
        None,
        None,
        model=1,
        is_dsv4_nvfp4=True,
        extra_fp4=False,
        value_dim=512,
    )._replace(variant=0)
    sm, shared = device_caps(f["q"].device)
    original = module.dsv4_nvfp4_resolve_attention(m, 3, cpb, sm, shared, False)
    grouped = module.dsv4_nvfp4_grouped32_attention_plan(original)
    workspace = [
        torch.empty(tuple(shape), dtype=getattr(torch, str(dtype)), device="cuda")
        for shape, dtype, _, _ in grouped.workspace()
    ]
    mid, mlse, lse = workspace
    completion = torch.zeros(f["q"].shape[:2], dtype=torch.int32, device="cuda")
    common = (
        f["caches"][0],
        f["indices"][0],
        mid,
        mlse,
        f["context"],
        lse,
        512**-0.5,
        f["lengths"][0],
        f["sink"],
        f["caches"][1],
        f["indices"][1],
        f["lengths"][1],
        math.log(2),
    )

    def reference(plan=original):
        f["prologue"]()
        module.dsv4_nvfp4_execute_attention(plan, f["rotated"], *common)
        f["post"]()
        return f["out"], f["scales"], lse

    def fused():
        module.dsv4_nvfp4_execute_fused_mega_attention(
            grouped,
            f["q"],
            *common,
            f["positions"],
            f["cs"],
            f["out"].view(torch.uint8),
            f["scales"],
            completion,
        )
        return f["out"], f["scales"], lse

    return reference, lambda: reference(grouped), fused, grouped, completion
