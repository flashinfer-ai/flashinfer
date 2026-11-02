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

Bit-identity of routingCustom's small-batch kernels (256 experts, top-8, 1-64 tokens:
routingIndicesSmallBatchKernel up to 18 tokens, routingIndicesSmallBatchClusterKernel above) with
the static-block / dyn-block / split kernels they replace.

The switch FLASHINFER_ROUTING_SMALL_BATCH_KERNEL=0 restores the previous kernels and is
read on every routing call, so each case runs both paths in one process:
- routing: expert ids, BF16 weights, CTA maps, padded size, and the permutation. The layout is
  compared exactly up to 16 tokens (block kernels, ascending token order inside an expert); from
  17 tokens the previous path places tokens inside an expert in atomic order, so the permutation
  is compared as per-expert row sets plus a consistent (token, k) -> row mapping;
- the NVFP4 MoE output (FromLogits and precomputed top-k, finalized and not), bit for bit,
  including forced tactics of every tile size and CUDA-graph replay.
"""

import contextlib
import os
import zlib
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
from flashinfer.quantization.nvfp4_quantization_utils import NVFP4_4OVER6_CODE_FROM_ENV
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    RoutingMethodType,
    SfLayout,
)
from flashinfer.utils import get_compute_capability

NUM_EXPERTS = 256
TOP_K = 8
MAX_TOKENS = 64  # SmallBatchKernelMaxNumTokens
SINGLE_CTA_MAX_TOKENS = 18  # SmallBatchKernelSingleCtaMaxNumTokens
# Token counts of the routing comparisons: the kernel's window plus the counts around it (outside
# the window both settings run the previous kernels, which the comparison then pins as identical).
TOKEN_COUNTS = range(1, MAX_TOKENS + 9)
EXACT_LAYOUT_MAX_TOKENS = 16  # previous path: block kernels up to here, then atomics
SWITCH = "FLASHINFER_ROUTING_SMALL_BATCH_KERNEL"
TILES = [8, 16, 32, 64, 128, 192, 256]
# A GLM-5-class per-rank MoE shape at TP4 (routing: DeepSeek-V3, n_group 1, scale 2.5).
HIDDEN = 6144
INTER = 512

METHODS = {
    "deepseek_v3": SimpleNamespace(
        type=int(RoutingMethodType.DeepSeekV3), n_group=1, topk_group=1, scale=2.5
    ),
    "minimax2": SimpleNamespace(
        type=int(RoutingMethodType.MiniMax2), n_group=None, topk_group=None, scale=1.0
    ),
}


@pytest.fixture(autouse=True)
def require_trtllm_gen_moe():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("trtllm-gen fused MoE requires SM100 / SM103")


def small_batch_kernels_dispatched() -> bool:
    """The small-batch kernels run on SM10x only; other architectures keep the previous kernels
    with the switch on, so there the comparisons run the same kernels twice."""
    return get_compute_capability(torch.device("cuda"))[0] == 10


@contextlib.contextmanager
def small_batch_kernel(enabled: bool):
    saved = os.environ.get(SWITCH)
    os.environ[SWITCH] = "1" if enabled else "0"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(SWITCH, None)
        else:
            os.environ[SWITCH] = saved


def stable_seed(*parts):
    return zlib.crc32(repr(parts).encode()) % (2**31)


def bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

LOGIT_CASES = ["randn", "glm", "ties", "all_equal", "cancel", "hot"]


def make_logits(
    case, num_tokens, seed, logits_dtype=torch.float32, bias_dtype=torch.float32
):
    """(logits, bias) for one routing case."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    # A GLM-5-like e_score_correction_bias: an offset of ~8 with a spread of a few bf16 ULPs.
    glm_bias = 8.0 + 0.03 * randn(NUM_EXPERTS)
    if case == "randn":
        logits, bias = 3.0 * randn(num_tokens, NUM_EXPERTS), 0.5 * randn(NUM_EXPERTS)
    elif case == "glm":
        logits, bias = 2.0 * randn(num_tokens, NUM_EXPERTS), glm_bias
    elif case == "ties":
        # Many exactly equal keys: ties go to the lower expert index.
        logits = torch.round(randn(num_tokens, NUM_EXPERTS)) / 2
        bias = torch.zeros(NUM_EXPERTS, device="cuda")
    elif case == "all_equal":
        logits = torch.zeros(num_tokens, NUM_EXPERTS, device="cuda")
        bias = torch.zeros(NUM_EXPERTS, device="cuda")
    elif case == "cancel":
        # Tiny sigmoids under a large bias: weights must use the un-biased sigmoid.
        logits, bias = -30.0 + 5.0 * randn(num_tokens, NUM_EXPERTS), glm_bias
    elif case == "hot":
        # Every token picks expert 3: one expert spans several CTA tiles.
        logits, bias = randn(num_tokens, NUM_EXPERTS), 0.1 * randn(NUM_EXPERTS)
        logits[:, 3] += 20.0
    else:
        raise ValueError(case)
    return logits.to(logits_dtype).contiguous(), bias.to(bias_dtype).contiguous()


def make_topk(num_tokens, seed, case="distinct"):
    """Precomputed (ids int32, weights bf16) of shape [num_tokens, 8], distinct ids per token;
    "hot": every token lists expert 5 first (one expert spans several CTA tiles)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    keys = torch.rand(num_tokens, NUM_EXPERTS, generator=gen, device="cuda")
    if case == "hot":
        keys[:, 5] = -1.0
    ids = torch.argsort(keys, dim=1)[:, :TOP_K].to(torch.int32)
    weights = torch.rand(num_tokens, TOP_K, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    return ids.contiguous(), weights.contiguous()


def pack_topk(ids, weights):
    """trtllm-gen packed top-k: (expert << 16) | bf16 weight bits."""
    return (
        (ids.to(torch.int32) << 16) | (bits(weights).to(torch.int32) & 0xFFFF)
    ).contiguous()


# ---------------------------------------------------------------------------------------------
# Routing (staged BF16 entry: Routing::Runner -> routingCustom, outputs returned)
# ---------------------------------------------------------------------------------------------

_STAGED = {}


def _staged_operands(num_tokens, local_num):
    """Shape-only BF16 operands of the staged routing entry (the GEMMs never run)."""
    key = (num_tokens, local_num)
    if key not in _STAGED:
        _STAGED[key] = SimpleNamespace(
            hs=torch.zeros(num_tokens, 128, dtype=torch.bfloat16, device="cuda"),
            w1=torch.zeros(local_num, 256, 128, dtype=torch.bfloat16, device="cuda"),
            w2=torch.zeros(local_num, 128, 128, dtype=torch.bfloat16, device="cuda"),
        )
    return _STAGED[key]


def run_routing(
    *,
    logits=None,
    bias=None,
    ids=None,
    weights=None,
    packed=None,
    num_tokens,
    tile,
    method="deepseek_v3",
    local_offset=0,
    local_num=NUM_EXPERTS,
):
    m = METHODS[method]
    ops = _staged_operands(num_tokens, local_num)
    empty_ids = torch.empty(0, TOP_K, dtype=torch.int32, device="cuda")
    empty_w = torch.empty(0, TOP_K, dtype=torch.bfloat16, device="cuda")
    if packed is not None:
        expert_indices, expert_weights = packed, empty_w
    elif ids is not None:
        expert_indices, expert_weights = ids, weights
    else:
        expert_indices, expert_weights = empty_ids, empty_w
    replay = (
        torch.full((num_tokens, TOP_K), -2, dtype=torch.int16, device="cuda")
        if logits is not None
        else None
    )
    out = get_trtllm_moe_sm100_module().moe_op.trtllm_moe_run_routing(
        logits,
        bias,
        expert_indices,
        expert_weights,
        ops.hs,
        ops.w1,
        ops.w2,
        NUM_EXPERTS,
        TOP_K,
        m.n_group,
        m.topk_group,
        128,  # intermediate_size
        local_offset,
        local_num,
        m.scale,
        m.type,
        True,  # use_shuffled_weight
        0,  # weight_layout: MajorK
        True,  # enable_pdl
        [tile, 0],
        int(ActivationType.Swiglu),
        True,  # norm_topk_prob (unused by these methods)
        replay,
    )
    outs = list(out)
    t = {i: torch.from_dlpack(outs[i]) for i in range(1, 7)}
    torch.cuda.synchronize()
    # With caller weights (precomputed ids) the launcher borrows them: slot 0 is then empty.
    borrowed = ids is not None and packed is None
    return SimpleNamespace(
        weights=weights.clone() if borrowed else torch.from_dlpack(outs[0]).clone(),
        e2p=t[1][: num_tokens * TOP_K].clone(),
        p2t=t[2].clone(),
        cta_batch=t[3].clone(),
        cta_limit=t[4].clone(),
        num_ctas=int(t[5].item()),
        padded=int(t[6].item()),
        replay=replay,
    )


def assert_same_routing(
    ref, new, ids, *, num_tokens, tile, local_offset, local_num, tag
):
    """`ids` [num_tokens, 8]: the expert of every (token, k) pair, -1 when not routed."""
    assert new.num_ctas == ref.num_ctas, f"{tag}: num_non_exiting_ctas"
    assert new.padded == ref.padded, f"{tag}: total_num_padded_tokens"
    n = ref.num_ctas
    assert torch.equal(new.cta_batch[:n], ref.cta_batch[:n]), (
        f"{tag}: cta_idx_xy_to_batch_idx"
    )
    assert torch.equal(new.cta_limit[:n], ref.cta_limit[:n]), (
        f"{tag}: cta_idx_xy_to_mn_limit"
    )
    # Tile padding: -1 rows at the same places.
    p2t_new, p2t_ref = new.p2t[: ref.padded], ref.p2t[: ref.padded]
    assert torch.equal(p2t_new == -1, p2t_ref == -1), f"{tag}: tile padding rows"

    ids = ids.reshape(-1).long()
    routed = (ids >= local_offset) & (ids < local_offset + local_num)
    for name, r in (("new", new), ("ref", ref)):
        e2p = r.e2p.long()
        assert torch.equal(e2p >= 0, routed), f"{tag}: {name} unrouted pairs not -1"
        if bool(routed.any()):
            rows = e2p[routed]
            expert = r.cta_batch.long()[rows // tile] + local_offset
            assert torch.equal(expert, ids[routed]), (
                f"{tag}: {name} row in wrong expert tile"
            )
            tok = torch.arange(num_tokens, device="cuda").repeat_interleave(TOP_K)
            assert torch.equal(r.p2t.long()[rows], tok[routed]), f"{tag}: {name} p2t"
    # Same rows per expert (the previous path's order inside an expert may differ above 16).
    assert torch.equal(
        torch.sort(new.e2p[routed]).values, torch.sort(ref.e2p[routed]).values
    ), f"{tag}: permuted row sets"
    if num_tokens <= EXACT_LAYOUT_MAX_TOKENS:
        assert torch.equal(new.e2p, ref.e2p), (
            f"{tag}: expanded_idx_to_permuted_idx (exact)"
        )
        assert torch.equal(p2t_new, p2t_ref), (
            f"{tag}: permuted_idx_to_token_idx (exact)"
        )


def check_logits_case(
    logits, bias, *, tile, method="deepseek_v3", local_offset=0, local_num=256, tag=""
):
    T = logits.shape[0]
    kw = dict(
        num_tokens=T,
        tile=tile,
        method=method,
        local_offset=local_offset,
        local_num=local_num,
    )
    with small_batch_kernel(False):
        ref = run_routing(logits=logits, bias=bias, **kw)
    with small_batch_kernel(True):
        new = run_routing(logits=logits, bias=bias, **kw)
    assert bool((ref.replay >= 0).all()), (
        f"{tag}: previous path did not record the experts"
    )
    assert torch.equal(new.replay, ref.replay), (
        f"{tag}: routing_replay_out (expert ids, k order)"
    )
    assert torch.equal(bits(new.weights), bits(ref.weights)), f"{tag}: expert weights"
    assert_same_routing(
        ref,
        new,
        ref.replay.int(),
        num_tokens=T,
        tile=tile,
        local_offset=local_offset,
        local_num=local_num,
        tag=tag,
    )


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("case", LOGIT_CASES)
@pytest.mark.parametrize("method", list(METHODS))
def test_routing_from_logits(method, case, tile):
    for T in TOKEN_COUNTS:
        logits, bias = make_logits(case, T, stable_seed(method, case, tile, T))
        check_logits_case(
            logits,
            bias,
            tile=tile,
            method=method,
            tag=f"{method}/{case}/tile{tile}/T{T}",
        )


@pytest.mark.parametrize("tile", [8, 16])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_routing_from_logits_dtypes(logits_dtype, bias_dtype, tile):
    for T in TOKEN_COUNTS:
        logits, bias = make_logits(
            "glm", T, stable_seed("dt", tile, T), logits_dtype, bias_dtype
        )
        check_logits_case(
            logits, bias, tile=tile, tag=f"{logits_dtype}/{bias_dtype}/tile{tile}/T{T}"
        )


@pytest.mark.parametrize("tile", [8, 64])
@pytest.mark.parametrize("window", [(64, 64), (128, 32), (0, 128)])
def test_routing_from_logits_expert_parallel(window, tile):
    local_offset, local_num = window
    for T in TOKEN_COUNTS:
        logits, bias = make_logits("glm", T, stable_seed("ep", window, tile, T))
        check_logits_case(
            logits,
            bias,
            tile=tile,
            local_offset=local_offset,
            local_num=local_num,
            tag=f"ep{window}/tile{tile}/T{T}",
        )


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("fmt", ["ids", "packed"])
@pytest.mark.parametrize("case", ["distinct", "hot"])
def test_routing_precomputed(case, fmt, tile):
    for T in TOKEN_COUNTS:
        ids, weights = make_topk(T, stable_seed("pre", case, fmt, tile, T), case)
        inputs = (
            dict(packed=pack_topk(ids, weights))
            if fmt == "packed"
            else dict(ids=ids, weights=weights)
        )
        with small_batch_kernel(False):
            ref = run_routing(num_tokens=T, tile=tile, **inputs)
        with small_batch_kernel(True):
            new = run_routing(num_tokens=T, tile=tile, **inputs)
        tag = f"{case}/{fmt}/tile{tile}/T{T}"
        if fmt == "packed":
            assert torch.equal(bits(new.weights), bits(ref.weights)), (
                f"{tag}: unpacked weights"
            )
            assert torch.equal(bits(new.weights), bits(weights)), (
                f"{tag}: weights != packed scores"
            )
        assert_same_routing(
            ref,
            new,
            ids,
            num_tokens=T,
            tile=tile,
            local_offset=0,
            local_num=NUM_EXPERTS,
            tag=tag,
        )


@pytest.mark.parametrize("window", [(64, 64), (192, 64)])
@pytest.mark.parametrize("fmt", ["ids", "packed"])
def test_routing_precomputed_expert_parallel(fmt, window):
    local_offset, local_num = window
    for T in TOKEN_COUNTS:
        ids, weights = make_topk(T, stable_seed("pre-ep", fmt, window, T))
        inputs = (
            dict(packed=pack_topk(ids, weights))
            if fmt == "packed"
            else dict(ids=ids, weights=weights)
        )
        kw = dict(num_tokens=T, tile=8, local_offset=local_offset, local_num=local_num)
        with small_batch_kernel(False):
            ref = run_routing(**kw, **inputs)
        with small_batch_kernel(True):
            new = run_routing(**kw, **inputs)
        assert_same_routing(
            ref,
            new,
            ids,
            num_tokens=T,
            tile=8,
            local_offset=local_offset,
            local_num=local_num,
            tag=f"{fmt}/ep{window}/T{T}",
        )


@pytest.mark.parametrize("fmt", ["ids", "packed"])
def test_routing_precomputed_invalid_and_duplicate_ids(fmt):
    """Out-of-range ids are not routed; an expert listed twice for a token is routed once,
    from its lowest k. (The previous kernels disagree on duplicates: the block kernels keep one
    pair, chosen by a race; the cluster kernel routes every copy.)"""
    if not small_batch_kernels_dispatched():
        pytest.skip("the small-batch routing kernels run on SM10x only")
    for T in [1, 6, 16, 18, 19, 24, 32, 33, 48, 64]:
        ids, weights = make_topk(T, stable_seed("bad", fmt, T))
        ids[0, 1] = -1
        ids[-1, 7] = NUM_EXPERTS + 3 if fmt == "ids" else -5
        ids[T // 2, 5] = ids[T // 2, 2]  # duplicate: pair k=5 must map to -1
        inputs = (
            dict(packed=pack_topk(ids, weights))
            if fmt == "packed"
            else dict(ids=ids, weights=weights)
        )
        with small_batch_kernel(True):
            new = run_routing(num_tokens=T, tile=8, **inputs)
        expected = ids.clone()
        expected[T // 2, 5] = -1
        expected = torch.where((expected >= 0) & (expected < NUM_EXPERTS), expected, -1)
        assert_same_routing(
            new,
            new,
            expected,
            num_tokens=T,
            tile=8,
            local_offset=0,
            local_num=NUM_EXPERTS,
            tag=f"{fmt}/T{T}",
        )
        counts = torch.bincount(expected[expected >= 0].long(), minlength=NUM_EXPERTS)
        assert new.padded == int(((counts + 7) // 8 * 8).sum()), (
            f"{fmt}/T{T}: padded size"
        )


def _routing_kernels(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted(
        {
            e.name
            for e in prof.events()
            if "routing" in e.name.lower() and "Kernel" in e.name
        }
    )


@pytest.mark.parametrize("T", [1, 6, 16, 18, 19, 32, 64, 65])
def test_switch_selects_kernels(T):
    """Guards the premise of the comparisons: the switch really changes the kernels, and the
    small-batch kernels serve exactly the calls in their token windows, in a single launch
    (SM10x; elsewhere the switch leaves the kernels unchanged)."""
    logits, bias = make_logits("glm", T, 1)

    def call():
        run_routing(logits=logits, bias=bias, num_tokens=T, tile=8)

    try:
        with small_batch_kernel(True):
            new = _routing_kernels(call)
        with small_batch_kernel(False):
            old = _routing_kernels(call)
    except Exception as e:  # noqa: BLE001 - no CUPTI in this environment
        pytest.skip(f"torch profiler unavailable: {e}")
    if not new and not old:
        pytest.skip("torch profiler recorded no kernels")
    small = [k for k in new if "SmallBatch" in k]
    if not small_batch_kernels_dispatched():
        assert new == old, (new, old)
    elif T <= MAX_TOKENS:
        assert len(new) == 1 and len(small) == 1, new
        cluster = "SmallBatchClusterKernel" in small[0]
        assert cluster == (T > SINGLE_CTA_MAX_TOKENS), new
    else:
        assert not small, new
    assert not any("SmallBatch" in k for k in old), old


# ---------------------------------------------------------------------------------------------
# NVFP4 MoE output
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fp4_weights():
    gen = torch.Generator(device="cuda").manual_seed(11)

    def fp4(*shape):
        return torch.randint(
            0, 256, shape, generator=gen, device="cuda", dtype=torch.uint8
        )

    def scales(*shape):
        return (0.5 + torch.rand(*shape, generator=gen, device="cuda")).to(
            torch.float8_e4m3fn
        )

    return dict(
        gemm1_weights=fp4(NUM_EXPERTS, 2 * INTER, HIDDEN // 2),
        gemm1_weights_scale=scales(NUM_EXPERTS, 2 * INTER, HIDDEN // 16),
        gemm2_weights=fp4(NUM_EXPERTS, HIDDEN, INTER // 2),
        gemm2_weights_scale=scales(NUM_EXPERTS, HIDDEN, INTER // 16),
        output1_scale_scalar=torch.full((NUM_EXPERTS,), 1e-3, device="cuda"),
        output1_scale_gate_scalar=torch.full((NUM_EXPERTS,), 1e-2, device="cuda"),
        output2_scale_scalar=torch.full((NUM_EXPERTS,), 1e-2, device="cuda"),
    )


def make_activations(T, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hs = torch.randint(
        0, 256, (T, HIDDEN // 2), generator=gen, device="cuda", dtype=torch.uint8
    )
    hs_scale = (0.5 + torch.rand(T, HIDDEN // 16, generator=gen, device="cuda")).to(
        torch.float8_e4m3fn
    )
    return hs, hs_scale


def moe_kwargs(weights, hs, hs_scale, do_finalize):
    m = METHODS["deepseek_v3"]
    return dict(
        hidden_states=hs,
        hidden_states_scale=hs_scale,
        hidden_states_scale_layout=SfLayout.layout_linear,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_bias=None,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=m.n_group,
        topk_group=m.topk_group,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=m.scale,
        routing_method_type=m.type,
        do_finalize=do_finalize,
        enable_pdl=True,
        activation_type=int(ActivationType.Swiglu),
        **weights,
    )


def outputs(res, do_finalize):
    """Order-independent outputs: the finalized output, or (each (token, k)'s FC2 row, the
    weights) for a deferred finalize (rows inside an expert may be placed in another order)."""
    if do_finalize:
        return [res[0].clone()]
    gemm2, weights, e2p = res[0], res[1], res[2]
    idx = e2p.reshape(-1).long()
    rows = torch.where((idx >= 0).view(-1, 1), gemm2[idx.clamp(min=0)], 0)
    return [rows, weights.clone()]


def assert_same_outputs(new, ref, ref2, tag):
    for i, (a, b, b2) in enumerate(zip(new, ref, ref2, strict=True)):
        if not torch.equal(bits(b), bits(b2)):
            pytest.skip(
                f"{tag}: the previous path is not run-to-run deterministic here"
            )
        assert torch.equal(bits(a), bits(b)), (
            f"{tag}: output {i} differs (max |d| {(a.float() - b.float()).abs().max().item():.3g})"
        )


@pytest.mark.parametrize("do_finalize", [True, False])
def test_fp4_moe_from_logits(fp4_weights, do_finalize):
    for T in range(1, MAX_TOKENS + 1):
        logits, bias = make_logits("glm", T, stable_seed("moe", T))
        hs, hs_scale = make_activations(T, stable_seed("act", T))
        kw = moe_kwargs(fp4_weights, hs, hs_scale, do_finalize)

        def call():
            return outputs(trtllm_fp4_block_scale_moe(logits, bias, **kw), do_finalize)

        with small_batch_kernel(False):
            ref, ref2 = call(), call()
        with small_batch_kernel(True):
            new = call()
        assert_same_outputs(new, ref, ref2, f"T{T} finalize={do_finalize}")


@pytest.mark.parametrize("fmt", ["packed", "tuple"])
def test_fp4_moe_precomputed(fp4_weights, fmt):
    for T in [1, 3, 4, 5, 8, 16, 17, 24, 33, 48, 64]:
        ids, weights = make_topk(T, stable_seed("moe-pre", fmt, T))
        topk = pack_topk(ids, weights) if fmt == "packed" else (ids, weights)
        hs, hs_scale = make_activations(T, stable_seed("act", T))
        kw = moe_kwargs(fp4_weights, hs, hs_scale, True)

        def call():
            return outputs(trtllm_fp4_block_scale_routed_moe(topk, None, **kw), True)

        with small_batch_kernel(False):
            ref, ref2 = call(), call()
        with small_batch_kernel(True):
            new = call()
        assert_same_outputs(new, ref, ref2, f"{fmt}/T{T}")


def _first_config_per_tile(T):
    configs = get_trtllm_moe_sm100_module().moe_op.trtllm_get_valid_moe_configs(
        int(DtypeTrtllmGen.E2m1),
        int(DtypeTrtllmGen.E2m1),
        int(Fp8QuantizationType.NoneFp8),
        TOP_K,
        HIDDEN,
        HIDDEN,
        INTER,
        NUM_EXPERTS,
        int(ActivationType.Swiglu),
        True,  # use_shuffled_weight
        0,  # weight_layout: MajorK
        False,  # use_per_token_scaling
        T,
        False,  # has_gemm1_lora_delta
    )
    first = {}
    for tile, cfg in ([int(v) for v in c] for c in configs):
        first.setdefault(tile, [tile, cfg])
    return [first[t] for t in sorted(first)]


def _run_fp4_ffi(kw, logits, bias, tactic):
    """trtllm_fp4_block_scale_moe through the FFI with a forced [tile, config] tactic."""
    T = kw["hidden_states"].shape[0]
    output = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.empty(T, TOP_K, dtype=torch.int32, device="cuda")
    topk_weights = torch.empty(T, TOP_K, dtype=torch.bfloat16, device="cuda")
    get_trtllm_moe_sm100_module().moe_op.trtllm_fp4_block_scale_moe(
        0,  # RoutingInputMode.FromLogits
        logits,
        topk_ids,
        topk_weights,
        bias,
        kw["hidden_states"],
        kw["hidden_states_scale"],
        kw["gemm1_weights"],
        kw["gemm1_weights_scale"],
        None,
        None,
        None,
        None,
        None,
        kw["gemm2_weights"],
        kw["gemm2_weights_scale"],
        None,
        kw["output1_scale_scalar"],
        kw["output1_scale_gate_scalar"],
        kw["output2_scale_scalar"],
        None,
        NUM_EXPERTS,
        TOP_K,
        0,
        kw["n_group"],
        kw["topk_group"],
        INTER,
        0,
        NUM_EXPERTS,
        kw["routed_scaling_factor"],
        kw["routing_method_type"],
        True,  # do_finalize
        True,  # enable_pdl
        int(ActivationType.Swiglu),
        output,
        list(tactic),
        True,
        None,
        [],
        [],
        False,
        None,  # valid_hidden_size
        None,  # valid_intermediate_size
        None,  # workspace_buffer
        False,  # query_workspace_size
        NVFP4_4OVER6_CODE_FROM_ENV,
    )
    return [output]


def test_fp4_moe_every_tile(fp4_weights):
    seen_tiles = set()
    for T in [1, 3, 6, 8, 12, 16, 17, 24, 32, 33, 48, 64]:
        logits, bias = make_logits("glm", T, stable_seed("tile", T))
        hs, hs_scale = make_activations(T, stable_seed("act", T))
        kw = moe_kwargs(fp4_weights, hs, hs_scale, True)
        for tactic in _first_config_per_tile(T):
            seen_tiles.add(tactic[0])
            with small_batch_kernel(False):
                ref = _run_fp4_ffi(kw, logits, bias, tactic)
                ref2 = _run_fp4_ffi(kw, logits, bias, tactic)
            with small_batch_kernel(True):
                new = _run_fp4_ffi(kw, logits, bias, tactic)
            assert_same_outputs(new, ref, ref2, f"T{T} tactic {tactic}")
    assert {8, 16}.issubset(seen_tiles), f"tiles covered: {sorted(seen_tiles)}"


@pytest.mark.parametrize("T", [1, 6, 12, 18, 19, 33, 64])
def test_fp4_moe_cuda_graph(fp4_weights, T):
    """Capture the small-batch path, replay on fresh inputs, compare with the previous path
    run eagerly on the same inputs."""
    logits, bias = make_logits("glm", T, 5)
    hs, hs_scale = make_activations(T, 6)
    kw = moe_kwargs(fp4_weights, hs, hs_scale, True)
    with small_batch_kernel(True):
        trtllm_fp4_block_scale_moe(logits, bias, **kw)  # warm up outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = trtllm_fp4_block_scale_moe(logits, bias, **kw)[0]
    for step in range(3):
        new_logits, _ = make_logits("glm", T, 100 + step)
        new_hs, new_hs_scale = make_activations(T, 200 + step)
        logits.copy_(new_logits)
        hs.copy_(new_hs)
        hs_scale.copy_(new_hs_scale)
        graph.replay()
        torch.cuda.synchronize()
        with small_batch_kernel(False):
            ref = trtllm_fp4_block_scale_moe(logits, bias, **kw)[0].clone()
            ref2 = trtllm_fp4_block_scale_moe(logits, bias, **kw)[0].clone()
        assert_same_outputs(
            [captured.clone()], [ref], [ref2], f"graph T{T} replay {step}"
        )
