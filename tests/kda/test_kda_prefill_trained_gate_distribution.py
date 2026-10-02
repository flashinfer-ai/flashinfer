# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded-gate KDA prefill under trained deep-layer gate statistics.

A real Kimi-K3 layer-44 prefill (64 tokens after an 8K cached prefix, nonzero
recurrent state, ``lower_bound=-5``) exposed a BF16 fused-M128 inverse
composition bug: the 8-row block inverse was composed from a stale fragment,
so every route built on that body drifted from the recurrence from token 8
onwards (output rel L2 above 60, non-finite values on many real calls) while
synthetic ``randn`` gates never triggered it.  This test reproduces the trained
statistics synthetically (per-head decay rate ``exp(A_log)`` near 1, deep
negative ``dt_bias`` so most gates sit in ``(-0.5, 0]``, wide beta logits up to
sigmoid ~0.99, nonzero initial state) and checks the ``recurrent_kda`` facade
(BF16 state pool), the generated FP32-indexed portfolio behind the same facade
(257-slot FP32 state pool) and the prepared BF16 export (FP32 state pool)
against an FP64 token-by-token recurrence.  Non-finite output is a hard
failure.
"""

import math

import pytest

import torch
import torch.nn.functional as F

from flashinfer import prepare_bf16_kda_prefill
from flashinfer.kda import recurrent_kda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake KDA prefill requires SM100a or SM103a",
)

HEAD_DIM = 128
LOWER_BOUND = -5.0
# Per-head trained statistics of one 12-head TP rank of a deep layer, kept
# aligned by head: the heads whose decay rate exp(A_log) sits closest to 1.0
# also carry beta near 1 (mean 0.95-0.97), which is the combination that
# exposed the bug.  Values are statistics, not checkpoint tensors.
_A_LOG_PER_HEAD = (
    0.0089,
    0.4862,
    0.0643,
    0.2757,
    -0.0391,
    0.1571,
    0.6433,
    0.4869,
    0.1956,
    0.3648,
    0.7447,
    0.5918,
)
_BETA_LOGIT_MEAN_PER_HEAD = (
    3.65,
    -1.82,
    -2.68,
    -1.39,
    2.99,
    -0.22,
    -5.11,
    -4.32,
    -0.41,
    -3.29,
    -5.56,
    -7.12,
)
# dt_bias quantiles (probability, value) of the same layer
_DT_BIAS_QUANTILES = (
    (0.000, -8.2005),
    (0.001, -7.9175),
    (0.010, -7.1049),
    (0.100, -6.5265),
    (0.500, -4.6909),
    (0.900, -2.8283),
    (0.990, -1.9639),
    (0.999, -1.3914),
    (1.000, -1.2937),
)


def _quantile_values(anchors, count, generator, device):
    probabilities = torch.linspace(0.0, 1.0, count, device=device)
    ps = torch.tensor([p for p, _ in anchors], dtype=torch.float32, device=device)
    vs = torch.tensor([v for _, v in anchors], dtype=torch.float32, device=device)
    upper = torch.searchsorted(ps, probabilities, right=True).clamp_(
        1, len(anchors) - 1
    )
    lower = upper - 1
    frac = (probabilities - ps[lower]) / (ps[upper] - ps[lower])
    values = vs[lower] + frac * (vs[upper] - vs[lower])
    return values[torch.randperm(count, generator=generator, device=device)]


def _per_head(anchors, heads, device):
    return torch.tensor(
        [anchors[h % len(anchors)] for h in range(heads)],
        dtype=torch.float32,
        device=device,
    )


def trained_gate_inputs(*, lengths, heads, seed, device="cuda"):
    """Synthetic operands following the layer-44 fixture statistics."""
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = sum(lengths)
    shape = (1, tokens, heads, HEAD_DIM)

    def randn(*size, scale=1.0, shift=0.0):
        return torch.randn(size, generator=gen, device=device) * scale + shift

    # Trained q / k / v are far from isotropic: within a chunk the keys of one
    # head share a direction (mean pairwise |cos| 0.84-0.99 in the captured
    # layer; q 0.81-0.97, v 0.66-0.96).  Random 128-d vectors are nearly
    # orthogonal, which makes the intra-chunk Gram matrix (and therefore the
    # block-inverse composition) numerically irrelevant and hides the bug.
    def correlated(rms, cos):
        # x_t = d + s * n_t with ||d|| = 1 and n_t ~ N(0, I): E[cos(x_i, x_j)] = 1 / (1 + D s^2),
        # so s = sqrt((1 - cos) / (cos * D)).  The result is rescaled to the trained per-element rms.
        spread = math.sqrt((1.0 - cos) / (cos * HEAD_DIM))
        direction = F.normalize(randn(1, 1, heads, HEAD_DIM), dim=-1)
        values = direction + spread * randn(*shape)
        return (
            values
            * (rms * math.sqrt(HEAD_DIM) / math.sqrt(1.0 + HEAD_DIM * spread * spread))
        ).bfloat16()

    q = correlated(0.09, 0.90)
    k = correlated(0.22, 0.93)
    v = correlated(0.06, 0.80).clamp(-1.0, 1.0)
    # self-check of the contract: the keys must actually share a direction inside a chunk,
    # otherwise the block-inverse composition is numerically irrelevant and the test proves nothing.
    kn = F.normalize(k[0, :16, 0].float(), dim=-1)
    gram = kn @ kn.T
    mean_cos = (
        gram[torch.triu(torch.ones_like(gram, dtype=torch.bool), 1)].mean().item()
    )
    assert 0.85 <= mean_cos <= 0.97, (
        f"key direction sharing off contract: mean cos {mean_cos:.3f}"
    )
    # raw gate projection: mean -0.62, std 0.88
    g = randn(*shape, scale=0.88, shift=-0.62).bfloat16()
    # beta logits: per-head means of the trained layer, per-token spread 1.5
    beta = (
        randn(1, tokens, heads, scale=1.5)
        + _per_head(_BETA_LOGIT_MEAN_PER_HEAD, heads, device)
    ).bfloat16()
    A_log = _per_head(_A_LOG_PER_HEAD, heads, device).contiguous()
    dt_bias = (
        _quantile_values(_DT_BIAS_QUANTILES, heads * HEAD_DIM, gen, device)
        .reshape(heads, HEAD_DIM)
        .contiguous()
    )
    # nonzero recurrent state after a long prefix (rms ~0.014 with sparse peaks ~1)
    state = randn(len(lengths), heads, HEAD_DIM, HEAD_DIM, scale=0.014)
    peaks = torch.rand(state.shape, generator=gen, device=device) < 2e-4
    state = torch.where(peaks, randn(*state.shape, scale=0.5), state)
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state=state,
        offsets=offsets,
    )


def fp64_reference(inp):
    """Token-by-token bounded-gate KDA recurrence in FP64 (BF16 operand rounding as the kernels see it)."""
    q, k, v, g, beta = (inp[n] for n in ("q", "k", "v", "g", "beta"))
    qn = F.normalize(q.float(), dim=-1).bfloat16().double()
    kn = F.normalize(k.float(), dim=-1).bfloat16().double()
    rate = inp["A_log"].double().exp().reshape(1, 1, -1, 1)
    biased = g.double() + inp["dt_bias"].double().reshape(1, 1, *inp["dt_bias"].shape)
    decay = (LOWER_BOUND * torch.sigmoid(rate * biased)).exp()
    active = beta.float().sigmoid().double()
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    final = []
    offsets = inp["offsets"]
    for seq, (start, end) in enumerate(zip(offsets, offsets[1:], strict=False)):
        state = inp["state"][seq].double()
        for token in range(start, end):
            state = state * decay[0, token, :, None, :]
            residual = (
                v[0, token].double() - (state * kn[0, token, :, None, :]).sum(-1)
            ) * active[0, token, :, None]
            state = state + residual[:, :, None] * kn[0, token, :, None, :]
            out[0, token] = (state * qn[0, token, :, None, :]).sum(-1) * HEAD_DIM**-0.5
        final.append(state)
    return out, torch.stack(final)


def _check(name, actual, reference, tol=0.01):
    assert torch.isfinite(actual).all(), f"{name}: non-finite values"
    rel = (actual.double() - reference).norm() / reference.norm()
    assert rel < tol, f"{name}: rel L2 {rel:.4g} >= {tol}"
    return float(rel)


CASES = [
    pytest.param((64,), 12, 4407, id="bs1_t64"),
    pytest.param((128,), 12, 4408, id="bs1_t128"),
    pytest.param((64,) * 64, 12, 4409, id="bs64_t64"),
    pytest.param((128,) * 64, 12, 4410, id="bs64_t128"),
]


@pytest.mark.parametrize(("lengths", "heads", "seed"), CASES)
@pytest.mark.parametrize("checkpoints", [False, True], ids=["nocp", "cp64"])
def test_facade_bf16_state_matches_fp64_recurrence(lengths, heads, seed, checkpoints):
    inp = trained_gate_inputs(lengths=lengths, heads=heads, seed=seed)
    expected_out, expected_final = fp64_reference(inp)
    n = len(lengths)
    pool = torch.zeros(
        (n + 2, heads, HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    indices = torch.arange(n, device="cuda", dtype=torch.int32) + 1
    pool[indices.long()] = inp["state"].bfloat16()
    cu = torch.tensor(inp["offsets"], device="cuda", dtype=torch.int64)
    kwargs = {}
    if checkpoints:
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + (length + 63) // 64)
        kwargs = dict(
            state_checkpoints=torch.empty(
                (starts[-1], heads, HEAD_DIM, HEAD_DIM),
                device="cuda",
                dtype=torch.bfloat16,
            ),
            checkpoint_cu_starts=torch.tensor(starts, device="cuda", dtype=torch.int64),
            checkpoint_every_n_tokens=64,
        )
    result = recurrent_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        scale=None,
        initial_state=pool,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        beta_is_logit=True,
        backend="cake",
        **kwargs,
    )
    out = result[0]
    torch.cuda.synchronize()
    # the recurrence reference is exact; BF16 inputs/state allow 1e-2
    _check("output", out, expected_out)
    _check("final_state", pool[indices.long()], expected_final)
    if checkpoints:
        # the first checkpoint of every sequence is its (BF16-rounded) initial state
        first = kwargs["state_checkpoints"][kwargs["checkpoint_cu_starts"][:-1].long()]
        _check("checkpoint0", first, inp["state"].bfloat16().double(), tol=1e-6)


def _run_prepared_bf16_export(lengths, heads, seed, checkpoints):
    """Launch the prepared export on an FP32 pool; return the FP64 verdict inputs."""
    inp = trained_gate_inputs(lengths=lengths, heads=heads, seed=seed)
    expected_out, expected_final = fp64_reference(inp)
    n = len(lengths)
    pool = torch.zeros(
        (n + 2, heads, HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.float32
    )
    indices = torch.arange(n, device="cuda", dtype=torch.int32) + 1
    pool[indices.long()] = inp["state"]
    cu = torch.tensor(inp["offsets"], device="cuda", dtype=torch.int64)
    out = torch.empty_like(inp["q"])
    kwargs = {}
    if checkpoints:
        starts = [0]
        for length in lengths:
            starts.append(starts[-1] + (length + 63) // 64)
        # BF16 checkpoint observations (the serving contract for the bounded
        # gate); FP32 exact intermediate states are a separate specialization.
        kwargs = dict(
            state_checkpoints=torch.empty(
                (starts[-1], heads, HEAD_DIM, HEAD_DIM),
                device="cuda",
                dtype=torch.bfloat16,
            ),
            checkpoint_cu_starts=torch.tensor(starts, device="cuda", dtype=torch.int64),
            checkpoint_every_n_tokens=64,
        )
    call = prepare_bf16_kda_prefill(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["g"],
        inp["beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        out=out,
        initial_state=pool,
        final_state=pool,
        scale=None,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu,
        sequence_lengths=list(lengths),
        state_indices=indices,
        beta_is_logit=True,
        **kwargs,
    )
    try:
        call.launch()
        torch.cuda.synchronize()
    finally:
        close = getattr(call, "close", None)
        if close is not None:
            close()
    _check("output", out, expected_out)
    _check("final_state", pool[indices.long()], expected_final)
    if checkpoints:
        first = kwargs["state_checkpoints"][kwargs["checkpoint_cu_starts"][:-1].long()]
        _check("checkpoint0", first, inp["state"].bfloat16().double(), tol=1e-6)
    return call


@pytest.mark.parametrize(("lengths", "heads", "seed"), CASES)
@pytest.mark.parametrize("checkpoints", [False, True], ids=["nocp", "cp64"])
def test_prepared_bf16_export_matches_fp64_recurrence(
    lengths, heads, seed, checkpoints
):
    _run_prepared_bf16_export(lengths, heads, seed, checkpoints)


# Long bounded sequences with checkpoint rows on the FP32 pool: serving's
# radix-cache shape.  The one-wave M64 value split's FP32-pool body re-derived
# the chunk state from its BF16 projection copy and drifted under these
# statistics (worst-head final-state rel L2 0.05 at 2241 tokens, 0.19 at 8192
# against FP32 Triton on B200 and GB300; SGLang #34299 follow-up, CAKE-736
# round 7).  The round-9 body accumulates the BF16 decay correction onto the
# FP32 state instead of re-deriving it (delta decay; FP32 chunk
# carrier); these rows fail on the round-7 split body and pass on the fixed one.
LONG_CASES = [
    pytest.param((2048,), 12, 4411, id="bs1_t2048"),
    pytest.param((2241,), 12, 4412, id="bs1_t2241"),
    pytest.param((1024,) * 4, 12, 4413, id="bs4_t1024"),
]


@pytest.mark.parametrize(("lengths", "heads", "seed"), LONG_CASES)
def test_prepared_bf16_export_long_bounded_rows_keep_fp32_state_carrier(
    lengths, heads, seed
):
    call = _run_prepared_bf16_export(lengths, heads, seed, checkpoints=True)
    assert str(call.schedule) == "fused_m64_independent_dvsplit_fp32_state", str(
        call.schedule
    )


# The FP32-indexed generated portfolio (``flashkda_generated_indexed_*``) is a
# separate set of fused-M128 / persistent-M128 bodies behind
# ``recurrent_kda(backend="cake")`` with a 257-slot FP32 state pool.  It had the
# same stale inverse composition as the facade bodies above.  Its dispatcher
# replays exact frozen rows only (sequence lengths, heads, SM count and the
# state-slot content), so each case names its frozen state slots as an
# arithmetic progression mod 257.  On SM100 the rows reach all 12 sm100a
# bodies that compose the block inverse (``direct_m128``, ``direct_m128_n16``,
# the persistent routes and every M128 stage of ``affine_split_m128``);
# ``h6_t64_control`` runs a body without the composition.  Known issues that
# are not the stale composition, tracked as xfail:
# - the BT16 prepare/chain route (``bt16_prepare_chain_m64``, both targets) is
#   non-finite under these statistics;
# - on SM103 the H96 ``direct_m128`` body with the tensor-core state decay
#   (``bf16_fused_m128_610103ee26``) stays wrong after the repack, and the
#   ``source599_vtile_m128`` bodies drift past 1e-2 on long rows.
_SM103_STATE_DECAY_BODY = (
    "SM103 H96 direct_m128 state-decay body is wrong under trained-gate "
    "statistics after the repack (separate fix)"
)
_SM103_VTILE_DRIFT = (
    "SM103 source599_vtile_m128 body drifts past 1e-2 on long rows under "
    "trained-gate statistics (separate fix)"
)
INDEXED_CASES = [
    # (lengths, heads, packed, first state slot, slot stride, SM103 xfail)
    pytest.param((64,), 12, False, 91, 0, None, id="h12_t64"),
    pytest.param((63,), 12, False, 74, 0, None, id="h12_t63_n16"),
    pytest.param((512,) * 32, 12, True, 6, 177, None, id="h12_32x512"),
    pytest.param((1024,) * 8, 6, True, 76, 155, None, id="h6_8x1024"),
    pytest.param((512,), 96, False, 142, 0, _SM103_STATE_DECAY_BODY, id="h96_t512"),
    pytest.param((128,) * 8, 96, True, 193, 199, None, id="h96_8x128"),
    pytest.param((1024,) * 8, 96, True, 227, 203, _SM103_VTILE_DRIFT, id="h96_8x1024"),
    pytest.param((8192,), 6, False, 8, 0, None, id="h6_t8192_affine"),
    pytest.param((16384,), 12, False, 229, 0, None, id="h12_t16384_affine"),
    pytest.param((64,), 6, False, 144, 0, None, id="h6_t64_control"),
    pytest.param(
        (512,),
        12,
        False,
        195,
        0,
        None,
        id="h12_t512_bt16",
        marks=pytest.mark.xfail(
            strict=True,
            reason="BT16 prepare/chain route is non-finite under trained-gate "
            "statistics (separate fix)",
        ),
    ),
]


@pytest.mark.parametrize(
    ("lengths", "heads", "packed", "first_slot", "slot_stride", "sm103_issue"),
    INDEXED_CASES,
)
def test_indexed_fp32_state_matches_fp64_recurrence(
    request, lengths, heads, packed, first_slot, slot_stride, sm103_issue
):
    from flashinfer.jit import flash_kda_indexed

    if sm103_issue is not None and torch.cuda.get_device_capability() == (10, 3):
        request.applymarker(pytest.mark.xfail(reason=sm103_issue, strict=False))
    if torch.cuda.get_device_properties(0).multi_processor_count != 148:
        pytest.skip("the frozen indexed rows were exported for 148-SM parts")
    inp = trained_gate_inputs(lengths=lengths, heads=heads, seed=4420 + heads)
    expected_out, expected_final = fp64_reference(inp)
    n = len(lengths)
    capacity = flash_kda_indexed._EXPECTED_STATE_POOL_CAPACITY
    pool = torch.zeros(
        (capacity, heads, HEAD_DIM, HEAD_DIM), device="cuda", dtype=torch.float32
    )
    indices = torch.tensor(
        [(first_slot + slot_stride * i) % capacity for i in range(n)],
        device="cuda",
        dtype=torch.int32,
    )
    pool[indices.long()] = inp["state"]
    before = pool.clone()
    cu = (
        torch.tensor(inp["offsets"], device="cuda", dtype=torch.int64)
        if packed
        else None
    )
    common = dict(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        initial_state=pool,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        beta_is_logit=True,
    )
    assert flash_kda_indexed.flash_kda_indexed_prefill_is_eligible(
        **common,
        num_spec_tokens=None,
        num_accepted_tokens=None,
        output=None,
        initial_state_source=None,
        initial_state_indices=None,
        seq_order=None,
        prefill_workspace=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
    )
    out, final_state = recurrent_kda(
        **common, scale=None, output_final_state=True, backend="cake"
    )
    torch.cuda.synchronize()
    assert final_state is pool
    _check("output", out, expected_out)
    _check("final_state", pool[indices.long()], expected_final)
    unselected = torch.ones(capacity, dtype=torch.bool, device="cuda")
    unselected[indices.long()] = False
    assert torch.equal(pool[unselected], before[unselected])
