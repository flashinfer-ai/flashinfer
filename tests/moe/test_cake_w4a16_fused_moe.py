# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cake W4A16 (BF16 activations x NVFP4 experts) route of ``fused_moe``.

Every GPU test compares the Cake route on one fixture against (a) FlashInfer's
own CuTe DSL W4A16 "Split" kernels (bit statistics: bit-exact fraction, BF16
ulp histogram, SQNR) and (b) an FP32 torch reference with no intermediate
rounding (the acceptance band of FlashInfer's W4A16 oracle tests, ``atol =
2e-3 * max|y_ref| + 2 ulp_bf16(max|y_ref|)``, ``rtol = 0.05``,
and the SQNR gate ``SQNR_cake >= SQNR_fi - 0.2 dB``).  Fixtures follow the
shape rows of the two model presets: NumPy PCG64 routing (seed ``2026102300 +
row``, token-major ``choice(E, 8, replace=False)``, scores uniform ``[0.25,
1.25)`` normalised per token), PCG64 BF16 inputs (seed ``2026102400 + row``),
and directly sampled E2M1 codes with E4M3 scales in ``0x28..0x3F`` (seed
``2026102501``; one stream in the order w13 codes, w13 scales, w2 codes, w2
scales).  The host-only tests at the end run anywhere.

GPU tests skip when there is no CUDA device, when the device is not an exact
sm_100a / sm_103a part, or when the generated loader
``flashinfer.jit.cake_fused_moe_w4a16`` does not import (the ``cake_programs``
fixture).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pytest
import torch

from flashinfer.fused_moe import cake_w4a16 as cake
from flashinfer.tllm_enums import ActivationType

ROUTING_SEED0 = 2026102300
INPUT_SEED0 = 2026102400
WEIGHT_SEED = 2026102501
ATOL_FRAC = 2e-3
RTOL = 0.05
SQNR_MARGIN_DB = 0.2


@dataclass(frozen=True)
class Preset:
    name: str
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    num_local_experts: int
    local_expert_offset: int


# G: a 6144-hidden layer as rank 0 of an 8-way expert-parallel split (tokens of
# every rank are visible; routes outside experts [0, 32) are masked by moe_sort).
# D: a 7168-hidden layer on one device (all 256 experts local).
PRESETS = {
    "G": Preset("G", 6144, 2048, 256, 8, 32, 0),
    "D": Preset("D", 7168, 2048, 256, 8, 256, 0),
}
ROWS = {"G": (8, 32, 64, 128, 512, 4096), "D": (1, 8, 32, 128, 512, 4096)}
GOAL_ROWS = [("G", t) for t in ROWS["G"]] + [("D", t) for t in ROWS["D"]]


def _row_index(preset: str, tokens: int) -> int:
    return ROWS[preset].index(tokens) + (0 if preset == "G" else len(ROWS["G"]))


def _device_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in (
        (10, 0),
        (10, 3),
    )


gpu_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)
blackwell_required = pytest.mark.skipif(
    torch.cuda.is_available() and not _device_ok(),
    reason="the Cake W4A16 programs ship for sm_100a and sm_103a only",
)
# The loader check runs at test time (fixture ``cake_programs``), not at import.
gpu_route = [
    gpu_required,
    blackwell_required,
    pytest.mark.usefixtures("cake_programs"),
]


# --------------------------------------------------------------------------- fixtures


@dataclass
class Weights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


def make_weights(p: Preset, device: torch.device, seed: int = WEIGHT_SEED) -> Weights:
    """Canonical ``PrequantizedMoEWeights`` tensors of the rank's expert slice,
    drawn on the CPU from one PCG64 stream in the order w13 codes, w13 scales,
    w2 codes, w2 scales (the whole model's experts are drawn; the slice is cut
    afterwards so every rank sees the same global weights)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out = []
    lo, hi = p.local_expert_offset, p.local_expert_offset + p.num_local_experts
    for rows, cols in ((2 * p.intermediate, p.hidden), (p.hidden, p.intermediate)):
        packed = rng.integers(
            0, 256, size=(p.num_experts, rows, cols // 2), dtype=np.uint8
        )
        scales = rng.integers(
            0x28, 0x40, size=(p.num_experts, rows, cols // 16), dtype=np.uint8
        )
        out.append(
            (
                torch.from_numpy(packed[lo:hi]).to(device).contiguous(),
                torch.from_numpy(scales[lo:hi])
                .to(device)
                .contiguous()
                .view(torch.float8_e4m3fn),
            )
        )
    return Weights(out[0][0], out[0][1], out[1][0], out[1][1])


def make_routing(
    p: Preset,
    tokens: int,
    row: int,
    variant: str = "uniform",
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Global expert ids (int32 ``[T, K]``, ``-1`` = masked) and FP32 scores.

    ``uniform``: eight distinct experts per token, scores normalised per token.
    ``hot``: Zipf(1.2) expert popularity over a random permutation.
    ``one_expert``: slot 0 of every token is expert 7 (capacity path).
    ``pad``: uniform, then 0..K slots of each token are ``-1`` at random slot
    positions; masked slots carry a NaN score (the route must never read them)
    and some tokens are fully masked (their output is exactly zero).
    """
    rng = np.random.Generator(np.random.PCG64(ROUTING_SEED0 + row))
    idx = np.empty((tokens, p.top_k), dtype=np.int64)
    if variant == "hot":
        ranks = rng.permutation(p.num_experts)
        pop = 1.0 / (np.arange(p.num_experts) + 1.0) ** 1.2
        prob = np.empty(p.num_experts)
        prob[ranks] = pop / pop.sum()
    for t in range(tokens):
        if variant == "hot":
            idx[t] = rng.choice(p.num_experts, size=p.top_k, replace=False, p=prob)
        elif variant == "one_expert":
            others = rng.choice(p.num_experts - 1, size=p.top_k - 1, replace=False)
            idx[t, 0] = 7
            idx[t, 1:] = np.where(others >= 7, others + 1, others)
        else:
            idx[t] = rng.choice(p.num_experts, size=p.top_k, replace=False)
    scores = rng.uniform(0.25, 1.25, size=(tokens, p.top_k)).astype(np.float32)
    scores /= scores.sum(axis=1, keepdims=True)
    if variant == "pad":
        n_mask = rng.integers(0, p.top_k + 1, size=tokens)
        for t in range(tokens):
            if n_mask[t]:
                pos = rng.choice(p.top_k, size=n_mask[t], replace=False)
                idx[t, pos] = -1
                scores[t, pos] = np.nan
    device = torch.device("cpu") if device is None else device
    ids = torch.from_numpy(idx.astype(np.int32)).to(device).contiguous()
    return ids, torch.from_numpy(scores).to(device).contiguous()


def make_inputs(p: Preset, tokens: int, row: int, device: torch.device) -> torch.Tensor:
    rng = np.random.Generator(np.random.PCG64(INPUT_SEED0 + row))
    x = rng.standard_normal((tokens, p.hidden), dtype=np.float32)
    return torch.from_numpy(x).to(torch.bfloat16).to(device).contiguous()


# --------------------------------------------------------------------------- oracle

_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _dequant(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """E2M1 x E4M3 of one expert as FP32 (exact; the product is representable in BF16)."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = (
        torch.stack((packed & 0xF, packed >> 4), dim=-1)
        .reshape(packed.shape[0], -1)
        .long()
    )
    values = lut[codes].reshape(*scales.shape, 16) * scales.float().unsqueeze(-1)
    return values.reshape(packed.shape[0], -1)


class _ieee_fp32:
    """True FP32 GEMMs inside the reference (no TF32, no reduced-precision reductions)."""

    def __enter__(self):
        m = torch.backends.cuda.matmul
        self.saved = (
            m.allow_tf32,
            m.allow_bf16_reduced_precision_reduction,
            torch.backends.cudnn.allow_tf32,
        )
        m.allow_tf32 = False
        m.allow_bf16_reduced_precision_reduction = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(m, "fp32_precision"):
            self.saved_precision = m.fp32_precision
            m.fp32_precision = "ieee"
        return self

    def __exit__(self, *exc):
        m = torch.backends.cuda.matmul
        (
            m.allow_tf32,
            m.allow_bf16_reduced_precision_reduction,
            torch.backends.cudnn.allow_tf32,
        ) = self.saved
        if hasattr(self, "saved_precision"):
            m.fp32_precision = self.saved_precision


@torch.no_grad()
def reference_fp32(
    x: torch.Tensor,
    ids: torch.Tensor,
    scores: torch.Tensor,
    w: Weights,
    p: Preset,
    *,
    fc1_alpha: Optional[torch.Tensor] = None,
    fc2_alpha: Optional[torch.Tensor] = None,
    limit: Optional[float] = None,
) -> torch.Tensor:
    """FP32 reference with no intermediate rounding: exact dequant, FP32 FC1,
    alpha, optional clamp, ``up * (gate * sigmoid(gate))``, FP32 FC2, alpha,
    routing weight after FC2, FP64 sum over the live slots.  Returns FP32
    ``[T, H]``.  Masked (``-1``) slots never contribute, whatever their score."""
    tokens, hidden = x.shape
    y = torch.zeros((tokens, hidden), dtype=torch.float64, device=x.device)
    xf = x.float()
    with _ieee_fp32():
        for local in range(p.num_local_experts):
            expert = p.local_expert_offset + local
            sel = (ids == expert).nonzero(as_tuple=False)
            if sel.numel() == 0:
                continue
            rows, slots = sel[:, 0], sel[:, 1]
            w13 = _dequant(w.w13[local], w.w13_scale[local])
            w2 = _dequant(w.w2[local], w.w2_scale[local])
            fc1 = xf[rows] @ w13.t()
            if fc1_alpha is not None:
                fc1 = fc1 * fc1_alpha[local].float()
            gate, up = fc1[:, : p.intermediate], fc1[:, p.intermediate :]
            if limit is not None:
                gate = torch.clamp(gate, max=limit)
                up = torch.clamp(up, min=-limit, max=limit)
            h = up * (gate * torch.sigmoid(gate))
            fc2 = h @ w2.t()
            if fc2_alpha is not None:
                fc2 = fc2 * fc2_alpha[local].float()
            y.index_add_(
                0, rows, (fc2 * scores[rows, slots].float().unsqueeze(1)).double()
            )
    return y.float()


def bf16_ulp(x: float) -> float:
    """BF16 spacing at magnitude ``x`` (8 significand bits, the hidden one included)."""
    if x <= 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(x)) - 7)


def oracle_atol(amax_ref: float) -> float:
    """FlashInfer's W4A16 oracle tolerance: ``2e-3 * amax`` covers the NVFP4 decode
    and the accumulation-order noise, the extra ``2 * bf16_ulp(amax)`` one BF16
    rounding flip of a per-(token, slot) FC2 term before the top-k sum, which the
    relative part of the compare cannot absorb where the terms cancel."""
    return ATOL_FRAC * amax_ref + 2.0 * bf16_ulp(amax_ref)


def band(y: torch.Tensor, y_ref: torch.Tensor) -> Dict[str, float]:
    ref = y_ref.float()
    amax = float(ref.abs().max()) if ref.numel() else 0.0
    atol = oracle_atol(amax)
    diff = (y.float() - ref).abs()
    ok = diff <= atol + RTOL * ref.abs()
    rel_l2 = float(
        diff.double().pow(2).sum().sqrt()
        / ref.double().pow(2).sum().sqrt().clamp_min(1e-300)
    )
    return {
        "atol": atol,
        "max_abs_err": float(diff.max()) if ref.numel() else 0.0,
        "mismatch": int((~ok).sum()),
        "rel_l2": rel_l2,
    }


def sqnr_db(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    ref = y_ref.double()
    noise = y.double() - ref
    signal, error = float(ref.pow(2).sum()), float(noise.pow(2).sum())
    return 10.0 * math.log10(signal / max(error, 1e-300)) if signal > 0 else math.nan


def ulp_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, object]:
    """Bit agreement of two BF16 tensors: bit-exact fraction and |ulp| histogram."""

    def mono(t: torch.Tensor) -> torch.Tensor:
        v = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(v < 0, -(v & 0x7FFF), v)

    d = (mono(a) - mono(b)).abs()
    n = max(d.numel(), 1)
    hist = {
        "0": int((d == 0).sum()),
        "1": int((d == 1).sum()),
        "2": int((d == 2).sum()),
        "3-8": int(((d >= 3) & (d <= 8)).sum()),
        ">8": int((d > 8).sum()),
    }
    return {
        "bit_exact_frac": hist["0"] / n,
        "ulp_hist": hist,
        "max_ulp": int(d.max()) if d.numel() else 0,
    }


# --------------------------------------------------------------------------- arms


def fi_split_weights(w: Weights, p: Preset):
    """FlashInfer's Split layout from the canonical tensors by byte permutation
    only (the halves become [up, gate], 64-row interleave, swizzled block
    scales); neither path re-quantizes."""
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.prepare import _interleave_linear_and_gate

    i_size = p.intermediate

    def up_gate(t: torch.Tensor) -> torch.Tensor:
        return torch.cat((t[:, i_size:], t[:, :i_size]), dim=1)

    def split_scale(sf_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        return convert_sf_to_mma_layout(
            block_scale_interleave(sf_u8.flatten(0, 1).contiguous()),
            m=rows,
            k=cols,
            num_groups=p.num_local_experts,
            sf_vec_size=16,
        )

    w1 = _interleave_linear_and_gate(up_gate(w.w13), group_size=64, dim=1)
    w1_sf = _interleave_linear_and_gate(
        up_gate(w.w13_scale.view(torch.uint8)), group_size=64, dim=1
    )
    return {
        "w1_weight": w1.contiguous(),
        "w1_weight_sf": split_scale(w1_sf, 2 * i_size, p.hidden),
        "w2_weight": w.w2,
        "w2_weight_sf": split_scale(w.w2_scale.view(torch.uint8), p.hidden, i_size),
    }


def run_fi(
    x: torch.Tensor,
    ids: torch.Tensor,
    scores: torch.Tensor,
    split: Dict[str, torch.Tensor],
    p: Preset,
    *,
    fc1_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    limit: Optional[float],
) -> torch.Tensor:
    """FlashInfer's W4A16 chain (moe_sort, moe_permute, two CuTe DSL grouped
    GEMMs, deterministic ``moe_unpermute``) with its default tactic."""
    from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import (
        DEFAULT_W4A16_MOE_TACTIC,
        launch_w4a16_moe,
    )
    from flashinfer.tllm_enums import DEFAULT_SWIGLU_LIMIT

    out = torch.empty((x.shape[0], p.hidden), dtype=torch.bfloat16, device=x.device)
    return launch_w4a16_moe(
        x=x,
        token_selected_experts=ids,
        token_final_scales=scores,
        w1_weight=split["w1_weight"],
        w1_weight_sf=split["w1_weight_sf"],
        w1_alpha=fc1_alpha,
        w2_weight=split["w2_weight"],
        w2_weight_sf=split["w2_weight_sf"],
        w2_alpha=fc2_alpha,
        num_experts=p.num_experts,
        num_local_experts=p.num_local_experts,
        local_expert_offset=p.local_expert_offset,
        moe_output=out,
        use_fused_finalize=False,
        enable_pdl=True,
        activation_type=ActivationType.Swiglu,
        swiglu_limit=DEFAULT_SWIGLU_LIMIT if limit is None else float(limit),
        tactic=DEFAULT_W4A16_MOE_TACTIC,
    )


def workspace_for(
    p: Preset, max_tokens: int, device: torch.device, fill: Optional[int] = None
) -> torch.Tensor:
    """A prepared workspace; ``fill`` sets every byte first (the prepare step
    then rewrites only the default-alpha regions)."""
    nbytes = cake.cake_w4a16_workspace_size(
        max_tokens, p.hidden, p.intermediate, p.num_local_experts, p.top_k
    )
    ws = torch.empty((nbytes,), dtype=torch.uint8, device=device)
    if fill is not None:
        ws.fill_(fill)
    cake.cake_w4a16_prepare_workspace(
        ws,
        max_num_tokens=max_tokens,
        hidden_size=p.hidden,
        intermediate_size=p.intermediate,
        num_local_experts=p.num_local_experts,
        top_k=p.top_k,
    )
    return ws


def run_cake(
    x: torch.Tensor,
    ids: torch.Tensor,
    scores: torch.Tensor,
    w: Weights,
    p: Preset,
    workspace: torch.Tensor,
    *,
    fc1_alpha: Optional[torch.Tensor] = None,
    fc2_alpha: Optional[torch.Tensor] = None,
    limit: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    cell: Optional[cake.Cell] = None,
) -> torch.Tensor:
    return cake.cake_w4a16_fused_moe(
        x,
        ids,
        scores,
        w.w13,
        w.w13_scale,
        w.w2,
        w.w2_scale,
        num_experts=p.num_experts,
        local_expert_offset=p.local_expert_offset,
        output=output,
        workspace_buffer=workspace,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        swiglu_limit=limit,
        cell=cell,
    )


def compare(
    tag: str,
    y_cake: torch.Tensor,
    y_fi: torch.Tensor,
    y_ref: torch.Tensor,
) -> Dict[str, object]:
    """The acceptance band of both arms, the SQNR gate and the bit statistics;
    printed as one JSON line for the evidence record."""
    assert y_cake.dtype == torch.bfloat16 and y_cake.shape == y_ref.shape
    assert torch.isfinite(y_cake).all(), "the Cake route produced a non-finite value"
    assert torch.isfinite(y_fi).all(), "the FlashInfer arm produced a non-finite value"
    cake_band = band(y_cake, y_ref)
    fi_band = band(y_fi, y_ref)
    sqnr_cake, sqnr_fi = sqnr_db(y_ref, y_cake), sqnr_db(y_ref, y_fi)
    record = {
        "case": tag,
        "cake": {**cake_band, "sqnr_db": sqnr_cake},
        "fi": {**fi_band, "sqnr_db": sqnr_fi},
        "cake_vs_fi": ulp_stats(y_cake, y_fi),
    }
    print("CAKE_W4A16_STATS " + json.dumps(record))
    torch.testing.assert_close(y_cake.float(), y_ref, atol=cake_band["atol"], rtol=RTOL)
    assert sqnr_cake >= sqnr_fi - SQNR_MARGIN_DB, (
        f"{tag}: SQNR {sqnr_cake:.3f} dB is below FlashInfer's {sqnr_fi:.3f} dB - {SQNR_MARGIN_DB}"
    )
    return record


# --------------------------------------------------------------------------- fixtures (pytest)


@pytest.fixture(scope="module")
def cake_programs() -> None:
    """Skip the GPU tests unless the generated loader imports and publishes every
    program when they run, and the CUDA toolkit can run them."""
    if not cake.cake_w4a16_available():
        pytest.skip(
            f"the Cake W4A16 programs are not present ({cake.LOADER_MODULE} does not "
            "import or publishes no program)"
        )
    unavailable = cake.cake_w4a16_toolkit_unavailable()
    if unavailable is not None:
        pytest.skip(unavailable)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda", 0)


@pytest.fixture(scope="module")
def weights_by_preset(device):
    cache: Dict[str, Tuple[Weights, Dict[str, torch.Tensor]]] = {}

    def get(name: str) -> Tuple[Weights, Dict[str, torch.Tensor]]:
        if name not in cache:
            p = PRESETS[name]
            w = make_weights(p, device)
            cache[name] = (w, fi_split_weights(w, p))
        return cache[name]

    yield get
    cache.clear()
    torch.cuda.empty_cache()


def _ones(p: Preset, device: torch.device) -> torch.Tensor:
    return torch.ones((p.num_local_experts,), dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- GPU tests


@pytest.mark.parametrize(
    "preset,tokens", GOAL_ROWS, ids=[f"{n}-T{t}" for n, t in GOAL_ROWS]
)
class TestGoalRows:
    pytestmark = gpu_route

    def test_row(self, preset, tokens, device, weights_by_preset):
        p = PRESETS[preset]
        w, split = weights_by_preset(preset)
        row = _row_index(preset, tokens)
        x = make_inputs(p, tokens, row, device)
        ids, scores = make_routing(p, tokens, row, "uniform", device)
        ones = _ones(p, device)
        workspace = workspace_for(p, tokens, device)
        y_cake = run_cake(x, ids, scores, w, p, workspace)
        y_fi = run_fi(
            x, ids, scores, split, p, fc1_alpha=ones, fc2_alpha=ones, limit=None
        )
        y_ref = reference_fp32(x, ids, scores, w, p)
        compare(f"{preset}-T{tokens}-uniform", y_cake, y_fi, y_ref)
        # A second eager call on the same inputs is bit-identical (no atomics, no order dependence).
        again = run_cake(x, ids, scores, w, p, workspace)
        assert torch.equal(again.view(torch.int16), y_cake.view(torch.int16))


@pytest.mark.parametrize("variant", ["pad", "hot", "one_expert"])
class TestBoundaryRouting:
    pytestmark = gpu_route

    def test_variant(self, variant, device, weights_by_preset):
        p = PRESETS["G"]
        tokens = 128
        w, split = weights_by_preset("G")
        row = _row_index("G", tokens)
        x = make_inputs(p, tokens, row, device)
        ids, scores = make_routing(p, tokens, row, variant, device)
        ones = _ones(p, device)
        workspace = workspace_for(p, tokens, device)
        y_cake = run_cake(x, ids, scores, w, p, workspace)
        y_fi = run_fi(
            x, ids, scores, split, p, fc1_alpha=ones, fc2_alpha=ones, limit=None
        )
        y_ref = reference_fp32(x, ids, scores, w, p)
        compare(f"G-T{tokens}-{variant}", y_cake, y_fi, y_ref)
        if variant == "pad":
            # Fully masked tokens (and tokens without a local route) write exactly zero.
            local = (ids >= p.local_expert_offset) & (
                ids < p.local_expert_offset + p.num_local_experts
            )
            dead = ~local.any(dim=1)
            assert dead.any(), "the pad fixture is expected to fully mask some tokens"
            assert torch.equal(
                y_cake[dead].view(torch.int16),
                torch.zeros_like(y_cake[dead]).view(torch.int16),
            )


class TestNumericsKnobs:
    pytestmark = gpu_route

    def test_alpha_and_clamp(self, device, weights_by_preset):
        p = PRESETS["G"]
        tokens = 128
        w, split = weights_by_preset("G")
        row = _row_index("G", tokens)
        x = make_inputs(p, tokens, row, device)
        ids, scores = make_routing(p, tokens, row, "uniform", device)
        rng = np.random.Generator(np.random.PCG64(WEIGHT_SEED + 1))
        fc1_alpha = torch.from_numpy(
            rng.uniform(0.5, 1.5, p.num_local_experts).astype(np.float32)
        ).to(device)
        fc2_alpha = torch.from_numpy(
            rng.uniform(0.5, 1.5, p.num_local_experts).astype(np.float32)
        ).to(device)
        limit = 7.0
        workspace = workspace_for(p, tokens, device)
        y_cake = run_cake(
            x,
            ids,
            scores,
            w,
            p,
            workspace,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            limit=limit,
        )
        y_fi = run_fi(
            x,
            ids,
            scores,
            split,
            p,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            limit=limit,
        )
        y_ref = reference_fp32(
            x, ids, scores, w, p, fc1_alpha=fc1_alpha, fc2_alpha=fc2_alpha, limit=limit
        )
        compare(f"G-T{tokens}-alpha-clamp{limit}", y_cake, y_fi, y_ref)
        # The clamp changes the result on this fixture (otherwise the case proves nothing).
        y_plain = run_cake(
            x, ids, scores, w, p, workspace, fc1_alpha=fc1_alpha, fc2_alpha=fc2_alpha
        )
        assert not torch.equal(y_plain.view(torch.int16), y_cake.view(torch.int16))

    def test_clamp_as_cpu_tensor_and_device_tensor_rejected(
        self, device, weights_by_preset
    ):
        p = PRESETS["G"]
        tokens = 8
        w, _ = weights_by_preset("G")
        row = _row_index("G", tokens)
        x = make_inputs(p, tokens, row, device)
        ids, scores = make_routing(p, tokens, row, "uniform", device)
        workspace = workspace_for(p, tokens, device)
        y_float = run_cake(x, ids, scores, w, p, workspace, limit=7.0)
        y_tensor = run_cake(x, ids, scores, w, p, workspace, limit=torch.tensor([7.0]))
        assert torch.equal(y_float.view(torch.int16), y_tensor.view(torch.int16))
        with pytest.raises(ValueError, match="CPU tensor"):
            run_cake(
                x,
                ids,
                scores,
                w,
                p,
                workspace,
                limit=torch.tensor([7.0], device=device),
            )


class TestWorkspaceAndEdges:
    pytestmark = gpu_route

    def test_zero_tokens(self, device, weights_by_preset):
        p = PRESETS["G"]
        w, _ = weights_by_preset("G")
        workspace = workspace_for(p, 8, device)
        x = torch.empty((0, p.hidden), dtype=torch.bfloat16, device=device)
        ids = torch.empty((0, p.top_k), dtype=torch.int32, device=device)
        scores = torch.empty((0, p.top_k), dtype=torch.float32, device=device)
        y = run_cake(x, ids, scores, w, p, workspace)
        assert y.shape == (0, p.hidden) and y.dtype == torch.bfloat16
        out = torch.empty((0, p.hidden), dtype=torch.bfloat16, device=device)
        assert run_cake(x, ids, scores, w, p, workspace, output=out) is out

    def test_stale_workspace_rows_do_not_leak(self, device, weights_by_preset):
        """Padding rows of the permuted input and of the FC1 handoff hold whatever
        the workspace held; a NaN-filled workspace must give the same bits as a
        zero-filled one."""
        p = PRESETS["G"]
        tokens = 64
        w, _ = weights_by_preset("G")
        row = _row_index("G", tokens)
        x = make_inputs(p, tokens, row, device)
        ids, scores = make_routing(p, tokens, row, "uniform", device)
        y_zero = run_cake(
            x, ids, scores, w, p, workspace_for(p, tokens, device, fill=0)
        ).clone()
        # 0x7F7F is a BF16 NaN and 0x7F7F7F7F a large positive int32: every
        # padding row and every stale table entry starts poisoned.
        y_nan = run_cake(
            x, ids, scores, w, p, workspace_for(p, tokens, device, fill=0x7F)
        )
        assert torch.isfinite(y_nan).all()
        assert torch.equal(y_zero.view(torch.int16), y_nan.view(torch.int16))
        # A smaller call after a larger one on the same workspace sees the larger
        # call's rows as stale padding and must give the small call's own bits.
        shared = workspace_for(p, 512, device)
        big_row = _row_index("G", 512)
        run_cake(
            make_inputs(p, 512, big_row, device),
            *make_routing(p, 512, big_row, "uniform", device),
            w,
            p,
            shared,
        )
        y_after = run_cake(x, ids, scores, w, p, shared)
        assert torch.equal(y_zero.view(torch.int16), y_after.view(torch.int16))

    def test_workspace_too_small_is_rejected(self, device, weights_by_preset):
        p = PRESETS["G"]
        w, _ = weights_by_preset("G")
        row = _row_index("G", 128)
        x = make_inputs(p, 128, row, device)
        ids, scores = make_routing(p, 128, row, "uniform", device)
        small = torch.empty((4096,), dtype=torch.uint8, device=device)
        with pytest.raises(ValueError, match="workspace_buffer holds"):
            run_cake(x, ids, scores, w, p, small)

    def test_every_cell_matches_reference(self, device, weights_by_preset):
        """Every shipped cell (including the 2-CTA pairs) on one fixture: the
        cell changes the tiling, never the numerics beyond the band."""
        p = PRESETS["G"]
        tokens = 512
        w, split = weights_by_preset("G")
        row = _row_index("G", tokens)
        x = make_inputs(p, tokens, row, device)
        ids, scores = make_routing(p, tokens, row, "uniform", device)
        ones = _ones(p, device)
        workspace = workspace_for(p, tokens, device)
        y_fi = run_fi(
            x, ids, scores, split, p, fc1_alpha=ones, fc2_alpha=ones, limit=None
        )
        y_ref = reference_fp32(x, ids, scores, w, p)
        for cell in cake.CELLS:
            y_cake = run_cake(x, ids, scores, w, p, workspace, cell=cell)
            compare(f"G-T{tokens}-cell-{cell.suffix}", y_cake, y_fi, y_ref)


class TestCudaGraph:
    pytestmark = gpu_route

    def test_capture_replay_bit_identity_and_input_change(
        self, device, weights_by_preset
    ):
        p = PRESETS["G"]
        tokens = 128
        w, _ = weights_by_preset("G")
        row = _row_index("G", tokens)
        x = make_inputs(p, tokens, row, device)
        ids, scores = make_routing(p, tokens, row, "uniform", device)
        workspace = workspace_for(p, tokens, device)
        out = torch.empty((tokens, p.hidden), dtype=torch.bfloat16, device=device)
        eager = run_cake(x, ids, scores, w, p, workspace, output=out).clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run_cake(x, ids, scores, w, p, workspace, output=out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_cake(x, ids, scores, w, p, workspace, output=out)
        out.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
        # New inputs through the same captured addresses: replay equals eager on them.
        other_row = _row_index("G", 32)
        x.copy_(make_inputs(p, tokens, other_row, device))
        new_ids, new_scores = make_routing(p, tokens, other_row, "uniform", device)
        ids.copy_(new_ids)
        scores.copy_(new_scores)
        y_ref = reference_fp32(x, ids, scores, w, p)
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        eager_new = run_cake(x, ids, scores, w, p, workspace)
        assert torch.equal(replayed.view(torch.int16), eager_new.view(torch.int16))
        torch.testing.assert_close(
            replayed.float(), y_ref, atol=band(replayed, y_ref)["atol"], rtol=RTOL
        )


class TestPublicRoute:
    pytestmark = gpu_route

    def test_cutlass_fused_moe_backend_cake(self, device, weights_by_preset):
        """``cutlass_fused_moe(backend="cake")`` reaches the W4A16 route and
        agrees bit for bit with the module entry."""
        from flashinfer.fused_moe import (
            cutlass_fused_moe,
            cutlass_fused_moe_workspace_size,
        )

        p = PRESETS["G"]
        tokens = 32
        w, _ = weights_by_preset("G")
        row = _row_index("G", tokens)
        x = make_inputs(p, tokens, row, device)
        ids, scores = make_routing(p, tokens, row, "uniform", device)
        nbytes = cutlass_fused_moe_workspace_size(
            tokens,
            p.hidden,
            p.intermediate,
            p.num_experts,
            p.top_k,
            x_dtype=torch.bfloat16,
            weight_dtype=torch.uint8,
            ep_size=p.num_experts // p.num_local_experts,
            ep_rank=p.local_expert_offset // p.num_local_experts,
            backend="cake",
        )
        assert nbytes == cake.cake_w4a16_workspace_size(
            tokens, p.hidden, p.intermediate, p.num_local_experts, p.top_k
        )
        workspace = torch.empty((nbytes,), dtype=torch.uint8, device=device)
        out = torch.empty((tokens, p.hidden), dtype=torch.bfloat16, device=device)
        result = cutlass_fused_moe(
            x,
            ids,
            scores,
            w.w13,
            w.w2,
            torch.bfloat16,
            [w.w13_scale, w.w2_scale],
            ep_size=p.num_experts // p.num_local_experts,
            ep_rank=p.local_expert_offset // p.num_local_experts,
            output=out,
            backend="cake",
            workspace_buffer=workspace,
        )
        assert result is out
        expected = run_cake(x, ids, scores, w, p, workspace_for(p, tokens, device))
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


# --------------------------------------------------------------------------- host-only tests


class TestHostLogic:
    """Pure host arithmetic of the route; runs without a device or programs."""

    @pytest.mark.parametrize(
        "preset,tokens,expected",
        [
            ("G", 8, cake.Cell(8)),
            ("G", 32, cake.Cell(8)),
            ("G", 64, cake.Cell(8)),
            ("G", 128, cake.Cell(16)),
            ("G", 512, cake.Cell(32)),
            ("G", 4096, cake.Cell(128, 2)),
            ("D", 1, cake.Cell(8)),
            ("D", 8, cake.Cell(8)),
            ("D", 32, cake.Cell(8)),
            ("D", 128, cake.Cell(16)),
            ("D", 512, cake.Cell(32)),
            ("D", 4096, cake.Cell(128, 2)),
        ],
    )
    def test_seed_cell_table(self, preset, tokens, expected):
        p = PRESETS[preset]
        assert cake.select_cell(tokens, p.top_k, p.num_experts) == expected

    def test_available_follows_the_loader(self, monkeypatch):
        """``cake_w4a16_available`` asks the shipped loader: false when it does not
        import or publishes a program short, true when every program is there."""

        def no_loader():
            raise ImportError(cake.LOADER_MODULE)

        monkeypatch.setattr(cake._Programs, "_loader", staticmethod(no_loader))
        assert not cake.cake_w4a16_available()

        class Short:
            FC1_N8 = object()

        monkeypatch.setattr(cake._Programs, "_loader", staticmethod(lambda: Short))
        assert not cake.cake_w4a16_available()

        class Full:
            pass

        for identifier in cake.PROGRAM_IDENTIFIERS:
            setattr(Full, identifier, object())
        monkeypatch.setattr(cake._Programs, "_loader", staticmethod(lambda: Full))
        assert cake.cake_w4a16_available()

    def test_toolkit_gate(self, monkeypatch):
        """``cake_w4a16_toolkit_unavailable`` names the toolkit floor and the
        toolkit found below CUDA 13.0 and is ``None`` from 13.0 on; the loader
        refuses to build a program on an older toolkit before it reads the
        program's launch record."""
        from packaging.version import Version

        from flashinfer.jit import cpp_ext

        assert cake.CAKE_W4A16_MIN_CUDA == "13.0"
        monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: Version("12.9"))
        reason = cake.cake_w4a16_toolkit_unavailable()
        assert reason is not None
        assert "CUDA 13.0" in reason and "12.9" in reason

        class Program:
            targets = ("sm_100a", "sm_103a")

            @staticmethod
            def target_for(compute_capability):
                return "sm_100a"

            @staticmethod
            def launch_record():
                raise AssertionError("the toolkit gate runs before the launch record")

        class Loader:
            FC1_N8 = Program()

        monkeypatch.setattr(cake._Programs, "_loader", staticmethod(lambda: Loader))
        monkeypatch.setattr(cake, "get_compute_capability", lambda device: (10, 0))
        with pytest.raises(RuntimeError, match="FC1_N8: .*CUDA 13.0"):
            cake._Programs().handle("FC1", cake.Cell(8), torch.device("cpu"))

        for version in ("13.0", "13.4"):
            monkeypatch.setattr(
                cpp_ext, "get_cuda_version", lambda v=version: Version(v)
            )
            assert cake.cake_w4a16_toolkit_unavailable() is None

    def test_cells_and_identifiers(self):
        assert len(cake.CELLS) == 5
        assert len(cake.PROGRAM_IDENTIFIERS) == 10
        assert cake.PROGRAM_IDENTIFIERS == (
            "FC1_N8",
            "FC1_N16",
            "FC1_N32",
            "FC1_N64",
            "FC1_N128_CG2",
            "FC2_N8",
            "FC2_N16",
            "FC2_N32",
            "FC2_N64",
            "FC2_N128_CG2",
        )
        for n, cta_group in ((192, 1), (8, 2), (128, 1), (64, 2), (24, 1)):
            with pytest.raises(ValueError):
                cake.Cell(n, cta_group)
        assert cake.Cell(128, 2).suffix == "N128_CG2" and cake.Cell(64).suffix == "N64"
        without_limit = tuple(a for a in cake.FC1_ARGS if a != "swiglu_limit")
        assert without_limit == cake.FC2_ARGS

    def test_workspace_size_monotonic_and_covers_every_cell(self):
        p = PRESETS["G"]
        sizes = [
            cake.cake_w4a16_workspace_size(
                t, p.hidden, p.intermediate, p.num_local_experts, p.top_k
            )
            for t in (1, 8, 128, 512, 4096)
        ]
        assert sizes == sorted(sizes)
        for cell in cake.CELLS:
            _, total = cake._layout(
                512, p.hidden, p.intermediate, p.num_local_experts, p.top_k, cell
            )
            assert total <= sizes[3]
        with pytest.raises(ValueError):
            cake.cake_w4a16_workspace_size(
                8, 6144 + 64, p.intermediate, p.num_local_experts, p.top_k
            )

    def test_form_detection(self):
        """The form check reads host metadata only and returns False for a geometry
        the kernels do not take, so the other backend="cake" route reports it."""
        hidden, inter, experts, top_k = 256, 256, 2, 8
        w13 = torch.empty((experts, 2 * inter, hidden // 2), dtype=torch.uint8)
        w2 = torch.empty((experts, hidden, inter // 2), dtype=torch.uint8)
        scale = torch.empty(
            (experts, 2 * inter, hidden // 16), dtype=torch.float8_e4m3fn
        )
        x = torch.empty((4, hidden), dtype=torch.bfloat16)
        routes = torch.empty((4, top_k), dtype=torch.int32)
        w4a16 = {
            "activation_type": ActivationType.Swiglu,
            "input": x,
            "fc1_expert_weights": w13,
            "fc2_expert_weights": w2,
            "token_selected_experts": routes,
            "quant_scales": [scale, scale],
        }
        assert cake.is_cake_w4a16_form(w4a16)
        assert not cake.is_cake_w4a16_form(
            {**w4a16, "activation_type": ActivationType.Situ}
        )
        assert not cake.is_cake_w4a16_form({**w4a16, "quant_scales": [w13, w13]})
        narrow = torch.empty((4, hidden + 64), dtype=torch.bfloat16)
        assert not cake.is_cake_w4a16_form({**w4a16, "input": narrow})
        wide_k = torch.empty((4, 40), dtype=torch.int32)
        assert not cake.is_cake_w4a16_form({**w4a16, "token_selected_experts": wide_k})
        size_query = {
            "activation_type": ActivationType.Swiglu,
            "x_dtype": torch.bfloat16,
            "weight_dtype": torch.uint8,
            "hidden_size": 3584,
            "intermediate_size": 2048,
            "top_k": 8,
        }
        assert cake.is_cake_w4a16_form(size_query)
        assert not cake.is_cake_w4a16_form({**size_query, "weight_dtype": torch.int64})
        # The other cake route's own geometry (intermediate 384, top-16): not this form.
        assert not cake.is_cake_w4a16_form(
            {**size_query, "intermediate_size": 384, "top_k": 16}
        )
        assert not cake.is_cake_w4a16_form({**size_query, "top_k": 33})

    def test_routing_fixture_is_reproducible(self):
        p = PRESETS["G"]
        a = make_routing(p, 16, 3, "pad")
        b = make_routing(p, 16, 3, "pad")
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1].view(torch.int32), b[1].view(torch.int32))
        assert (a[0] == -1).any() and torch.isnan(a[1][a[0] == -1]).all()
        ids, _ = make_routing(p, 16, 3, "uniform")
        assert all(len(set(row.tolist())) == p.top_k for row in ids)
