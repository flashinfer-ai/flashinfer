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

"""Multi-rank integration tests for the Cake SM120 MXFP4 x MXFP8 hybrid-transport MegaMoE.

Launch on 4 or 8 sm_120 GPUs of one host (NCCL 2.30.7 with the Device API), or on
16 ranks over two hosts (``torchrun --nnodes 2``):

    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_sm120_w4a8_cake_mega.py \
        -m "gpu_4 and arch_sm120" -v

``CAKE_SM120_MEGAMOE_CASES`` selects ``shape:variant[:l<L>]`` cases (default
:data:`DEFAULT_CASES`; shapes in :data:`SHAPES`) and ``CAKE_SM120_MEGAMOE_TABLE`` the
transport table (default ``all_gin``; ``all_lsa`` / ``lsa_where_reachable`` / ``auto``).

The device source is parametric (EP world size, experts, top-k, hidden, intermediate,
row capacity, shared expert), so the cases cover the DeepSeek-V4-Flash routed layer
the kernel was tuned on (E=256, H=4096, I=2048, top-k 6) and the routed-expert
geometries of DeepSeek-V3 (E=256, H=7168, I=2048, top-k 8), Qwen3-235B-A22B
(E=128, H=4096, I=1536, top-k 8) and Qwen3-30B-A3B (E=128, H=2048, I=768, top-k 8),
each JIT compiled for the launched world size, plus a ``local3`` case with three
experts per rank (a local expert count that is not a multiple of 4).  The weights are random MXFP4 (FP8
e4m3 for the shared expert), the activations Gaussian, quantized to FP8 e4m3 with
UE8M0 block-32 scales.  The SwiGLU clamp is a compile-time parameter
(``swiglu_limit``): the default cases use DeepSeek-V4's ``L = 10`` with weight scales
that keep the W1 outputs O(1), i.e. inside the clamp range; the ``:l0`` cases build the
kernel with ``L = 0`` (standard unclamped SwiGLU, DeepSeek-V3 / Qwen3) and scale the
routed W1 weights up so that a large share of the gate / up values lies outside
``[-10, 10]`` -- the oracle with the V4 clamp must then disagree with the kernel
(checked).  The comparison is against a pure-torch oracle on the kernel's numerical
boundary, as the kernel's own validators define it:

* W1 accumulates in FP32 and rounds to BF16;
* SwiGLU clamps ``gate`` to ``<= L`` and ``up`` to ``[-L, L]`` (no clamp for
  ``L = 0``), uses the FP32 sigmoid ``1 / (1 + exp2(-x * log2(e)))`` and multiplies
  by the FP32 route weight (1.0 for the shared expert);
* the intermediate requantizes to e4m3 with one UE8M0 scale per 32 values
  (scale = next power of two ``>= amax * (1 / 448)``);
* W2 accumulates in FP32 and rounds to BF16 per route (and for the shared expert);
* the combine sums the BF16 route partials in FP32 in slot order, adds the BF16
  shared-expert partial in FP32 and rounds once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch

# DeepSeek-V4-Flash routed layer (the default geometry; the real-weight fixture test uses it).
HIDDEN = 4096
INTERMEDIATE = 2048
EXPERTS = 256
TOP_K = 6
MAX_ROWS = 4096
ATOL = RTOL = 0.1
REL_L2_MAX = 0.02
MAX_ABS_REFERENCE_FRACTION = 0.02
LOG2_E = 1.4426950408889634
# SwiGLU clamp of the kernel build (MOE_SWIGLU_LIMIT); DeepSeek-V4-Flash = 10, 0 = unclamped.
DEFAULT_SWIGLU_LIMIT = 10
FP8_MAX = 448.0
MIN_WORLD_SIZE = 2


@dataclass(frozen=True)
class Shape:
    """Routed-expert geometry of one test case (``experts`` = physical experts)."""

    name: str
    hidden: int
    intermediate: int
    experts: int
    top_k: int
    # > 0: the expert count scales with the EP world size (experts = per-rank x ranks).
    experts_per_rank: int = 0

    def for_world(self, world_size: int) -> "Shape":
        if not self.experts_per_rank:
            return self
        return Shape(
            self.name,
            self.hidden,
            self.intermediate,
            self.experts_per_rank * world_size,
            self.top_k,
        )


SHAPES = {
    "v4flash": Shape("v4flash", 4096, 2048, 256, 6),
    "dsv3": Shape("dsv3", 7168, 2048, 256, 8),
    "qwen3_235b": Shape("qwen3_235b", 4096, 1536, 128, 8),
    "qwen3_30b": Shape("qwen3_30b", 2048, 768, 128, 8),
    "local3": Shape("local3", 4096, 2048, 0, 6, experts_per_rank=3),
}
V4FLASH = SHAPES["v4flash"]
# ``shape:variant`` builds with the V4 clamp (L = 10); ``shape:variant:l<L>`` with MOE_SWIGLU_LIMIT=L.
DEFAULT_CASES = (
    "v4flash:noshared,v4flash:fp8shared,dsv3:fp8shared,"
    "qwen3_235b:noshared,qwen3_30b:noshared,"
    "dsv3:noshared:l0,qwen3_235b:fp8shared:l0,qwen3_30b:fp8shared:l0,"
    "local3:fp8shared"
)
# Weight scales 2^-9 .. 2^-6 keep the random-weight W1 outputs O(1), so the
# SwiGLU clamps stay mostly inactive and the test discriminates the math.
WEIGHT_EXPONENT_RANGE = (118, 122)
SHARED_EXPONENT_RANGE = (116, 119)
# Unclamped (L != 10) cases scale the routed W1 weights by 2^3 so that a large share of the
# gate / up values leaves [-10, 10] (std ~ 10-18 over the test geometries).
HOT_W13_EXPONENT_SHIFT = 3
MIN_OUTSIDE_CLAMP_FRACTION = 0.1


@dataclass(frozen=True)
class Case:
    """One test configuration: geometry, variant and SwiGLU clamp of the device build."""

    shape: Shape
    variant: str
    swiglu_limit: int = DEFAULT_SWIGLU_LIMIT

    @property
    def w13_exponent_shift(self) -> int:
        return (
            0 if self.swiglu_limit == DEFAULT_SWIGLU_LIMIT else HOT_W13_EXPONENT_SHIFT
        )

    @property
    def name(self) -> str:
        suffix = (
            ""
            if self.swiglu_limit == DEFAULT_SWIGLU_LIMIT
            else f"-l{self.swiglu_limit}"
        )
        return f"{self.shape.name}-{self.variant}{suffix}"


def _world() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def skip_unless_ep_launch() -> tuple[int, int]:
    rank, world_size = _world()
    if world_size < MIN_WORLD_SIZE:
        pytest.skip(f"requires a torchrun launch with at least {MIN_WORLD_SIZE} ranks")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from flashinfer.moe_ep.kernel_src.sm120.cake_mxfp4_megamoe import (
        cake_sm120_megamoe_available,
    )
    from flashinfer.utils import is_sm120a_supported

    if not is_sm120a_supported(torch.device("cuda")):
        pytest.skip("needs sm_120 (compute capability 12.0)")
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if torch.cuda.device_count() < local_world_size:
        pytest.skip(f"needs {local_world_size} visible GPUs, one per local rank")
    if not cake_sm120_megamoe_available():
        pytest.skip("needs NCCL 2.30.7 with the Device API (set NCCL_HOME)")
    return rank, world_size


def init_distributed() -> tuple[int, int]:
    import torch.distributed as dist

    rank, world_size = skip_unless_ep_launch()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", device_id=torch.device("cuda", torch.cuda.current_device())
        )
    return rank, world_size


def _variants() -> list[str]:
    raw = os.environ.get("CAKE_SM120_MEGAMOE_VARIANTS", "noshared,fp8shared")
    return [v.strip() for v in raw.split(",") if v.strip()]


def _cases() -> list[Case]:
    raw = os.environ.get("CAKE_SM120_MEGAMOE_CASES", DEFAULT_CASES)
    cases = []
    for item in raw.split(","):
        if item.strip():
            shape, variant, *limit = item.strip().split(":")
            swiglu_limit = (
                int(limit[0].removeprefix("l")) if limit else DEFAULT_SWIGLU_LIMIT
            )
            cases.append(Case(SHAPES[shape], variant, swiglu_limit))
    return cases


def table_spec() -> str:
    value = os.environ.get("CAKE_SM120_MEGAMOE_TABLE", "").strip()
    return value or "all_gin"


# --------------------------------------------------------------------------- problem


def _packed_e2m1_and_scales(
    shape_rows: int, k: int, generator: torch.Generator, exponent_shift: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random finite E2M1 codes (packed low nibble first) + UE8M0 exponents."""

    codes = torch.randint(
        0, 16, (shape_rows, k), dtype=torch.uint8, device="cuda", generator=generator
    )
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    exponents = torch.randint(
        WEIGHT_EXPONENT_RANGE[0] + exponent_shift,
        WEIGHT_EXPONENT_RANGE[1] + exponent_shift,
        (shape_rows, k // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    return packed, exponents


def _e4m3_and_scales(
    rows: int, k: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random finite e4m3 codes + UE8M0 exponents (the shared expert's operands)."""

    values = torch.randn(rows, k, device="cuda", generator=generator) * 4.0
    codes = values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    exponents = torch.randint(
        SHARED_EXPONENT_RANGE[0],
        SHARED_EXPONENT_RANGE[1],
        (rows, k // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    return codes, exponents


def expert_weights(
    expert: int, seed: int, shape: Shape = V4FLASH, w13_exponent_shift: int = 0
):
    """Packed MXFP4 ``(w13, w13_scale, w2, w2_scale)`` of one (logical) expert.

    Every expert draws from its own generator, so any rank can materialize any expert
    (EPLB replicas are byte copies of their logical expert).  ``w13_exponent_shift``
    scales the W1 weights by a power of two (unclamped-SwiGLU cases)."""

    generator = torch.Generator(device="cuda").manual_seed(seed * 4099 + expert)
    w13, w13_scale = _packed_e2m1_and_scales(
        2 * shape.intermediate, shape.hidden, generator, w13_exponent_shift
    )
    w2, w2_scale = _packed_e2m1_and_scales(shape.hidden, shape.intermediate, generator)
    return w13, w13_scale, w2, w2_scale


def local_weights(
    rank: int,
    world_size: int,
    seed: int,
    *,
    shared: bool,
    shape: Shape = V4FLASH,
    physical_to_logical: list[int] | None = None,
    w13_exponent_shift: int = 0,
):
    """This rank's weight pack: physical experts ``rank * local ... (rank + 1) * local - 1``.

    ``physical_to_logical`` (EPLB) maps each physical expert to the logical expert whose
    weights it carries; by default physical == logical."""

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        CakePrequantizedMoEWeightsWithSharedExpert,
    )

    local = shape.experts // world_size
    logical = [
        (physical_to_logical[p] if physical_to_logical is not None else p)
        for p in range(rank * local, (rank + 1) * local)
    ]
    parts = [expert_weights(e, seed, shape, w13_exponent_shift) for e in logical]
    w13, w13_scale, w2, w2_scale = (
        torch.stack([part[i] for part in parts]) for i in range(4)
    )
    routed = (
        w13,
        w2,
        w13_scale.view(torch.float8_e8m0fnu),
        w2_scale.view(torch.float8_e8m0fnu),
    )
    if not shared:
        return MoEWeightPack(*routed)
    # The shared expert is replicated: every rank draws it from the same seed.
    shared_generator = torch.Generator(device="cuda").manual_seed(seed + 7919)
    s_w13, s_w13_scale = _e4m3_and_scales(
        2 * shape.intermediate, shape.hidden, shared_generator
    )
    s_w2, s_w2_scale = _e4m3_and_scales(
        shape.hidden, shape.intermediate, shared_generator
    )
    return CakePrequantizedMoEWeightsWithSharedExpert(
        *routed,
        shared_w13=s_w13.view(torch.float8_e4m3fn),
        shared_w2=s_w2.view(torch.float8_e4m3fn),
        shared_w13_scale=s_w13_scale.view(torch.float8_e8m0fnu),
        shared_w2_scale=s_w2_scale.view(torch.float8_e8m0fnu),
    )


def make_inputs(
    rows: int,
    seed: int,
    *,
    masked: bool = False,
    shape: Shape = V4FLASH,
    logical_experts: int | None = None,
):
    """Gaussian activations and softmax top-k routing over ``logical_experts``
    (default ``shape.experts``)."""

    from flashinfer.moe_ep import MoEEpTensors

    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        rows, shape.hidden, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    logits = torch.randn(
        rows, logical_experts or shape.experts, device="cuda", generator=generator
    )
    topk_weights, topk_ids = torch.topk(
        torch.softmax(logits, dim=-1), shape.top_k, dim=-1
    )
    topk_weights = (
        (topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20))
        .float()
        .contiguous()
    )
    topk_ids = topk_ids.to(torch.int64).contiguous()
    if masked:
        topk_ids.fill_(-1)
        topk_weights.zero_()
    return MoEEpTensors(
        hidden_states=hidden_states, topk_ids=topk_ids, topk_weights=topk_weights
    )


def all_gather_cat(tensor: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    contiguous = tensor.contiguous()
    wire = (
        contiguous.view(torch.uint8) if contiguous.element_size() == 1 else contiguous
    )
    gathered = [torch.empty_like(wire) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, wire)
    out = torch.cat(gathered, dim=0)
    return out.view(contiguous.dtype) if contiguous.element_size() == 1 else out


# --------------------------------------------------------------------------- oracle


def _decode_ue8m0(exponents: torch.Tensor) -> torch.Tensor:
    return (exponents.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


def decode_fp4(packed: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """Low-nibble-first E2M1 ``[N, K//2]`` + UE8M0 ``[N, K//32]`` -> fp32 ``[N, K]``."""

    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    raw = packed.view(torch.uint8)
    codes = torch.empty(
        (*raw.shape[:-1], raw.shape[-1] * 2), dtype=torch.uint8, device=raw.device
    )
    codes[..., 0::2] = raw & 0x0F
    codes[..., 1::2] = raw >> 4
    values = magnitudes[(codes & 7).to(torch.int64)]
    values = torch.where(((codes & 8) != 0) & ((codes & 7) != 0), -values, values)
    return values * _decode_ue8m0(exponents).repeat_interleave(32, dim=-1)


def decode_fp8(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """e4m3 ``[N, K]`` + UE8M0 ``[N, K//32]`` -> fp32 ``[N, K]``."""

    values = codes.view(torch.float8_e4m3fn).float()
    return values * _decode_ue8m0(exponents).repeat_interleave(32, dim=-1)


def requantize_k32(values: torch.Tensor) -> torch.Tensor:
    """The kernel's e4m3 / UE8M0-K32 requantization, decoded back to fp32."""

    rows, cols = values.shape
    grouped = values.reshape(rows, cols // 32, 32)
    amax = grouped.abs().amax(dim=2)
    bits = (amax * (1.0 / FP8_MAX)).contiguous().view(torch.int32)
    exponent = (((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)).clamp(
        max=254
    )
    inverse = ((254 - exponent) << 23).view(torch.float32)
    codes = (grouped * inverse.unsqueeze(2)).to(torch.float8_e4m3fn)
    scale = (exponent << 23).view(torch.float32)
    return (codes.float() * scale.unsqueeze(2)).reshape(rows, cols)


def swiglu_kernel_boundary(
    fc1_bf16: torch.Tensor,
    route_weight: torch.Tensor,
    swiglu_limit: int = DEFAULT_SWIGLU_LIMIT,
) -> torch.Tensor:
    """SwiGLU with the kernel's clamp (``swiglu_limit``; 0 = none), FP32 exp2 sigmoid and
    FP32 route weight."""

    intermediate = fc1_bf16.shape[1] // 2
    gate = fc1_bf16[:, :intermediate].float()
    up = fc1_bf16[:, intermediate:].float()
    if swiglu_limit > 0:
        gate = gate.clamp(max=float(swiglu_limit))
        up = up.clamp(min=-float(swiglu_limit), max=float(swiglu_limit))
    sigmoid_gate = 1.0 / (1.0 + torch.exp2((0.0 - gate) * LOG2_E))
    return gate * sigmoid_gate * up * route_weight.unsqueeze(1)


def expert_partial(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    route_weight: torch.Tensor,
    swiglu_limit: int = DEFAULT_SWIGLU_LIMIT,
) -> torch.Tensor:
    """One expert's BF16 partial on the kernel boundary (dense fp32 weights)."""

    fc1 = (x @ w13.t()).to(torch.bfloat16)
    intermediate = requantize_k32(
        swiglu_kernel_boundary(fc1, route_weight, swiglu_limit)
    )
    return (intermediate @ w2.t()).to(torch.bfloat16)


def reference_output(
    x_dequant: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    decode_w13,
    decode_w2,
    shared: tuple[torch.Tensor, torch.Tensor] | None = None,
    swiglu_limit: int = DEFAULT_SWIGLU_LIMIT,
) -> torch.Tensor:
    """Slot-ordered FP32 sum of the routed BF16 partials (+ shared partial), one BF16 round.

    ``decode_w13(expert) -> fp32 [2I, H]`` (canonical gate rows then up rows) and
    ``decode_w2(expert) -> fp32 [H, I]`` supply the dense routed expert weights;
    ``shared`` is the dense ``(w13 [2I, H], w2 [H, I])`` shared expert.
    """

    rows, hidden = x_dequant.shape
    top_k = topk_ids.shape[1]
    partials = torch.zeros(
        rows, top_k, hidden, dtype=torch.float32, device=x_dequant.device
    )
    for expert in torch.unique(topk_ids[topk_ids >= 0]).tolist():
        row, slot = torch.where(topk_ids == expert)
        partials[row, slot] = expert_partial(
            x_dequant[row],
            decode_w13(expert),
            decode_w2(expert),
            topk_weights[row, slot],
            swiglu_limit,
        ).float()
    out = torch.zeros(rows, hidden, dtype=torch.float32, device=x_dequant.device)
    for slot in range(top_k):
        out += partials[:, slot]
    if shared is not None and rows:
        ones = torch.ones(rows, dtype=torch.float32, device=x_dequant.device)
        out += expert_partial(
            x_dequant, shared[0], shared[1], ones, swiglu_limit
        ).float()
    return out.to(torch.bfloat16)


def check_close(actual: torch.Tensor, reference: torch.Tensor) -> None:
    assert torch.isfinite(actual).all()
    if reference.numel() == 0:
        return
    # Element-wise bound scaled to the output range (the Cake harness' real-weight rule: max |error| <= 2 % of
    # max |reference|).  The fp32-matmul oracle and the kernel's MMA order disagree on a few e4m3 requant codes at
    # knife edges; with the unclamped ``:l0`` cases (W1 outputs far beyond +-10) one such code flip is several
    # times the fixed 0.1 bound on a near-cancelling output, so a fixed atol only fits O(1) outputs.
    scale = float(reference.float().abs().max())
    atol = max(ATOL, MAX_ABS_REFERENCE_FRACTION * scale)
    torch.testing.assert_close(actual.float(), reference.float(), atol=atol, rtol=RTOL)
    rel_l2 = torch.linalg.norm(actual.float() - reference.float()) / torch.linalg.norm(
        reference.float()
    ).clamp_min(1e-6)
    assert float(rel_l2) <= REL_L2_MAX, (
        f"relative L2 {float(rel_l2):.4f} > {REL_L2_MAX}"
    )


class Oracle:
    """Dense global weights (all ranks' routed experts + the shared expert)."""

    def __init__(
        self, weights, shared: bool, swiglu_limit: int = DEFAULT_SWIGLU_LIMIT
    ) -> None:
        self.swiglu_limit = swiglu_limit
        self.w13 = all_gather_cat(weights.w13)
        self.w2 = all_gather_cat(weights.w2)
        self.w13_scale = all_gather_cat(weights.w13_scale)
        self.w2_scale = all_gather_cat(weights.w2_scale)
        self.shared = (
            (
                decode_fp8(weights.shared_w13, weights.shared_w13_scale),
                decode_fp8(weights.shared_w2, weights.shared_w2_scale),
            )
            if shared
            else None
        )

    def _x(self, tensors) -> torch.Tensor:
        from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_staging import (
            dequantize_activations_k32,
            quantize_activations_k32,
        )

        return dequantize_activations_k32(
            *quantize_activations_k32(tensors.hidden_states)
        )

    def outside_clamp_fraction(self, tensors, limit: float = 10.0) -> float:
        """Share of the routed W1 outputs (BF16 gate / up values) with ``|v| > limit``."""

        x = self._x(tensors)
        outside = total = 0
        for expert in torch.unique(tensors.topk_ids[tensors.topk_ids >= 0]).tolist():
            row, _ = torch.where(tensors.topk_ids == expert)
            fc1 = (
                x[row] @ decode_fp4(self.w13[expert], self.w13_scale[expert]).t()
            ).to(torch.bfloat16)
            outside += int((fc1.float().abs() > limit).sum())
            total += fc1.numel()
        return outside / max(total, 1)

    def __call__(self, tensors, swiglu_limit: int | None = None) -> torch.Tensor:
        return reference_output(
            self._x(tensors),
            tensors.topk_ids,
            tensors.topk_weights,
            lambda e: decode_fp4(self.w13[e], self.w13_scale[e]),
            lambda e: decode_fp4(self.w2[e], self.w2_scale[e]),
            self.shared,
            self.swiglu_limit if swiglu_limit is None else swiglu_limit,
        )


# --------------------------------------------------------------------------- layer


def make_layer(
    rank: int,
    world_size: int,
    weights,
    *,
    variant: str,
    table,
    max_rows=MAX_ROWS,
    shape: Shape = V4FLASH,
    swiglu_limit: int = DEFAULT_SWIGLU_LIMIT,
):
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MegaConfig, MoEEpLayer
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )

    return MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size, rank=rank, device=torch.cuda.current_device()
        ),
        fleet_params=FleetParams(
            num_experts=shape.experts,
            max_tokens_per_rank=max_rows,
            token_hidden_size=shape.hidden,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                intermediate_size=shape.intermediate,
                top_k=shape.top_k,
                swiglu_limit=swiglu_limit,
                shared_expert="fp8_in_kernel" if variant == "fp8shared" else None,
                transport_table=table,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


@pytest.fixture(scope="module", params=_cases(), ids=lambda case: case.name)
def ep_layer(request):
    import torch.distributed as dist

    rank, world_size = init_distributed()
    case = request.param
    shape, variant = case.shape.for_world(world_size), case.variant
    if shape.experts % world_size:
        pytest.skip(f"{shape.name}: {shape.experts} experts on {world_size} ranks")
    shared = variant == "fp8shared"
    weights = local_weights(
        rank,
        world_size,
        seed=20260914,
        shared=shared,
        shape=shape,
        w13_exponent_shift=case.w13_exponent_shift,
    )
    oracle = Oracle(weights, shared, case.swiglu_limit)
    layer = make_layer(
        rank,
        world_size,
        weights,
        variant=variant,
        table=table_spec(),
        shape=shape,
        swiglu_limit=case.swiglu_limit,
    )
    layer.warmup(make_inputs(256, seed=1, shape=shape))
    torch.cuda.synchronize()
    yield rank, world_size, layer, oracle, shape
    # The layer owns collective transport resources: release them on every rank
    # before another variant (or test module) opens its own session.
    layer.destroy()
    dist.barrier()


def _launch(layer) -> dict:
    return layer._workspace.last_launch


def _protocol_clean(layer) -> None:
    assert layer._kernel.protocol_error(layer._workspace) == 0


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_transport_table_is_rank_uniform(ep_layer) -> None:
    import torch.distributed as dist

    rank, world_size, layer, _, shape = ep_layer
    ws = layer._workspace
    cells = list(ws.transport_table.cells)
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, (ws.transport_table.name, cells))
    assert all(g == gathered[0] for g in gathered)
    assert ws.copiers_active == (1 in cells)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_small_rows_use_m64_and_match_oracle(ep_layer) -> None:
    rank, world_size, layer, oracle, shape = ep_layer
    tensors = make_inputs(512, seed=100 + rank, shape=shape)
    out = layer(tensors)
    torch.cuda.synchronize()
    assert _launch(layer)["instance"].startswith("m64")
    check_close(out, oracle(tensors))
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_large_rows_use_m128_and_match_oracle(ep_layer) -> None:
    rank, world_size, layer, oracle, shape = ep_layer
    tensors = make_inputs(2048, seed=150 + rank, shape=shape)
    out = layer(tensors)
    torch.cuda.synchronize()
    assert _launch(layer)["instance"].startswith("m128")
    check_close(out, oracle(tensors))
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_heterogeneous_rows_including_empty_rank(ep_layer) -> None:
    """Per-rank row counts may differ in one call; a rank may hold no token at all."""

    rank, world_size, layer, oracle, shape = ep_layer
    pattern = (17, 1024, 0, 2048, 129, 640, 1, 777, 300, 0, 64, 4096, 5, 900, 2, 1500)
    rows = pattern[rank % len(pattern)]
    tensors = make_inputs(rows, seed=200 + rank, shape=shape)
    out = layer(tensors)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (rows, shape.hidden)
    assert _launch(layer)["collective_rows"] == max(
        pattern[r % len(pattern)] for r in range(world_size)
    )
    check_close(out, oracle(tensors))
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_schedule_switching_is_bitwise_stable(ep_layer) -> None:
    """M64 and M128 calls share the session state; alternating them changes nothing."""

    rank, world_size, layer, oracle, shape = ep_layer
    small = make_inputs(300, seed=500 + rank, shape=shape)
    large = make_inputs(1500, seed=550 + rank, shape=shape)
    first_small = layer(small).clone()
    first_large = layer(large).clone()
    again_small = layer(small).clone()
    again_large = layer(large).clone()
    torch.cuda.synchronize()
    assert torch.equal(first_small, again_small)
    assert torch.equal(first_large, again_large)
    check_close(first_small, oracle(small))
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_fully_masked_rank_returns_routed_zeros(ep_layer) -> None:
    """A rank whose routes are all masked (-1) gets only the shared expert (or zeros)."""

    rank, world_size, layer, oracle, shape = ep_layer
    tensors = make_inputs(
        256, seed=300 + rank, masked=(rank == world_size - 1), shape=shape
    )
    out = layer(tensors)
    torch.cuda.synchronize()
    reference = oracle(tensors)
    if rank == world_size - 1 and oracle.shared is None:
        assert torch.equal(out.float(), torch.zeros_like(out.float()))
    check_close(out, reference)
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_all_ranks_empty_call_is_skipped(ep_layer) -> None:
    rank, world_size, layer, _, shape = ep_layer
    out = layer(make_inputs(0, seed=600 + rank, shape=shape))
    torch.cuda.synchronize()
    assert tuple(out.shape) == (0, shape.hidden)
    assert _launch(layer).get("skipped")
    # The next real call still works.
    out = layer(make_inputs(64, seed=650 + rank, shape=shape))
    torch.cuda.synchronize()
    assert tuple(out.shape) == (64, shape.hidden)
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_swiglu_limit_is_compiled_in(ep_layer) -> None:
    """The kernel applies its build's clamp and not another one.

    For unclamped builds the data leaves the V4 clamp range, so the kernel must match
    the unclamped oracle and must not match the clamped one; for clamped builds the
    oracle's clamp is the build's."""

    rank, world_size, layer, oracle, shape = ep_layer
    tensors = make_inputs(512, seed=700 + rank, shape=shape)
    out = layer(tensors)
    torch.cuda.synchronize()
    check_close(out, oracle(tensors))
    if oracle.swiglu_limit != DEFAULT_SWIGLU_LIMIT:
        fraction = oracle.outside_clamp_fraction(tensors, float(DEFAULT_SWIGLU_LIMIT))
        assert fraction >= MIN_OUTSIDE_CLAMP_FRACTION, (
            f"only {fraction:.3f} of the W1 outputs leave the V4 clamp range"
        )
        clamped = oracle(tensors, swiglu_limit=DEFAULT_SWIGLU_LIMIT).float()
        rel_l2 = torch.linalg.norm(out.float() - clamped) / torch.linalg.norm(
            clamped
        ).clamp_min(1e-6)
        assert float(rel_l2) > 10 * REL_L2_MAX, (
            f"output matches the clamped oracle (relative L2 {float(rel_l2):.4f})"
        )
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_device_epoch_advances_once_per_launch(ep_layer) -> None:
    """The kernel owns its epoch: +1 per launched call (both schedules), +0 per skipped call."""

    rank, world_size, layer, _, shape = ep_layer
    start = layer._kernel.device_epoch(layer._workspace)
    rows = (64, 1024, 0, 64)  # m64, m128, all-ranks-empty (skipped), m64 again
    launched = 0
    for i, n in enumerate(rows):
        layer(make_inputs(n, seed=800 + 10 * i + rank, shape=shape))
        torch.cuda.synchronize()
        launched += 0 if _launch(layer).get("skipped") else 1
    assert launched == 3
    assert (
        layer._kernel.device_epoch(layer._workspace) == (start + launched) & 0xFFFFFFFF
    )
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_repeated_epochs_are_bitwise_identical(ep_layer) -> None:
    rank, world_size, layer, _, shape = ep_layer
    tensors = make_inputs(1024, seed=400 + rank, shape=shape)
    first = layer(tensors).clone()
    second = layer(tensors).clone()
    third = layer(tensors).clone()
    torch.cuda.synchronize()
    assert torch.equal(first, second) and torch.equal(first, third)
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_first_call_on_a_non_default_stream(ep_layer) -> None:
    """A layer first called on a non-blocking stream (serving frameworks run the forward on one) is correct.

    Its weight descriptors are encoded on that stream while the stream is stalled, so the carriers' zero
    fill is still queued; an upload not ordered on the caller's stream would be overwritten before launch.
    """

    import torch.distributed as dist

    rank, world_size, layer, _, shape = ep_layer
    variant = layer._megakernel_config.variant
    swiglu_limit = layer._megakernel_config.swiglu_limit
    weights = local_weights(
        rank, world_size, seed=20261011, shared=variant == "fp8shared", shape=shape
    )
    oracle = Oracle(weights, variant == "fp8shared", swiglu_limit)
    tensors = make_inputs(512, seed=1100 + rank, shape=shape)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # Same geometry as `layer`: shares its pooled workspace, only the layer descriptors are new.
        fresh = make_layer(
            rank,
            world_size,
            weights,
            variant=variant,
            table=table_spec(),
            shape=shape,
            swiglu_limit=swiglu_limit,
        )
        torch.cuda._sleep(int(2e8))
        out = fresh(tensors)
    side.synchronize()
    check_close(out, oracle(tensors))
    assert fresh._kernel.protocol_error(fresh._workspace) == 0
    fresh.destroy()
    dist.barrier()


def _bucketed(layer, tensors, bound: int) -> torch.Tensor:
    """One invocation with a rank-uniform token bucket: no routing all-gather."""

    layer.stage_inputs(tensors, compile_tokens_per_rank=bound)
    return layer.compute_staged(output=None)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_bucket_skips_the_routing_exchange_bitwise(ep_layer) -> None:
    """With ``compile_tokens_per_rank`` the call has no host synchronization (no routing
    all-gather, grid from the bound) and returns the same bits as the exchanged call, for
    heterogeneous rows, an empty rank and both row schedules -- with the exact largest
    row count and with a coarser bucket (the workspace capacity, M128 schedule)."""

    rank, world_size, layer, oracle, shape = ep_layer
    capacity = layer._workspace.max_rows
    for pattern in (
        (17, 300, 0, 511, 129, 64, 1, 77),
        (2048, 0, 1500, 4096, 640, 1, 900, 3000),
    ):
        rows = [pattern[r % len(pattern)] for r in range(world_size)]
        tensors = make_inputs(rows[rank], seed=1200 + rank, shape=shape)
        exchanged = layer(tensors).clone()
        assert _launch(layer)["grid_rule"] == "routing"
        torch.cuda.synchronize()
        for bound in (max(rows), capacity):
            torch.cuda.set_sync_debug_mode("error")
            try:
                out = _bucketed(layer, tensors, bound)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
            launch = _launch(layer)
            assert launch["grid_rule"] == "bound"
            assert launch["collective_rows"] == bound
            assert launch["instance"].startswith(
                "m64" if bound <= layer._workspace.m64_max_rows else "m128"
            )
            assert torch.equal(out, exchanged)
            check_close(out, oracle(tensors))
            _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_bucket_with_routing_onto_one_rank(ep_layer) -> None:
    """Every route of every rank lands on rank 0's experts (and the next ones when a rank
    owns fewer than top-k): the bound-sized grid leaves most workers without a tile on the
    other ranks; the output is still exact."""

    rank, world_size, layer, oracle, shape = ep_layer
    rows = 96
    tensors = make_inputs(rows, seed=1300 + rank, shape=shape)
    pool = max(shape.experts // world_size, shape.top_k)
    generator = torch.Generator(device="cuda").manual_seed(1350 + rank)
    order = torch.rand(rows, pool, device="cuda", generator=generator).argsort(dim=-1)
    tensors.topk_ids = order[:, : shape.top_k].to(torch.int64).contiguous()
    exchanged = layer(tensors).clone()
    routing_grid = _launch(layer)["grid"]
    out = _bucketed(layer, tensors, rows)
    torch.cuda.synchronize()
    assert _launch(layer)["grid"] >= routing_grid
    assert torch.equal(out, exchanged)
    check_close(out, oracle(tensors))
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_bucket_all_empty_and_invalid(ep_layer) -> None:
    from flashinfer.moe_ep import MoEEpConfigError

    rank, world_size, layer, _, shape = ep_layer
    out = _bucketed(layer, make_inputs(0, seed=1400 + rank, shape=shape), 0)
    assert tuple(out.shape) == (0, shape.hidden)
    assert _launch(layer).get("skipped")
    tensors = make_inputs(32, seed=1450 + rank, shape=shape)
    # Rejected on every rank before any collective: below this rank's tokens, or above
    # the workspace capacity.
    with pytest.raises(MoEEpConfigError, match="compile_tokens_per_rank"):
        _bucketed(layer, tensors, 31)
    with pytest.raises(MoEEpConfigError, match="compile_tokens_per_rank"):
        _bucketed(layer, tensors, layer._workspace.max_rows + 1)
    out = _bucketed(layer, tensors, 32)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (32, shape.hidden)
    _protocol_clean(layer)


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_cuda_graph_buckets_replay_in_mixed_order(ep_layer) -> None:
    """The serving pattern: one graph per token bucket, captured on one workspace after the
    layer ran eagerly, all kept alive and replayed in a mixed order (the same sequence on
    every rank) with new inputs and heterogeneous per-rank rows under each bucket, both
    row schedules; every replay returns the eager bits.  The graphs share the workspace
    output buffer, so each replay's result is copied out before the next call."""

    import torch.distributed as dist

    rank, world_size, layer, oracle, shape = ep_layer
    m64_max_rows = layer._workspace.m64_max_rows
    buckets = (64, m64_max_rows, 2048)
    # Heterogeneous rows under each bucket; the last rank holds no token.
    rows = [
        0 if rank == world_size - 1 else bucket - (7 * rank) % (bucket // 2)
        for bucket in buckets
    ]
    statics = []
    graphs = []
    for b, bucket in enumerate(buckets):
        static = make_inputs(rows[b], seed=1500 + 10 * b + rank, shape=shape)
        _bucketed(layer, static, bucket)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _bucketed(layer, static, bucket)
        torch.cuda.synchronize()
        dist.barrier()
        statics.append((static, captured))
        graphs.append(graph)
    start = layer._kernel.device_epoch(layer._workspace)
    order = (2, 0, 1, 1, 2, 0, 0, 1)
    for i, b in enumerate(order):
        static, captured = statics[b]
        fresh = make_inputs(rows[b], seed=1600 + 10 * i + rank, shape=shape)
        static.hidden_states.copy_(fresh.hidden_states)
        static.topk_ids.copy_(fresh.topk_ids)
        static.topk_weights.copy_(fresh.topk_weights)
        graphs[b].replay()
        replayed = captured.clone()
        eager = _bucketed(layer, fresh, buckets[b]).clone()
        torch.cuda.synchronize()
        assert torch.equal(replayed, eager), (i, buckets[b])
        check_close(replayed, oracle(fresh))
    assert (
        layer._kernel.device_epoch(layer._workspace)
        == (start + 2 * len(order)) & 0xFFFFFFFF
    )
    _protocol_clean(layer)
    del graphs
    dist.barrier()


@pytest.mark.arch_sm120
@pytest.mark.gpu_4
def test_cuda_graph_capture_needs_bucket(ep_layer) -> None:
    from flashinfer.moe_ep import MoEEpConfigError

    rank, world_size, layer, _, shape = ep_layer
    tensors = make_inputs(64, seed=1700 + rank, shape=shape)
    _bucketed(layer, tensors, 64)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(MoEEpConfigError, match="compile_tokens_per_rank"),
        torch.cuda.graph(graph),
    ):
        layer(tensors)
    torch.cuda.synchronize()
    out = _bucketed(layer, tensors, 64)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (64, shape.hidden)
    _protocol_clean(layer)
