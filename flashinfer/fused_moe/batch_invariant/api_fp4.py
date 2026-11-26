# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp4 MoE classes: ``moe.nvfp4.<act>`` (W4A4: NVFP4 experts with one per-tensor global scale per expert and GEMM,
per-token NVFP4 activations; the static-intermediate class ``moe.nvfp4.<act>.static``) and ``moe.mxfp4.<act>`` (MXFP4 experts
and MXFP4 activations, one UE8M0 scale per 32 elements on both operands, no global or per-token scale), on the bf16 classes'
routing, permutation and combine (:mod:`.api`) and the shared grouped block-scaled bodies (:mod:`.blocks`). The backend's
third pair, MXFP4 x MXFP8 (``moe.mxfp4w8a.<act>``: MXFP4 experts prepared here, the canonical MXFP8 activation pack), shares
the pair descriptor and the weight preparation of this module and runs on the MX op (:mod:`.api_mxfp4w8a`).

One pipeline, parametrised by the pair (:class:`Fp4Pair`): the pair's quantization of ``x`` (its quantizer class, or the
caller's pack) -> the deterministic permutation of the expanded rows (expert order, then token-slot order) with the scale
factors at 128-row atoms -> the grouped block-scaled GEMM1 with the pair's chain -> the intermediate quantized by the same
class -> the grouped block-scaled GEMM2 -> the slot-order combine. Every output bit is a function of the token's own row, its
``k`` experts and routing weights and the expert weights; the specifications are :mod:`.spec_nvfp4` / :mod:`.spec_mxfp4`, the
Lean statements the ``MoENvfp4`` / ``MoEMxfp4`` modules of the mode.

Two packings of the expanded rows are placements of the same bits (``path``): ``"dense"`` (the default: the fp4 rows back
to back, the scale factors and per-token scales at the 128-row-padded positions) and ``"padded"`` (every group's rows start
at a 128-row atom; the padding rows are zero-filled, computed and never stored).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from ...batch_invariant.quantization.layout import padded_sf_cols, sf_numel
from ...batch_invariant.quantization.mx_spec import MXFP4 as MXFP4_FORMAT
from ...batch_invariant.support import (
    BatchInvariantUnsupportedError,
    LazyNames,
    unsupported,
)
from ...gemm.batch_invariant.api import part_of, sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.schedule import chunks_for, fp4_class_for
from ...gemm.batch_invariant.spec import MX_MIXED_FMTS, MXFP4_FMTS, NVFP4_FMTS
from . import spec as S
from .api import (
    _permute_plan_run,
    interleave_fc1,
    launch_handle,
    padded_rows_bound,
    permute_plan,
    route,
)
from .blocks.epilogue import (
    MXFP4_GEMM2_CHAIN,
    GroupedEpilogueBlock,
    Mxfp4Chain,
    W4A4Chain,
)
from .blocks.epilogue_mx import MxChain
from .schedule import launch_tile_bound
from .spec_mxfp4 import CLASS as MX_CLASS
from .spec_mxfp4 import MX_BLOCK, expert_classes_mx
from .spec_mxfp4w8a import CLASS as MIXED_CLASS
from .spec_mxfp4w8a import granule_of
from .spec_nvfp4 import CLASS as NV_CLASS
from .spec_nvfp4 import (
    E2M1_MAX,
    E4M3_MAX,
    INTERMEDIATES,
    NVFP4_BLOCK,
    TE_GS_INV,
)

#: The block-scaled chain blocks: the pair's GEMM1 / GEMM2 chains (``.at(in_fmt)`` places them on an instruction class).
Fp4Chain = Union[W4A4Chain, Mxfp4Chain, MxChain]
#: 16-byte rows of the packed fp4 operands (the TMA alignment) and whole scale blocks / scale-factor column groups.
ALIGN_FP4 = 32
MAX_TOP_K = 16
PATHS = ("dense", "padded")
#: Rows past the live rows in the dense packing's A buffers (and past the padded rows in its scale-factor buffers) so the
#: last tiles' TMA boxes stay in bounds; never stored. A placement.
DENSE_TAIL = 256
# the NVFP4 intermediate's scale factors reach the padded positions by the row-mapped quantizer (one launch) or by the
# quantizer and a re-tiling move (two launches; the same bytes): two spellings of one placement
INTERMEDIATE_RETILE = "mapped"
RETILES = ("mapped", "scatter")
# a bf16 NVFP4 input reaches the grouped layout by the quantizer and the fp4 scatter (two launches) or by the quantizer
# storing into the grouped layout itself (one launch; the same bytes): two spellings of one placement
INPUT_SCATTER = "fused"
INPUT_SCATTERS = ("fused", "scatter")
GROUP_ALIGN = 128  # the scale-factor atom: the group alignment of the scale factors
_OUT = {"bf16": torch.bfloat16, "f32": torch.float32}


def expert_classes(
    H: int, I: int, family: Optional[str], gated: bool = True
) -> Tuple[str, str]:
    """The NVFP4 instruction class of each expert weight by the dense rule (`schedule.fp4_class_for`): GEMM1 on ``(2I, H)``
    (``(I, H)`` ungated), GEMM2 on ``(H, I)``; a function of the weight shapes and the architecture family only."""
    return fp4_class_for(2 * I if gated else I, H, family), fp4_class_for(H, I, family)


@dataclass(frozen=True)
class Fp4Pair:
    """A quantization pair of the fp4 backend: ``name`` tags the weight view and the kernel objects; ``cls`` is the registry
    class; ``block`` the elements per scale; ``fmts`` the instruction classes; ``scaled`` whether the pair carries a per-tensor
    alpha per expert and GEMM and a per-token scale per row (the NVFP4 pair, which also serves the static-intermediate
    class) -- the MX pairs' scales are the block scales alone; ``act_fmt`` the activations' form: ``"fp4"`` (bf16 rows the
    pair's class quantizes, or its packed fp4 rows) or ``"mxfp8"`` (the canonical MXFP8 pack of the mixed pair)."""

    name: str
    cls: str
    block: int
    fmts: Tuple[str, ...]
    scaled: bool
    act_fmt: str = field(default="fp4", kw_only=True)

    @property
    def mixed(self) -> bool:
        """The mixed MXFP4-weight x MXFP8-activation pair (its op is :mod:`.api_mxfp4w8a`'s)."""
        return self.act_fmt == "mxfp8"

    def sf_numel(self, rows: int, K: int) -> int:
        """Bytes of the 128x4 swizzled scale buffer of ``rows`` rows of ``K`` elements."""
        return sf_numel(rows, K, self.block, "128x4")

    def sf_cols(self, K: int) -> int:
        """Scale columns of a row of ``K`` elements in the 128x4 layout (whole 4-column groups)."""
        return padded_sf_cols(K, self.block, "128x4")

    def classes(
        self, H: int, I: int, family: Optional[str], gated: bool = True
    ) -> Tuple[str, str]:
        """The instruction class of each expert GEMM: the dense rule per weight shape (NVFP4), the part's (the MX pairs)."""
        if self.scaled:
            return expert_classes(H, I, family, gated)
        if self.mixed:
            from .api_mxfp4w8a import expert_classes_mixed

            return expert_classes_mixed(family)
        return expert_classes_mx(family)

    def check_shape(
        self,
        H: Optional[int],
        I: int,
        E: int,
        k: Optional[int] = None,
        gated: bool = True,
        in_fmt: Optional[str] = None,
    ) -> None:
        check_shape_fp4(self, H, I, E, k, gated, in_fmt)

    def prepare_activations(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The layer's activation pack of the pair: bf16 rows as given and no scale for the fp4 pairs (the class quantizes
        the rows inside the op: per token for NVFP4, per 32-element block for MXFP4); the canonical MXFP8 pack by the
        batch-invariant quantizer for the mixed pair (e4m3 ``[T, H]``, the token-major UE8M0 scales ``[T, H / 32]``)."""
        rows = hidden_states.to(torch.bfloat16).contiguous()
        if self.mixed:
            from .api_mxfp8 import quantize_activations_mxfp8

            return quantize_activations_mxfp8(rows)
        return rows, None

    def quantize(
        self,
        x: torch.Tensor,
        gs_inv: Union[float, torch.Tensor] = TE_GS_INV,
        out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """The pair's quantization of bf16 rows ``[M, K]``: ``(x4 [M, K/2] uint8, sf flat 128x4 uint8, pts [M] f32 or None)``
        -- per token for NVFP4 (``gs_inv`` the recipe's inverse base scale), per 32-element block for MXFP4 (no per-token
        scale). ``out = (x4, sf)`` preallocated buffers."""
        if self.scaled:
            return quantize_activations(x, gs_inv, out)
        x4, sf = quantize_rows_mx(x, out)
        return x4, sf, None


NVFP4 = Fp4Pair("nvfp4", NV_CLASS, NVFP4_BLOCK, NVFP4_FMTS, True)
MXFP4 = Fp4Pair("mxfp4", MX_CLASS, MX_BLOCK, MXFP4_FMTS, False)
MXFP4W8A = Fp4Pair(
    "mxfp4w8a", MIXED_CLASS, MX_BLOCK, MX_MIXED_FMTS, False, act_fmt="mxfp8"
)
#: The pairs by their view tag.
PAIRS_BY_NAME: Dict[str, Fp4Pair] = {p.name: p for p in (NVFP4, MXFP4, MXFP4W8A)}


def pair_of(chain: GroupedEpilogueBlock) -> Fp4Pair:
    """The pair of a block-scaled chain block (its class)."""
    if isinstance(chain, Mxfp4Chain):
        return MXFP4
    if isinstance(chain, W4A4Chain):
        return NVFP4
    if isinstance(chain, MxChain) and chain.in_fmt in MX_MIXED_FMTS:
        return MXFP4W8A
    raise TypeError(f"not a block-scaled chain of the fp4 pairs: {chain!r}")


def check_shape_fp4(
    pair: Fp4Pair,
    H: Optional[int],
    I: int,
    E: int,
    k: Optional[int] = None,
    gated: bool = True,
    in_fmt: Optional[str] = None,
) -> None:
    """The complete acceptance rule of a pair's classes on ``(H, I, E, k)``: ``H`` and ``I`` positive multiples of 32 (16-byte
    rows of the packed fp4 operands, whole scale blocks) -- and, for the mixed pair on the instruction class ``in_fmt``, of the
    class's K granule (128 on the SM100 / SM103 class, whose TMA unpacks the e2m1 weight in 128-element runs; ``in_fmt`` None:
    the 32-element rule alone, the class entering where the device is known) -- ``1 <= E <= 1024``, ``1 <= k <= min(E, 16)``,
    and the expert weight shapes unsplit by the dense class's chunk rule (the grouped classes take one fold per weight).
    Everything else about a shape is placement. ``H=None`` checks the rules the layer's configuration fixes (the hidden size
    enters at the first call). Raises :class:`BatchInvariantUnsupportedError`; the class never routes a shape elsewhere."""
    from .blocks.scheduler import MAX_E

    if in_fmt is not None and (not pair.mixed or in_fmt not in pair.fmts):
        raise ValueError(f"{pair.name}: no instruction class {in_fmt!r}")
    gran = granule_of(in_fmt) if in_fmt is not None else ALIGN_FP4

    def bad(what: str, limit: str):
        raise unsupported(
            pair.cls,
            f"{what} ("
            + (f"H={H}, " if H is not None else "")
            + f"I={I}, E={E}"
            + (f", k={k}" if k is not None else "")
            + ")",
            limit,
        )

    if I <= 0 or I % ALIGN_FP4 or (H is not None and (H <= 0 or H % ALIGN_FP4)):
        bad(
            f"hidden size {H} / intermediate size {I}"
            if H is not None
            else f"intermediate size {I}",
            f"positive multiples of {ALIGN_FP4} (16-byte rows of the quantized activations and weights)",
        )
    if gran != ALIGN_FP4 and (I % gran or (H is not None and H % gran)):
        bad(
            f"hidden size {H} / intermediate size {I} on the {in_fmt} class"
            if H is not None
            else f"intermediate size {I} on the {in_fmt} class",
            f"multiples of {gran} (the SM100 / SM103 instruction reads the e2m1 weight through the TMA's unpacking format in "
            f"{gran}-element runs)",
        )
    if not 1 <= E <= MAX_E:
        bad(f"{E} experts", f"1 <= num_experts <= {MAX_E}")
    if k is not None and not 1 <= k <= min(E, MAX_TOP_K):
        bad(f"top_k {k}", f"1 <= top_k <= min(num_experts, {MAX_TOP_K})")
    if H is None:
        return
    n1 = 2 * I if gated else I
    fmt = in_fmt if in_fmt is not None else (pair.fmts[0] if pair.mixed else pair.name)
    if chunks_for(n1, H, fmt) != 1 or chunks_for(H, I, fmt) != 1:
        bad(
            "the expert weight shapes",
            "unsplit expert weights (the dense class would K-split this shape)",
        )


def _require_cuda_pack(pair: Fp4Pair, dev: torch.device) -> None:
    """The pack lives on a CUDA device of the classes (the quantizer class runs there): a refusal by name, after the
    device-free rules of the preparation and before any kernel."""
    if dev.type != "cuda":
        raise unsupported(
            pair.cls,
            f"expert weights quantized on {dev}",
            "a CUDA device for the pack (the weights' device, or device=...): the quantizer class runs there",
        )


def _scale_vector(t, name: str, E: int, device: torch.device) -> torch.Tensor:
    """A per-expert vector of ``E`` multipliers as a contiguous fp32 ``[E]`` on ``device``, a copy of the caller's values
    (the view's tensors never alias the caller's: its reciprocal is formed once beside it). The rule, checked before the
    reciprocal with the mode's error: a floating tensor (or Python numbers) of shape ``[E]`` -- one element broadcasts to
    the experts -- on the weights' device or the host, every value finite, positive and normal (their reciprocals are
    factors of the class's chains)."""
    limit = (
        f"{name}: finite positive normal floating multipliers of shape [{E}] (one per expert; one element broadcasts) "
        f"on the weights' device or the host"
    )
    if isinstance(t, torch.Tensor):
        if not t.is_floating_point():
            raise unsupported(NV_CLASS, f"{name} of dtype {t.dtype}", limit)
        if t.device.type != "cpu" and t.device != torch.device(device):
            raise unsupported(
                NV_CLASS, f"{name} on {t.device} with weights on {device}", limit
            )
    v = torch.as_tensor(t, dtype=torch.float32)
    if tuple(v.shape) not in ((), (1,), (E,)):
        raise unsupported(NV_CLASS, f"{name} of shape {tuple(v.shape)}", limit)
    v = v.reshape(-1).expand(E).to(device).clone(memory_format=torch.contiguous_format)
    tiny = torch.finfo(torch.float32).tiny
    if not bool(torch.isfinite(v).all()) or not bool((v >= tiny).all()):
        raise unsupported(
            NV_CLASS,
            f"{name} with a value that is not finite, positive and normal (at least {tiny:.4g})",
            limit,
        )
    return v


def _global_scale_of(amax: torch.Tensor) -> torch.Tensor:
    """FlashInfer's per-tensor global scale from an amax: ``448 * 6 / amax``, one for a zero amax, a NaN amax propagated."""
    # the unit scale for the all-zero tensor alone: ``amax > 0`` would select it for a NaN amax too
    return torch.where(amax == 0, torch.ones_like(amax), E4M3_MAX * E2M1_MAX / amax).to(
        torch.float32
    )


def fp4_global_scale(w: torch.Tensor) -> torch.Tensor:
    """FlashInfer's per-tensor global scale of a weight: ``448 * 6 / amax(w)``, one for an all-zero tensor (the convention of
    the package's weight preparation; a 1-element f32 tensor); a NaN amax (a NaN weight) propagates, so the validator of
    the computed scales (:func:`fp4_global_scales`) refuses the expert."""
    return _global_scale_of(w.detach().to(torch.float32).abs().amax().reshape(1))


def fp4_global_scales(
    w: torch.Tensor, name: str, device: Optional[torch.device] = None
) -> torch.Tensor:
    """The per-expert global scales of the experts ``w [E, N, K]``: :func:`fp4_global_scale` of each expert's bf16 values
    (a wider dtype is rounded to bf16 first, as the quantizer sees it), as a contiguous fp32 ``[E]`` on ``device`` (the
    weights' by default), under the rule of the caller's scales (:func:`_scale_vector`): every scale finite, positive and
    normal, else the mode's error naming the weight and the expert -- an expert whose amax is below ``448 * 6 / FLT_MAX``
    has an infinite scale and a zero alpha, which would give NaN rows where a block of its weights is all zero and a
    silent zero contribution otherwise; a NaN weight gives a NaN scale. One expert's slice at a time is moved to ``device``
    and cast, and its fp32 magnitudes live alone: never a full-size copy of the experts (CPU-resident and fp32 weights
    included)."""
    dev = w.device if device is None else torch.device(device)
    amax = torch.stack(
        [
            w[e]
            .detach()
            .to(device=dev, dtype=torch.bfloat16)
            .to(torch.float32)
            .abs()
            .amax()
            for e in range(w.shape[0])
        ]
    )
    gs = _global_scale_of(amax)
    tiny = torch.finfo(torch.float32).tiny
    bad = ~(torch.isfinite(gs) & (gs >= tiny))
    if bool(bad.any()):
        e = int(bad.nonzero()[0])
        raise unsupported(
            NV_CLASS,
            f"{name}: expert {e} with an amax of {float(amax[e]):.3g} (its per-tensor global scale 448 * 6 / amax is not "
            "finite, positive and normal)",
            f"experts with an amax of at least {E4M3_MAX * E2M1_MAX / torch.finfo(torch.float32).max:.3g}, or "
            "caller-supplied global_scales",
        )
    return gs


@dataclass
class MoEWeightsFp4:
    """The fp4 expert weights as the kernels consume them: ``w1 [E, N1, H/2]`` packed fp4 (``N1 = 2I``, gate / up
    interleaved, for a gated chain; ``I`` ungated), ``w1_sf [E, sf_numel(N1, H)]`` 128x4 swizzled scales per expert, likewise
    ``w2 [E, H, I/2]`` / ``w2_sf``; the GEMM1 chain ``act`` with its ``params``. The NVFP4 pair adds the alphas ``w1_alpha`` /
    ``w2_alpha [E]`` (``1 / global_scale``, multipliers as ``mm_fp4`` takes them) and the intermediate's class: ``per_token``
    (``moe.nvfp4.<act>``) or ``static`` (``moe.nvfp4.<act>.static``: GEMM1's epilogue quantizes the fp32 gated value per
    16-column block with the expert's static scale ``gs2 [E]``, a multiplier in FlashInfer's ``fc2_input_scale`` convention;
    ``inv2 = 1 / gs2`` formed once, read by GEMM2 as given)."""

    pair: Fp4Pair
    w1: torch.Tensor
    w1_sf: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    hidden_size: int
    intermediate_size: int
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()
    w1_alpha: Optional[torch.Tensor] = None
    w2_alpha: Optional[torch.Tensor] = None
    intermediate: str = "per_token"
    gs2: Optional[torch.Tensor] = None
    inv2: Optional[torch.Tensor] = None

    @staticmethod
    def prepare(
        pair: Fp4Pair,
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        interleaved: bool = False,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
        global_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        intermediate: str = "per_token",
        intermediate_scales: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> "MoEWeightsFp4":
        """Quantize the experts (``fc1 [E, 2I, H]`` rows ``[up, gate]``, or ``[E, I, H]`` ungated; ``fc2 [E, H, I]``; bf16, or
        a wider floating dtype cast to bf16 one expert at a time) with the pair's quantizer class: NVFP4 per expert and GEMM
        at one per-tensor global scale (``global_scales = (gs1 [E], gs2 [E])``, or FlashInfer's ``448 * 6 / amax``), MXFP4
        with one UE8M0 scale per 32 elements of a row. The gate / up interleave is a row permutation applied before the
        quantization. ``act`` defaults to the class's default chain of the gating. ``intermediate="static"`` selects the
        NVFP4 static class with the per-expert ``intermediate_scales [E]`` (f32 multipliers; default ones) and is refused
        for the MX pair. The pack lives on ``device`` (the weights' by default). The per-expert global scales, the
        caller's or the computed ones, are validated before any expert is quantized (finite, positive and normal:
        :func:`fp4_global_scales`). The experts are quantized one at a time into the preallocated pack, each expert's
        slice moved to the pack's device and cast to bf16 right before its quantization (a same-device bf16 slice as it
        is): the peak memory beyond the inputs is the fp4 pack plus one expert's temporaries (its moved or cast rows, its
        interleaved bf16 rows, the fp32 magnitudes of its amax; a strided view's expert made contiguous) for CPU-resident
        and fp32 inputs too, the bits those of each expert quantized alone."""
        from ...batch_invariant.quantization.nvfp4_api import nvfp4_quantize_rows

        if intermediate not in INTERMEDIATES:
            raise ValueError(
                f"intermediate must be one of {INTERMEDIATES}, not {intermediate!r}"
            )
        if not pair.scaled and (
            intermediate != "per_token" or intermediate_scales is not None
        ):
            raise unsupported(
                pair.cls,
                f"intermediate={intermediate!r} with intermediate scales"
                if intermediate_scales is not None
                else f"intermediate={intermediate!r}",
                "the intermediate quantized per 32-element block by the class itself (the static class is the NVFP4 pair's)",
            )
        if intermediate_scales is not None and intermediate != "static":
            raise unsupported(
                pair.cls,
                f"intermediate_scales with intermediate={intermediate!r}",
                "intermediate scales with intermediate='static' alone (the per-token class derives its scales from the rows)",
            )
        act = act if act is not None else ("silu" if gated else "relu2")
        if act not in (S.ACTS if gated else S.UNGATED_ACTS):
            raise ValueError(f"no {'gated' if gated else 'ungated'} chain {act!r}")
        key: Tuple[Optional[float], ...] = S.params_key(params)
        if len(key) != len(S.ACT_PARAMS.get(act, ())):
            raise ValueError(
                f"chain {act!r} takes parameters {S.ACT_PARAMS.get(act, ())}, got {key!r}"
            )
        if (
            not fc1.is_floating_point()
            or not fc2.is_floating_point()
            or fc1.dim() != 3
            or fc2.dim() != 3
        ):
            raise TypeError(
                "the fp4 MoE classes take floating expert weights fc1 [E, N1, H], fc2 [E, H, I] (bf16, or cast to bf16 one "
                "expert at a time)"
            )
        E, N1, H = fc1.shape
        I = N1 // 2 if gated else N1
        if tuple(fc2.shape) != (E, H, I) or (gated and N1 % 2):
            raise ValueError(
                f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
            )
        dev = fc1.device if device is None else torch.device(device)
        in_fmt = None
        if pair.mixed and dev.type == "cuda":
            from .api_mxfp4w8a import mx_class_of

            in_fmt = mx_class_of(
                dev
            )  # the pack's part names the mixed class, hence the granule
        check_shape_fp4(pair, H, I, E, gated=gated, in_fmt=in_fmt)
        # the pack, preallocated and filled one expert at a time (the quantizer writes every byte of its expert's rows and
        # scale buffer; the global scale and the interleave are per expert, so the bits are those of the expert quantized
        # alone): the peak beyond the inputs and the pack is one expert's temporaries -- its slice moved to the pack's
        # device and cast to bf16, its interleaved or contiguous rows and the fp32 magnitudes of its amax -- never a
        # full-size bf16 or fp32 copy of the experts (a strided view is made contiguous one expert at a time)
        w1 = torch.empty(E, N1, H // 2, dtype=torch.uint8, device=dev)
        w1_sf = torch.empty(E, pair.sf_numel(N1, H), dtype=torch.uint8, device=dev)
        w2 = torch.empty(E, H, I // 2, dtype=torch.uint8, device=dev)
        w2_sf = torch.empty(E, pair.sf_numel(H, I), dtype=torch.uint8, device=dev)

        def expert(w: torch.Tensor, e: int) -> torch.Tensor:
            """Expert ``e``'s slice on the pack's device as contiguous bf16, moved and cast right before its quantization
            (a contiguous bf16 slice already there is returned as it is)."""
            return w[e].to(device=dev, dtype=torch.bfloat16).contiguous()

        def fc1_rows(e: int) -> torch.Tensor:
            """Expert ``e``'s GEMM1 rows, contiguous: the gate / up interleave applied to this expert alone."""
            if interleaved or not gated:
                return expert(fc1, e)
            return interleave_fc1(expert(fc1, e).unsqueeze(0))[0]

        def fc2_rows(e: int) -> torch.Tensor:
            return expert(fc2, e)

        if not pair.scaled:
            _require_cuda_pack(pair, dev)
            for e in range(E):
                quantize_rows_mx(fc1_rows(e), out=(w1[e], w1_sf[e]))
                quantize_rows_mx(fc2_rows(e), out=(w2[e], w2_sf[e]))
            return MoEWeightsFp4(pair, w1, w1_sf, w2, w2_sf, H, I, gated, act, key)
        # the per-expert global scales, the caller's or the computed ones, under one rule before any expert is quantized
        if global_scales is not None:
            gs1_all = _scale_vector(global_scales[0], "global_scales[0]", E, dev)
            gs2_all = _scale_vector(global_scales[1], "global_scales[1]", E, dev)
        else:
            gs1_all = fp4_global_scales(fc1, "w1 / fc1", dev)
            gs2_all = fp4_global_scales(fc2, "w2 / fc2", dev)
        gs_inter = inv_inter = None
        if intermediate == "static":
            gs_inter = (
                torch.ones(E, dtype=torch.float32, device=dev)
                if intermediate_scales is None
                else _scale_vector(intermediate_scales, "intermediate_scales", E, dev)
            )
            inv_inter = torch.reciprocal(
                gs_inter
            ).contiguous()  # div.rn: the quantizer's reciprocal
        # div.rn: the alphas as mm_fp4 takes them
        alpha1 = torch.reciprocal(gs1_all).contiguous()
        alpha2 = torch.reciprocal(gs2_all).contiguous()
        _require_cuda_pack(pair, dev)
        for e in range(E):
            # the expert's scale as its own 1-element tensor: the quantizer's scale operand is 16-byte aligned
            nvfp4_quantize_rows(
                fc1_rows(e), gs1_all[e : e + 1].clone(), "128x4", out=(w1[e], w1_sf[e])
            )
            nvfp4_quantize_rows(
                fc2_rows(e), gs2_all[e : e + 1].clone(), "128x4", out=(w2[e], w2_sf[e])
            )
        return MoEWeightsFp4(
            pair,
            w1,
            w1_sf,
            w2,
            w2_sf,
            H,
            I,
            gated,
            act,
            key,
            alpha1,
            alpha2,
            intermediate,
            gs_inter,
            inv_inter,
        )

    @property
    def num_experts(self) -> int:
        return self.w1.shape[0]

    @property
    def n1(self) -> int:
        """GEMM1's weight rows: ``2I`` gated, ``I`` ungated."""
        return self.w1.shape[1]

    def classes(self, family: Optional[str]) -> Tuple[str, str]:
        """The instruction class of each GEMM on this part."""
        return self.pair.classes(
            self.hidden_size, self.intermediate_size, family, self.gated
        )

    @property
    def static(self) -> bool:
        return self.intermediate == "static"

    @property
    def chain1(self) -> Fp4Chain:
        """The GEMM1 chain of the class: the gated ``act`` on the (gate, up) pair or the ungated ``act`` on the accumulator,
        with the parameters; bf16 output, or the fused block quantization (``nvfp4``) of the static class. The mixed pair's
        is the MX chain (the bf16 classes' chain on the dequantised accumulators, placed on the launch's instruction class)."""
        kind = "glu" if self.gated else "act"
        if self.pair.mixed:
            return MxChain(kind, self.act, "bf16", self.params, self.pair.fmts[0])
        if not self.pair.scaled:
            return Mxfp4Chain(kind, self.act, "bf16", self.params)
        return W4A4Chain(
            kind, self.act, "nvfp4" if self.static else "bf16", self.params, "w4a4"
        )

    @property
    def chain1_quant(self) -> MxChain:
        """The mixed pair's fused-requantization spelling of ``chain1``: the same chain, its bf16-rounded value quantized by
        the ``mxfp8.quant`` class in the epilogue (codes and 128x4 block scales out, no bf16 intermediate)."""
        assert self.pair.mixed, self.pair
        return MxChain(
            "glu" if self.gated else "act",
            self.act,
            "mxfp8",
            self.params,
            self.pair.fmts[0],
        )

    @property
    def chain2(self) -> Fp4Chain:
        """The GEMM2 chain: NVFP4 ``cvt_bf16(mul(acc, mul(alpha2[e], pts2)))`` (the static class's with ``inv2[e]``), MXFP4
        and the mixed pair ``cvt_bf16(acc)``."""
        if self.pair.mixed:
            return MIXED_GEMM2_CHAIN
        if not self.pair.scaled:
            return MXFP4_GEMM2_CHAIN
        return GEMM2_STATIC_CHAIN if self.static else GEMM2_CHAIN

    def view(self) -> Dict[str, object]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for the backend (``quant`` names the pair:
        the pairs share the view key; ``intermediate`` names the intermediate's class for both pairs, the MX pair's being
        the per-token class alone)."""
        v: Dict[str, object] = {
            "quant": self.pair.name,
            "gemm1_weights": self.w1,
            "gemm1_weights_scale": self.w1_sf,
            "gemm2_weights": self.w2,
            "gemm2_weights_scale": self.w2_sf,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
            "intermediate": self.intermediate,
        }
        if self.pair.scaled:
            v.update(
                gemm1_alpha=self.w1_alpha,
                gemm2_alpha=self.w2_alpha,
                intermediate_scale=self.gs2,
                intermediate_inv_scale=self.inv2,
            )
        return v

    @staticmethod
    def from_view(view: Dict[str, object]) -> "MoEWeightsFp4":
        """The weights of a view :meth:`view` wrote; the pair is read from its ``quant`` key, never assumed."""
        pair = PAIRS_BY_NAME[str(view["quant"])]
        gated = bool(view.get("gated", True))
        return MoEWeightsFp4(
            pair,
            view["gemm1_weights"],  # type: ignore[arg-type]
            view["gemm1_weights_scale"],  # type: ignore[arg-type]
            view["gemm2_weights"],  # type: ignore[arg-type]
            view["gemm2_weights_scale"],  # type: ignore[arg-type]
            int(view["hidden_size"]),  # type: ignore[call-overload]
            int(view["intermediate_size"]),  # type: ignore[call-overload]
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),  # type: ignore[arg-type]
            view.get("gemm1_alpha"),  # type: ignore[arg-type]
            view.get("gemm2_alpha"),  # type: ignore[arg-type]
            str(view.get("intermediate", "per_token")),
            view.get("intermediate_scale"),  # type: ignore[arg-type]
            view.get("intermediate_inv_scale"),  # type: ignore[arg-type]
        )


GEMM2_CHAIN = W4A4Chain("plain", "silu", "bf16", (), "w4a4")
#: GEMM2 of the static class: ``cvt_bf16(mul(acc, mul(alpha2[e], inv2[e])))``, the scale once per tile.
GEMM2_STATIC_CHAIN = W4A4Chain("plain", "silu", "bf16", (), "w4a4s")
#: The unscaled oracles: the raw fp32 fold of either pair.
F32_CHAIN = W4A4Chain("plain", "silu", "f32", (), "none")
MX_F32_CHAIN = Mxfp4Chain("plain", "silu", "f32", ())
#: GEMM2 of the mixed pair: the dequantised accumulator converted to bf16 (the instruction class is the launch's).
MIXED_GEMM2_CHAIN = MxChain("plain", "silu", "bf16", (), MX_MIXED_FMTS[0])


def _operands(
    name: str,
    pair: Fp4Pair,
    pts,
    alpha,
    rows_sf: Optional[int],
    rows: int,
    dense_a: bool,
) -> int:
    """The pair's scale operands and the padded row count of the scale factors: the NVFP4 pair's ``pts`` (f32, one per padded
    row) and ``alpha [E]`` (f32), none for the MX pair, which names ``rows_sf`` itself on the dense packing."""
    if pair.scaled:
        if (
            pts is None
            or alpha is None
            or pts.dtype != torch.float32
            or alpha.dtype != torch.float32
        ):
            raise ValueError(
                f"{name}: the NVFP4 chains take alpha [E] and pts (float32)"
            )
        if rows_sf is not None and rows_sf != pts.numel():
            raise ValueError(f"{name}: rows_sf must be the row count of pts")
        rows_sf = pts.numel()
    elif pts is not None or alpha is not None:
        raise ValueError(f"{name}: the MXFP4 chains take no alpha / pts")
    if dense_a:
        if rows_sf is None or rows_sf % GROUP_ALIGN or rows_sf < rows - DENSE_TAIL:
            raise ValueError(
                f"{name}: dense A takes the scale factors (and row scales) at the padded positions (rows_sf a multiple of 128)"
            )
    else:
        if rows % GROUP_ALIGN or (rows_sf is not None and rows_sf != rows):
            raise ValueError(
                f"{name}: the padded rows are a multiple of the 128-row scale-factor atom"
            )
        rows_sf = rows
    return rows_sf


def _load_kernels() -> Dict[str, Any]:
    from .blocks.compile import (
        can_implement_grouped_blockscaled,
        can_implement_swapped_blockscaled,
        get_compiled_grouped,
        get_compiled_swapped,
    )
    from .combine import get_compiled_combine, warps_per_token
    from .permute import warps_per_row
    from .permute_nvfp4 import (
        get_compiled_scatter_fp4,
        get_compiled_scatter_sf,
        scatter_lanes,
        scatter_sf_lanes,
    )

    return dict(
        can_implement_grouped_blockscaled=can_implement_grouped_blockscaled,
        can_implement_swapped_blockscaled=can_implement_swapped_blockscaled,
        get_compiled_grouped=get_compiled_grouped,
        get_compiled_swapped=get_compiled_swapped,
        get_compiled_combine=get_compiled_combine,
        warps_per_token=warps_per_token,
        warps_per_row=warps_per_row,
        get_compiled_scatter_fp4=get_compiled_scatter_fp4,
        get_compiled_scatter_sf=get_compiled_scatter_sf,
        scatter_lanes=scatter_lanes,
        scatter_sf_lanes=scatter_sf_lanes,
    )


_k = LazyNames(_load_kernels)


def _fp4_kernel(
    config: Knobs, chain: Fp4Chain, kind: str, packing: str, sf_src: str
) -> Any:
    """The compiled fp4 body of a record: the gather record at its scale source, the swapped body, or the grouped body at
    ``packing``."""
    if kind == "gather":
        return _k.get_compiled_swapped(config, chain, sf_src=sf_src)
    if kind == "swapped":
        return _k.get_compiled_swapped(config, chain)
    return _k.get_compiled_grouped(config, chain, packing=packing)


def _fp4_handle(
    device: torch.device,
    config: Knobs,
    chain: Fp4Chain,
    kind: str,
    packing: str = "dense",
    sf_src: str = "128x4",
):
    """The launch handle of an fp4 body (the compiled kernel with the device's co-resident cluster count for the record)."""
    return launch_handle(
        device,
        config,
        (kind, chain, packing, sf_src),
        _fp4_kernel,
        config,
        chain,
        kind,
        packing,
        sf_src,
    )


def _vec(t: torch.Tensor) -> torch.Tensor:
    """A scale operand as the kernel reads it: a contiguous vector (the prepared weights' alphas and the pack's row scales
    already are one)."""
    return t if (t.dim() == 1 and t.is_contiguous()) else t.reshape(-1).contiguous()


def _fp4_launch(
    pair: Fp4Pair,
    a4: torch.Tensor,
    a_sf: torch.Tensor,
    w4: torch.Tensor,
    w_sf: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: Fp4Chain,
    in_fmt: str,
    rows: int,
    E: int,
    N: int,
    *,
    pts: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
    aux: Optional[torch.Tensor] = None,
    dense_a: bool = True,
    rows_sf: int = 0,
    src: Optional[torch.Tensor] = None,
    k_div: int = 1,
    sf_src: str = "128x4",
) -> None:
    """One fp4 body launch on prepared operands (the launch of :func:`grouped_gemm_fp4` and of the gather record): the
    gather record (``src`` given: the token rows through the plan's map, their scales at ``sf_src``), the swapped body or
    the grouped body at the packing; the NVFP4 chains take ``alpha [E]`` and ``pts``, the static class ``out_sf`` / ``aux``;
    the persistent grid is the handle's cluster count clipped to the launch's tile bound. The callers validated the
    operands."""
    static = chain.needs_aux
    kind = "gather" if src is not None else ("swapped" if config.swap_ab else "grouped")
    h = _fp4_handle(
        a4.device,
        config,
        chain.at(in_fmt),
        kind,
        "dense" if dense_a else "padded",
        sf_src,
    )
    grid = min(h.clusters, launch_tile_bound(config, rows, E, N))
    a3, sfa, w3, wsf = a4.unsqueeze(-1), a_sf.reshape(-1), w4.permute(1, 2, 0), w_sf.reshape(-1)  # fmt: skip
    head: Tuple[Any, ...]
    if kind == "gather":
        head = (a3, sfa, src, k_div, w3, wsf, out)
    elif kind == "swapped":
        head = (a3, sfa, w3, wsf, out)
    else:
        head = (a3, w3, sfa, wsf, out)
    # the grouped body's raster order is a launch argument; the swapped body's is in its object
    tail = (grid, h.raster) if kind == "grouped" else (grid,)
    if not pair.scaled:
        if kind == "grouped":
            h.kernel(*head, counts, rows_sf, *tail)
        else:
            h.kernel(*head, counts, *tail)
    elif static:
        if out_sf is None:
            out_sf = _dummy_sf(
                a4.device
            )  # GEMM2 of the static class: the scale-byte operand is unread
        h.kernel(
            *head, out_sf.reshape(-1), counts, _vec(alpha), _vec(pts), _vec(aux), *tail
        )
    else:
        h.kernel(*head, counts, _vec(alpha), _vec(pts), *tail)


def grouped_gemm_fp4(
    a4: torch.Tensor,
    a_sf: torch.Tensor,
    w4: torch.Tensor,
    w_sf: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: Fp4Chain,
    in_fmt: str,
    *,
    pts: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    dense_a: bool = False,
    rows_sf: Optional[int] = None,
    out_sf: Optional[torch.Tensor] = None,
    aux: Optional[torch.Tensor] = None,
    src: Optional[torch.Tensor] = None,
    k_div: int = 1,
    sf_layout: Optional[str] = None,
) -> torch.Tensor:
    """``out[p] = chain(a4[p] @ w4[e(p)].T, ...)`` on the expert segments ``counts`` (int32 ``[E]``, group order = expert order),
    the block scales of both operands applied inside the instruction. Padded packing (default): ``a4 [rows, K/2]`` uint8 with
    every group's rows starting at a multiple of 128, ``a_sf`` its 128x4 swizzled scales. Dense packing (``dense_a``): the
    groups' rows back to back (any row count), ``a_sf`` (and ``pts``) at the padded positions of ``rows_sf`` rows, ``out
    [rows]`` dense. ``w4 [E, N, K/2]`` uint8, ``w_sf [E, sf_numel(N, K)]``. The NVFP4 chains take ``alpha [E]`` f32 and ``pts``
    (f32, the padded rows; GEMM2 of the static class leaves it unread) and the static class ``aux`` (``gs2 [E]`` for GEMM1,
    which writes the packed codes to ``out`` (uint8) and the scale bytes to ``out_sf``; ``inv2 [E]`` for GEMM2); the MX chains
    take none. The record selects the body: a dense record runs the grouped body on either packing, a swapped record
    (``config.swap_ab``: :mod:`.blocks.swap`) the swapped decode body on the dense packing (``a4`` carries the token tile's
    rows past the live rows: ``DENSE_TAIL``). The gather record (``config.gather_a``): ``a4 [T, K/2]`` the unpermuted token
    rows, ``a_sf`` their scale factors (the flat 128x4 buffer of the ``T`` rows, or the linear ``[T, K/block]`` bytes), ``pts
    [T]`` (or one entry: the unit scale), ``src [>= Tk]`` int32 the plan's slot of every dense row (``src[p] // k_div`` its
    token), ``out`` given (it sizes the dense rows); no permuted tensor, the same bits. ``sf_layout`` names the gather record's
    scale layout where the shape does not (:func:`gather_sf_layout`)."""
    from .api import _check_grouped_chain, _check_grouped_operands, _grouped_out

    pair = pair_of(chain)
    name = f"grouped_gemm_{pair.name}"
    if config.gather_a:
        return _grouped_gemm_fp4_gather(
            pair, a4, a_sf, w4, w_sf, counts, config, chain, in_fmt,
            pts, alpha, out, out_sf, aux, src, k_div, sf_layout, rows_sf,
        )  # fmt: skip
    if src is not None:
        raise ValueError(f"{name}: src belongs to the gather record (config.gather_a)")
    rows, Kh, E, N, _ = _check_grouped_operands(
        name,
        a4,
        w4,
        counts,
        a4.shape[0] if a4.dim() == 2 else None,
        None,
        torch.uint8,
        ALIGN_FP4 // 2,
        "packed fp4 uint8",
        n_align=8,
    )
    K = 2 * Kh
    rows_sf = _operands(name, pair, pts, alpha, rows_sf, rows, dense_a)
    if a_sf.dtype != torch.uint8 or a_sf.numel() < pair.sf_numel(rows_sf, K):
        raise ValueError(
            f"{name}: a_sf must hold the 128x4 swizzled scales of the (padded) rows"
        )
    if w_sf.dtype != torch.uint8 or w_sf.numel() < E * pair.sf_numel(N, K):
        raise ValueError(
            f"{name}: w_sf must hold E consecutive 128x4 swizzled scale buffers of (N, K)"
        )
    if pair.scaled and alpha.numel() != E:
        raise ValueError(f"{name}: alpha must hold one f32 per expert")
    if in_fmt not in pair.fmts or chunks_for(N, K, in_fmt) != 1:
        raise ValueError(
            f"{name}: the grouped class takes unsplit expert weight shapes"
        )
    if config.swap_ab:
        if not dense_a:
            raise ValueError(
                f"{name}: the swapped decode body reads the dense packing (dense_a=True)"
            )
        if not _k.can_implement_swapped_blockscaled(config, N, K, E, chain, in_fmt):
            raise ValueError(
                f"{name}: swapped record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
            )
    elif not _k.can_implement_grouped_blockscaled(config, N, K, E, chain, in_fmt):
        raise ValueError(
            f"{name}: record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
        )
    n_out = _check_grouped_chain(name, N, chain)
    static = chain.needs_aux
    if chain.static_quant:
        n_out //= 2  # the packed codes: two per byte
        out_dtype = torch.uint8
        if (
            out_sf is None
            or out_sf.dtype != torch.uint8
            or out_sf.numel() < pair.sf_numel(rows_sf, N // 2 if chain.pair else N)
        ):
            raise ValueError(
                f"{name}: the static class's GEMM1 takes out_sf, the flat 128x4 scale bytes of the (padded) rows"
            )
    else:
        out_dtype = _OUT[chain.out_fmt]
    if static:
        if aux is None or aux.dtype != torch.float32 or aux.numel() != E:
            raise ValueError(
                f"{name}: the static class's chains take aux, the per-expert static scales [E] (gs2 for GEMM1, inv2 for GEMM2)"
            )
        if out_sf is None:
            out_sf = _dummy_sf(
                a4.device
            )  # GEMM2 of the static class: the scale-byte operand is unread
    if out is None:
        out = torch.zeros(rows, n_out, dtype=out_dtype, device=a4.device)
    out = _grouped_out(name, out, rows, n_out, out_dtype, a4.device)
    if rows == 0:
        return out
    # the persistent grid clipped to the launch's tile bound (a placement: which cluster computes a tile, never the bits)
    _fp4_launch(
        pair, a4, a_sf, w4, w_sf, out, counts, config, chain, in_fmt, rows, E, N,
        pts=pts, alpha=alpha, out_sf=out_sf, aux=aux, dense_a=dense_a, rows_sf=rows_sf,
    )  # fmt: skip
    return out


#: The layouts of a pre-quantized input's scale factors: the 128x4 swizzled buffer of the mode's quantizers, the row-major bytes.
SF_LAYOUTS = ("128x4", "linear")


def sf_layouts_collide(T: int, K: int, block: int) -> bool:
    """Whether the linear ``[T, K / block]`` layout and the 128x4 swizzled buffer of ``T`` rows have one shape: the quantizers
    return the buffer as ``(rows padded to 128, scale columns padded to 4)``, so a tensor of that shape names no layout."""
    return T % GROUP_ALIGN == 0 and (K // block) % 4 == 0


def gather_sf_layout(
    a_sf: torch.Tensor,
    T: int,
    K: int,
    block: int = 16,
    layout: Optional[str] = None,
    cls: str = NV_CLASS,
) -> str:
    """The layout of a scale operand of ``T`` rows of ``K`` elements: ``layout`` names it (``"128x4"`` / ``"linear"``, a tensor
    named linear with another shape refused); otherwise the shape does where it can -- a tensor that is not a 2-D
    ``[T, K / block]`` is the flat 128x4 buffer, a 2-D ``[T, K / block]`` tensor the linear layout, unless the swizzled buffer
    has that shape too (:func:`sf_layouts_collide`), which is refused: the caller names the layout
    (``QuantConfig(swizzled_scale_factors=...)`` through the layer, ``sf_layout`` on the op) or flattens the quantizer's
    buffer. An empty pack (``T == 0``) is the 128x4 buffer of no rows: the two layouts are one empty buffer."""
    if layout is not None:
        if layout not in SF_LAYOUTS:
            raise ValueError(f"sf_layout must be one of {SF_LAYOUTS}, not {layout!r}")
        if layout == "linear" and (
            a_sf.dim() != 2 or tuple(a_sf.shape) != (T, K // block)
        ):
            raise unsupported(
                cls,
                f"scale factors of shape {tuple(a_sf.shape)} for {T} rows named linear",
                f"the linear [{T}, {K // block}] layout (QuantConfig(swizzled_scale_factors=False) through the layer, "
                "sf_layout='linear' on the op)",
            )
        return layout
    if T == 0 or a_sf.dim() != 2 or tuple(a_sf.shape) != (T, K // block):
        return "128x4"
    if sf_layouts_collide(T, K, block):
        raise unsupported(
            cls,
            f"scale factors of shape {tuple(a_sf.shape)} for {T} rows without a named layout",
            "the layout named by the caller: the quantizer's 128x4 buffer flattened, or QuantConfig(swizzled_scale_factors="
            "True / False) through the layer and sf_layout='128x4' / 'linear' on the op (at a row count that is a multiple "
            "of 128 the swizzled buffer and the linear layout have the same shape)",
        )
    return "linear"


def gather_form_ok(K: int, sf_src: str, block: int = 16) -> bool:
    """The gather form's host rule on the scale source: the linear layout's rows are whole 4-byte groups (the kernel copies a
    row's scales in 4-byte groups), the 128x4 buffer always."""
    return sf_src == "128x4" or (K // block) % 4 == 0


def _grouped_gemm_fp4_gather(
    pair, a4, a_sf, w4, w_sf, counts, config, chain, in_fmt, pts, alpha, out, out_sf, aux, src, k_div, sf_layout=None,
    rows_sf=None,
) -> torch.Tensor:  # fmt: skip
    """The gather record of :func:`grouped_gemm_fp4` (its operand rules, the token rows in place of the permuted rows; the
    swapped body). The static class's ``out_sf`` holds the scale bytes of the padded rows: ``rows_sf`` (the caller's padded
    row count, a multiple of 128) or the plan's host bound :func:`padded_rows_bound` of ``T x k_div`` slots, which the
    caller's count may not undercut."""
    from .api import _check_grouped_chain, _check_grouped_operands, _grouped_out

    name = f"grouped_gemm_{pair.name}"
    T, Kh, E, N, _ = _check_grouped_operands(
        name,
        a4,
        w4,
        counts,
        a4.shape[0] if a4.dim() == 2 else None,
        None,
        torch.uint8,
        ALIGN_FP4 // 2,
        "packed fp4 uint8",
        n_align=8,
    )
    K = 2 * Kh
    if out is None or src is None:
        raise ValueError(
            f"{name}: the gather record reads the token rows through the plan's map: src and out"
        )
    if (
        src.dtype != torch.int32
        or src.dim() != 1
        or not src.is_contiguous()
        or k_div < 1
    ):
        raise ValueError(
            f"{name}: src must be a contiguous int32 [>= Tk] and k_div >= 1"
        )
    sf_src = gather_sf_layout(a_sf, T, K, pair.block, sf_layout, pair.cls)
    if a_sf.dtype != torch.uint8 or (
        sf_src == "128x4" and a_sf.numel() < pair.sf_numel(T, K)
    ):
        raise ValueError(
            f"{name}: a_sf must hold the 128x4 swizzled scales of the T token rows or their linear [T, K/{pair.block}] bytes"
        )
    if not gather_form_ok(K, sf_src, pair.block):
        raise ValueError(
            f"{name}: the linear scale layout's rows must be whole 4-byte groups (K a multiple of {4 * pair.block})"
        )
    if w_sf.dtype != torch.uint8 or w_sf.numel() < E * pair.sf_numel(N, K):
        raise ValueError(
            f"{name}: w_sf must hold E consecutive 128x4 swizzled scale buffers of (N, K)"
        )
    if pair.scaled:
        if (
            alpha is None
            or pts is None
            or alpha.dtype != torch.float32
            or alpha.numel() != E
            or pts.dtype != torch.float32
            or pts.numel() not in (1, T)
        ):
            raise ValueError(
                f"{name}: alpha [E] and pts [T] (or one unit entry) must be float32"
            )
    elif pts is not None or alpha is not None:
        raise ValueError(f"{name}: the MXFP4 chains take no alpha / pts")
    if in_fmt not in pair.fmts or chunks_for(N, K, in_fmt) != 1:
        raise ValueError(
            f"{name}: the grouped class takes unsplit expert weight shapes"
        )
    if not config.swap_ab or not _k.can_implement_swapped_blockscaled(
        config, N, K, E, chain, in_fmt
    ):
        raise ValueError(
            f"{name}: gather record {config.describe()} does not implement ({N}, {K}, {E}, {chain}, {in_fmt})"
        )
    n_out = _check_grouped_chain(name, N, chain)
    static = chain.needs_aux
    rows = out.shape[0]
    if chain.static_quant:
        n_out //= 2
        out_dtype = torch.uint8
        # the epilogue stores a dense row's scale bytes at its padded position, below the plan's padded row count, whose host
        # bound is the groups of T x k_div slots each padded to the atom (the box form takes the count from its pts rows)
        bound = padded_rows_bound(T, k_div, E)
        rows_sf = bound if rows_sf is None else rows_sf
        if rows_sf % GROUP_ALIGN or rows_sf < bound:
            raise ValueError(
                f"{name}: rows_sf must be a multiple of 128 and at least the padded-row bound {bound} of T x k_div slots"
            )
        n_sf = N // 2 if chain.pair else N
        if (
            out_sf is None
            or out_sf.dtype != torch.uint8
            or out_sf.numel() < pair.sf_numel(rows_sf, n_sf)
        ):
            raise ValueError(
                f"{name}: the static class's GEMM1 takes out_sf, the flat 128x4 scale bytes of the padded rows "
                f"(at least sf_numel({rows_sf}, {n_sf}) bytes)"
            )
    else:
        out_dtype = _OUT[chain.out_fmt]
    if static:
        if aux is None or aux.dtype != torch.float32 or aux.numel() != E:
            raise ValueError(
                f"{name}: the static class's chains take aux, the per-expert static scales [E]"
            )
        if out_sf is None:
            out_sf = _dummy_sf(a4.device)
    out = _grouped_out(name, out, rows, n_out, out_dtype, a4.device)
    if rows == 0 or T == 0:
        return out
    _fp4_launch(
        pair, a4, a_sf, w4, w_sf, out, counts, config, chain, in_fmt, rows, E, N,
        pts=pts, alpha=alpha, out_sf=out_sf, aux=aux, src=src, k_div=k_div, sf_src=sf_src,
    )  # fmt: skip
    return out


def permute_plan_dual(
    experts: torch.Tensor, E: int, pad_align: int = GROUP_ALIGN
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(counts [E], dest [T, k], src [Tk], dest_pad [T, k], pos_pad [Tk])``: the dense plan plus the positions with the
    groups padded to ``pad_align`` rows (``dest_pad[i]``) and the padded position of every dense row (``pos_pad[p]``): the
    dense packing's row map (A rows dense, scale factors and row scales at the padded positions) -- the shared plan's dual
    form (:func:`.api.permute_plan` with ``dual=True``)."""
    counts, dest, src, dest_pad, pos_pad = permute_plan(
        experts, E, dual=True, pad_align=pad_align
    )
    return counts, dest, src, dest_pad, pos_pad


_GS_INV: Dict[
    Tuple[str, float], torch.Tensor
] = {}  # the recipe's constant per device, filled once (no fill launch per call)
_DUMMY_SF: Dict[str, torch.Tensor] = {}
_UNIT: Dict[str, torch.Tensor] = {}


def _device_key(device) -> str:
    """The constant caches' key for a device: its indexed form, so ``"cuda"`` and the current device's ``"cuda:N"`` share one
    entry."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return str(d)


def _const(cache: Dict, key, fill, device: torch.device) -> torch.Tensor:
    """A constant operand of the kernels, cached per device: built on the host (``fill(cpu)``) and copied with a blocking copy,
    so its bytes are on the device before the call returns and any stream may read it -- a captured call too (no node in the
    graph). First met under CUDA graph capture it is a device fill (``fill(device)``): a graph-pool allocation whose fill runs
    only at replay, so it serves that capture alone and is not cached (as the bf16 op's ``_const``)."""
    t = cache.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return fill(device)
        t = cache[key] = fill(torch.device("cpu")).to(device)
    return t


def _dummy_sf(device: torch.device) -> torch.Tensor:
    """A 512-byte placeholder for an unread scale-byte operand (one atom: the kernels' alignment assumption), per device."""
    return _const(
        _DUMMY_SF,
        _device_key(device),
        lambda d: torch.zeros(512, dtype=torch.uint8, device=d),
        device,
    )


def _placeholder(device: torch.device) -> torch.Tensor:
    """A one-element fp32 tensor for the scatter kernels' unread per-token-scale operands, per device (no fill per call)."""
    return _const(
        _UNIT,
        _device_key(device),
        lambda d: torch.ones(1, dtype=torch.float32, device=d),
        device,
    )


def route_plan_dual(
    logits: torch.Tensor,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    pad_align: int = GROUP_ALIGN,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """:func:`.api.route` and :func:`permute_plan_dual` in one launch for small token counts (:func:`.api.route_plan` with
    ``dual=True``): ``(experts [T, k] int32, weights [T, k] f32, counts [E] int32, dest [T, k] int32, src [T k] int32,
    dest_pad [T, k] int32, pos_pad [T k] int32)``, the same bits and integers as the two calls."""
    from .api import route_plan

    return route_plan(
        logits, top_k, method, bias, scaling_factor, dual=True, pad_align=pad_align
    )


def _gs_inv(gs_inv: Union[float, torch.Tensor], device: torch.device) -> torch.Tensor:
    """The recipe's inverse base scale as a one-element fp32 tensor on ``device``: a tensor as given; a number from the
    per-device cache (:func:`_const`: never a tensor first made under CUDA graph capture)."""
    if isinstance(gs_inv, torch.Tensor):
        return gs_inv.to(device=device, dtype=torch.float32).reshape(1)
    v = float(gs_inv)
    return _const(
        _GS_INV,
        (_device_key(device), v),
        lambda d: torch.full((1,), v, dtype=torch.float32, device=d),
        device,
    )


def quantize_activations(
    x: torch.Tensor, gs_inv: Union[float, torch.Tensor] = TE_GS_INV, out=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The NVFP4 class's per-token quantization of bf16 rows ``[M, K]``: ``(x4 [M, K/2] uint8, sf flat 128x4 uint8, pts [M]
    f32)``; ``gs_inv`` the recipe's inverse base scale. ``out = (x4, sf)`` preallocated buffers."""
    from ...batch_invariant.quantization.nvfp4_api import nvfp4_quantize_rows

    M, K = x.shape
    dev = x.device
    if M == 0:
        return (
            torch.empty(0, K // 2, dtype=torch.uint8, device=dev),
            torch.zeros(NVFP4.sf_numel(0, K), dtype=torch.uint8, device=dev),
            torch.empty(0, dtype=torch.float32, device=dev),
        )
    x4, sf, pts = nvfp4_quantize_rows(
        x, _gs_inv(gs_inv, dev), "128x4", per_token=True, out=out, enable_pdl=True
    )
    return x4, sf.reshape(-1), pts


def quantize_rows_mx(
    x: torch.Tensor, out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The MXFP4 class's quantization of bf16 rows ``[M, K]`` (``mxfp4.quant``: one UE8M0 scale per 32 elements of a row):
    ``(x4 [M, K/2] uint8, sf flat 128x4 uint8)``. ``out = (x4, sf)`` preallocated buffers."""
    from ...batch_invariant.quantization.mx_api import mx_quantize

    M, K = x.shape
    dev = x.device
    if M == 0:
        return (
            torch.empty(0, K // 2, dtype=torch.uint8, device=dev),
            torch.zeros(MXFP4.sf_numel(0, K), dtype=torch.uint8, device=dev),
        )
    x4, sf = mx_quantize(x, MXFP4_FORMAT, "128x4", out=out, enable_pdl=True)
    return x4, sf.reshape(-1)


def quantize_rows_mapped(
    x: torch.Tensor,
    gs_inv: Union[float, torch.Tensor],
    row_map: torch.Tensor,
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The NVFP4 class's per-token quantization of bf16 rows ``[M, K]`` with the scale factors and per-row scales stored at
    row ``row_map[r]`` of the padded buffers ``out = (x4 [M, K/2], sf_pad flat 128x4, pts_pad)`` while the codes stay at row
    ``r``: the quantizer's kernel term for term (:mod:`.blocks.quant_nvfp4`), one launch in place of the quantizer and the
    scale re-tiling."""
    from ...batch_invariant.quantization.mx_api import device_sm_count
    from ...batch_invariant.quantization.nvfp4_schedule import default_config
    from .blocks.quant_nvfp4 import get_compiled_rowmap_quant

    M, K = x.shape
    x4, sf_pad, pts_pad = out
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0
    assert x4.shape == (M, K // 2) and x4.is_contiguous() and x4.dtype == torch.uint8
    assert (
        row_map.dtype == torch.int32
        and row_map.is_contiguous()
        and row_map.numel() >= M
    )
    assert sf_pad.dtype == torch.uint8 and pts_pad.dtype == torch.float32
    if M == 0:
        return
    cfg = default_config(
        K, M, x.element_size(), True, device_sm_count(x.device), "128x4"
    )
    kernel = get_compiled_rowmap_quant("bf16", K, cfg, True)
    kernel(x, x4, sf_pad.view(-1), pts_pad, _gs_inv(gs_inv, x.device), row_map, M, K)


def quantize_scatter_rows(
    x: torch.Tensor,
    gs_inv: Union[float, torch.Tensor],
    dest: torch.Tensor,
    dest_pad: torch.Tensor,
    k: int,
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The NVFP4 class's per-token quantization of bf16 rows ``[T, K]`` stored into the grouped layout: token ``t``'s codes at
    the dense rows ``dest[t k + j]`` of ``out[0]`` (``[rows, K/2]``), its scale factors and per-token scale at the padded
    positions ``dest_pad[t k + j]`` of ``out[1]`` (flat 128x4) and ``out[2]``: the quantizer's kernel term for term
    (:mod:`.blocks.quant_nvfp4`), one launch in place of the quantizer and the fp4 scatter."""
    from ...batch_invariant.quantization.mx_api import device_sm_count
    from ...batch_invariant.quantization.nvfp4_schedule import default_config
    from .blocks.quant_nvfp4 import get_compiled_scatter_quant

    T, K = x.shape
    x_perm4, sf_pad, pts_pad = out
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0
    assert (
        x_perm4.dtype == torch.uint8
        and x_perm4.is_contiguous()
        and x_perm4.shape[1] == K // 2
    )
    assert (
        dest.dtype == torch.int32
        and dest_pad.dtype == torch.int32
        and dest.numel() >= T * k
    )
    assert dest.is_contiguous() and dest_pad.is_contiguous()
    assert sf_pad.dtype == torch.uint8 and pts_pad.dtype == torch.float32
    if T == 0:
        return
    cfg = default_config(
        K, T, x.element_size(), True, device_sm_count(x.device), "128x4"
    )
    kernel = get_compiled_scatter_quant("bf16", K, cfg, k, True)
    kernel(
        x,
        x_perm4,
        sf_pad.view(-1),
        pts_pad,
        _gs_inv(gs_inv, x.device),
        dest,
        dest_pad,
        T,
        K,
    )


def _part(device: torch.device) -> Tuple[int, str]:
    return part_of(device)


def _input(
    w: MoEWeightsFp4, x, gs_inv, dev_hint=None, sf_layout: Optional[str] = None
) -> Tuple[
    int,
    torch.device,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    str,
    bool,
]:
    """The op's input forms: bf16 rows, or the pair's pre-quantized pack -- NVFP4 ``(x4, x_sf, x_pts or None)``, MXFP4 ``(x4,
    x_sf)``. Returns ``(T, device, x_rows, x4, x_sf, x_pts, sf_src, unit_pts)``: a bf16 input keeps its rows (``x4`` None where
    the dense path quantizes straight into the grouped layout); a pack's scale factors are the class's flat 128x4 buffer or the
    row-major ``[T, H / block]`` bytes (``sf_src``: ``sf_layout`` as given, else from the shape where it names one layout);
    ``unit_pts``: the NVFP4 per-tensor pack without per-token scales."""
    pair, H = w.pair, w.hidden_size
    op = f"fused_moe_{pair.name}"
    if isinstance(x, torch.Tensor):
        if (
            x.dtype != torch.bfloat16
            or x.dim() != 2
            or x.stride(1) != 1
            or x.stride(0) % 8
        ):
            raise TypeError(
                f"{op}: x must be a row-major, 16-byte-aligned bf16 [T, H] or a quantized pack"
            )
        if x.shape[1] != H:
            raise ValueError(f"{op}: x has H = {x.shape[1]}, the weights H = {H}")
        x = x.contiguous()
        if x.data_ptr() % 16:
            raise unsupported(
                pair.cls,
                "bf16 rows at a storage offset that is not a multiple of 16 bytes",
                "16-byte aligned rows (the kernels' vector loads and TMA boxes)",
            )
        return x.shape[0], x.device, x, None, None, None, "128x4", False
    if pair.scaled:
        x4, x_sf, x_pts = x
    else:
        if len(x) == 3 and x[2] is not None:
            raise TypeError(f"{op}: the MXFP4 pair takes no per-token scale")
        x4, x_sf = x[0], x[1]
        x_pts = None
    T = x4.shape[0]
    sf_src = gather_sf_layout(x_sf, T, H, pair.block, sf_layout, pair.cls)
    x_sf = x_sf.contiguous().reshape(-1)
    unit_pts = x_pts is None
    if (
        x4.dtype != torch.uint8
        or tuple(x4.shape) != (T, H // 2)
        or not x4.is_contiguous()
        or x_sf.dtype != torch.uint8
        or (sf_src == "128x4" and x_sf.numel() < pair.sf_numel(T, H))
        or (
            not unit_pts
            and (tuple(x_pts.shape) != (T,) or x_pts.dtype != torch.float32)
        )
    ):
        raise TypeError(
            f"{op}: the quantized pack is (x4 [T, H/2] uint8, x_sf 128x4 uint8 of >= sf_numel(T, H) bytes or the linear "
            f"[T, H/{pair.block}] bytes"
            + (", x_pts [T] f32 or None)" if pair.scaled else ")")
        )
    if x4.data_ptr() % 16 or x_sf.data_ptr() % 16:
        raise unsupported(
            pair.cls,
            "packed codes or scale bytes at a storage offset that is not a multiple of 16 bytes",
            "16-byte aligned codes and scale bytes (the kernels' vector loads and TMA boxes)",
        )
    if pair.scaled and unit_pts:
        x_pts = _gs_inv(
            1.0, x4.device
        )  # a placeholder the unit-scale scatter never reads
    elif pair.scaled:
        x_pts = (
            x_pts.contiguous()
        )  # a strided per-token scale is read as given by the scatter: a placement
    return T, x4.device, None, x4, x_sf, x_pts, sf_src, unit_pts


def fused_moe_fp4(
    x: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: MoEWeightsFp4,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    in_fmts: Optional[Tuple[str, str]] = None,
    path: str = "dense",
    gs_inv: Union[float, torch.Tensor] = TE_GS_INV,
    return_intermediates: bool = False,
    swap: Optional[bool] = None,
    plan: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    round_scales: bool = False,
    gather: Optional[bool] = None,
    sf_layout: Optional[str] = None,
):
    """The fp4 MoE FFN for pre-routed tokens: ``x`` bf16 ``[T, H]`` (quantized here by the pair's class; ``gs_inv`` the NVFP4
    recipe's inverse base scale) or the pair's quantized pack (NVFP4 ``(x4 [T, H/2] uint8, x_sf, x_pts [T] f32 or None: unit
    scales, the per-tensor pack)``, MXFP4 ``(x4, x_sf)``; ``x_sf`` the 128x4 swizzled buffer or the linear ``[T, H/block]``
    bytes; the buffer's pad columns are zero, as every FlashInfer quantizer writes them: the scatter and the gather copy a
    row's last 4-byte group as given); ``token_selected_experts [T, k]`` int32, ``token_final_scales [T, k]`` f32 (``round_scales``: the combine takes their
    bf16 rounding, the pre-routed packed-word convention); the output bf16 ``[T, H]`` in token order. ``in_fmts`` names the
    instruction class per GEMM (default: the pair's rule on this part); ``config1`` / ``config2`` the grouped records (any
    record of the space gives the same bits; the dispatcher's pick by default); ``path`` the packing of the expanded rows
    (``"dense"`` / ``"padded"``: placements); ``swap`` the dense path's body -- None: the dispatcher's ranking, True / False: the
    swapped / the grouped body on both GEMMs (placements); explicit records select the body themselves (a swapped record needs
    the dense path). ``plan``: the dense packing's precomputed plan, the outputs of :func:`route_plan_dual` after the ids and
    weights (one launch saved). ``gather``: GEMM1's swapped records in their gather form (the body reads the token rows through
    the plan's map: no scatter launch, no permuted tensor) -- False excludes it, True forces it, None (the default) applies the
    structural rule (:func:`~.schedule_fp4.gather_form_rule`); an explicit ``config1`` selects the form itself (``gather_a``).
    ``sf_layout`` names a pack's scale layout (``"128x4"`` / ``"linear"``); without it the shape must name one
    (:func:`gather_sf_layout`: a 2-D ``[T, H / block]`` tensor at a row count that is a multiple of 128 is refused)."""
    if w.pair.mixed:
        raise TypeError(
            "fused_moe_fp4 runs the fp4 pairs; the MXFP4 x MXFP8 pair's op is api_mxfp4w8a.fused_moe_mxfp4w8a"
        )
    from .combine import get_compiled_combine, warps_per_token
    from .permute import warps_per_row
    from .permute_nvfp4 import get_compiled_scatter_fp4, scatter_lanes
    from .schedule_fp4 import default_configs_fp4, gather_form_rule

    pair = w.pair
    op = f"fused_moe_{pair.name}"
    if path not in PATHS:
        raise ValueError(f"{op}: path must be one of {PATHS}, not {path!r}")
    H, I, E = w.hidden_size, w.intermediate_size, w.num_experts
    T, dev, x_rows, x4, x_sf, x_pts, sf_src, unit_pts = _input(
        w, x, gs_inv, sf_layout=sf_layout
    )
    k = token_selected_experts.shape[1]
    check_shape_fp4(pair, H, I, E, k, w.gated)
    if (
        tuple(token_selected_experts.shape) != (T, k)
        or token_selected_experts.dtype != torch.int32
    ):
        raise TypeError(f"{op}: token_selected_experts must be an int32 [T, k]")
    if (
        tuple(token_final_scales.shape) != (T, k)
        or token_final_scales.dtype != torch.float32
    ):
        raise TypeError(f"{op}: token_final_scales must be a float32 [T, k]")
    experts = token_selected_experts.contiguous()
    scales = token_final_scales.contiguous()
    if out is None:
        out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    if (
        tuple(out.shape) != (T, H)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
        or out.stride(0) % 8
    ):
        raise ValueError(f"{op}: out must be a row-major, 16-byte-aligned bf16 [T, H]")
    sm_count, family = _part(dev)
    in1, in2 = in_fmts if in_fmts is not None else w.classes(family)
    chain1, chain2 = w.chain1, w.chain2
    static = w.static
    scaled = pair.scaled
    # a bf16 NVFP4 input on the dense path is quantized straight into the grouped layout (the fused form); the MX pair's
    # quantizer stores the class's layout and the scatter moves it (two launches)
    fused_in = (
        x_rows is not None and scaled and path == "dense" and INPUT_SCATTER == "fused"
    )
    gsi = _gs_inv(gs_inv, dev)
    if x_rows is not None and not fused_in:
        x4, x_sf, x_pts = pair.quantize(x_rows, gsi)
    explicit1 = (
        config1 is not None
    )  # the caller's record (a gather record is a request for the form)
    if config1 is None or config2 is None:
        ranked1 = config1 is None
        c1, c2 = default_configs_fp4(
            T,
            k,
            H,
            I,
            E,
            (in1, in2),
            chain1,
            chain2,
            sm_count,
            family,
            swap=swap if path == "dense" else False,
            gather=(gather or False) if path == "dense" else False,
        )
        config1, config2 = config1 or c1, config2 or c2
        if ranked1 and gather is None and path == "dense":
            # the rule: a standalone scatter launch (a pre-quantized input; the MX pair's quantizer) folded into the narrow
            # swapped tile's gather form
            config1 = gather_form_rule(config1, x4 is not None)
    if path != "dense" and (swap or config1.swap_ab or config2.swap_ab):
        raise ValueError(
            f"{op}: the swapped decode body reads the dense packing (path='dense')"
        )
    if config2.gather_a or (path != "dense" and (gather or config1.gather_a)):
        raise ValueError(
            f"{op}: the gather form is GEMM1's on the dense packing (path='dense')"
        )
    gather1 = config1.gather_a
    if gather1 and not gather_form_ok(
        H, "128x4" if x_rows is not None else sf_src, pair.block
    ):
        if T and (explicit1 or gather):
            # the caller asked for the gather form (a gather record, or gather=True): the layout's rule is reported
            raise unsupported(
                pair.cls,
                f"the gather form with the linear scale layout at H = {H} ({H // pair.block} scale bytes per row)",
                f"the linear layout's rows as whole 4-byte groups (H a multiple of {4 * pair.block}) or the 128x4 "
                "layout for the gather form; the dispatcher's own pick takes the box form here",
            )
        # the dispatcher's pick, or an empty batch (no row to transport): the box form through the scatter
        config1 = dataclasses.replace(config1, gather_a=False)
        gather1 = False
    if path != "dense" and plan is not None:
        raise ValueError(
            f"{op}: the precomputed plan is the dense packing's (path='dense')"
        )
    if path == "dense":
        # the records' acceptance (the grouped entry's rule, for a caller's records)
        for cfg, N, K, chain, fmt in (
            (config1, w.w1.shape[1], H, chain1, in1),
            (config2, H, I, chain2, in2),
        ):
            if fmt not in pair.fmts or chunks_for(N, K, fmt) != 1:
                raise ValueError(
                    f"{op}: the grouped class takes unsplit expert weight shapes"
                )
            if cfg.gather_a:
                ok = cfg.swap_ab and _k.can_implement_swapped_blockscaled(
                    cfg, N, K, E, chain, fmt
                )
                kind = "gather record"
            elif cfg.swap_ab:
                ok = _k.can_implement_swapped_blockscaled(cfg, N, K, E, chain, fmt)
                kind = "swapped record"
            else:
                ok = _k.can_implement_grouped_blockscaled(cfg, N, K, E, chain, fmt)
                kind = "record"
            if not ok:
                raise ValueError(
                    f"{op}: {kind} {cfg.describe()} does not implement ({N}, {K}, {E}, {chain}, {fmt})"
                )
        return _fused_moe_fp4_run(
            w, x_rows, x4, x_sf, x_pts, sf_src, unit_pts, experts, scales, out, config1, config2, in1, in2,
            T, H, k, E, I, gsi, plan=plan, round_scales=round_scales, return_intermediates=return_intermediates,
        )  # fmt: skip
    Tk = T * k
    Tk_pad = padded_rows_bound(T, k, E)
    pcols_h = pair.sf_cols(H)
    unit = _placeholder(dev)
    wpr = warps_per_row(Tk, H // 32, sm_count)
    # the scale operands of the GEMMs: the NVFP4 pair's (pts rows, alphas), none for the MX pair
    alpha1, alpha2 = (w.w1_alpha, w.w2_alpha) if scaled else (None, None)
    combine = get_compiled_combine(
        k, "bf16", "bf16", warps_per_token(T, H, sm_count), round_scales
    )

    def scatter(
        x4_, x_sf_, x_pts_, dest_, dest_pad_, x_perm4_, x_perm_sf_, x_perm_pts_
    ):
        """The fp4 scatter of a quantized input to the grouped layout (its scale rows re-tiled into the atoms)."""
        if scaled:
            kern = get_compiled_scatter_fp4(
                k, wpr, sf_src, unit_pts, **scatter_lanes(H // 2, pcols_h, wpr)
            )
            kern(
                x4_,
                x_sf_,
                x_pts_,
                dest_,
                dest_pad_,
                x_perm4_,
                x_perm_sf_,
                x_perm_pts_,
                Tk,
                H // 2,
                pcols_h,
            )
        else:
            kern = get_compiled_scatter_fp4(
                k,
                wpr,
                sf_src,
                False,
                True,
                MX_BLOCK,
                **scatter_lanes(H // 2, pcols_h, wpr),
            )
            kern(
                x4_,
                x_sf_,
                unit,
                dest_,
                dest_pad_,
                x_perm4_,
                x_perm_sf_,
                unit,
                Tk,
                H // 2,
                pcols_h,
            )

    # the padded packing: every group's rows start at a 128-row atom; the padding rows are zero (computed, never stored)
    from .api import PLAN_MULTI_MIN_N
    from .permute import (
        PLAN_BLOCK,
        get_compiled_plan,
        get_compiled_plan_multi,
        plan_threads,
    )

    n = Tk
    dest = torch.empty(n, dtype=torch.int32, device=dev)
    src = torch.full((Tk_pad,), -1, dtype=torch.int32, device=dev)
    counts = torch.empty(E, dtype=torch.int32, device=dev)
    flat = experts.reshape(-1)
    if n == 0:
        counts.zero_()
    elif n >= PLAN_MULTI_MIN_N:
        nb = (n + PLAN_BLOCK - 1) // PLAN_BLOCK
        hist_blk = torch.empty(nb * E, dtype=torch.int32, device=dev)
        get_compiled_plan_multi(GROUP_ALIGN, False)(
            flat, counts, dest, src, dest, dest, hist_blk, n, E, nb
        )
    else:
        get_compiled_plan(GROUP_ALIGN, False, threads=plan_threads(n))(
            flat, counts, dest, src, dest, dest, n, E
        )
    dest = dest.view(T, k)
    x_perm4 = torch.zeros(Tk_pad, H // 2, dtype=torch.uint8, device=dev)
    x_perm_sf = torch.zeros(pair.sf_numel(Tk_pad, H), dtype=torch.uint8, device=dev)
    x_perm_pts = (
        torch.zeros(Tk_pad, dtype=torch.float32, device=dev) if scaled else None
    )
    if Tk:
        flat_dest = dest.reshape(-1)
        scatter(x4, x_sf, x_pts, flat_dest, flat_dest, x_perm4, x_perm_sf, x_perm_pts)
    if static:
        # the static class on the padded packing: the codes and scale bytes at the padded rows (the group's rows start at an atom)
        a4 = torch.zeros(Tk_pad, I // 2, dtype=torch.uint8, device=dev)
        a_sf = torch.zeros(pair.sf_numel(Tk_pad, I), dtype=torch.uint8, device=dev)
        grouped_gemm_fp4(
            x_perm4, x_perm_sf, w.w1, w.w1_sf, counts, config1, chain1, in1,
            pts=x_perm_pts, alpha=alpha1, out=a4, out_sf=a_sf, aux=w.gs2,
        )  # fmt: skip
        a, a_pts = None, x_perm_pts
        y = grouped_gemm_fp4(
            a4, a_sf, w.w2, w.w2_sf, counts, config2, chain2, in2, pts=a_pts, alpha=alpha2, aux=w.inv2
        )  # fmt: skip
    else:
        a = grouped_gemm_fp4(
            x_perm4, x_perm_sf, w.w1, w.w1_sf, counts, config1, chain1, in1, pts=x_perm_pts, alpha=alpha1
        )  # fmt: skip
        a4, a_sf, a_pts = pair.quantize(
            a, gsi
        )  # the class, second pass (padding rows included: zeros)
        y = grouped_gemm_fp4(
            a4, a_sf, w.w2, w.w2_sf, counts, config2, chain2, in2, pts=a_pts, alpha=alpha2
        )  # fmt: skip
    if T:
        combine(y, dest, scales, out, T, H)
    if return_intermediates:
        return out, dict(
            counts=counts, dest=dest, src=src, x4=x4, x_sf=x_sf, x_pts=x_pts,
            x_perm4=x_perm4, x_perm_sf=x_perm_sf, x_perm_pts=x_perm_pts,
            a=a, a4=a4, a_sf=a_sf, a_pts=a_pts, y=y,
            config1=config1, config2=config2, in_fmts=(in1, in2), Tk_pad=Tk_pad, path=path,
        )  # fmt: skip
    return out


def _fused_moe_fp4_run(
    w: MoEWeightsFp4,
    x_rows: Optional[torch.Tensor],
    x4: Optional[torch.Tensor],
    x_sf: Optional[torch.Tensor],
    x_pts: Optional[torch.Tensor],
    sf_src: str,
    unit_pts: bool,
    experts: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    config1: Knobs,
    config2: Knobs,
    in1: str,
    in2: str,
    T: int,
    H: int,
    k: int,
    E: int,
    I: int,
    gsi: torch.Tensor,
    *,
    plan: Optional[Tuple[torch.Tensor, ...]] = None,
    round_scales: bool = False,
    return_intermediates: bool = False,
):
    """:func:`fused_moe_fp4` on the dense packing with validated operands and the records chosen (``config1``'s gather form
    admitted by the pack's scale layout): the runner's launch (its pack proved every rule the public entry checks, so
    nothing is checked twice per call) and the public entry's body. ``x_rows`` the bf16 rows (quantized here), or the pack
    ``x4`` / ``x_sf`` (the flat 128x4 bytes or the linear rows: ``sf_src``) / ``x_pts`` (the unit placeholder when
    ``unit_pts``)."""
    pair, dev = w.pair, out.device
    sm_count = sm_count_of(dev)
    chain1, chain2 = w.chain1, w.chain2
    static = w.static
    scaled = pair.scaled
    gather1 = config1.gather_a
    Tk = T * k
    Tk_pad = padded_rows_bound(T, k, E)
    pcols_h, pcols_i = pair.sf_cols(H), pair.sf_cols(I)
    unit = _placeholder(dev)
    wpr = _k.warps_per_row(Tk, H // 32, sm_count)
    # the scale operands of the GEMMs: the NVFP4 pair's (pts rows, alphas), none for the MX pair
    alpha1, alpha2 = (w.w1_alpha, w.w2_alpha) if scaled else (None, None)
    combine = _k.get_compiled_combine(
        k, "bf16", "bf16", _k.warps_per_token(T, H, sm_count), round_scales
    )

    def scatter(
        x4_, x_sf_, x_pts_, dest_, dest_pad_, x_perm4_, x_perm_sf_, x_perm_pts_
    ):
        """The fp4 scatter of a quantized input to the grouped layout (its scale rows re-tiled into the atoms)."""
        if scaled:
            kern = _k.get_compiled_scatter_fp4(
                k, wpr, sf_src, unit_pts, **_k.scatter_lanes(H // 2, pcols_h, wpr)
            )
            kern(
                x4_,
                x_sf_,
                x_pts_,
                dest_,
                dest_pad_,
                x_perm4_,
                x_perm_sf_,
                x_perm_pts_,
                Tk,
                H // 2,
                pcols_h,
            )
        else:
            kern = _k.get_compiled_scatter_fp4(
                k,
                wpr,
                sf_src,
                False,
                True,
                MX_BLOCK,
                **_k.scatter_lanes(H // 2, pcols_h, wpr),
            )
            kern(
                x4_,
                x_sf_,
                unit,
                dest_,
                dest_pad_,
                x_perm4_,
                x_perm_sf_,
                unit,
                Tk,
                H // 2,
                pcols_h,
            )

    counts, dest, src, dest_pad, pos_pad = (
        plan if plan is not None else _permute_plan_run(experts.reshape(-1), T, k, E, 1, True, GROUP_ALIGN)
    )  # fmt: skip  # ``plan``: ``route_plan_dual``'s outputs, the same integers
    # the fp4 rows back to back (a tail past Tk for the last tiles' TMA boxes); the scale factors and the row scales at
    # the padded positions (one 128x4 atom per (expert, 128-row tile); a tail past Tk_pad likewise)
    Tk_a, Tk_sf = Tk + DENSE_TAIL, Tk_pad + DENSE_TAIL
    x_perm4 = x_perm_sf = x_perm_pts = None
    n1 = w.w1.shape[1]
    if gather1:
        # the gather form: GEMM1 reads the token rows through the plan's map; a bf16 input is quantized over its T rows
        if x4 is None:
            x4, x_sf, x_pts = pair.quantize(x_rows, gsi)
        g_a, g_pts = x4, x_pts
        g_sf = x_sf.view(T, H // pair.block) if sf_src == "linear" else x_sf
    else:
        x_perm4 = torch.empty(Tk_a, H // 2, dtype=torch.uint8, device=dev)
        x_perm_sf = torch.empty(pair.sf_numel(Tk_sf, H), dtype=torch.uint8, device=dev)
        x_perm_pts = (
            torch.empty(Tk_sf, dtype=torch.float32, device=dev) if scaled else None
        )
        if Tk and x4 is None:
            quantize_scatter_rows(
                x_rows,
                gsi,
                dest.reshape(-1),
                dest_pad.reshape(-1),
                k,
                (x_perm4, x_perm_sf, x_perm_pts),
            )
        elif Tk:
            scatter(
                x4,
                x_sf,
                x_pts,
                dest.reshape(-1),
                dest_pad.reshape(-1),
                x_perm4,
                x_perm_sf,
                x_perm_pts,
            )
        g_a, g_sf, g_pts = x_perm4, x_perm_sf, x_perm_pts
    g_src = src if gather1 else None
    a4 = torch.empty(Tk_a, I // 2, dtype=torch.uint8, device=dev)
    # zero-filled: the writers (the static epilogue, the quantizers' re-tiling) store a row's real scale columns only, and
    # GEMM2's scale boxes read the padded columns of a row's last 4-byte group against the zero-filled tail of its k-tile
    a_sf_pad = torch.zeros(pair.sf_numel(Tk_sf, I), dtype=torch.uint8, device=dev)
    a = a_sf = a_pts = a_pts_pad = None
    run1 = Tk_a > 0 and (
        not gather1 or T > 0
    )  # the gather record's launch needs token rows
    if static:
        # the static class: GEMM1's epilogue writes the intermediate's codes (dense rows) and scale bytes (the padded
        # positions) itself -- no bf16 intermediate, no quantizer launch; GEMM2's scale is mul(alpha2[e], inv2[e])
        if run1:
            _fp4_launch(
                pair, g_a, g_sf, w.w1, w.w1_sf, a4, counts, config1, chain1, in1, Tk_a, E, n1,
                pts=g_pts, alpha=alpha1, out_sf=a_sf_pad, aux=w.gs2, rows_sf=Tk_sf, src=g_src, k_div=k, sf_src=sf_src,
            )  # fmt: skip
        a_pts_pad = (
            x_perm_pts  # GEMM2's row-scale operand is unread by the static chain
            if x_perm_pts is not None
            else torch.empty(Tk_sf, dtype=torch.float32, device=dev)
        )
    else:
        a = torch.empty(Tk_a, I, dtype=torch.bfloat16, device=dev)
        if run1:
            _fp4_launch(
                pair, g_a, g_sf, w.w1, w.w1_sf, a, counts, config1, chain1, in1, Tk_a, E, n1,
                pts=g_pts, alpha=alpha1, rows_sf=Tk_sf, src=g_src, k_div=k, sf_src=sf_src,
            )  # fmt: skip
        # the class's second pass over the live rows: the codes stay dense; the scale factors (and the row scales) are
        # re-tiled to the padded positions for GEMM2's atom-aligned scale boxes (a byte move)
        if scaled:
            a_pts_pad = torch.empty(Tk_sf, dtype=torch.float32, device=dev)
            if Tk and INTERMEDIATE_RETILE == "mapped":
                quantize_rows_mapped(
                    a[:Tk], gsi, pos_pad, (a4[:Tk], a_sf_pad, a_pts_pad)
                )
            elif Tk:
                a_sf = torch.empty(pair.sf_numel(Tk, I), dtype=torch.uint8, device=dev)
                _, _, a_pts = quantize_activations(a[:Tk], gsi, out=(a4[:Tk], a_sf))
                _k.get_compiled_scatter_sf(1, spl=_k.scatter_sf_lanes(pcols_i))(
                    a_sf, a_pts, pos_pad, a_sf_pad, a_pts_pad, Tk, pcols_i
                )
        else:
            a_sf = torch.empty(pair.sf_numel(Tk, I), dtype=torch.uint8, device=dev)
            if Tk:
                quantize_rows_mx(a[:Tk], out=(a4[:Tk], a_sf))
                _k.get_compiled_scatter_sf(1, True, spl=_k.scatter_sf_lanes(pcols_i))(
                    a_sf, unit, pos_pad, a_sf_pad, unit, Tk, pcols_i
                )
    y = torch.empty(Tk_a, H, dtype=torch.bfloat16, device=dev)
    if Tk_a:
        _fp4_launch(
            pair, a4, a_sf_pad, w.w2, w.w2_sf, y, counts, config2, chain2, in2, Tk_a, E, H,
            pts=a_pts_pad if scaled else None, alpha=alpha2, aux=w.inv2 if static else None, rows_sf=Tk_sf,
        )  # fmt: skip
    if T:
        combine(y, dest, scales, out, T, H)
    if return_intermediates:
        return out, dict(
            counts=counts, dest=dest, src=src, dest_pad=dest_pad, pos_pad=pos_pad,
            x4=x4, x_sf=x_sf, x_pts=x_pts,
            x_perm4=None if x_perm4 is None else x_perm4[:Tk],
            x_perm_sf=None if x_perm_sf is None else x_perm_sf[: pair.sf_numel(Tk_pad, H)],
            x_perm_pts=None if x_perm_pts is None else x_perm_pts[:Tk_pad],
            a=None if a is None else a[:Tk], a4=a4[:Tk], a_sf=a_sf, a_pts=a_pts,
            a_sf_pad=a_sf_pad[: pair.sf_numel(Tk_pad, I)],
            a_pts_pad=None if a_pts_pad is None else a_pts_pad[:Tk_pad],
            y=y[:Tk], config1=config1, config2=config2, in_fmts=(in1, in2), Tk_pad=Tk_pad, path="dense",
        )  # fmt: skip
    return out


def moe_ffn_fp4(
    x: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    router_logits: torch.Tensor,
    w: MoEWeightsFp4,
    top_k: int,
    method: Union[S.RoutingMethod, str] = S.SOFTMAX_RENORM,
    bias: Optional[torch.Tensor] = None,
    scaling_factor: Optional[Union[float, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
) -> torch.Tensor:
    """Routing (the bf16 classes' kernel) + the fp4 FFN."""
    experts, weights = route(router_logits, top_k, method, bias, scaling_factor)
    return fused_moe_fp4(x, experts, weights, w, out, config1, config2)


__all__ = [
    "ALIGN_FP4",
    "BatchInvariantUnsupportedError",
    "DENSE_TAIL",
    "F32_CHAIN",
    "Fp4Pair",
    "GEMM2_CHAIN",
    "GEMM2_STATIC_CHAIN",
    "GROUP_ALIGN",
    "MAX_TOP_K",
    "MXFP4",
    "MX_F32_CHAIN",
    "MoEWeightsFp4",
    "NVFP4",
    "PAIRS_BY_NAME",
    "PATHS",
    "TE_GS_INV",
    "check_shape_fp4",
    "expert_classes",
    "fp4_global_scale",
    "fp4_global_scales",
    "fused_moe_fp4",
    "grouped_gemm_fp4",
    "moe_ffn_fp4",
    "padded_rows_bound",
    "pair_of",
    "permute_plan",
    "permute_plan_dual",
    "quantize_activations",
    "quantize_rows_mapped",
    "quantize_rows_mx",
    "quantize_scatter_rows",
    "route",
    "route_plan_dual",
    "sf_layouts_collide",
    "gather_sf_layout",
    "SF_LAYOUTS",
    "sm_count_of",
]
