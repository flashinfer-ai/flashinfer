# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 x MXFP8 MoE classes ``moe.mxfp8.<act>``: the expert weights and the activations quantized by the
``mxfp8.quant`` class (e4m3 values, one UE8M0 scale per 32 elements), the two GEMMs as grouped placements of the dense
MXFP8 class of the part (the block scales inside the block-scaled step), the bf16 classes' chains, routing, deterministic
permutation and slot-order combine.

The mixed MXFP4-weight x MXFP8-activation classes (``moe.mxfp4w8a.<act>``, the fp4 backend's third pair) run the same
pipeline on the byte-packed e2m1 experts of :class:`.api_fp4.MoEWeightsFp4` at the mixed instruction class of the part
(``in_fmt`` in ``spec_mxfp4w8a.MX_CLASSES``: the K granule is the class's, the swapped body stages the record's own k-block,
the gather and decode forms are the one-format classes'); :mod:`.api_mxfp4w8a` is their entry.

``MoEWeightsMXFP8.prepare`` quantizes the experts one at a time into the 128x4 swizzled scale layout the instruction reads;
``quantize_activations_mxfp8`` is the class's per-token activation quantization into the layer's canonical MXFP8 pack
(e4m3 ``[T, H]`` with the token-major UE8M0 scales ``[T, H / 32]``, the pack ``TrtllmFp8BlockConfig`` and
``CutlassMxfp8Config`` prepare: the linear layout alone, no swizzled buffer and no per-token scale); ``fused_moe_mxfp8``
runs the FFN on pre-routed tokens from that pack (the caller's bytes taken as given). Every output bit is a function of
the token's own quantized row, its ``k`` experts and routing weights, and the quantized expert weights: batch size,
composition, row position, expert load and the grouped records never enter the dataflow (the specification is
:mod:`.spec_mxfp8`). The arithmetic is exact: GEMM1's chain is the bf16 classes' (every step correctly rounded, no SFU
table enters), the UE8M0 block scales are operands of the instruction and the quantizer's chain is the declared one; the
instruction class of the part (the SM107 step folds 64 elements, the SM100 / SM103 step 32) is a dimension of the class.

The 128x4 scale buffers the kernels read (the experts', the fused requantization's) carry zero pad columns where a row's
scale columns are not a multiple of 4 (``K`` not a multiple of 128): the quantizer writes them, the fused epilogue's
buffer is zero-filled before the launch where they exist, and GEMM2's scale boxes read them against the zero-filled tail
of a k-tile.

The expanded rows are permuted into groups padded to whole 128-row scale-factor atoms (the swizzled scale layout is affine
only inside an atom); one scatter launch moves the e4m3 rows and re-tiles the linear scale bytes into the atoms, zero-filling
the padding rows. The GEMM outputs are allocated uninitialised: a padding row of the intermediate holds unspecified bytes,
is quantized to some codes and read by the padded tiles, and no stored value depends on it (the step is row-local, the
store predicate is the group's real end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import dataclasses

import torch

from ...batch_invariant.quantization.layout import padded_sf_cols as _padded_sf_cols
from ...batch_invariant.quantization.layout import sf_numel as _sf_numel
from ...batch_invariant.quantization.mx_api import mxfp8_quantize_batch_invariant
from ...batch_invariant.support import (
    LazyNames,
    check_device,
    compute_capability,
    unsupported,
)
from ...gemm.batch_invariant.api import sm_count_of
from ...gemm.batch_invariant.generator import Knobs
from ...gemm.batch_invariant.heuristic import family_of
from ...gemm.batch_invariant.schedule import chunks_for, mx_class_for
from . import spec as S
from .api import (
    _check_fused_inputs,
    _check_grouped_chain,
    _check_grouped_operands,
    _grouped_out,
    _permute_plan_run,
    combine_rows,
    interleave_fc1,
    launch_handle,
    permute_plan,
)
from .api import padded_rows_bound
from .blocks.epilogue import GroupedEpilogueBlock
from .blocks.epilogue_mx import MxChain
from .blocks.scheduler import MAX_E
from .blocks.swap import PLAN_MAX_E, PLAN_MAX_SLOTS
from .schedule_mxfp8 import GEMM2_CHAIN
from .spec_mxfp4w8a import MX_CLASSES as MX_MIXED_CLASSES
from .spec_mxfp8 import BLOCK, CLASS, MX_CLASSES

#: The padded group alignment (rows): one 128x4 scale-factor atom.
GROUP_ALIGN = 128
#: The dense packing's tail rows: the A buffers carry this many rows past the live ones (the scale buffers past the padded
#: rows) so the last tiles' TMA boxes stay inside the buffers (an out-of-bounds box costs the TMA engine far more than the
#: rows it zero-fills); never stored. A 256-row cluster tile of the padded body on the dense packing is the largest box.
DENSE_TAIL = 256
PACKINGS = ("padded", "dense")
ALIGN_E4M3 = 16  # e4m3 elements per 16-byte row: the TMA alignment of every operand
ALIGN_E2M1_BYTES = 16  # bytes per 16-byte row of the mixed classes' packed e2m1 weight (two codes per byte)
_OUT = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def _load_kernels() -> Dict[str, Any]:
    from .blocks.compile import get_compiled_grouped, get_compiled_swapped
    from .mx_permute import get_compiled_scatter_mx, scatter_mx_lanes
    from .permute import warps_per_row
    from .schedule import launch_tile_bound, rows_per_expert
    from .schedule_mxfp8 import (
        can_implement_grouped_mxfp8,
        can_implement_swapped_mxfp8,
        default_configs_mx,
        gather_form_rule,
        packing_for,
        swap_k_block_arg,
    )

    return dict(
        get_compiled_grouped=get_compiled_grouped,
        get_compiled_swapped=get_compiled_swapped,
        get_compiled_scatter_mx=get_compiled_scatter_mx,
        scatter_mx_lanes=scatter_mx_lanes,
        warps_per_row=warps_per_row,
        launch_tile_bound=launch_tile_bound,
        rows_per_expert=rows_per_expert,
        can_implement_grouped_mxfp8=can_implement_grouped_mxfp8,
        can_implement_swapped_mxfp8=can_implement_swapped_mxfp8,
        default_configs_mx=default_configs_mx,
        gather_form_rule=gather_form_rule,
        packing_for=packing_for,
        swap_k_block_arg=swap_k_block_arg,
    )


_k = LazyNames(_load_kernels)


def mx_class_of(device: torch.device) -> str:
    """The instruction class a device runs: a function of the architecture family only (``mxfp8_k64`` on SM107,
    ``mxfp8`` on SM100 / SM103), never of the shape."""
    return mx_class_for("mxfp8", family_of(compute_capability(device)))


def _is_mixed(in_fmt: str) -> bool:
    """The mixed MXFP4 x MXFP8 classes (the e2m1 weight), against the MXFP8 classes."""
    if in_fmt in MX_CLASSES:
        return False
    if in_fmt in MX_MIXED_CLASSES:
        return True
    raise ValueError(f"unknown MX instruction class {in_fmt!r}")


#: The compute capabilities the classes are stated for (the dense MXFP8 GEMM class's parts).
MOE_MXFP8_ARCHS: Tuple[Tuple[int, int], ...] = ((10, 0), (10, 3), (10, 7))


def sf_numel(rows: int, K: int) -> int:
    """Bytes of the 128x4 swizzled UE8M0 buffer of a ``(rows, K)`` operand (rows padded to 128, scale columns to 4)."""
    return _sf_numel(rows, K, BLOCK, "128x4")


def padded_sf_cols(K: int) -> int:
    return _padded_sf_cols(K, BLOCK, "128x4")


def check_shape_mxfp8(H: int, I: int, E: int, k: Optional[int] = None) -> None:
    """The complete acceptance rule of the class on ``(H, I, E, k)``; anything else raises
    :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`. Everything else about a shape (the padded group
    layout, the tile scheduler, the records, the per-expert tensor maps) is a placement.

    * ``H`` and ``I`` positive multiples of 32: whole 32-element scale blocks of the quantized ``x`` (K = H), of the
      quantized intermediate (K = I) and of both expert weights, hence 16-byte e4m3 rows (the gate / up pairing needs
      ``I % 8``, implied);
    * ``1 <= E <= MAX_E`` (the routing kernel's buckets, the group tables), ``1 <= k <= min(E, MAX_TOP_K)`` (the shared
      bounds of the MoE classes);
    * ``chunks_for(2I, H) == chunks_for(H, I) == 1`` for the MX class: the grouped classes take unsplit expert weight shapes
      (the dense class K-splits a weight only when ``K`` is very large against ``N``)."""
    if H <= 0 or I <= 0 or H % BLOCK or I % BLOCK:
        raise unsupported(
            CLASS,
            f"hidden_size {H}, intermediate_size {I}",
            f"positive multiples of {BLOCK} (whole UE8M0 scale blocks: 16-byte e4m3 rows of the activations, the intermediate and the weights)",
        )
    if not 1 <= E <= MAX_E:
        raise unsupported(CLASS, f"num_experts {E}", f"1 <= num_experts <= {MAX_E}")
    if k is not None and not 1 <= k <= min(E, S.MAX_TOP_K):
        raise unsupported(
            CLASS, f"top_k {k}", f"1 <= top_k <= min(num_experts, {S.MAX_TOP_K})"
        )
    if chunks_for(2 * I, H, "mxfp8") != 1 or chunks_for(H, I, "mxfp8") != 1:
        raise unsupported(
            CLASS,
            f"expert weights (2I = {2 * I}, H = {H}) / (H = {H}, I = {I})",
            "weight shapes the dense MXFP8 class folds in one chunk (the grouped class takes unsplit weights)",
        )


# ----------------------------------------------------------------------------- the quantizer (the declared class)


def quantize_mxfp8_rows(
    x: torch.Tensor,
    swizzled: bool,
    enable_pdl: bool = False,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ``mxfp8.quant`` class on bf16 rows ``[M, K]``: ``(codes [M, K] float8_e4m3fn, sf)`` with ``sf`` the flat uint8
    UE8M0 bytes in the 128x4 swizzled layout (``sf_numel(M, K)``: the block-scaled GEMMs' operand) or the token-major
    ``[M, K / 32]`` bytes (the activation pack). ``valid`` (int32 ``[M]``, negative = skip): only the rows marked live are
    read and written (the padded operand's padding rows hold unspecified bytes no stored value depends on)."""
    codes, sf = mxfp8_quantize_batch_invariant(
        x, "128x4" if swizzled else "linear", BLOCK, enable_pdl, valid=valid
    )
    sf = sf.view(torch.uint8)
    return codes, sf.reshape(-1) if swizzled else sf.reshape(x.shape[0], -1)


def quantize_activations_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The class's activation quantization: ``(xq [T, H] float8_e4m3fn, xsf [T, H / 32] uint8)``, the scales token-major
    (the layer's canonical MXFP8 activation pack)."""
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        raise TypeError("quantize_activations_mxfp8 takes a bf16 [T, H]")
    if x.shape[1] % BLOCK:
        raise unsupported(CLASS, f"hidden_size {x.shape[1]}", f"a multiple of {BLOCK}")
    return quantize_mxfp8_rows(x.contiguous(), swizzled=False)


def check_activation_pack(
    x: torch.Tensor,
    sf: Optional[torch.Tensor],
    cls: str,
    *,
    sf_dtypes: Tuple[torch.dtype, ...] = (torch.uint8, torch.float8_e8m0fnu),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The canonical MXFP8 activation pack as the runners take it, validated in the class's terms: the e4m3 rows ``[T, H]``
    (``H`` whole scale blocks) and their token-major UE8M0 scales in the linear ``[T, H / 32]`` shape, taken as given (a
    flattened or swizzled buffer is refused, never reshaped into rows; the empty pack takes any empty scale tensor; the
    scale bytes typed as one of ``sf_dtypes``), both contiguous and 16-byte aligned; ``(x, sf)`` with ``sf`` as uint8."""
    if x.dtype != torch.float8_e4m3fn or x.ndim != 2:
        raise unsupported(
            cls,
            f"hidden_states_q {x.dtype} of rank {x.ndim}",
            "the MXFP8 activation pack: an e4m3 [num_tokens, hidden_size] with its UE8M0 scales",
        )
    if sf is None:
        raise unsupported(
            cls,
            "no hidden_states_scale",
            f"the token-major UE8M0 scales [num_tokens, hidden_size / {BLOCK}]",
        )
    T, H = x.shape
    if sf.dtype not in sf_dtypes:
        raise unsupported(
            cls,
            f"hidden_states_scale {sf.dtype}",
            "UE8M0 scale bytes ("
            + " / ".join(str(d).replace("torch.", "") for d in sf_dtypes)
            + ")",
        )
    if H % BLOCK:
        raise unsupported(
            cls,
            f"hidden size {H}",
            f"a positive multiple of {BLOCK} (whole UE8M0 scale blocks)",
        )
    sf = sf.view(torch.uint8)
    if T == 0 and sf.numel() == 0:
        sf = sf.reshape(
            0, H // BLOCK
        )  # the empty pack: no scale byte, whatever its shape
    if tuple(sf.shape) != (T, H // BLOCK):
        raise unsupported(
            cls,
            f"hidden_states_scale of shape {tuple(sf.shape)} for {T} rows of {H}",
            f"the token-major [{T}, {H // BLOCK}] UE8M0 scales (the linear layout of the canonical MXFP8 pack; a "
            "swizzled or flattened buffer is not reshaped into rows)",
        )
    x, sf = x.contiguous(), sf.contiguous()
    if x.data_ptr() % 16 or sf.data_ptr() % 16:
        raise unsupported(
            cls,
            "hidden_states_q or hidden_states_scale at a storage offset that is not a multiple of 16 bytes",
            "16-byte aligned rows and scale bytes (the kernels' vector loads and TMA boxes)",
        )
    return x, sf


# ----------------------------------------------------------------------------- weights

#: The weight view's pair tag (``quant``): the fp8-block backend's views share one key, so a view names its pair.
VIEW_PAIR = "mxfp8"


@dataclass
class MoEWeightsMXFP8:
    """The MXFP8 expert weights as the kernels consume them: ``w1 [E, N1, H]`` e4m3 (``N1 = 2I`` gate / up interleaved
    for a gated chain, ``I`` as given for an ungated one), ``w1_sf [E, sf_numel(N1, H)]`` uint8 (one 128x4 swizzled
    UE8M0 buffer per expert), ``w2 [E, H, I]`` e4m3, ``w2_sf [E, sf_numel(H, I)]``; the GEMM1 chain ``act`` (a gated
    chain of ``spec.ACTS`` or an ungated one of ``spec.UNGATED_ACTS``) with its ``params``."""

    w1: torch.Tensor
    w1_sf: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    gated: bool = True
    act: str = "silu"
    params: Tuple[Optional[float], ...] = ()

    @staticmethod
    def prepare(
        fc1: torch.Tensor,
        fc2: torch.Tensor,
        interleaved: bool = False,
        gated: bool = True,
        act: Optional[str] = None,
        params: Sequence[Optional[float]] = (),
        device: Optional[torch.device] = None,
    ) -> "MoEWeightsMXFP8":
        """Quantize the experts (``fc1 [E, 2I, H]`` rows ``[up, gate]`` for a gated chain, ``[E, I, H]`` for an ungated
        one; ``fc2 [E, H, I]``; bf16, or a wider floating dtype cast to bf16 one expert at a time) with the ``mxfp8.quant``
        class into the 128x4 swizzled scale layout. The gate / up interleave is a row permutation applied before the
        quantization (a row's block scales are the row's own). ``act`` defaults to the class's default chain of the gating
        (``silu`` gated, ``relu2`` ungated). The pack lives on ``device`` (the weights' by default), a CUDA device of the
        class: the experts are quantized one at a time into the preallocated pack, each expert's slice moved to the pack's
        device and cast to bf16 right before its quantization (a same-device bf16 slice as it is), so the peak memory
        beyond the inputs is the pack plus one expert's temporaries for CPU-resident, fp32 and strided inputs too, and
        the bits are those of each expert quantized alone (a row's quantization is row-local)."""
        from ...batch_invariant.quantization.mx_api import mx_quantize
        from ...batch_invariant.quantization.mx_spec import MXFP8 as MXFP8_FORMAT

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
                "the MXFP8 MoE classes take floating expert weights fc1 [E, N1, H], fc2 [E, H, I] (bf16, or cast to "
                "bf16 one expert at a time)"
            )
        E, N1, H = fc1.shape
        I = N1 // 2 if gated else N1
        if tuple(fc2.shape) != (E, H, I) or (gated and N1 % 2):
            raise ValueError(
                f"fc2 {tuple(fc2.shape)} does not match fc1 {tuple(fc1.shape)} (gated={gated})"
            )
        check_shape_mxfp8(H, I, E)
        dev = fc1.device if device is None else torch.device(device)
        if dev.type != "cuda":
            raise unsupported(
                CLASS,
                f"expert weights quantized on {dev}",
                "a CUDA device for the pack (the weights' device, or device=...): the quantizer class runs there",
            )
        check_device(dev, CLASS, MOE_MXFP8_ARCHS)
        # the pack, preallocated and filled one expert at a time: the quantizer writes every byte of its expert's rows and
        # scale buffer (the pad columns zero), so the peak beyond the inputs and the pack is one expert's temporaries
        w1 = torch.empty(E, N1, H, dtype=torch.uint8, device=dev)
        w1_sf = torch.empty(E, sf_numel(N1, H), dtype=torch.uint8, device=dev)
        w2 = torch.empty(E, H, I, dtype=torch.uint8, device=dev)
        w2_sf = torch.empty(E, sf_numel(H, I), dtype=torch.uint8, device=dev)

        def expert(w: torch.Tensor, e: int) -> torch.Tensor:
            """Expert ``e``'s slice on the pack's device as contiguous bf16, moved and cast right before its quantization."""
            return w[e].to(device=dev, dtype=torch.bfloat16).contiguous()

        for e in range(E):
            rows = expert(fc1, e)
            if gated and not interleaved:
                rows = interleave_fc1(rows.unsqueeze(0))[0]
            mx_quantize(rows, MXFP8_FORMAT, "128x4", out=(w1[e], w1_sf[e]))
            mx_quantize(expert(fc2, e), MXFP8_FORMAT, "128x4", out=(w2[e], w2_sf[e]))
        return MoEWeightsMXFP8(
            w1.view(torch.float8_e4m3fn),
            w1_sf,
            w2.view(torch.float8_e4m3fn),
            w2_sf,
            gated,
            act,
            key,
        )

    @property
    def num_experts(self) -> int:
        return self.w1.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w1.shape[2]

    @property
    def intermediate_size(self) -> int:
        return self.w2.shape[2]

    @property
    def chain1(self) -> MxChain:
        """The GEMM1 epilogue chain of the class: the gated ``act`` on the (gate, up) pair or the ungated ``act`` on the
        single accumulator, with the parameters (the bf16 classes' chain: no scale; the instruction class is the launch's)."""
        return MxChain("glu" if self.gated else "act", self.act, "bf16", self.params)

    @property
    def chain1_quant(self) -> MxChain:
        """The fused-requantization spelling of ``chain1``: the same chain, its bf16-rounded value quantized by the
        ``mxfp8.quant`` class in the epilogue (codes and 128x4 block scales out, no bf16 intermediate)."""
        return MxChain("glu" if self.gated else "act", self.act, "mxfp8", self.params)

    def view(self) -> Dict[str, object]:
        """The weight view a :class:`flashinfer.fused_moe.MoEWeightPack` stores for the backend (``quant`` names the pair:
        the backend's views share one key)."""
        return {
            "quant": VIEW_PAIR,
            "w1": self.w1,
            "w1_sf": self.w1_sf,
            "w2": self.w2,
            "w2_sf": self.w2_sf,
            "gated": self.gated,
            "act": self.act,
            "params": self.params,
        }

    @staticmethod
    def from_view(view: Dict[str, object]) -> "MoEWeightsMXFP8":
        """The weights of a view :meth:`view` wrote; the pair is read from its ``quant`` key, never assumed."""
        pair = view.get("quant")
        if pair != VIEW_PAIR:
            raise unsupported(
                CLASS,
                "a weight view without its quantization pair (no `quant` key)"
                if pair is None
                else f"a weight view of the pair {pair!r}",
                "the view of BatchInvariantFp8BlockConfig.prepare_weights(..., quant=QuantConfig(weight=MXFP8, "
                f"activation=MXFP8)), which names the pair (`quant` = {VIEW_PAIR!r})",
            )
        gated = bool(view.get("gated", True))
        return MoEWeightsMXFP8(
            view["w1"],  # type: ignore[arg-type]
            view["w1_sf"],  # type: ignore[arg-type]
            view["w2"],  # type: ignore[arg-type]
            view["w2_sf"],  # type: ignore[arg-type]
            gated,
            str(view.get("act", "silu" if gated else "relu2")),
            tuple(view.get("params", ())),  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------------- the grouped GEMM


def _bytes(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return t
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return t.view(torch.uint8)
    raise TypeError(
        f"{name}: expected float8_e4m3fn / UE8M0 / uint8 bytes, got {t.dtype}"
    )


def grouped_gemm_mxfp8(
    a: torch.Tensor,
    a_sf: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    counts: torch.Tensor,
    config: Knobs,
    chain: GroupedEpilogueBlock,
    in_fmt: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
    dense_a: bool = False,
    sf_rows: Optional[int] = None,
    k_block: Optional[int] = None,
    src: Optional[torch.Tensor] = None,
    k_div: int = 1,
    rows: Optional[int] = None,
    in_kernel: Optional[Tuple[torch.Tensor, int, torch.Tensor]] = None,
    n_live: Optional[int] = None,
) -> torch.Tensor:
    """``out[p] = chain(a[p] @ w[e(p)].T)`` with the UE8M0 block scales applied inside the block-scaled step, on the padded
    expert segments ``counts`` (int32 ``[E]``; the rows of group ``e`` start at the exclusive prefix of the counts rounded
    up to 128): ``a [Tk_pad, K]`` e4m3, ``a_sf`` its 128x4 swizzled scales (``sf_numel(Tk_pad, K)`` bytes), ``w [E, N,
    K]`` e4m3 -- or, at a mixed instruction class, ``w [E, N, K / 2]`` the byte-packed e2m1 weight (uint8, two codes per byte;
    K in whole granules of the class, the swapped record's own k-block, no gather or decode form) -- ``w_sf [E, sf_numel(N,
    K)]``. ``chain``: a bf16 class chain (no scale) or the fused requantization chain
    (``out_fmt == "mxfp8"``: returns ``(codes, sf)``); ``in_fmt`` the instruction class (the device's by default); ``out
    [Tk_pad, Nout]`` of the chain's dtype (rows past a group's real end are never stored). ``dense_a``: the dense packing --
    ``a`` the groups' rows back to back (any row count; the rows past the last group never stored), ``a_sf`` the scales at
    the padded positions (``sf_numel(sf_rows, K)`` bytes, ``sf_rows`` the padded rows incl. the tail; derived from the
    buffer when not given), the output's rows ``a``'s; the fused requantization's scales land at the padded positions. The
    record selects the body: a dense record runs the padded body (on either packing), a swapped record (``config.swap_ab``:
    128 weight rows x 8 / 16 / 32 tokens, the weights on the MMA's M side, :mod:`.blocks.swap`) the swapped body on
    the dense packing (``a`` carries at least the token tile's rows past the live rows: ``DENSE_TAIL``). Both bodies are the
    unified grouped bodies (:mod:`.blocks.kernel` / :mod:`.blocks.swap`) on the block-scaled operand block at the MX
    chain's instruction class. The gather record (``config.gather_a``, a swapped record): ``a [T, K]`` the unpermuted token
    rows, ``a_sf`` their token-major ``[T, K / 32]`` scale bytes (the activation pack's layout), ``src [>= Tk]`` int32 the
    plan's slot of every dense row (``src[p] // k_div`` its token), ``rows`` the dense rows of the output (the live rows and
    the tail); the body reads the rows itself (no scatter launch), the fused requantization's scale bytes land at the padded
    positions of ``sf_rows`` rows as in the box form. ``n_live``: the live rows at most (the routing's slots) -- on the dense
    packing the scale rows must then cover the padded positions of that many rows (:func:`padded_rows_bound`), the
    positions the body reads and the fused requantization writes; the gather record takes it (``src`` and the output cover
    the slots)."""
    in_fmt = in_fmt or mx_class_of(a.device)
    mixed = _is_mixed(in_fmt)
    name = "grouped_gemm_mxfp4w8a" if mixed else "grouped_gemm_mxfp8"
    if config.gather_a and mixed:
        raise ValueError(
            f"{name}: the gather form is the one-format classes' (the mixed classes read the token box)"
        )
    if config.gather_a:
        return _grouped_gemm_mxfp8_gather(
            a,
            a_sf,
            w,
            w_sf,
            counts,
            config,
            chain,
            in_fmt,
            out,
            src,
            k_div,
            rows,
            sf_rows,
            in_kernel,
            n_live,
        )
    if src is not None or rows is not None or in_kernel is not None:
        raise ValueError(
            f"{name}: src, rows and in_kernel belong to the gather record (config.gather_a)"
        )

    # the shared operand rules at the class's alignment: K in whole scale blocks (whole granules of the mixed class), N at
    # the weight-row rule, every row 16 bytes; the mixed classes' weight stored as two e2m1 codes per byte
    if mixed:
        from .spec_mxfp4w8a import granule_of

        w_kw: dict = dict(
            k_align=granule_of(in_fmt),
            w_dtype=torch.uint8,
            w_pack=2,
            w_align=ALIGN_E2M1_BYTES,
        )
    else:
        w_kw = dict(k_align=BLOCK)
    _, K, E, N, Tk = _check_grouped_operands(
        name,
        a,
        w,
        counts,
        a.shape[0],
        None,
        torch.float8_e4m3fn,
        ALIGN_E4M3,
        "e4m3",
        n_align=8,
        **w_kw,
    )
    if not dense_a and Tk % GROUP_ALIGN:
        raise ValueError(
            f"{name}: the padded rows are a multiple of the 128-row scale-factor atom"
        )
    a_sf, w_sf = _bytes(a_sf, "a_sf").reshape(-1), _bytes(w_sf, "w_sf").reshape(-1)
    if dense_a:
        if sf_rows is None:
            sf_rows = (a_sf.numel() // padded_sf_cols(K)) // GROUP_ALIGN * GROUP_ALIGN
        if sf_rows % GROUP_ALIGN or sf_rows < Tk:
            raise ValueError(
                f"{name}: the dense packing's scale rows are whole 128-row atoms covering the rows"
            )
        if n_live is not None and sf_rows < padded_rows_bound(n_live, 1, E):
            raise ValueError(
                f"{name}: the dense packing's scale rows must cover the padded positions of the live rows "
                f"(at least padded_rows_bound({n_live}, 1, {E}) = {padded_rows_bound(n_live, 1, E)} rows)"
            )
    else:
        if sf_rows is not None and sf_rows != Tk:
            raise ValueError(
                f"{name}: the padded packing's scale rows are the operand's rows"
            )
        sf_rows = Tk
    if a_sf.numel() < sf_numel(sf_rows, K) or w_sf.numel() < E * sf_numel(N, K):
        raise ValueError(
            f"{name}: the scale-factor buffers are smaller than the operands' 128x4 layouts"
        )
    chain = MxChain.of(chain, in_fmt)
    n_out = _check_grouped_chain(name, N, chain)
    kb = 0
    if config.swap_ab:
        if not dense_a:
            raise ValueError(
                f"{name}: the swapped decode body reads the dense packing (the scale factors at the padded positions)"
            )
        # the stage's K extent: two atoms when they divide K (`swap_k_block`; ``k_block`` overrides), a placement; the
        # mixed classes' swapped records carry their stage width as a field of the record (the dense class's rule)
        kb = 0 if mixed else _k.swap_k_block_arg(in_fmt, K, k_block)
        if mixed and k_block:
            raise ValueError(
                f"{name}: the mixed classes' swapped records carry their own k-block"
            )
        if not _k.can_implement_swapped_mxfp8(config, N, K, E, chain, in_fmt, kb):
            raise ValueError(
                f"{name}: swapped record {config.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()}, {in_fmt}, k_block={kb or 'default'})"
            )
        kind = "swapped"
    else:
        if not _k.can_implement_grouped_mxfp8(config, N, K, E, chain, in_fmt):
            raise ValueError(
                f"{name}: record {config.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()}, {in_fmt})"
            )
        kind = "grouped"
    packing = "dense" if dense_a else "padded"
    w3 = w.permute(1, 2, 0)
    if chain.quant:
        # the fused requantization: the codes and the 128x4 scale bytes of the bf16-rounded chain, written by the epilogue
        if out is not None:
            raise ValueError(
                f"{name}: the fused requantization allocates its two outputs"
            )
        codes = _grouped_out(name, None, Tk, n_out, torch.uint8, a.device)
        sf = _quant_sf_buffer(sf_rows, n_out, a.device)
        if Tk:
            # the host rules above come before the JIT build
            _mx_launch(kind, a, a_sf, w3, w_sf, (codes, sf), counts, config, chain, Tk, E, N, sf_rows, kb, packing)  # fmt: skip
        return codes.view(torch.float8_e4m3fn), sf
    out = _grouped_out(name, out, Tk, n_out, _OUT[chain.out_fmt], a.device)
    if Tk:
        _mx_launch(kind, a, a_sf, w3, w_sf, (out,), counts, config, chain, Tk, E, N, sf_rows, kb, packing)  # fmt: skip
    return out


def _mx_kernel(
    config: Knobs,
    chain: MxChain,
    kind: str,
    kb: int,
    packing: str,
    plan_in_kernel: bool,
) -> Any:
    """The compiled MX body of a record: the gather record (``sf_src="linear"``, with or without the in-kernel plan), the
    swapped body at the stage width ``kb``, or the padded body at ``packing``."""
    if kind == "gather":
        return _k.get_compiled_swapped(
            config, chain, sf_src="linear", plan_in_kernel=plan_in_kernel
        )
    if kind == "swapped":
        return _k.get_compiled_swapped(config, chain, k_block=kb)
    return _k.get_compiled_grouped(config, chain, packing=packing)


def _mx_launch(
    kind: str,
    a: torch.Tensor,
    a_sf8: torch.Tensor,
    w3: torch.Tensor,
    w_sf8: torch.Tensor,
    outs: Tuple[torch.Tensor, ...],
    counts: torch.Tensor,
    config: Knobs,
    chain: MxChain,
    rows: int,
    E: int,
    N: int,
    sf_rows: int = 0,
    kb: int = 0,
    packing: str = "dense",
    src: Optional[torch.Tensor] = None,
    k_div: int = 1,
    in_kernel: Optional[Tuple[torch.Tensor, int, torch.Tensor]] = None,
) -> None:
    """One MX body launch on prepared operands (the launch of :func:`grouped_gemm_mxfp8` and of the gather record): ``kind``
    ``gather`` (the token rows through ``src``, or the in-kernel plan ``in_kernel = (ids, n_slots, dest)``), ``swapped`` (the
    token box at the stage width ``kb``) or ``grouped`` (the padded body at ``packing``, its scale rows ``sf_rows``); ``a_sf8``
    / ``w_sf8`` the flat scale bytes, ``outs`` the output (or the codes and scales of the fused requantization), ``rows`` the
    rows the tile bound counts. The persistent grid is the handle's cluster count clipped to that bound. The callers
    validated the operands."""
    pk = in_kernel is not None
    h = launch_handle(
        a.device, config, (kind, chain, kb, packing, pk), _mx_kernel, config, chain, kind, kb, packing, pk
    )  # fmt: skip
    grid = min(h.clusters, _k.launch_tile_bound(config, rows, E, N))
    a3 = a.unsqueeze(-1)
    if kind == "gather":
        if in_kernel is not None:
            ids, n_slots, dest = in_kernel
            h.kernel(
                a3, a_sf8, ids, n_slots, k_div, w3, w_sf8, *outs, counts, dest, grid
            )
        else:
            h.kernel(a3, a_sf8, src, k_div, w3, w_sf8, *outs, counts, grid)
    elif kind == "swapped":
        h.kernel(a3, a_sf8, w3, w_sf8, *outs, counts, grid)
    else:
        h.kernel(a3, w3, a_sf8, w_sf8, *outs, counts, sf_rows, grid, h.raster)


def _quant_sf_buffer(sf_rows: int, n_out: int, device: torch.device) -> torch.Tensor:
    """The fused requantization's 128x4 scale buffer of ``(sf_rows, n_out)``: zero-filled where a row's scale columns leave
    pad columns in their last 4-byte group (``n_out`` not a multiple of 128) -- the epilogue stores a row's real scale
    columns only, and GEMM2's scale boxes read the pad columns against the zero-filled tail of its k-tile -- and
    uninitialised otherwise (every byte of a live row is stored; the padding rows are row-local)."""
    alloc = torch.zeros if padded_sf_cols(n_out) != n_out // BLOCK else torch.empty
    return alloc(sf_numel(sf_rows, n_out), dtype=torch.uint8, device=device)


def gather_form_ok(K: int) -> bool:
    """The gather form's scale source: the pack's row-major UE8M0 bytes are read in 4-byte block groups, so a row's ``K / 32``
    bytes must be whole groups."""
    return K % (4 * BLOCK) == 0


def decode_form_ok(T: int, k: int, E: int, H: int) -> bool:
    """The decode form's shape rule: the routing's slots fit the in-kernel plan's registers (one warp's) and its membership
    masks, the pack's scale rows are whole 4-byte groups."""
    return T * k <= PLAN_MAX_SLOTS and E <= PLAN_MAX_E and gather_form_ok(H)


def decode_form_default(
    T: int,
    k: int,
    E: int,
    H: int,
    config1,
    config2,
    path: str = "dense",
    requant: str = "fused",
) -> bool:
    """Whether the fused op takes the decode form by default for this routing and these records: the dense packing with the
    fused requantization, both records swapped, GEMM1's at the gather rule of the routing's rows per expert
    (`schedule_mxfp8.gather_form_rule`) and the routing within the in-kernel plan's bounds (`decode_form_ok`)."""
    from .schedule import rows_per_expert
    from .schedule_mxfp8 import gather_form_rule

    return bool(
        path == "dense"
        and requant == "fused"
        and config1.swap_ab
        and config2.swap_ab
        and gather_form_rule(config1, rows_per_expert(T, k, E)).gather_a
        and decode_form_ok(T, k, E, H)
    )


def _grouped_gemm_mxfp8_gather(
    a,
    a_sf,
    w,
    w_sf,
    counts,
    config,
    chain,
    in_fmt,
    out,
    src,
    k_div,
    rows,
    sf_rows,
    in_kernel=None,
    n_live=None,
):
    """The gather record of :func:`grouped_gemm_mxfp8` (its operand rules, the token rows in place of the permuted rows);
    ``in_kernel = (ids, n_slots, dest)``: the decode form, the plan built by the kernel from the ids (``src`` unread, ``counts``
    and ``dest`` written by it). The live rows are at most ``n_live`` (the launch form; the decode form's ``n_slots``): ``src``
    and the output cover them, the scale rows their padded positions."""
    T, K, E, N, _ = _check_grouped_operands(
        "grouped_gemm_mxfp8",
        a,
        w,
        counts,
        a.shape[0],
        None,
        torch.float8_e4m3fn,
        ALIGN_E4M3,
        "e4m3",
        k_align=BLOCK,
        n_align=8,
    )
    if rows is None or (src is None and in_kernel is None):
        raise ValueError(
            "grouped_gemm_mxfp8: the gather record reads the token rows through the plan's map: src and rows"
        )
    if in_kernel is None:
        if n_live is None:
            raise ValueError(
                "grouped_gemm_mxfp8: the gather record takes n_live, the routing's slots (the live rows at most)"
            )
        if (
            src.dtype != torch.int32
            or src.dim() != 1
            or not src.is_contiguous()
            or k_div < 1
            or src.numel() < n_live
            or rows < n_live
        ):
            raise ValueError(
                "grouped_gemm_mxfp8: src must be a contiguous int32 [>= n_live], k_div >= 1 and rows >= n_live"
            )
    else:
        ids, n_slots, dest = in_kernel
        if (
            ids.dtype != torch.int32
            or ids.dim() != 1
            or not ids.is_contiguous()
            or dest.dtype != torch.int32
            or dest.numel() < n_slots
            or ids.numel() < n_slots
            or rows < n_slots
            or counts.dtype != torch.int32
            or not decode_form_ok(n_slots, 1, counts.numel(), K)
        ):
            raise ValueError(
                "grouped_gemm_mxfp8: the in-kernel plan takes contiguous int32 ids [>= n_slots] and dest [>= n_slots], "
                f"int32 counts [E], rows >= n_slots, at most {PLAN_MAX_SLOTS} slots and {PLAN_MAX_E} experts"
            )
        n_live = n_slots
    a_sf8 = _bytes(a_sf, "a_sf")
    if a_sf8.numel() != T * (K // BLOCK) or not a_sf8.is_contiguous():
        raise ValueError(
            f"grouped_gemm_mxfp8: a_sf must be the token-major [T, {K // BLOCK}] UE8M0 scales of the T token rows"
        )
    if not gather_form_ok(K):
        raise ValueError(
            "grouped_gemm_mxfp8: the gather form reads the scale rows in 4-byte groups (K a multiple of 128)"
        )
    w_sf8 = _bytes(w_sf, "w_sf").reshape(-1)
    if w_sf8.numel() < E * sf_numel(N, K):
        raise ValueError(
            "grouped_gemm_mxfp8: w_sf must hold E consecutive 128x4 swizzled scale buffers of (N, K)"
        )
    if (
        sf_rows is None
        or sf_rows % GROUP_ALIGN
        or sf_rows < rows
        or sf_rows < padded_rows_bound(n_live, 1, E)
    ):
        raise ValueError(
            "grouped_gemm_mxfp8: the gather record takes the dense packing's scale rows (whole 128-row atoms covering the "
            f"rows and the padded positions of the live rows: at least padded_rows_bound({n_live}, 1, {E}) = "
            f"{padded_rows_bound(n_live, 1, E)})"
        )
    in_fmt = in_fmt or mx_class_of(a.device)
    if in_fmt not in MX_CLASSES:
        raise ValueError(f"grouped_gemm_mxfp8: unknown instruction class {in_fmt!r}")
    chain = MxChain.of(chain, in_fmt)
    if not config.swap_ab or not _k.can_implement_swapped_mxfp8(
        config, N, K, E, chain, in_fmt, 0
    ):
        raise ValueError(
            f"grouped_gemm_mxfp8: gather record {config.describe()} does not implement ({N}, {K}, {E}, {chain.describe()}, {in_fmt})"
        )
    n_out = _check_grouped_chain("grouped_gemm_mxfp8", N, chain)
    w3, a_sf_flat = w.permute(1, 2, 0), a_sf8.reshape(-1)
    if chain.quant:
        if out is not None:
            raise ValueError(
                "grouped_gemm_mxfp8: the fused requantization allocates its two outputs"
            )
        codes = _grouped_out(
            "grouped_gemm_mxfp8", None, rows, n_out, torch.uint8, a.device
        )
        sf = _quant_sf_buffer(sf_rows, n_out, a.device)
        if rows and T:
            _mx_launch(
                "gather", a, a_sf_flat, w3, w_sf8, (codes, sf), counts, config, chain, rows, E, N,
                src=src, k_div=k_div, in_kernel=in_kernel,
            )  # fmt: skip
        return codes.view(torch.float8_e4m3fn), sf
    out = _grouped_out(
        "grouped_gemm_mxfp8", out, rows, n_out, _OUT[chain.out_fmt], a.device
    )
    if rows and T:
        _mx_launch(
            "gather", a, a_sf_flat, w3, w_sf8, (out,), counts, config, chain, rows, E, N,
            src=src, k_div=k_div, in_kernel=in_kernel,
        )  # fmt: skip
    return out


# ----------------------------------------------------------------------------- the permutations


def permute_plan_padded(
    experts: torch.Tensor, E: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(counts [E] int32, dest [T, k] int32, src [Tk_pad] int32)``: the bf16 classes' deterministic permutation
    (expert order, then token-slot order) with the group offsets rounded up to 128 rows; ``src`` has
    ``padded_rows_bound`` rows and ``-1`` in every padding slot (the shared plan's padded form)."""
    counts, dest, src = permute_plan(experts, E, align=GROUP_ALIGN)
    return counts, dest, src


def scatter_mx(k: int, Tk: int, H: int, dev: torch.device):
    """The MX row scatter's compiled kernel for ``Tk`` expanded rows of ``H`` e4m3 bytes: the warps per row from the row count
    and the part, the lane's static bounds from the row width (placements of the same bytes)."""
    wpr = _k.warps_per_row(Tk, H // 16, sm_count_of(dev))
    return _k.get_compiled_scatter_mx(
        k, wpr, **_k.scatter_mx_lanes(H, padded_sf_cols(H), wpr)
    )


def permute_plan_dual_padded(
    experts: torch.Tensor, E: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(counts [E], dest [T, k], src [Tk], dest_pad [T, k], pos_pad [Tk])``: the dense plan plus the positions with the
    groups padded to 128 rows (``dest_pad[i]``) and the padded position of every dense row (``pos_pad[p]``): the dense
    packing's row map (the rows dense, the scale factors at the padded positions; the shared plan's dual form)."""
    counts, dest, src, dest_pad, pos_pad = permute_plan(
        experts, E, dual=True, pad_align=GROUP_ALIGN
    )
    return counts, dest, src, dest_pad, pos_pad


# ----------------------------------------------------------------------------- the op


def fused_moe_mxfp8(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w: Any,
    out: Optional[torch.Tensor] = None,
    config1: Optional[Knobs] = None,
    config2: Optional[Knobs] = None,
    in_fmt: Optional[str] = None,
    return_intermediates: bool = False,
    requant: str = "fused",
    plan: Optional[Tuple[torch.Tensor, ...]] = None,
    path: Optional[str] = None,
    w_round: bool = False,
    gather: Optional[bool] = None,
    decode: Optional[bool] = None,
) -> Any:
    """The MXFP8 x MXFP8 MoE FFN for pre-routed tokens (``w`` a :class:`MoEWeightsMXFP8`), and the mixed MXFP4 x MXFP8 FFN
    at a mixed instruction class ``in_fmt`` (``w`` the fp4 backend's :class:`.api_fp4.MoEWeightsFp4` of the mixed pair; the
    box form alone: ``gather`` and ``decode`` are the one-format classes'; :func:`.api_mxfp4w8a.fused_moe_mxfp4w8a` is the
    entry): ``x`` a bf16 ``[T, H]`` (quantized here per token by the
    ``mxfp8.quant`` class) or the quantized pack ``(xq [T, H] e4m3, xsf [T, H / 32] uint8 / UE8M0)`` taken as given;
    ``token_selected_experts [T, k]`` int32, ``token_final_scales [T, k]`` fp32; the output bf16 ``[T, H]``. ``in_fmt``
    names the instruction class (the device's by default); ``config1`` / ``config2`` the records (any record of either
    space gives the same bits; the dispatcher's picks by default). ``path``: the packing of the expanded rows (a placement:
    the same bits) -- ``"padded"`` (every group padded to whole 128-row atoms, the GEMMs' A boxes read the atoms) or
    ``"dense"`` (the rows back to back, the scale factors at the padded positions: the swapped decode body's packing, which
    the padded body reads too; the fused requantization only); None: the dense packing when either record is a swapped
    one, else the padded one (`schedule_mxfp8.packing_for`). ``plan``: the padded plan's ``(counts, dest, src)`` or the
    dual plan's ``(counts, dest, src, dest_pad, pos_pad)`` from ``route_plan`` (the same integers as the plans here, one
    launch with the routing); its form must match the packing. ``return_intermediates``: the live buffers (``counts``,
    ``dest``, ``src``, ``xq``, ``xsf``, ``x_perm`` / ``x_perm_sf``, ``a``, ``aq`` / ``a_sf``, ``y``; the dense packing adds
    ``dest_pad`` / ``pos_pad`` and the row counts ``Tk_a`` / ``Tk_sf``), the records used, the packing and the bodies.
    ``w_round``: the f32 weights enter the fold as their bf16 rounding, by the combine kernel (the pre-routed packed-word
    convention), as in the bf16 op. ``gather``: GEMM1's record reads the token rows itself through the plan's map (the
    gather form of a swapped record on the dense packing: no scatter launch) -- None applies the structural rule
    (`schedule_mxfp8.gather_form_rule`: the narrow swapped tiles), True forces it on a swapped GEMM1 record, False the box
    form; a placement of the same fold. ``decode``: the decode form -- GEMM1's gather record builds the plan itself from the
    ids (no plan launch, no scatter launch; its first CTA stores the counts and the row map for GEMM2 and the combine):
    None takes it where the rule holds (both records swapped, the fused requantization, no precomputed plan,
    `decode_form_ok`), True forces it (a refusal outside the rule), False never; the same integers and bits."""
    H, I, E = w.hidden_size, w.intermediate_size, w.num_experts
    mixed = getattr(w, "pair", None) is not None
    name = "fused_moe_mxfp4w8a" if mixed else "fused_moe_mxfp8"
    if isinstance(x, torch.Tensor):
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != H:
            raise TypeError(
                f"{name}: x must be a bf16 [T, {H}] or the quantized pack (xq, xsf)"
            )
        dev, T = x.device, x.shape[0]
        if T:
            xq, xsf = quantize_activations_mxfp8(x)
        else:
            xq = torch.empty(0, H, dtype=torch.float8_e4m3fn, device=dev)
            xsf = torch.empty(0, H // BLOCK, dtype=torch.uint8, device=dev)
    else:
        xq, xsf = x
        dev, T = xq.device, xq.shape[0]
    if xq.dtype != torch.float8_e4m3fn or tuple(xq.shape) != (T, H):
        raise TypeError(f"{name}: xq must be an e4m3 [T, {H}]")
    xsf8 = _bytes(xsf, "xsf")
    if T == 0 and xsf8.numel() == 0:
        xsf8 = xsf8.reshape(
            0, H // BLOCK
        )  # the empty pack: no scale byte, whatever its shape
    if tuple(xsf8.shape) != (T, H // BLOCK) or not xsf8.is_contiguous():
        raise TypeError(
            f"{name}: xsf must be the token-major [T, {H // BLOCK}] UE8M0 scales of the activation pack (the "
            f"linear layout; a swizzled or flattened buffer is not reshaped into rows), got {tuple(xsf8.shape)}"
        )
    # the shared fused-entry rules (the rows' alignment at the class's scale block, the ids and weights, the output)
    T, H, k, experts, scales, out = _check_fused_inputs(
        name,
        xq,
        token_selected_experts,
        token_final_scales,
        out,
        H,
        torch.float8_e4m3fn,
        BLOCK,
        "e4m3",
    )
    cls = w.pair.cls if mixed else CLASS
    if xq.data_ptr() % 16 or xsf8.data_ptr() % 16:
        raise unsupported(
            cls,
            "quantized rows or scale bytes at a storage offset that is not a multiple of 16 bytes",
            "16-byte aligned rows and scale bytes (the kernels' vector loads and TMA boxes)",
        )
    if mixed:
        from .api_mxfp4w8a import mx_class_of as mixed_class_of

        in_fmt = in_fmt or mixed_class_of(dev)
        if not _is_mixed(in_fmt):
            raise ValueError(f"{name}: {in_fmt!r} is not a mixed instruction class")
        # the pair's complete shape rule at the class (the granule of the part)
        w.pair.check_shape(H, I, E, k, w.gated, in_fmt=in_fmt)
        if gather or decode:
            raise ValueError(
                f"{name}: the gather and decode forms are the one-format classes' (the mixed classes read the token box)"
            )
        gather = decode = False
    else:
        check_shape_mxfp8(H, I, E, k)
        in_fmt = in_fmt or mx_class_of(dev)
        if _is_mixed(in_fmt):
            raise ValueError(f"{name}: {in_fmt!r} is a mixed instruction class")
    if T == 0:  # no tokens: nothing to route, the (0, H) output as it stands
        if return_intermediates:
            return out, dict(
                counts=torch.zeros(E, dtype=torch.int32, device=dev),
                in_fmt=in_fmt,
                Tk_pad=0,
            )
        return out
    if requant not in ("fused", "standalone"):
        raise ValueError(f"requant must be 'fused' or 'standalone', got {requant!r}")
    if config1 is None or config2 is None:
        # the standalone requantization is a padded-path spelling: its dispatcher ranks the padded records only; the mixed
        # classes rank with their own dispatcher (the fused spelling GEMM1 runs)
        sm_count, family = _part(dev)
        if mixed:
            from .schedule_mxfp4w8a import default_configs_mx as default_configs_mixed

            c1, c2 = default_configs_mixed(
                T, k, H, I, E, w.chain1_quant, in_fmt, sm_count, family,
                swap=None if (requant == "fused" and path != "padded") else False,
            )  # fmt: skip
        else:
            c1, c2 = _k.default_configs_mx(
                T,
                k,
                H,
                I,
                E,
                w.chain1,
                in_fmt,
                sm_count,
                family,
                swap=None if (requant == "fused" and path != "padded") else False,
            )
        config1, config2 = config1 or c1, config2 or c2
    if path is None:
        path = _k.packing_for(config1, config2)
    if path not in PACKINGS:
        raise ValueError(f"{name}: unknown packing {path!r}")
    if path == "padded" and (config1.swap_ab or config2.swap_ab):
        raise ValueError(
            f"{name}: a swapped record reads the dense packing (path='dense')"
        )
    if gather is None and path == "dense" and requant == "fused":
        config1 = _k.gather_form_rule(config1, _k.rows_per_expert(T, k, E))
    elif gather:
        if not config1.swap_ab or path != "dense" or requant != "fused":
            raise ValueError(
                f"{name}: the gather form is GEMM1's swapped record on the dense packing with the fused requantization"
            )
        config1 = dataclasses.replace(config1, gather_a=True)
    elif gather is False:
        config1 = dataclasses.replace(config1, gather_a=False)
    if config1.gather_a and not gather_form_ok(H):
        # the pack's scale rows are not whole 4-byte groups: the box form through the scatter
        config1 = dataclasses.replace(config1, gather_a=False)
    if config2.gather_a:
        raise ValueError(f"{name}: the gather form is GEMM1's")
    if decode and gather is not False:
        config1 = dataclasses.replace(
            config1, gather_a=True
        )  # the decode form is the gather record's
    decode_ok = (
        plan is None
        and path == "dense"
        and requant == "fused"
        and config1.swap_ab
        and config2.swap_ab
        and config1.gather_a
        and decode_form_ok(T, k, E, H)
    )
    if decode and not decode_ok:
        raise ValueError(
            f"{name}: the decode form takes two swapped records on the dense packing with the fused requantization, "
            "no precomputed plan, the gather form and a routing within the in-kernel plan's bounds"
        )
    in_kernel = bool(decode) if decode is not None else decode_ok
    if plan is not None and len(plan) != (5 if path == "dense" else 3):
        raise ValueError(
            f"{name}: the plan's form (padded: 3 tensors, dual: 5) must match the packing"
        )
    # the records' acceptance (the grouped entry's rule, for a caller's records)
    c1_chain = MxChain.of(w.chain1_quant if requant == "fused" else w.chain1, in_fmt)
    for cfg, N, K, chain in (
        (config1, w.w1.shape[1], H, c1_chain),
        (config2, H, I, MxChain.of(GEMM2_CHAIN, in_fmt)),
    ):
        if cfg.gather_a:
            ok = cfg.swap_ab and _k.can_implement_swapped_mxfp8(
                cfg, N, K, E, chain, in_fmt, 0
            )
            kind = "gather record"
        elif cfg.swap_ab:
            kb = 0 if mixed else _k.swap_k_block_arg(in_fmt, K, None)
            ok = _k.can_implement_swapped_mxfp8(cfg, N, K, E, chain, in_fmt, kb)
            kind = "swapped record"
        else:
            ok = _k.can_implement_grouped_mxfp8(cfg, N, K, E, chain, in_fmt)
            kind = "record"
        if not ok:
            raise ValueError(
                f"{name}: {kind} {cfg.describe()} does not implement (N={N}, K={K}, E={E}, {chain.describe()}, {in_fmt})"
            )
    return _fused_moe_mxfp8_run(
        xq, xsf8, experts, scales, w, out, config1, config2, in_fmt, path, requant, in_kernel, T, H, k, E, I,
        plan=plan, return_intermediates=return_intermediates, w_round=w_round,
    )  # fmt: skip


def _mx_forms(
    config1: Knobs,
    config2: Knobs,
    T: int,
    k: int,
    E: int,
    H: int,
    path: str,
    plan_given: bool,
) -> Tuple[Knobs, bool]:
    """GEMM1's form at the dispatcher's records on the dense packing with the fused requantization (the runner's case:
    :func:`fused_moe_mxfp8`'s rules with ``gather`` and ``decode`` unset): the gather record where the structural rule and
    the pack's scale rows admit it, else the box form; the decode form (the plan built in the kernel) when both records are
    swapped, GEMM1's the gather record, no plan was given and the routing fits the in-kernel plan. Returns ``(config1,
    in_kernel)``."""
    if path == "dense":
        config1 = _k.gather_form_rule(config1, _k.rows_per_expert(T, k, E))
        if config1.gather_a and not gather_form_ok(H):
            config1 = dataclasses.replace(config1, gather_a=False)
    in_kernel = (
        not plan_given
        and path == "dense"
        and config1.swap_ab
        and config2.swap_ab
        and config1.gather_a
        and decode_form_ok(T, k, E, H)
    )
    return config1, in_kernel


def _fused_moe_mxfp8_run(
    xq: torch.Tensor,
    xsf8: torch.Tensor,
    experts: torch.Tensor,
    scales: torch.Tensor,
    w: Any,
    out: torch.Tensor,
    config1: Knobs,
    config2: Knobs,
    in_fmt: str,
    path: str,
    requant: str,
    in_kernel: bool,
    T: int,
    H: int,
    k: int,
    E: int,
    I: int,
    *,
    plan: Optional[Tuple[torch.Tensor, ...]] = None,
    return_intermediates: bool = False,
    w_round: bool = False,
) -> Any:
    """:func:`fused_moe_mxfp8` on validated operands (``xq`` e4m3 ``[T, H]``, ``xsf8`` its ``[T, H / 32]`` scale bytes) with
    the records, the packing, the requantization spelling and the plan form chosen: the runner's launch (its pack proved every
    rule the public entry checks, so nothing is checked twice per call) and the public entry's body."""
    mixed = getattr(w, "pair", None) is not None
    dev = xq.device
    nb, pcols = H // BLOCK, padded_sf_cols(H)
    w1_3, w2_3 = w.w1.permute(1, 2, 0), w.w2.permute(1, 2, 0)
    w1sf8, w2sf8 = (
        _bytes(w.w1_sf, "w_sf").reshape(-1),
        _bytes(w.w2_sf, "w_sf").reshape(-1),
    )
    n1 = w.w1.shape[1]
    chain1q = MxChain.of(w.chain1_quant, in_fmt)
    chain2 = MxChain.of(GEMM2_CHAIN, in_fmt)
    kb1 = 0 if (mixed or not config1.swap_ab) else _k.swap_k_block_arg(in_fmt, H, None)
    kb2 = 0 if (mixed or not config2.swap_ab) else _k.swap_k_block_arg(in_fmt, I, None)
    kind1 = "swapped" if config1.swap_ab else "grouped"
    kind2 = "swapped" if config2.swap_ab else "grouped"
    if path == "dense":
        # the dense packing fed by the dense TMA box: the rows back to back, the scale factors at the padded positions; the A
        # buffers carry DENSE_TAIL rows past the live rows so the last tiles' boxes stay inside them (never stored), the scale
        # buffers past the padded rows
        if requant != "fused":
            raise ValueError(
                "fused_moe_mxfp8: the dense packing runs the fused requantization"
            )
        Tk = T * k
        Tk_pad = padded_rows_bound(T, k, E)
        Tk_a, Tk_sf = Tk + DENSE_TAIL, Tk_pad + DENSE_TAIL
        x_perm = x_perm_sf = None
        # GEMM1's fused requantization: the codes and the 128x4 scale bytes of the bf16-rounded chain, written by the epilogue
        aq8 = torch.empty(Tk_a, I, dtype=torch.uint8, device=dev)
        a_sf = _quant_sf_buffer(Tk_sf, I, dev)
        if in_kernel:
            # the decode form: GEMM1 builds the plan from the ids itself and stores the counts and the row map
            counts = torch.empty(E, dtype=torch.int32, device=dev)
            dest = torch.empty(Tk, dtype=torch.int32, device=dev)
            src = dest_pad = pos_pad = None
            if Tk_a and T:
                _mx_launch(
                    "gather", xq, xsf8.reshape(-1), w1_3, w1sf8, (aq8, a_sf), counts, config1, chain1q, Tk_a, E, n1,
                    k_div=k, in_kernel=(experts.reshape(-1), Tk, dest),
                )  # fmt: skip
            dest = dest.view(T, k)
            if return_intermediates:
                # the maps the kernel never materialises, for the intermediates (the same integers as its dest and counts)
                _, _, src, dest_pad, pos_pad = permute_plan_dual_padded(experts, E)
        else:
            counts, dest, src, dest_pad, pos_pad = (
                plan if plan is not None else _permute_plan_run(experts.reshape(-1), T, k, E, 1, True, GROUP_ALIGN)
            )  # fmt: skip
        if in_kernel:
            pass
        elif config1.gather_a:
            # the gather form: GEMM1 reads the token rows and their scale bytes through the plan's map (no scatter launch)
            if Tk_a and T:
                _mx_launch(
                    "gather", xq, xsf8.reshape(-1), w1_3, w1sf8, (aq8, a_sf), counts, config1, chain1q, Tk_a, E, n1,
                    src=src, k_div=k,
                )  # fmt: skip
        else:
            x_perm = torch.empty(Tk_a, H, dtype=torch.float8_e4m3fn, device=dev)
            x_perm_sf = torch.empty(sf_numel(Tk_sf, H), dtype=torch.uint8, device=dev)
            scatter_mx(k, Tk, H, dev)(
                xq.view(torch.uint8),
                xsf8.reshape(-1),
                dest.reshape(-1),
                dest_pad.reshape(-1),
                x_perm.view(torch.uint8),
                x_perm_sf,
                Tk,
                H,
                nb,
                pcols,
            )
            if Tk_a:
                _mx_launch(
                    kind1, x_perm, x_perm_sf, w1_3, w1sf8, (aq8, a_sf), counts, config1, chain1q, Tk_a, E, n1,
                    Tk_sf, kb1, "dense",
                )  # fmt: skip
        aq = aq8.view(torch.float8_e4m3fn)
        y = torch.empty(Tk_a, H, dtype=torch.bfloat16, device=dev)
        if Tk_a:
            _mx_launch(
                kind2, aq, a_sf, w2_3, w2sf8, (y,), counts, config2, chain2, Tk_a, E, H, Tk_sf, kb2, "dense"
            )  # fmt: skip  # [Tk_a, H] bf16
        combine_rows(y, dest, scales, out, k, w_round)
        if return_intermediates:
            return out, dict(
                counts=counts,
                dest=dest,
                src=src,
                dest_pad=dest_pad,
                pos_pad=pos_pad,
                xq=xq,
                xsf=xsf8,
                x_perm=x_perm,
                x_perm_sf=x_perm_sf,
                plan="in_kernel" if in_kernel else "launch",
                a=None,
                aq=aq,
                a_sf=a_sf,
                y=y,
                config1=config1,
                config2=config2,
                in_fmt=in_fmt,
                Tk_pad=Tk_pad,
                Tk_a=Tk_a,
                Tk_sf=Tk_sf,
                path=path,
                requant=requant,
                swap=(config1.swap_ab, config2.swap_ab),
            )
        return out
    # ``plan``: ``route_plan(..., align=GROUP_ALIGN)``'s outputs (the same integers as the padded plan, one launch with the routing)
    counts, dest, src = (
        plan if plan is not None else _permute_plan_run(experts.reshape(-1), T, k, E, GROUP_ALIGN, False, GROUP_ALIGN)
    )  # fmt: skip
    Tk_pad = src.shape[0]
    x_perm = torch.empty(Tk_pad, H, dtype=torch.float8_e4m3fn, device=dev)
    x_perm_sf = torch.empty(sf_numel(Tk_pad, H), dtype=torch.uint8, device=dev)
    # one launch over the live rows: the e4m3 rows and the UE8M0 scales to their padded positions (the scales re-tiled into
    # the swizzled atoms); the padding rows are never written (unspecified bytes the MMA reads into rows no tile stores)
    scatter_mx(k, T * k, H, dev)(
        xq.view(torch.uint8),
        xsf8.reshape(-1),
        dest.reshape(-1),
        dest.reshape(-1),
        x_perm.view(torch.uint8),
        x_perm_sf,
        T * k,
        H,
        nb,
        pcols,
    )
    if requant == "fused":
        # the class's second quantization inside GEMM1's epilogue: no bf16 intermediate
        a = None
        aq8 = torch.empty(Tk_pad, I, dtype=torch.uint8, device=dev)
        a_sf = _quant_sf_buffer(Tk_pad, I, dev)
        if Tk_pad:
            _mx_launch(
                "grouped", x_perm, x_perm_sf, w1_3, w1sf8, (aq8, a_sf), counts, config1, chain1q, Tk_pad, E, n1,
                Tk_pad, 0, "padded",
            )  # fmt: skip
        aq = aq8.view(torch.float8_e4m3fn)
    else:
        a = torch.empty(Tk_pad, I, dtype=torch.bfloat16, device=dev)  # [Tk_pad, I] bf16
        if Tk_pad:
            _mx_launch(
                "grouped", x_perm, x_perm_sf, w1_3, w1sf8, (a,), counts, config1, MxChain.of(w.chain1, in_fmt), Tk_pad,
                E, n1, Tk_pad, 0, "padded",
            )  # fmt: skip
        aq, a_sf = quantize_mxfp8_rows(
            a, swizzled=True, enable_pdl=True, valid=src
        )  # the quantizer class, second pass, the live rows
    y = torch.empty(Tk_pad, H, dtype=torch.bfloat16, device=dev)  # [Tk_pad, H] bf16
    if Tk_pad:
        _mx_launch(
            "grouped", aq, a_sf, w2_3, w2sf8, (y,), counts, config2, chain2, Tk_pad, E, H, Tk_pad, 0, "padded"
        )  # fmt: skip
    combine_rows(y, dest, scales, out, k, w_round)
    if return_intermediates:
        return out, dict(
            counts=counts,
            dest=dest,
            src=src,
            xq=xq,
            xsf=xsf8,
            x_perm=x_perm,
            x_perm_sf=x_perm_sf,
            a=a,
            aq=aq,
            a_sf=a_sf,
            y=y,
            config1=config1,
            config2=config2,
            in_fmt=in_fmt,
            Tk_pad=Tk_pad,
            path=path,
            requant=requant,
            swap=(False, False),
        )
    return out


def _part(device: torch.device) -> Tuple[int, str]:
    from ...gemm.batch_invariant.api import part_of

    return part_of(device)
