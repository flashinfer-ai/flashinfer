# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""FP4 pair descriptors shared by the host runner and the kernel API.

Keep kernel imports inside methods so importing the runner for cubin packaging
or backend discovery does not require CUTLASS DSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from ...batch_invariant.quantization.layout import padded_sf_cols, sf_numel
from ...gemm.batch_invariant.spec import MXFP4_FMTS, NVFP4_FMTS
from .spec_mxfp4 import CLASS as MX_CLASS
from .spec_mxfp4 import MX_BLOCK, expert_classes_mx
from .spec_nvfp4 import CLASS as NV_CLASS
from .spec_nvfp4 import NVFP4_BLOCK, TE_GS_INV


@dataclass(frozen=True)
class Fp4Pair:
    """A quantization pair of the fp4 backend: ``name`` tags the weight view and the kernel objects; ``cls`` is the registry
    class; ``block`` the elements per scale; ``fmts`` the instruction classes; ``scaled`` whether the pair carries a per-tensor
    alpha per expert and GEMM and a per-token scale per row (the NVFP4 pair, which also serves the static-intermediate
    class) -- the MX pair's scales are the block scales alone."""

    name: str
    cls: str
    block: int
    fmts: Tuple[str, ...]
    scaled: bool

    def sf_numel(self, rows: int, K: int) -> int:
        """Bytes of the 128x4 swizzled scale buffer of ``rows`` rows of ``K`` elements."""
        return sf_numel(rows, K, self.block, "128x4")

    def sf_cols(self, K: int) -> int:
        """Scale columns of a row of ``K`` elements in the 128x4 layout (whole 4-column groups)."""
        return padded_sf_cols(K, self.block, "128x4")

    def classes(
        self, H: int, I: int, family: Optional[str], gated: bool = True
    ) -> Tuple[str, str]:
        """The instruction class of each expert GEMM: the dense rule per weight shape (NVFP4), the part's (MXFP4)."""
        if self.scaled:
            from .api_fp4 import expert_classes

            return expert_classes(H, I, family, gated)
        return expert_classes_mx(family)

    def check_shape(
        self,
        H: Optional[int],
        I: int,
        E: int,
        k: Optional[int] = None,
        gated: bool = True,
    ) -> None:
        from .api_fp4 import check_shape_fp4

        check_shape_fp4(self, H, I, E, k, gated)

    def prepare_activations(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The layer's activation pack of the pair: bf16 rows as given and no scale, for both served pairs (the class
        quantizes the rows inside the op: per token for NVFP4, per 32-element block for MXFP4). A pair whose activations
        arrive quantized by the caller declares its own form here."""
        return hidden_states.to(torch.bfloat16).contiguous(), None

    def quantize(
        self,
        x: torch.Tensor,
        gs_inv: Union[float, torch.Tensor] = TE_GS_INV,
        out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """The pair's quantization of bf16 rows ``[M, K]``: ``(x4 [M, K/2] uint8, sf flat 128x4 uint8, pts [M] f32 or None)``
        -- per token for NVFP4 (``gs_inv`` the recipe's inverse base scale), per 32-element block for MXFP4 (no per-token
        scale). ``out = (x4, sf)`` preallocated buffers."""
        from .api_fp4 import quantize_activations, quantize_rows_mx

        if self.scaled:
            return quantize_activations(x, gs_inv, out)
        x4, sf = quantize_rows_mx(x, out)
        return x4, sf, None


NVFP4 = Fp4Pair("nvfp4", NV_CLASS, NVFP4_BLOCK, NVFP4_FMTS, True)
MXFP4 = Fp4Pair("mxfp4", MX_CLASS, MX_BLOCK, MXFP4_FMTS, False)
