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

from typing import NamedTuple, Optional

import torch

from ..api_logging import flashinfer_api
from ..tllm_enums import SfLayout
from .packbits import get_quantization_module


class CsfScales(NamedTuple):
    r"""Block-scale matrices compressed per 16-row slab (CSF).

    Each slab of ``fixed`` holds its 16 rows' base bytes, then the rows' 4-bit codes, two per
    byte, low nibble first; a scale byte is its row's base plus its code. ``exceptions`` holds
    the bytes outside their row's 16-value window, matrix after matrix, in position order:
    each the flat ``row * cols + col`` position within its matrix in the low 24 bits and the
    byte in the high 8. Slab ``s`` of the ``num_matrices * rows // 16`` slabs owns
    ``exceptions[exc_offsets[s]:exc_offsets[s + 1]]``.

    Attributes
    ----------
    fixed : torch.Tensor
        uint8 ``[num_matrices, rows // 16, 16 * (1 + cols // 2)]``.
    exceptions : torch.Tensor
        uint32 ``[num_exceptions]``.
    exc_offsets : torch.Tensor
        int32 ``[num_matrices * rows // 16 + 1]``.
    rows, cols : int
        The shape of each matrix.
    """

    fixed: torch.Tensor
    exceptions: torch.Tensor
    exc_offsets: torch.Tensor
    rows: int
    cols: int


@flashinfer_api
def encode_csf_scales(scales: torch.Tensor) -> CsfScales:
    r"""Compress block-scale matrices into CSF slabs.

    A row's base is the lowest start among the 16-value windows that cover the most of its
    bytes, the rule of CSF checkpoints, so encoding a checkpoint's decoded scales gives back
    its ``nvfp4_csf_fixed`` and ``nvfp4_csf_exceptions`` streams.

    Parameters
    ----------
    scales : torch.Tensor
        Contiguous CUDA ``[rows, cols]`` or ``[num_matrices, rows, cols]`` block scales in a
        one-byte dtype such as ``float8_e4m3fn``, with ``rows`` a multiple of 16, ``cols``
        even and ``rows * cols <= 2**24``.

    Returns
    -------
    CsfScales
        The compressed scales. Sizing the exceptions reads their count back to the host, so
        encoding cannot run under CUDA graph capture.
    """
    if not scales.is_cuda:
        raise ValueError(f"CSF scales must be on a CUDA device, got {scales.device}")
    if scales.element_size() != 1:
        raise ValueError(f"CSF scales must have a one-byte dtype, got {scales.dtype}")
    device = scales.device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("encode_csf_scales cannot run under CUDA graph capture")
        scales = scales.view(torch.uint8)
        if scales.dim() == 2:
            scales = scales.unsqueeze(0)
        num_matrices, rows, cols = scales.shape
        fixed = torch.empty(
            num_matrices,
            rows // 16,
            16 * (1 + cols // 2),
            dtype=torch.uint8,
            device=device,
        )
        row_exceptions = torch.empty(
            num_matrices * rows, dtype=torch.int32, device=device
        )
        module = get_quantization_module()
        module.csf_encode_rows(scales, fixed, row_exceptions)
        row_offsets = torch.nn.functional.pad(
            row_exceptions.cumsum(0, dtype=torch.int64), (1, 0)
        )
        num_exceptions = int(row_offsets[-1])
        if num_exceptions >= 1 << 31:
            raise ValueError(f"{num_exceptions} CSF exceptions overflow int32 offsets")
        exceptions = torch.empty(num_exceptions, dtype=torch.uint32, device=device)
        module.csf_encode_exceptions(scales, fixed, row_offsets, exceptions)
        return CsfScales(
            fixed, exceptions, row_offsets[::16].to(torch.int32), rows, cols
        )


@flashinfer_api
def expand_csf_scales(
    csf: CsfScales,
    out: torch.Tensor,
    token_selected_experts: Optional[torch.Tensor] = None,
    layout: SfLayout = SfLayout.layout_linear,
    swap_row_halves: bool = False,
    local_expert_offset: int = 0,
) -> torch.Tensor:
    r"""Expand CSF block scales into a scale buffer, one matrix per expert.

    Nothing is read back to the host, so the expansion can run under CUDA graph capture, with
    the routing changing between replays.

    Parameters
    ----------
    csf : CsfScales
        Scales from :func:`encode_csf_scales`, one matrix per expert. Expansion takes about
        192 bytes of shared memory per column, so ``cols`` can be at most 864 on SM80, 1200
        on SM90 and SM100, and 518 on SM86, SM89 and SM120.
    out : torch.Tensor
        Contiguous destination in a one-byte dtype on the scales' device, written in place:

        - ``layout_128x4``: ``[num_experts, round_up(rows, 128), round_up(cols, 4)]`` in the
          128x4 swizzled layout of the fused MoE kernels, the padding written as zeros.
        - ``layout_linear``: ``[num_experts, rows, cols]``.
    token_selected_experts : Optional[torch.Tensor]
        Contiguous int32 expert ids of any shape, such as the ``[num_tokens, top_k]`` routing
        of a fused MoE call. Only these experts are expanded and the rest of ``out`` keeps its
        bytes; ids outside ``[local_expert_offset, local_expert_offset + num_experts)`` are
        skipped. ``None`` expands every expert.
        An expert named more than once is expanded once: in each 128-row tile, every id is
        compared with all the ids before it, so ``N`` ids take ``O(N^2)`` comparisons per tile.
        ``None`` costs the same for any routing. Pass ids only for small routings; with 512
        experts on SM80, for example, 640 ids expanded faster than ``None`` and 1280 slower.
    layout : SfLayout
        ``SfLayout.layout_128x4`` or ``SfLayout.layout_linear``.
    swap_row_halves : bool
        Swap the two halves of each matrix's rows, e.g. to turn ``[w1; w3]`` checkpoint
        order into the ``[w3; w1]`` order of the fused MoE kernels. Needs ``rows % 32 == 0``.
    local_expert_offset : int
        The id in ``token_selected_experts`` of ``csf``'s first expert. Under expert
        parallelism the routing holds global ids while ``csf`` holds this rank's experts, so
        pass the global id of this rank's first expert.

    Returns
    -------
    torch.Tensor
        ``out``.
    """
    if out.element_size() != 1:
        raise ValueError(f"out must have a one-byte dtype, got {out.dtype}")
    with torch.cuda.device(out.device):
        get_quantization_module().csf_expand(
            csf.fixed,
            csf.exceptions,
            csf.exc_offsets,
            token_selected_experts,
            out.view(torch.uint8),
            int(layout),
            swap_row_halves,
            local_expert_offset,
        )
    return out
