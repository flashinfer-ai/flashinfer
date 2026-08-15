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

"""SM120 CuteDSL backend for the MXFP8-activation x MXFP4-weight grouped MoE GEMM.

Zero-padding mode with token-packed ``a`` and a CSR ``m_indptr``, matching the
``moe_gemm_*_nt_groupwise`` family. The kernels live in
``flashinfer.grouped_mm.kernels.cutedsl.sm120_moe``; this module is the public entry.
"""

from typing import Any, Literal, Optional, Tuple

import torch

from ...api_logging import flashinfer_api
from ...autotuner import AutoTuner
from ...utils import supported_compute_capability
from .._sm120_moe_autotune import SM120_MOE_TUNING_CONFIG, Sm120MoeTunableRunner
from ..kernels.cutedsl.sm120_moe.core._common import ceil_div
from ..kernels.cutedsl.sm120_moe.core.sm120_blockscaled_layout import (
    Sm120SfConfigMxfp8Mxfp4,
)
from ..kernels.cutedsl.sm120_moe.core.sm120_gemm_builder import StoreMethod
from ..kernels.cutedsl.sm120_moe.moe_gemm_mxfp8_mxfp4 import GRANK_A, make_args
from ._arm import (
    TACTICS,
    CutedslSm120GroupedMxfp8Mxfp4Op,
    _check_a_scale_granularity,
    _op,
    compiled_kernel,
    select_tile,
    split_tactic,
)

_MXFP8_MXFP4_TACTIC_SCHEMA_VERSION = 1
_SCALE_CONTRACT = "mxfp8_mxfp4_mn_major_int32_ue8m0"
# The A side admits both the DeepGEMM-style 128 and the OCP MXFP8 32; the B side is fixed at 32 by
# the MXFP4 spec, so it is not a parameter.
_SUPPORTED_A_GRANULARITY = (128, 32)


def _encoded_tactics() -> tuple:
    """`(schema, bm, bn, store)` as ints, so the autotuner's cache can round-trip a tactic."""
    return tuple(
        (_MXFP8_MXFP4_TACTIC_SCHEMA_VERSION, bm, bn, StoreMethod[store].value)
        for bm, bn, store in TACTICS
    )


def _decode_tactic(tactic: tuple):
    _, bm, bn, store_value = tactic
    store = StoreMethod(store_value)
    return split_tactic((bm, bn, store.name))


def _check_scale_granularity_mnk(scale_granularity_mnk: Tuple[int, int, int]) -> int:
    """Validate the A-side granularity and return its K component."""
    if len(scale_granularity_mnk) != 3:
        raise ValueError(
            f"scale_granularity_mnk must be a 3-tuple (m_gran, n_gran, k_gran); "
            f"got length {len(scale_granularity_mnk)}"
        )
    if scale_granularity_mnk[0] != 1 or scale_granularity_mnk[1] != 1:
        raise ValueError(
            f"scale_granularity_mnk[0:2] must both be 1 (per-token scaling along M and N); "
            f"got {scale_granularity_mnk[:2]}"
        )
    if scale_granularity_mnk[2] not in _SUPPORTED_A_GRANULARITY:
        raise ValueError(
            f"scale_granularity_mnk[2] (k_gran) describes the activation side and must be one of "
            f"{_SUPPORTED_A_GRANULARITY}; got {scale_granularity_mnk[2]}"
        )
    return scale_granularity_mnk[2]


def _check_m_indptr(m_indptr: torch.Tensor, num_experts: int) -> None:
    """Metadata shape only; value invariants would force a GPU->CPU sync on a low-latency path."""
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be torch.int32; got {m_indptr.dtype}")
    if tuple(m_indptr.shape) != (num_experts + 1,):
        raise ValueError(
            f"m_indptr must have shape ({num_experts + 1},); got {tuple(m_indptr.shape)}"
        )


class _CutedslSm120Mxfp8Mxfp4MoeRunner(Sm120MoeTunableRunner):
    """Tunable view of the arm. The base keys the cache on granularity already."""

    def __init__(
        self,
        out: torch.Tensor,
        scale_granularity_mnk: Tuple[int, int, int],
        scale_major_mode: str,
        grank_a: int,
    ) -> None:
        super().__init__(
            out,
            False,
            scale_granularity_mnk,
            scale_major_mode,
            _encoded_tactics(),
            _MXFP8_MXFP4_TACTIC_SCHEMA_VERSION,
            _SCALE_CONTRACT,
        )
        self._grank_a = grank_a

    def get_valid_tactics(self, inputs, profile) -> list:
        """The base returns everything; feasibility here depends on the shape, so filter."""
        a, b = inputs[0], inputs[1]
        n, k = int(b.shape[1]), int(b.shape[2]) * 2
        out_dtype = self._out.dtype if self._out is not None else torch.bfloat16
        valid = []
        for tactic in self._tactics:
            tile, store = _decode_tactic(tactic)
            if CutedslSm120GroupedMxfp8Mxfp4Op.can_implement(
                n=n, k=k, tile=tile, out_dtype=out_dtype, store=store, grank_a=self._grank_a
            ):
                valid.append(tactic)
        return valid

    def is_valid_tactic(self, tactic: Any, inputs=None) -> bool:
        """Four ints, not three: slot 0 is the schema version and the store method is an axis."""
        if type(tactic) is not tuple or len(tactic) != 4:
            return False
        if any(type(value) is not int for value in tactic):
            return False
        return tactic in self._tactics

    def get_cache_key_extras(self, inputs) -> tuple:
        return super().get_cache_key_extras(inputs) + (self._grank_a,)

    def _launch(self, inputs, out, is_gated, tactic) -> None:
        a, b, a_scale, b_scale, m_indptr = inputs
        n, k = int(b.shape[1]), int(b.shape[2]) * 2
        props = torch.cuda.get_device_properties(a.device)
        grid_x = props.multi_processor_count
        if self.is_valid_tactic(tactic):
            tile, store = _decode_tactic(tactic)
        else:
            tile, store = (
                select_tile(
                    total_rows=int(a.shape[0]),
                    n=n,
                    k=k,
                    num_experts=int(b.shape[0]),
                    num_sms=grid_x,
                ),
                None,
            )
        op = _op(n, k, tuple(tile), out.dtype, store, self._grank_a)
        args = make_args(a, b, a_scale, b_scale, out, m_indptr)
        compiled_kernel(
            args, op=op, grid_x=grid_x, sm_version=f"sm_{props.major}{props.minor}"
        )(*args)


@supported_compute_capability([120, 121])
@flashinfer_api
def moe_gemm_mxfp8_mxfp4_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 1, 128),
    scale_major_mode: Literal["MN"] = "MN",
    backend: Literal["cutedsl"] = "cutedsl",
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Grouped GEMM with MXFP8 activations and packed MXFP4 weights in zero-padding mode.

    Currently only supported on NVIDIA RTX PRO 6000 Blackwell (SM120) architecture.

    Zero-padding mode accepts token-packed input ``a`` (no per-expert pre-padding along M)
    with a CSR cumsum group descriptor ``m_indptr``. It targets decoding, where the
    per-expert M can be as small as 1 and DeepGEMM-style contiguous padding would waste
    memory and compute.

    Parameters
    ----------
    a: torch.Tensor
        Row-major activation tensor, shape ``(cum_m, k)``, data type ``torch.float8_e4m3fn``.
        Token-packed across experts; ``cum_m`` is the cumulative sum of the segment lengths.

    b: torch.Tensor
        Column-major weight tensor, shape ``(num_experts, n, k // 2)``, data type
        ``torch.uint8``: two E2M1 values per byte.

    a_scale: torch.Tensor
        UE8M0 activation scales, ``torch.uint8`` viewing INT32 storage (4 scales per INT32
        along K). MN-major with M contiguous: shape ``(k_align_a, m_padded * 4)`` where
        ``k_align_a = ceil(k / (k_gran * 4))`` and ``m_padded = (cum_m + num_experts * 3) // 4 * 4``.
        Expert ``i``'s scales start at column ``(m_indptr[i] + 3 * i) // 4 * 4``. The 4-column
        padding is what keeps the TMA global stride 16-byte aligned; it is not optional.

    b_scale: torch.Tensor
        UE8M0 weight scales, ``torch.uint8`` viewing INT32 storage, MN-major with N
        contiguous: shape ``(num_experts, k_packs_b, n_padded * 4)`` where
        ``k_packs_b = ceil(k / 128)`` and ``n_padded = ceil(n / 4) * 4``. The weight side is
        fixed at block-32 by the MXFP4 spec.

    m_indptr: torch.Tensor
        Segment-length indptr, shape ``(num_experts + 1,)``, ``torch.int32``.
        ``m_indptr[0] = 0``, ``m_indptr[num_experts] = cum_m``.

    scale_granularity_mnk: Tuple[int, int, int]
        ``(m_granularity, n_granularity, k_granularity)`` of the **activation** scales.
        ``m`` and ``n`` must both be ``1`` (per-token). ``k`` is ``128`` (DeepGEMM-style
        production, default) or ``32`` (OCP MXFP8). The weight side takes no parameter: MXFP4
        fixes it at 32.

    scale_major_mode: Literal["MN"]
        Layout mode of the scale tensors. Only ``"MN"`` is supported.

    backend: Literal["cutedsl"]
        Backend selector. Only the CuteDSL backend is implemented for this dtype pair.

    out: Optional[torch.Tensor]
        Output tensor, shape ``(cum_m, n)``. Allocated when not given.

    out_dtype: Optional[torch.dtype]
        Output data type. Only ``torch.bfloat16`` is supported.

    Returns
    -------
    out: torch.Tensor
        The output tensor, shape ``(cum_m, n)``.

    Notes
    -----
    - Both scale tensors are UE8M0 bytes viewed over INT32 storage, so their trailing extent
      counts bytes: an ``(x, y)`` INT32 tensor is passed as ``(x, y * 4)`` ``uint8``.
    - The kernel derives the scales' K extent from ``scale_granularity_mnk`` rather than
      reading it off the tensor, so a mismatch between how ``a_scale`` was packed and what is
      declared here cannot be detected downstream; it is validated on entry instead.
    """
    if backend != "cutedsl":
        raise NotImplementedError(
            f'Only backend="cutedsl" is implemented for MXFP8 x MXFP4; got backend="{backend}"'
        )
    if scale_major_mode != "MN":
        raise NotImplementedError(
            f'Only scale_major_mode="MN" is supported; got "{scale_major_mode}"'
        )
    grank_a = _check_scale_granularity_mnk(scale_granularity_mnk)
    _check_m_indptr(m_indptr, num_experts=int(b.shape[0]))

    if out_dtype is None:
        out_dtype = out.dtype if out is not None else torch.bfloat16
    if out_dtype != torch.bfloat16:
        raise NotImplementedError(
            f"Only out_dtype=torch.bfloat16 is supported; got {out_dtype}"
        )

    n, k = int(b.shape[1]), int(b.shape[2]) * 2
    _check_a_scale_granularity(a_scale, k, grank_a)
    if out is None:
        out = torch.zeros((a.shape[0], n), dtype=out_dtype, device=a.device)

    inputs = [a, b, a_scale, b_scale, m_indptr]
    runner = _CutedslSm120Mxfp8Mxfp4MoeRunner(
        out, scale_granularity_mnk, scale_major_mode, grank_a
    )
    runner, tactic = AutoTuner.get().choose_one(
        "cutedsl_sm120_mxfp8_mxfp4_groupwise_moe",
        [runner],
        SM120_MOE_TUNING_CONFIG,
        inputs,
    )
    runner(inputs, tactic=tactic)
    return out
