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

"""The cutedsl backend behind moe_gemm_fp8_nt_groupwise: an autotuner runner plus its launcher."""

from typing import Any, Tuple

import torch

from ...autotuner import AutoTuner
from ..kernels.cutedsl.sm120_moe.core.sm120_gemm_builder import StoreMethod
from ..kernels.cutedsl.sm120_moe.moe_gemm_fp8 import make_args
from .._sm120_moe_autotune import SM120_MOE_TUNING_CONFIG, Sm120MoeTunableRunner
from ._arm import (
    TACTICS,
    CutedslSm120GroupedFp8Op,
    _op,
    compiled_kernel,
    select_tile,
    split_tactic,
)

_FP8_TACTIC_SCHEMA_VERSION = 1
_SCALE_CONTRACT = "fp8_1x128x128_mn_major_f32"


def _encoded_tactics() -> tuple:
    """`(schema, bm, bn, store)` as ints; the store method is an axis, so three slots do not fit."""
    return tuple(
        (_FP8_TACTIC_SCHEMA_VERSION, bm, bn, StoreMethod[store].value)
        for bm, bn, store in TACTICS
    )


def _decode_tactic(tactic: tuple):
    _, bm, bn, store_value = tactic
    return split_tactic((bm, bn, StoreMethod(store_value).name))


class _CutedslSm120Fp8MoeRunner(Sm120MoeTunableRunner):
    """Tunable view of the self-written fp8 arm."""

    def __init__(
        self,
        out: torch.Tensor,
        is_gated: bool,
        scale_granularity_mnk: Tuple[int, int, int],
        scale_major_mode: str,
    ) -> None:
        super().__init__(
            out,
            is_gated,
            scale_granularity_mnk,
            scale_major_mode,
            _encoded_tactics(),
            _FP8_TACTIC_SCHEMA_VERSION,
            _SCALE_CONTRACT,
        )

    def get_valid_tactics(self, inputs, profile) -> list:
        b = inputs[1]
        n, k = int(b.shape[1]), int(b.shape[2])
        out_dtype = self._out.dtype if self._out is not None else torch.bfloat16
        return [
            tactic
            for tactic in self._tactics
            if CutedslSm120GroupedFp8Op.can_implement(
                n=n,
                k=k,
                tile=_decode_tactic(tactic)[0],
                out_dtype=out_dtype,
                store=_decode_tactic(tactic)[1],
            )
        ]

    def is_valid_tactic(self, tactic: Any, inputs=None) -> bool:
        if type(tactic) is not tuple or len(tactic) != 4:
            return False
        if any(type(value) is not int for value in tactic):
            return False
        return tactic in self._tactics

    def _launch(self, inputs, out, is_gated, tactic) -> None:
        a, b, a_scale, b_scale, m_indptr = inputs
        n, k = int(b.shape[1]), int(b.shape[2])
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
        op = _op(n, k, tuple(tile), out.dtype, store)
        args = make_args(a, a_scale, b, b_scale, out, m_indptr)
        compiled_kernel(
            args, op=op, grid_x=grid_x, sm_version=f"sm_{props.major}{props.minor}"
        )(*args)


def launch_cutedsl_fp8_moe(
    inputs, out: torch.Tensor, scale_granularity_mnk, scale_major_mode, is_gated: bool
) -> None:
    """Entry point moe_gemm_fp8_nt_groupwise calls when backend="cutedsl"."""
    if is_gated:
        raise NotImplementedError(
            'backend="cutedsl" does not implement the fused gated (SwiGLU) path; '
            'use backend="cute" for is_gated=True'
        )
    runner = _CutedslSm120Fp8MoeRunner(
        out, is_gated, scale_granularity_mnk, scale_major_mode
    )
    runner, tactic = AutoTuner.get().choose_one(
        "cutedsl_sm120_fp8_groupwise_moe",
        [runner],
        SM120_MOE_TUNING_CONFIG,
        inputs,
    )
    runner(inputs, tactic=tactic)
