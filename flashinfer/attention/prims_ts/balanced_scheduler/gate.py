# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture-stable admission policy for balanced PrimsTS MLA decode."""

from __future__ import annotations

from collections.abc import Mapping
import os
from types import MappingProxyType
from typing import Optional

import torch


EXPECTED_MEAN_SEQ_LEN_ENV = "FLASHINFER_MLA_EXPECTED_MEAN_SEQ_LEN"
EXPECTED_MAX_SEQ_LEN_ENV = "FLASHINFER_MLA_EXPECTED_MAX_SEQ_LEN"

# B200 decode-only admission thresholds. These are routing policy, not the
# per-replay scheduler cost model. Recalibrate the table when a kernel change
# moves the balanced/default crossover. A missing head-count or dtype row is
# deliberately conservative and routes to the standard kernel.
BALANCED_MLA_GATE_THRESHOLDS: Mapping[tuple[int, str], tuple[int, int]] = (
    MappingProxyType(
        {
            (128, "bf16"): (2, 8_192),
            (64, "bf16"): (4, 32_768),
            (32, "bf16"): (2, 131_072),
            (16, "bf16"): (4, 262_144),
            (128, "fp8"): (4, 131_072),
            (64, "fp8"): (4, 262_144),
            (32, "fp8"): (4, 262_144),
            (16, "fp8"): (4, 524_288),
        }
    )
)

_GATE_DTYPE_KEYS = {
    (torch.bfloat16, torch.bfloat16): "bf16",
    (torch.float8_e4m3fn, torch.float8_e4m3fn): "fp8",
}
_MIN_CONCENTRATION_NUMERATOR = 2
_SUPPORTED_KV_LORA_RANK = 512
_SUPPORTED_QK_ROPE_HEAD_DIM = 64


def _resolve_expected_length(
    value: Optional[int],
    *,
    env_name: str,
    argument_name: str,
) -> Optional[int]:
    if value is None:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            return None
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"{env_name} must be a positive integer") from error
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{argument_name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{argument_name} must be positive")
    return value


def should_use_prims_ts_balanced_mla(
    *,
    batch_size: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    max_kv_len: int,
    max_seq_len_q: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    expected_mean_seq_len: Optional[int] = None,
    expected_max_seq_len: Optional[int] = None,
) -> bool:
    """Return the capture-stable balanced MLA routing decision.

    The two expected lengths describe one deployment or CUDA-graph bucket and
    must remain fixed from capture through every replay. Explicit arguments
    take precedence over :data:`EXPECTED_MEAN_SEQ_LEN_ENV` and
    :data:`EXPECTED_MAX_SEQ_LEN_ENV`.

    ``expected_mean_seq_len * batch_size`` estimates total K/V work, while
    ``expected_max_seq_len / expected_mean_seq_len`` estimates whether a long
    request exists for the balanced scheduler to redistribute. Balanced is
    admitted only when the expected concentration is at least two and both the
    calibrated batch and work thresholds are met.

    Unsupported query widths, head counts, dimensions, or dtype pairs route to
    the standard scheduler without requiring declarations. A supported auto
    configuration requires both declarations rather than guessing from live
    sequence lengths or a model context limit; a missing declaration raises
    :class:`ValueError` rather than returning ``False``.

    This predicate deliberately does not inspect the CUDA device or its
    calibration registry. A ``True`` result is only the route decision;
    automatic planning subsequently checks the exact-device calibration and
    falls back to ordinary MLA with a warning when none is registered.
    Explicit balanced planning remains fail-closed.
    """

    if max_seq_len_q != 1:
        return False
    if (
        kv_lora_rank != _SUPPORTED_KV_LORA_RANK
        or qk_rope_head_dim != _SUPPORTED_QK_ROPE_HEAD_DIM
    ):
        return False
    dtype_key = _GATE_DTYPE_KEYS.get((q_dtype, kv_dtype))
    thresholds = BALANCED_MLA_GATE_THRESHOLDS.get((num_heads, dtype_key))
    if thresholds is None:
        return False

    expected_mean_seq_len = _resolve_expected_length(
        expected_mean_seq_len,
        env_name=EXPECTED_MEAN_SEQ_LEN_ENV,
        argument_name="expected_mean_seq_len",
    )
    expected_max_seq_len = _resolve_expected_length(
        expected_max_seq_len,
        env_name=EXPECTED_MAX_SEQ_LEN_ENV,
        argument_name="expected_max_seq_len",
    )
    missing = []
    if expected_mean_seq_len is None:
        missing.append(EXPECTED_MEAN_SEQ_LEN_ENV)
    if expected_max_seq_len is None:
        missing.append(EXPECTED_MAX_SEQ_LEN_ENV)
    if missing:
        raise ValueError(
            "balanced PrimsTS MLA auto selection requires " + " and ".join(missing)
        )
    if expected_max_seq_len < expected_mean_seq_len:
        raise ValueError(
            "expected_max_seq_len must be greater than or equal to "
            "expected_mean_seq_len"
        )

    # max_kv_len is the plan's capture-stable capacity. A deployment-level
    # expected maximum above this particular bucket cannot contribute work to
    # the captured kernel, so cap it before evaluating concentration.
    effective_expected_max = min(expected_max_seq_len, max_kv_len)
    if effective_expected_max < _MIN_CONCENTRATION_NUMERATOR * expected_mean_seq_len:
        return False

    min_batch, min_tokens = thresholds
    return batch_size >= min_batch and expected_mean_seq_len * batch_size >= min_tokens


__all__ = [
    "BALANCED_MLA_GATE_THRESHOLDS",
    "EXPECTED_MAX_SEQ_LEN_ENV",
    "EXPECTED_MEAN_SEQ_LEN_ENV",
    "should_use_prims_ts_balanced_mla",
]
