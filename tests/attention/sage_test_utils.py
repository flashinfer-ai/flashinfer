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

"""Sage attention test helpers shared by the PrimTS decode and context kernel tests.

Scales follow the flat trtllm-gen layout of ``SageAttentionParams``; the
references dequantize 8-bit inputs under them and fold score columns into
online-softmax streams as both kernels do.
"""

import math

import pytest
import torch

from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_constants import (
    E4M3_MAX,
)
from flashinfer.attention.prims_ts.sage.config import _INT8_QK_COMPUTE_CAPABILITIES

FP8 = torch.float8_e4m3fn

# INT8 Q/K launch only where the tcgen05 MMA takes INT8 operands.
REQUIRES_INT8_QK_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in _INT8_QK_COMPUTE_CAPABILITIES,
    reason="INT8 Q/K Sage attention requires SM100",
)
# INT8 Q/K are drawn with this standard deviation and their scales divided by
# it, so dequantized INT8 and E4M3 inputs share one distribution.
INT8_STD = 40.0


@pytest.fixture(autouse=True)
def ieee_float32_references(monkeypatch: pytest.MonkeyPatch):
    """Keep FP32 references independent of the environment's TF32 preference."""
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous_precision)


def scale_slots(batch_size: int, seq_len: int, block_size: int) -> torch.Tensor:
    """Return the trtllm-gen flat scale slot ``b * S // blk + b + t // blk`` ``[B, S]``."""

    batch = torch.arange(batch_size, device="cuda")[:, None]
    token = torch.arange(seq_len, device="cuda")
    return batch * seq_len // block_size + batch + token // block_size


def random_scales(heads, batch_size, seq_len, block_size, low, high) -> torch.Tensor:
    """Uniform ``[low, high)`` scales of ``ceil(B * S / blk) + B - 1`` slots per head."""

    numel = -(-batch_size * seq_len // block_size) + batch_size - 1
    return torch.empty((heads, numel), device="cuda").uniform_(low, high)


def random_qk(shape, dtype: torch.dtype) -> torch.Tensor:
    """Random Q or K data; INT8 values are drawn with standard deviation ``INT8_STD``."""
    values = torch.randn(shape, device="cuda")
    if dtype == torch.int8:
        return (values * INT8_STD).round().clamp(-127, 127).to(torch.int8)
    return values.to(dtype)


def random_qk_scales(
    num_qo_heads: int,
    num_kv_heads: int,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    q_block_size: int,
    k_block_size: int,
    qk_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random positive Q and K scales; INT8 scales shrink by ``INT8_STD``."""
    unit = 1.0 / INT8_STD if qk_dtype == torch.int8 else 1.0
    q_scale = random_scales(
        num_qo_heads, batch_size, seq_len_q, q_block_size, 0.1 * unit, 0.4 * unit
    )
    k_scale = random_scales(
        num_kv_heads, batch_size, seq_len_kv, k_block_size, 0.5 * unit, 2.0 * unit
    )
    return q_scale, k_scale


def dequantize(x: torch.Tensor, scales: torch.Tensor, block_size: int):
    """Return FP32 ``[B, S, H, D]`` values of 8-bit data under flat-layout scales."""

    slots = scale_slots(x.shape[0], x.shape[1], block_size)
    return x.float() * scales[:, slots].permute(1, 2, 0).unsqueeze(-1)


def fold(
    state,
    logits: torch.Tensor,
    values: torch.Tensor,
    headroom: float,
    p_scale: float | None = None,
):
    """Fold one set of columns into an online-softmax stream ``(anchor, sum, acc)``.

    The stream keeps its anchor while the columns raise the row maximum by at
    most ``headroom`` binades, as the kernel does. The row sum takes FP32
    probabilities; the PV operand quantizes them to E4M3 at ``p_scale`` times
    their ratio to the anchor, by default ``E4M3_MAX * 2**-headroom``.
    """

    if p_scale is None:
        p_scale = E4M3_MAX * 2.0**-headroom

    if state is None:
        total = torch.zeros(logits.shape[:-1], device="cuda")
        acc = torch.zeros((*logits.shape[:-1], values.shape[-1]), device="cuda")
        state = (total - float("inf"), total, acc)
    running_max, total, acc = state
    new_max = torch.maximum(running_max, logits.amax(dim=-1))
    lag_log2 = (new_max - running_max) * math.log2(math.e)
    new_max = torch.where(
        running_max.isfinite() & (lag_log2 <= headroom), running_max, new_max
    )
    anchor = torch.where(new_max.isfinite(), new_max, 0.0)
    probabilities = torch.exp(logits - anchor.unsqueeze(-1)) * p_scale
    correction = torch.exp(running_max - anchor)
    return (
        new_max,
        total * correction + probabilities.sum(dim=-1),
        acc * correction.unsqueeze(-1) + probabilities.to(FP8).float() @ values,
    )


def reference_sage_scales(
    layout, params, *, batch_size: int, num_kv_heads: int
) -> torch.Tensor:
    """Build the prepared scales from the flat tensors as ``_SageScaleLayout`` describes them.

    Per sequence (batch entry and KV head): each K section's atoms in word
    order, a token past the section's length taking its last scale and
    padding words zero; each Q tile's rows in the section's row order, a row
    past the sequence's end taking the last token's scale of its head; then
    the ``sfV`` row and, with a mean, the ``v_mean`` row.
    """

    batch, hkv = batch_size, num_kv_heads
    sequences = torch.zeros((batch, hkv, layout.sequence_words), device="cuda")
    sources = (params.k_scale, getattr(params, "k_summary_scale", None))
    sources = sources[: len(layout.k_sections)]
    for section, scales in zip(layout.k_sections, sources, strict=True):
        atoms = torch.arange(section.atoms, device="cuda")[:, None]
        words = torch.arange(section.words_per_atom, device="cuda")
        tokens = (atoms * 64 + words * section.tokens_per_word).clamp(
            max=section.length - 1
        )
        slots = scale_slots(batch, section.length, section.source_block_size)
        columns = section.word_offset + atoms * section.atom_words + words
        sequences[:, :, columns] = scales[:, slots[:, tokens]].permute(1, 0, 2, 3)
    q_rows = layout.q_rows
    tiles = torch.arange(q_rows.tiles, device="cuda")[:, None]
    rows = torch.arange(q_rows.rows_per_tile, device="cuda")
    tokens, local_heads = q_rows.row_token_and_head(tiles, rows)
    tokens = tokens.clamp(max=q_rows.seq_len_q - 1)
    kv_heads = torch.arange(hkv, device="cuda")[:, None, None]
    heads = kv_heads * q_rows.heads_q_per_kv + local_heads
    slots = scale_slots(batch, q_rows.seq_len_q, q_rows.q_block_size)
    q_words = params.q_scale[heads[None], slots[:, tokens][:, None]]
    v_rows = layout.v_rows
    sequences[:, :, q_rows.word_offset : v_rows.word_offset] = q_words.flatten(2)
    rows = (params.v_scale,) + ((params.v_mean,) if layout.v_mean else ())
    sequences[:, :, v_rows.word_offset :] = torch.cat(rows, dim=-1)
    return sequences.flatten()
