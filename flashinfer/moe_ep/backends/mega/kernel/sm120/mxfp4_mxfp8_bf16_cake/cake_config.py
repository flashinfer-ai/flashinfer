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

"""Configuration for the Cake SM120 MXFP4 x MXFP8 persistent MegaMoE backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

SHARED_EXPERT_FP8_IN_KERNEL = "fp8_in_kernel"
TRANSPORT_TABLE_NAMES = ("all_gin", "all_lsa", "lsa_where_reachable")
TRANSPORT_TABLE_AUTO = "auto"
TRANSPORT_TABLE_TUNED = "tuned"
CODEC_OVERFLOW_BLOCKS_RANGE = (16, 256)

TransportTableSpec = Union[str, tuple]


@dataclass
class Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig:
    """Static model parameters of the Cake persistent MegaMoE kernel.

    ``intermediate_size`` is the post-SwiGLU width (``2 * intermediate_size``
    gate+up rows are consumed per expert).  ``top_k`` routes per token.

    ``shared_expert``:

    * ``None`` (default): routed top-k sum only; the caller adds the shared
      expert (``noshared`` kernel variant).
    * ``"fp8_in_kernel"``: the kernel also evaluates one shared expert on
      FP8 e4m3 weights and adds it to the output (``fp8shared`` variant).
      Requires the shared-expert weights in the weight pack
      (:class:`CakePrequantizedMoEWeightsWithSharedExpert`).

    ``transport_table`` selects, per pair of EP ranks, how the kernel moves
    dispatch payloads and expert results: ``SELF`` (same rank), ``LSA``
    (load/store over NVLink or PCIe peer mappings, driven by copier CTAs) or
    ``GIN`` (NCCL GPU-initiated networking puts).  Every pair that is LSA
    reachable may use either; pairs on different hosts use GIN.

    * ``"all_gin"`` (default): every remote pair uses GIN; the kernel runs its
      copier-free schedule instances (no SM is reserved for copies).
    * ``"all_lsa"`` / ``"lsa_where_reachable"``: a fixed table with LSA cells.
      Any LSA cell selects the copier instances, which reserve one copier CTA
      per LSA destination of a rank (the largest LSA count of any table row,
      compiled in as ``MOE_LSA_COPIERS``); each copier pushes the dispatch
      payload and the expert results of its destination with TMA bulk copies.
    * a flat tuple of ``world_size * world_size`` cells (``0`` SELF on the
      diagonal, ``1`` LSA, ``2`` GIN, symmetric): an explicit table, validated
      against the measured LSA reachability and GIN availability, e.g. LSA only
      between the two GPUs behind one PCIe switch.
    * ``"tuned"``: the table recorded for this deployment (GPU, geometry, variant,
      row capacity and peer topology) by the offline tuner
      ``python -m flashinfer.moe_ep.tune --dtype sm120_mxfp4_mxfp8 ...`` in the moe_ep
      knob cache (``FLASHINFER_MOE_EP_KNOB_CACHE``); ``all_gin`` when nothing is
      recorded.  Resolved once at workspace creation (rank 0 reads, broadcasts).
    * ``"auto"``: a setup-time probe scores the candidate tables (``all_gin``,
      ``all_lsa``, ``lsa_where_reachable``, ``lsa_where_faster``) under a
      concurrent all-to-all and keeps a mixed table only when it reproducibly
      beats the best uniform one by 3 %.  The choice is made by rank 0 and
      broadcast; ``transport_probe_report`` on the workspace records it.  Near a
      decision margin the probe may pick different tables on different launches,
      so production deployments should pin the table (or use the tuned table,
      see ``flashinfer.moe_ep.tune --dtype sm120_mxfp4_mxfp8``).

    Shapes: the device code is one parametric source compiled per deployment.  EP world
    size comes from :class:`~flashinfer.moe_ep.config.BootstrapConfig`; the global expert
    count, hidden size and ``max_tokens_per_rank`` from
    :class:`~flashinfer.moe_ep.config.FleetParams`; ``intermediate_size``, ``top_k`` and
    the variant from this config.  Supported region (checked in ``validate_init``, the
    same guards the source enforces at compile time): EP world size 2..32,
    ``num_experts % world_size == 0``
    with at most 128 experts per rank and 1024 in total, ``hidden_size`` a multiple of
    512 up to 7680, ``intermediate_size`` a multiple of 128, ``top_k`` from 1 to 9; the
    kernel's row capacity is ``max_tokens_per_rank`` rounded up to 128 (at most 65408).
    Above 16 EP ranks ``validate_init`` also compiles the device instances and rejects
    parameter sets whose build spills registers.

    ``codec_overflow_blocks`` (compile-time): the expert results return through a
    lossless BF16 block codec (64-value blocks, 96 B coded or 128 B raw); each
    return chunk reserves room for ``rows * codec_overflow_blocks`` raw blocks.
    A chunk that needs more sets the kernel's ``protocol_error`` flag and that call's
    output is not valid.  The flag is per call (cleared by the next launch) and the
    forward path does not read it: check it with the backend's
    ``protocol_error(workspace)`` (a host synchronization) after a call when needed,
    and discard the workspace after a call that set it.
    Default 16 (25 % of a row at ``hidden_size=4096``, 14 % at 7168;
    real DeepSeek-V4 activations need ~0.3 %); 16..256.  Each extra block costs
    ``128 * max_tokens_per_rank * top_k`` bytes per peer and ring slot of
    symmetric memory.

    Numerics: MXFP8 E4M3 activations (UE8M0 K32 scales) x MXFP4 E2M1 routed weights,
    SwiGLU ``silu(gate) * up``, FP8 requant of the intermediate, BF16 output; no expert
    biases.  ``swiglu_limit`` (a compile-time parameter of the device build) clamps the
    SwiGLU inputs to ``gate <= L`` and ``-L <= up <= L`` before the activation:
    ``10`` (default) is the DeepSeek-V4 clamp, ``0`` is standard unclamped SwiGLU
    (DeepSeek-V3, Qwen3).

    EPLB: pass ``num_experts`` = physical experts (logical experts plus redundant
    replicas, divisible by the EP world size) and physical expert ids in ``topk_ids``;
    each rank's weight pack holds its ``num_experts // world_size`` physical experts
    (replicas carry a copy of their logical expert's weights).  The logical-to-physical
    choice stays with the router.
    """

    intermediate_size: int = 2048
    top_k: int = 6
    swiglu_limit: int = 10
    codec_overflow_blocks: int = 16
    kernel_name: str = "sm120_mxfp4_mxfp8_bf16_cake"
    shared_expert: str | None = None
    transport_table: TransportTableSpec = "all_gin"
    fast_math: bool = True
    knobs: dict | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.swiglu_limit, int)
            or isinstance(self.swiglu_limit, bool)
            or self.swiglu_limit < 0
        ):
            raise ValueError(
                f"swiglu_limit must be a non-negative int (0 = unclamped SwiGLU), got {self.swiglu_limit!r}"
            )
        lo, hi = CODEC_OVERFLOW_BLOCKS_RANGE
        if (
            not isinstance(self.codec_overflow_blocks, int)
            or isinstance(self.codec_overflow_blocks, bool)
            or not lo <= self.codec_overflow_blocks <= hi
        ):
            raise ValueError(
                f"codec_overflow_blocks must be an int in [{lo}, {hi}], got {self.codec_overflow_blocks!r}"
            )
        if self.shared_expert not in (None, SHARED_EXPERT_FP8_IN_KERNEL):
            raise ValueError(
                f"shared_expert must be None or {SHARED_EXPERT_FP8_IN_KERNEL!r}, got {self.shared_expert!r}"
            )
        table = self.transport_table
        if isinstance(table, list):
            table = self.transport_table = tuple(table)
        if (
            table in (TRANSPORT_TABLE_AUTO, TRANSPORT_TABLE_TUNED)
            or table in TRANSPORT_TABLE_NAMES
        ):
            return
        if not isinstance(table, tuple) or not all(
            isinstance(c, int) and not isinstance(c, bool) and c in (0, 1, 2)
            for c in table
        ):
            raise ValueError(
                f"transport_table must be {TRANSPORT_TABLE_AUTO!r}, {TRANSPORT_TABLE_TUNED!r}, one of "
                f"{TRANSPORT_TABLE_NAMES}, or a flat tuple of 0/1/2 cells, got {table!r}"
            )

    @property
    def variant(self) -> str:
        return (
            "fp8shared"
            if self.shared_expert == SHARED_EXPERT_FP8_IN_KERNEL
            else "noshared"
        )


__all__ = [
    "CODEC_OVERFLOW_BLOCKS_RANGE",
    "SHARED_EXPERT_FP8_IN_KERNEL",
    "TRANSPORT_TABLE_AUTO",
    "TRANSPORT_TABLE_TUNED",
    "TRANSPORT_TABLE_NAMES",
    "Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig",
]
