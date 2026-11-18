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

"""Configuration for the SM120 MXFP4 x MXFP8 nosync MegaMoE kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass

TRANSPORTS = ("p2p", "auto", "nic")
TOKENS_PER_ITEM = (8, 16, 24, 32)
# The kernel is measured faster than cake and cutedsl up to 128 tokens per rank (RTX 6000D and
# RTX PRO 6000, DeepSeek-V4-Flash, DeepSeek-V4.1-Flash and GLM-5.3-Flash MoE shapes); its expert
# pools hold max_tokens_per_rank rows per (local expert, source rank).
MAX_TOKENS_PER_RANK = 128


@dataclass
class Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig:
    """Static parameters of the nosync kernel (at most 128 tokens per rank).

    Deployments normally reach it through :class:`NosyncSwitchConfig`, which runs it for
    small batches and the deployment's own backend for the rest.

    ``intermediate_size`` is the post-SwiGLU width (``w13`` has ``2 * intermediate_size``
    rows, gate rows then up rows). ``swiglu_limit`` clamps the SwiGLU inputs:
    ``silu(min(gate, limit)) * clamp(up, -limit, limit)``; ``inf`` for none.

    ``transport`` selects how ranks reach each other:

    * ``"p2p"`` (default): PCIe peer-to-peer stores between every pair (torch symmetric
      memory).
    * ``"auto"``: the buffer is an NCCL window; pairs NCCL maps for load/store use
      peer-to-peer stores, the others (e.g. across CPU sockets where that path is slow) send
      over the NIC with NCCL GIN.
    * ``"nic"``: NCCL GIN for every other rank.
    * a flat tuple of ``world_size * world_size`` cells in the cake backend's encoding
      (``0`` self, ``1`` peer-to-peer, ``2`` GIN, symmetric): the transport of each pair, e.g.
      the table the cake backend resolved for this deployment (``NosyncOptions`` passes it
      on). Without a GIN cell it is ``"p2p"``; a peer-to-peer cell must be a pair NCCL maps.

    ``"auto"``, ``"nic"`` and tables with GIN cells need NCCL 2.30.7 with the Device API and a GIN-capable network;
    every rank must use the same value. Which pairs NCCL maps depends on the NICs it may use:
    with ``NCCL_IB_HCA`` naming a single NIC every pair stayed mapped on an 8x RTX 6000D host
    with ConnectX-8 PCIe switches, so ``"auto"`` sent the cross-socket pairs over P2P too;
    naming every NIC the ranks can reach each other through moved them to GIN. That host's
    GPUs store to two remote PCIe switches at once at ~1.5 GB/s (~50 GB/s to one), so there
    EP8 needs the second setting (DSV4-Flash, 128 tokens per rank: 11 ms all-P2P, 0.75 ms).

    ``tokens_per_item`` is the number of tokens one work item multiplies per weight load (8,
    16, 24 or 32). ``None`` picks 16 when this rank's routed rows per local expert
    (``tokens * top_k / local_experts``) reach 3, else 8 (measured with cold L2 on RTX 6000D).
    Set it explicitly after sweeping a deployment's shapes.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm120_mxfp4_mxfp8_bf16_nosync"
    swiglu_limit: float = 10.0
    transport: str | tuple[int, ...] = "p2p"
    tokens_per_item: int | None = None
    fast_math: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.transport, (tuple, list)):
            cells = tuple(int(c) for c in self.transport)
            world = math.isqrt(len(cells))
            if (
                world * world != len(cells)
                or any(c not in (0, 1, 2) for c in cells)
                or any(
                    (cells[i * world + j] == 0) != (i == j)
                    or cells[i * world + j] != cells[j * world + i]
                    for i in range(world)
                    for j in range(world)
                )
            ):
                raise ValueError(
                    "a transport table is world_size * world_size cells, 0 on the diagonal "
                    f"only, 1 (P2P) or 2 (GIN) elsewhere, symmetric; got {self.transport!r}"
                )
            self.transport = cells
        elif self.transport not in TRANSPORTS:
            raise ValueError(
                f"transport must be one of {TRANSPORTS} or a per-pair table, "
                f"got {self.transport!r}"
            )
        if self.tokens_per_item is not None and (
            self.tokens_per_item not in TOKENS_PER_ITEM
        ):
            raise ValueError(
                f"tokens_per_item must be None or one of {TOKENS_PER_ITEM}, "
                f"got {self.tokens_per_item!r}"
            )
