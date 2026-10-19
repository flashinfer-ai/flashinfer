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

# Configuration of the experimental Cake W8A8 MXFP8 MegaMoE EP16 backend (SM103).

from __future__ import annotations

from dataclasses import dataclass

KERNEL_NAME = "sm103_mxfp8_mxfp8_bf16_cake"
INTERMEDIATE_SIZE = 5120
TOP_K = 8
MAX_ROUTES_PER_EXPERT = 32


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass
class Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig:
    """Static parameters of the Cake W8A8 MXFP8 MegaMoE EP16 route.

    The route has no free shape axis: every field below is a fixed contract
    value and ``__post_init__`` rejects anything else. The EP geometry it
    serves is EP world size 16, 512 experts (32 per rank), hidden size 3072,
    intermediate size 5120, top-8 routing, at most 64 tokens per rank and at
    most ``max_routes_per_expert`` (32) routes to one expert per forward.

    * ``intermediate_size`` must be 5120 and ``top_k`` must be 8 (read by
      :class:`~flashinfer.moe_ep.modes.mega_layer.MoEEpMegaLayer`).
    * ``fast_math`` must be ``True``: the device programs are compiled with
      fast math; ``False`` raises ``NotImplementedError``.
    * ``knobs`` must be ``None``: the route carries no tunable knobs.
    * ``max_routes_per_expert`` is a read-only contract constant exposed for
      callers and tests. Routing beyond it is not validated on the host (no
      per-call synchronization); the ``local_pack`` program counts such
      routes in a device-side overflow flag that ``routing_overflow(workspace)``
      exposes.
    """

    intermediate_size: int = INTERMEDIATE_SIZE
    top_k: int = TOP_K
    kernel_name: str = KERNEL_NAME
    fast_math: bool = True
    knobs: None = None
    max_routes_per_expert: int = MAX_ROUTES_PER_EXPERT

    def __post_init__(self) -> None:
        if not _is_int(self.intermediate_size) or (
            self.intermediate_size != INTERMEDIATE_SIZE
        ):
            raise ValueError(
                f"{KERNEL_NAME}: intermediate_size must be {INTERMEDIATE_SIZE}, "
                f"got {self.intermediate_size!r}"
            )
        if not _is_int(self.top_k) or self.top_k != TOP_K:
            raise ValueError(
                f"{KERNEL_NAME}: top_k must be {TOP_K}, got {self.top_k!r}"
            )
        if self.kernel_name != KERNEL_NAME:
            raise ValueError(
                f"kernel_name must be {KERNEL_NAME!r}, got {self.kernel_name!r}"
            )
        if self.knobs is not None:
            raise ValueError(
                f"{KERNEL_NAME} has no tunable knobs; knobs must be None, "
                f"got {self.knobs!r}"
            )
        if (
            not _is_int(self.max_routes_per_expert)
            or self.max_routes_per_expert != MAX_ROUTES_PER_EXPERT
        ):
            raise ValueError(
                f"{KERNEL_NAME}: max_routes_per_expert is a contract constant "
                f"({MAX_ROUTES_PER_EXPERT}), got {self.max_routes_per_expert!r}"
            )
        if self.fast_math is not True:
            raise NotImplementedError(
                f"{KERNEL_NAME} is compiled with fast math; fast_math=False is "
                "not supported"
            )


__all__ = [
    "INTERMEDIATE_SIZE",
    "KERNEL_NAME",
    "MAX_ROUTES_PER_EXPERT",
    "Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig",
    "TOP_K",
]
