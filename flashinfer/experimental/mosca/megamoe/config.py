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

# Configuration of the experimental MOSCA MegaMoE backend (SM100, FP8 x FP4).
#
# Lightweight on purpose: importing this package registers the backend with
# the moe_ep mega-kernel registry (core never imports it). The MOSCA changes
# and the validated coverage are listed in ../catalog.py.

from __future__ import annotations

import math
from dataclasses import dataclass

KERNEL_NAME = "sm100_fp8_fp4_bf16_mosca"

#: ``token_layout`` values. ``"partitioned"``: every rank passes its own tokens
#: and receives its own rows (the contract of ``sm100_fp8_fp4_bf16_deepgemm``).
#: ``"replicated"``: every rank passes the same full batch (attention TP), each
#: rank computes its ``torch.tensor_split`` slice, and the kernel all-gathers
#: the rows in-kernel, so every rank receives the full batch.
TOKEN_LAYOUTS = ("partitioned", "replicated")

#: MOSCA's tile rule for DeepSeek-V4-Pro at DP-attention + EP8 (DEP8): the
#: per-rank token ranges where DeepGEMM's heuristic picked a slower tile in
#: SGLang, with the shared expert in kernel. Outputs are unchanged with it, but
#: its speed is not validated on this backend (shared experts separate): at
#: T=64 it measured slower here. Pass it as ``tile_rule`` only to re-derive it.
DSV4_PRO_DEP8_TILE_RULE = (
    "t>=41&t<=56:block_m=32;t>=57&t<=80:block_m=64;"
    "t>=193&t<=451:block_m=96,store_block_m=16"
)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass
class Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig:
    """Kernel parameters of the MOSCA MegaMoE backend (experimental).

    The kernel is derived from DeepGEMM's SM100 ``fp8_fp4_mega_moe`` with MOSCA's
    changes (``../VENDOR.md``): FP8 (E4M3, per-32 UE8M0 scales) activations, packed FP4
    (E2M1, per-32 UE8M0 scales) expert weights, SwiGLU, BF16 combine, dispatch
    and combine over NVLink symmetric memory. Shape-parameterized like the
    stable ``Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig``; every MOSCA feature is
    a separate switch, and with all of them off the kernel is the upstream
    DeepGEMM kernel.

    * ``small_t``: when every token fits SM 0's first dispatch round, SM 0
      publishes the final per-expert counts and the cross-SM count round and
      its grid sync are skipped. Outputs are bitwise identical either way. The
      round holds ``(dispatch threads / 32) * (32 // top_k)`` tokens per rank:
      20 at top-6 with 128 dispatch threads (DeepSeek-V4-Pro), so the switch
      acts on decode-sized batches and is a no-op above that.
    * ``sparse_count`` (requires ``small_t``): only the SMs that own tokens add
      to the global send counters. Outputs are bitwise identical either way.
    * ``token_layout="replicated"``: the in-kernel all-gather (see
      ``TOKEN_LAYOUTS``). Requires BF16 combine (always the case here).
    * ``tile_rule``: MOSCA's per-call tile override, ``rule;rule;...`` with
      ``rule = [cond:]key=value,...``, ``cond = t<=N&t>=M...`` over the
      per-rank token count ``t``; keys ``block_m``, ``store_block_m``,
      ``num_stages``. Every value is validated against the kernel's static
      constraints; an invalid rule raises. ``None`` keeps DeepGEMM's heuristic.
      ``DSV4_PRO_DEP8_TILE_RULE`` is MOSCA's DEP8 rule for DeepSeek-V4-Pro;
      its speed is not validated on this backend (see the constant).
    * ``activation_clamp`` (required, finite): SwiGLU input clamp, and
      ``fast_math`` the kernel's fast-math template switch, as in the stable
      DeepGEMM backend. "No clamp" (+inf upstream) is not supported yet: the
      launcher instantiates the kernel in host code, and nvcc's host stub cannot
      spell a +inf template argument.
    * ``num_sms``: the persistent grid (an even number of SMs, at most the
      device's). ``None`` uses every SM, as upstream DeepGEMM does by default;
      SGLang hands DeepGEMM all but two SMs (``deep_gemm.set_num_sms``).
    * ``precompile`` (default on): when the workspace is prepared, build every
      kernel module the planner can select for the per-rank capacity, and have
      every EP rank confirm the build. ``forward`` then never compiles, and a
      failed build raises on every rank at warmup instead of leaving peers
      waiting in the kernel's NVLink barrier. Off (single rank only, refused
      for EP > 1, where one rank compiling inside ``forward`` would leave its
      peers in the barrier): modules are built on first use, and CUDA-graph
      capture of a launch configuration that was never run raises.
    """

    intermediate_size: int
    top_k: int
    activation_clamp: float
    kernel_name: str = KERNEL_NAME
    fast_math: bool = True
    small_t: bool = True
    sparse_count: bool = False
    token_layout: str = "partitioned"
    tile_rule: str | None = None
    num_sms: int | None = None
    precompile: bool = True
    knobs: None = None

    def __post_init__(self) -> None:
        if self.kernel_name != KERNEL_NAME:
            raise ValueError(
                f"kernel_name must be {KERNEL_NAME!r}, got {self.kernel_name!r}"
            )
        if not _is_int(self.intermediate_size) or self.intermediate_size <= 0:
            raise ValueError(
                f"{KERNEL_NAME}: intermediate_size must be a positive int, "
                f"got {self.intermediate_size!r}"
            )
        if self.intermediate_size % 128 != 0:
            raise ValueError(
                f"{KERNEL_NAME}: intermediate_size must be a multiple of 128, "
                f"got {self.intermediate_size}"
            )
        if not _is_int(self.top_k) or not 1 <= self.top_k <= 32:
            raise ValueError(
                f"{KERNEL_NAME}: top_k must be an int in [1, 32], got {self.top_k!r}"
            )
        if not isinstance(self.activation_clamp, (int, float)) or isinstance(
            self.activation_clamp, bool
        ):
            raise TypeError(
                f"{KERNEL_NAME}: activation_clamp must be a finite number "
                "(no clamp is not supported yet)"
            )
        clamp = float(self.activation_clamp)
        if not math.isfinite(clamp) or clamp < 0:
            raise ValueError(
                f"{KERNEL_NAME}: activation_clamp must be a finite non-negative "
                f"number (no clamp is not supported yet), got {self.activation_clamp!r}"
            )
        for name in ("fast_math", "small_t", "sparse_count", "precompile"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{KERNEL_NAME}: {name} must be a bool")
        if self.sparse_count and not self.small_t:
            raise ValueError(f"{KERNEL_NAME}: sparse_count requires small_t=True")
        if self.token_layout not in TOKEN_LAYOUTS:
            raise ValueError(
                f"{KERNEL_NAME}: token_layout must be one of {TOKEN_LAYOUTS}, "
                f"got {self.token_layout!r}"
            )
        if self.tile_rule is not None and not isinstance(self.tile_rule, str):
            raise TypeError(f"{KERNEL_NAME}: tile_rule must be a str or None")
        if self.num_sms is not None and (
            not _is_int(self.num_sms) or self.num_sms < 2 or self.num_sms % 2 != 0
        ):
            raise ValueError(
                f"{KERNEL_NAME}: num_sms must be None or an even int >= 2 "
                f"(2-CTA clusters), got {self.num_sms!r}"
            )
        if self.knobs is not None:
            raise ValueError(
                f"{KERNEL_NAME} has no tunable knobs (use tile_rule); knobs must "
                f"be None, got {self.knobs!r}"
            )

    @property
    def gathers_output(self) -> bool:
        """Whether the kernel runs the in-kernel all-gather."""
        return self.token_layout == "replicated"


__all__ = [
    "DSV4_PRO_DEP8_TILE_RULE",
    "KERNEL_NAME",
    "Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig",
    "TOKEN_LAYOUTS",
]
