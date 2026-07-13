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

"""Shared split-KV reduction-mode policy for FMHA and MLA decode."""

from collections.abc import Iterable


DIRECT_MODES = frozenset({"direct", "disabled"})
SEPARATE_MODES = frozenset({"gmem_separate", "gmem_reduction_with_separate_kernel"})
INLINE_MODES = frozenset({"gmem_inline", "gmem_reduction"})
CGA_MODES = frozenset({"cga_smem", "cga", "cga_smem_reduction"})

# Primary inline-to-separate crossover measured from the B200 FP8 reduction
# curves. A missing profile deliberately prefers the standalone reducer. The
# caller still owns profile support, shared-memory, and exact one-wave checks.
_FMHA_INLINE_SPLIT_THRESHOLD = {
    (8, 64): 17,
    (8, 128): 7,
    (8, 256): 12,
    (16, 64): 56,
    (16, 128): 25,
    (16, 256): 23,
    (32, 64): 56,
    (32, 128): 43,
    (32, 256): 54,
    (64, 64): 60,
    (64, 128): 59,
    (64, 256): 55,
    (128, 64): 60,
    (128, 256): 63,
}


def canonical_split_kv_mode(mode: str) -> str:
    """Return the canonical name for a production or selector mode spelling."""

    normalized = str(mode).strip().lower()
    if normalized in DIRECT_MODES:
        return "direct"
    if normalized in SEPARATE_MODES:
        return "gmem_separate"
    if normalized in INLINE_MODES:
        return "gmem_inline"
    if normalized in CGA_MODES:
        return "cga_smem"
    raise ValueError(f"unsupported reduction mode {mode!r}")


def select_split_kv_modes(
    *,
    family: str,
    topology: str,
    tile_size_q: int,
    head_dim: int,
    head_dim_per_cta_v: int | None,
    split_kv: int,
    available_modes: Iterable[str],
) -> tuple[str, ...]:
    """Return available split-KV modes in the order they should be tried.

    This policy only orders modes. Callers remain responsible for production
    support, SMEM limits, model coverage, and exact CGA one-wave residency.
    """

    if split_kv < 1:
        raise ValueError("split_kv must be positive")

    family = family.strip().lower()
    if family not in {"fmha_decode", "mla_decode"}:
        raise ValueError(f"unsupported decode family {family!r}")
    topology = topology.strip().lower()
    modes_by_name = {canonical_split_kv_mode(mode): mode for mode in available_modes}
    if split_kv == 1:
        direct = modes_by_name.get("direct")
        if direct is None:
            raise ValueError("split_kv=1 requires direct/disabled mode")
        return (direct,)

    mode_order: tuple[str, ...]
    if family == "fmha_decode":
        threshold = _FMHA_INLINE_SPLIT_THRESHOLD.get((tile_size_q, head_dim))
        first_gmem = (
            "gmem_inline"
            if threshold is not None and split_kv < threshold
            else "gmem_separate"
        )
        mode_order = (
            "cga_smem",
            first_gmem,
            "gmem_separate" if first_gmem == "gmem_inline" else "gmem_inline",
        )
    else:
        # Prefer CGA for every structurally capable 1CTA MLA split profile.
        # The profile factory rejects Keeps-MMA-AB and incomplete Q tiles,
        # applies the static SMEM budget, and the public planner performs the
        # exact cluster-size occupancy query before accepting CGA.  Keeping
        # shape values out of this ordering avoids a second, narrower support
        # matrix that can disagree with those authoritative checks.
        use_cga = topology == "1cta"
        mode_order = ("cga_smem", "gmem_separate") if use_cga else ("gmem_separate",)

    return tuple(modes_by_name[mode] for mode in mode_order if mode in modes_by_name)
