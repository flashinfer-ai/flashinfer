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

Internal SM90 BF16 grouped-GEMM tactic matrix and selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

FamilyMode = Literal["m64", "m128", "dual"]
KernelSchedule = Literal["pingpong", "cooperative"]

SUPPORTED_BLOCK_N: tuple[Literal[64, 128], ...] = (64, 128)
SUPPORTED_BLOCK_K: tuple[Literal[64, 128], ...] = (64, 128)
SUPPORTED_STAGES: tuple[Literal[2, 3, 4], ...] = (2, 3, 4)
SUPPORTED_CLUSTER_M: tuple[Literal[1, 2], ...] = (1, 2)

_SELECTOR_CALIBRATION = "bf16_ep4_20260817_r3"
_SELECTOR_CALIBRATION_ROUTING = "balanced"
# These expert-M boundaries are calibrated for H800 EP4 balanced routing.
_FAMILY_M64_MAX_EXPECTED_M = 64.0
_FAMILY_M128_MAX_EXPECTED_M = 128.0


@dataclass(frozen=True)
class Bf16GemmFamilyTactic:
    """One compiled CUTLASS row-family specialization."""

    block_m: Literal[64, 128]
    block_n: Literal[64, 128]
    block_k: Literal[64, 128]
    stages: Literal[2, 3, 4]
    cluster_m: Literal[1, 2]
    schedule: KernelSchedule

    def __post_init__(self) -> None:
        if self.block_m not in (64, 128):
            raise ValueError(f"block_m must be 64 or 128, got {self.block_m}")
        if self.block_n not in SUPPORTED_BLOCK_N:
            raise ValueError(
                f"block_n must be one of {SUPPORTED_BLOCK_N}, got {self.block_n}"
            )
        if self.block_k not in SUPPORTED_BLOCK_K:
            raise ValueError(
                f"block_k must be one of {SUPPORTED_BLOCK_K}, got {self.block_k}"
            )
        if self.stages not in SUPPORTED_STAGES:
            raise ValueError(
                f"stages must be one of {SUPPORTED_STAGES}, got {self.stages}"
            )
        if self.cluster_m not in SUPPORTED_CLUSTER_M:
            raise ValueError(
                f"cluster_m must be one of {SUPPORTED_CLUSTER_M}, got {self.cluster_m}"
            )
        if self.schedule not in ("pingpong", "cooperative"):
            raise ValueError(
                f"schedule must be 'pingpong' or 'cooperative', got {self.schedule!r}"
            )
        if self.block_m == 64 and self.schedule != "pingpong":
            raise ValueError("the M64 family supports only the pingpong schedule")

    @property
    def tag(self) -> str:
        """Return the stable compile-cache tag for this family."""
        schedule = "pp" if self.schedule == "pingpong" else "coop"
        return (
            f"m{self.block_m}n{self.block_n}k{self.block_k}"
            f"s{self.stages}c{self.cluster_m}{schedule}"
        )


@dataclass(frozen=True)
class Bf16GemmTactic:
    """A single-family or dual-family grouped-GEMM launch plan."""

    family_mode: FamilyMode
    m64: Bf16GemmFamilyTactic | None = None
    m128: Bf16GemmFamilyTactic | None = None
    swap_ab: bool = False

    def __post_init__(self) -> None:
        if self.family_mode not in ("m64", "m128", "dual"):
            raise ValueError(
                "family_mode must be 'm64', 'm128', or 'dual', "
                f"got {self.family_mode!r}"
            )
        if self.m64 is not None and self.m64.block_m != 64:
            raise ValueError("m64 must describe a BlockM=64 family")
        if self.m128 is not None and self.m128.block_m != 128:
            raise ValueError("m128 must describe a BlockM=128 family")
        expected = {
            "m64": (self.m64 is not None, self.m128 is None),
            "m128": (self.m64 is None, self.m128 is not None),
            "dual": (self.m64 is not None, self.m128 is not None),
        }[self.family_mode]
        if expected != (True, True):
            raise ValueError(
                f"family_mode={self.family_mode!r} has inconsistent family definitions"
            )
        if self.swap_ab and self.family_mode == "dual":
            raise ValueError("swap_ab supports only a single M-tile family")
        if self.swap_ab and any(
            family.schedule == "cooperative" and family.block_n < 128
            for family in self.families
        ):
            raise ValueError(
                "swap_ab cooperative tactics require BlockN=128 for the transposed M tile"
            )

    @property
    def family_mask(self) -> int:
        """Return the CUDA family bitmask for the plan."""
        return (1 if self.m64 is not None else 0) | (2 if self.m128 is not None else 0)

    @property
    def families(self) -> tuple[Bf16GemmFamilyTactic, ...]:
        """Return enabled families in launch order."""
        return tuple(family for family in (self.m128, self.m64) if family is not None)

    @property
    def tag(self) -> str:
        """Return the stable compile-cache tag for this launch plan."""
        families = "_".join(family.tag for family in self.families)
        return f"{self.family_mode}_{families}_sw{int(self.swap_ab)}"

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of the plan."""
        return {
            "tag": self.tag,
            "family_mode": self.family_mode,
            "family_mask": self.family_mask,
            "swap_ab": self.swap_ab,
            "families": [
                {
                    "block_m": family.block_m,
                    "block_n": family.block_n,
                    "block_k": family.block_k,
                    "stages": family.stages,
                    "cluster_m": family.cluster_m,
                    "schedule": family.schedule,
                }
                for family in self.families
            ],
        }


def _family_matrix(block_m: Literal[64, 128]) -> Iterator[Bf16GemmFamilyTactic]:
    schedules: tuple[KernelSchedule, ...] = (
        ("pingpong",) if block_m == 64 else ("pingpong", "cooperative")
    )
    for block_n in SUPPORTED_BLOCK_N:
        for block_k in SUPPORTED_BLOCK_K:
            for stages in SUPPORTED_STAGES:
                for cluster_m in SUPPORTED_CLUSTER_M:
                    for schedule in schedules:
                        family = Bf16GemmFamilyTactic(
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            stages=stages,
                            cluster_m=cluster_m,
                            schedule=schedule,
                        )
                        operand_bytes = (
                            family.stages
                            * (family.block_m + family.block_n)
                            * family.block_k
                            * 2
                        )
                        if operand_bytes <= 220 * 1024:
                            yield family


SUPPORTED_BF16_GEMM_FAMILY_TACTICS = tuple((*_family_matrix(64), *_family_matrix(128)))


def _plan_matrix() -> Iterator[Bf16GemmTactic]:
    m64_families = tuple(
        family for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS if family.block_m == 64
    )
    m128_families = tuple(
        family for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS if family.block_m == 128
    )
    for family in (*m64_families, *m128_families):
        mode: Literal["m64", "m128"] = "m64" if family.block_m == 64 else "m128"
        m64 = family if mode == "m64" else None
        m128 = family if mode == "m128" else None
        yield Bf16GemmTactic(mode, m64=m64, m128=m128)
        if family.schedule != "cooperative" or family.block_n == 128:
            yield Bf16GemmTactic(mode, m64=m64, m128=m128, swap_ab=True)
    for m64 in m64_families:
        for m128 in m128_families:
            if (
                m64.block_n,
                m64.block_k,
                m64.stages,
            ) != (
                m128.block_n,
                m128.block_k,
                m128.stages,
            ):
                continue
            yield Bf16GemmTactic("dual", m64=m64, m128=m128)


SUPPORTED_BF16_GEMM_TACTICS = tuple(_plan_matrix())
_TACTICS_BY_TAG = {tactic.tag: tactic for tactic in SUPPORTED_BF16_GEMM_TACTICS}
if len(_TACTICS_BY_TAG) != len(SUPPORTED_BF16_GEMM_TACTICS):
    raise RuntimeError("BF16 GEMM tactic tags must be unique")


DEFAULT_BF16_GEMM_TACTIC = Bf16GemmTactic(
    "dual",
    m64=Bf16GemmFamilyTactic(64, 128, 128, 3, 1, "pingpong"),
    m128=Bf16GemmFamilyTactic(128, 128, 128, 3, 1, "pingpong"),
)


def _unique_tactics(*tactics: Bf16GemmTactic) -> tuple[Bf16GemmTactic, ...]:
    return tuple(dict.fromkeys(tactics))


CORE_BF16_GEMM_TACTICS = _unique_tactics(
    Bf16GemmTactic("m64", m64=Bf16GemmFamilyTactic(64, 128, 128, 3, 1, "pingpong")),
    Bf16GemmTactic(
        "m64",
        m64=Bf16GemmFamilyTactic(64, 128, 128, 3, 1, "pingpong"),
        swap_ab=True,
    ),
    Bf16GemmTactic(
        "m128",
        m128=Bf16GemmFamilyTactic(128, 128, 128, 3, 1, "pingpong"),
    ),
    Bf16GemmTactic(
        "dual",
        m64=Bf16GemmFamilyTactic(64, 128, 128, 3, 1, "pingpong"),
        m128=Bf16GemmFamilyTactic(128, 128, 128, 3, 1, "pingpong"),
    ),
    Bf16GemmTactic("m64", m64=Bf16GemmFamilyTactic(64, 64, 128, 3, 1, "pingpong")),
    Bf16GemmTactic("m64", m64=Bf16GemmFamilyTactic(64, 128, 64, 3, 1, "pingpong")),
    Bf16GemmTactic("m64", m64=Bf16GemmFamilyTactic(64, 128, 128, 2, 1, "pingpong")),
    Bf16GemmTactic("m64", m64=Bf16GemmFamilyTactic(64, 128, 128, 4, 1, "pingpong")),
    Bf16GemmTactic("m64", m64=Bf16GemmFamilyTactic(64, 128, 128, 3, 2, "pingpong")),
    Bf16GemmTactic(
        "m128",
        m128=Bf16GemmFamilyTactic(128, 128, 128, 3, 2, "pingpong"),
    ),
    Bf16GemmTactic(
        "m128",
        m128=Bf16GemmFamilyTactic(128, 128, 128, 3, 2, "cooperative"),
    ),
    Bf16GemmTactic(
        "dual",
        m64=Bf16GemmFamilyTactic(64, 128, 128, 3, 1, "pingpong"),
        m128=Bf16GemmFamilyTactic(128, 128, 128, 3, 2, "cooperative"),
    ),
    DEFAULT_BF16_GEMM_TACTIC,
)
if not set(CORE_BF16_GEMM_TACTICS).issubset(SUPPORTED_BF16_GEMM_TACTICS):
    raise RuntimeError("core BF16 GEMM tactics must be present in the supported matrix")


def normalize_bf16_gemm_tactic(
    tactic: Bf16GemmTactic | str,
) -> Bf16GemmTactic:
    """Resolve a forced tactic object or canonical tactic tag."""
    if isinstance(tactic, Bf16GemmTactic):
        normalized = tactic.tag
    else:
        normalized = str(tactic).lower()
    try:
        canonical = _TACTICS_BY_TAG[normalized]
    except KeyError as error:
        raise ValueError(f"unknown BF16 GEMM tactic tag {tactic!r}") from error
    if isinstance(tactic, Bf16GemmTactic) and canonical != tactic:
        raise ValueError(f"BF16 GEMM tactic {tactic.tag!r} is not canonical")
    return canonical


def _make_selected_plan(
    *,
    family_mode: FamilyMode,
    block_n: Literal[64, 128],
    block_k: Literal[64, 128],
    stages: Literal[2, 3, 4],
    m128_cluster: Literal[1, 2],
    m128_schedule: KernelSchedule,
    swap_ab: bool,
) -> Bf16GemmTactic:
    m64 = (
        Bf16GemmFamilyTactic(64, block_n, block_k, stages, 1, "pingpong")
        if family_mode in ("m64", "dual")
        else None
    )
    m128 = (
        Bf16GemmFamilyTactic(
            128,
            block_n,
            block_k,
            stages,
            m128_cluster,
            m128_schedule,
        )
        if family_mode in ("m128", "dual")
        else None
    )
    return Bf16GemmTactic(
        family_mode,
        m64=m64,
        m128=m128,
        swap_ab=swap_ab,
    )


def select_sm90_push_bf16_gemm_tactic(
    *, expected_m: float, n: int, k: int, sm_count: int
) -> tuple[Bf16GemmTactic, str]:
    """Select an internal launch plan from static shape and load estimates."""
    expected_m = max(float(expected_m), 0.0)
    n = int(n)
    k = int(k)
    sm_count = int(sm_count)
    if n <= 0 or k <= 0:
        raise ValueError(f"N and K must be positive, got N={n}, K={k}")
    if sm_count <= 0:
        raise ValueError(f"sm_count must be positive, got {sm_count}")

    if expected_m <= _FAMILY_M64_MAX_EXPECTED_M:
        family_mode: FamilyMode = "m64"
    elif expected_m <= _FAMILY_M128_MAX_EXPECTED_M:
        family_mode = "m128"
    else:
        family_mode = "dual"
    block_n: Literal[64, 128] = 64 if n <= 2048 or n % 128 != 0 else 128
    block_k: Literal[64, 128] = 64 if k <= 2048 or k % 128 != 0 else 128
    stages: Literal[2, 3, 4] = 2 if expected_m <= 32 else 3
    m128_cluster: Literal[1, 2] = 1
    m128_schedule: KernelSchedule = "pingpong"
    swap_ab = family_mode != "dual" and expected_m <= 32 and n >= 4096
    tactic = _make_selected_plan(
        family_mode=family_mode,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m128_cluster=m128_cluster,
        m128_schedule=m128_schedule,
        swap_ab=swap_ab,
    )
    reason = (
        f"expected_m={expected_m:g} selected {family_mode}; N={n} selected N{block_n}; "
        f"K={k} selected K{block_k}; SMs={sm_count}; selected S{stages}; "
        f"swap_ab={int(swap_ab)}; "
        f"calibration={_SELECTOR_CALIBRATION}/{_SELECTOR_CALIBRATION_ROUTING}"
    )
    return tactic, reason


def estimate_sm90_push_bf16_expected_m(
    *, token_capacity: int, top_k: int, num_local_experts: int
) -> float:
    """Estimate uniform routes per local expert from one rank's token capacity."""
    token_capacity = int(token_capacity)
    top_k = int(top_k)
    num_local_experts = int(num_local_experts)
    if token_capacity < 0:
        raise ValueError(f"token_capacity must be nonnegative, got {token_capacity}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if num_local_experts <= 0:
        raise ValueError(f"num_local_experts must be positive, got {num_local_experts}")
    return token_capacity * top_k / num_local_experts


def bf16_gemm_cuda_flags(tactic: Bf16GemmTactic) -> tuple[str, ...]:
    """Return compile definitions for one BF16 GEMM launch plan."""
    tactic = normalize_bf16_gemm_tactic(tactic)

    def family_values(
        family: Bf16GemmFamilyTactic | None, block_m: int
    ) -> tuple[int, int, int, int, int]:
        if family is None:
            return (block_m, 64, 2, 1, 0)
        return (
            family.block_n,
            family.block_k,
            family.stages,
            family.cluster_m,
            int(family.schedule == "cooperative"),
        )

    m64 = family_values(tactic.m64, 64)
    m128 = family_values(tactic.m128, 128)
    return (
        f"-DSM90_PUSH_BF16_FAMILY_MASK={tactic.family_mask}",
        f"-DSM90_PUSH_BF16_M64_BLOCK_N={m64[0]}",
        f"-DSM90_PUSH_BF16_M64_BLOCK_K={m64[1]}",
        f"-DSM90_PUSH_BF16_M64_STAGES={m64[2]}",
        f"-DSM90_PUSH_BF16_M64_CLUSTER_M={m64[3]}",
        f"-DSM90_PUSH_BF16_M64_SCHEDULE={m64[4]}",
        f"-DSM90_PUSH_BF16_M128_BLOCK_N={m128[0]}",
        f"-DSM90_PUSH_BF16_M128_BLOCK_K={m128[1]}",
        f"-DSM90_PUSH_BF16_M128_STAGES={m128[2]}",
        f"-DSM90_PUSH_BF16_M128_CLUSTER_M={m128[3]}",
        f"-DSM90_PUSH_BF16_M128_SCHEDULE={m128[4]}",
        f"-DSM90_PUSH_BF16_SWAP_AB={int(tactic.swap_ab)}",
    )


__all__ = [
    "Bf16GemmFamilyTactic",
    "Bf16GemmTactic",
    "CORE_BF16_GEMM_TACTICS",
    "DEFAULT_BF16_GEMM_TACTIC",
    "SUPPORTED_BF16_GEMM_FAMILY_TACTICS",
    "SUPPORTED_BF16_GEMM_TACTICS",
    "bf16_gemm_cuda_flags",
    "estimate_sm90_push_bf16_expected_m",
    "normalize_bf16_gemm_tactic",
    "select_sm90_push_bf16_gemm_tactic",
]
