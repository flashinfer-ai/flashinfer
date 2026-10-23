# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

DV = 128
FWD_TARGET_FRAC = 0.7  # blackwell/fused_fwd.py:12

COMPUTE_VERSION_ARCH = {
    "9.0": "sm90",
    "10.0": "sm100",
    "10.3": "sm103",
    "12.0": "sm120",
}

BACKEND_OF = {
    "sm90": "hopper",
    "sm100": "blackwell",
    "sm103": "blackwell",
    "sm120": "blackwell_sm120",
}


@dataclass(frozen=True)
class LaunchShape:
    H: int
    n_part: int
    P: int

    @property
    def grid(self) -> int:
        return self.H * self.n_part  # == real_batch_size * H in the kernels


@dataclass(frozen=True)
class LaunchConfig:
    kernel: str
    block_DV: int

    @property
    def ways(self) -> int:
        # the kernel grid is ways * H * batch, e.g. hopper/cp_fwd.py:392
        return -(-DV // self.block_DV)

    @property
    def row(self) -> str:
        return coef_row(self.kernel, self.block_DV)


def coef_row(kernel: str, block_DV: int) -> str:
    return f"{kernel}_dv{int(block_DV)}" if kernel in DV_TILED else kernel


@dataclass(frozen=True)
class Rule:
    widths: tuple[
        int, ...
    ]  # every block_DV pick can return; coef_rows takes it as given
    pick: Callable[[LaunchShape], int]


_BLACKWELL_FWD = Rule(
    (64, 128),
    # blackwell/fused_fwd.py:861-864
    lambda shape: 128 if shape.grid >= int(FWD_TARGET_FRAC * shape.P) else 64,
)


def _hopper_fwd_pick(shape: LaunchShape) -> int:
    target = int(FWD_TARGET_FRAC * shape.P)
    if shape.grid >= target:
        return 128
    if shape.grid * 2 >= target:
        return 64
    return 32


_HOPPER_FWD = Rule((32, 64, 128), _hopper_fwd_pick)

BLOCK_DV: dict[str, dict[str, int | Rule]] = {
    "hopper": {
        "fused_fwd": _HOPPER_FWD,  # hopper/fused_fwd.py:730-736
        "correct_h0": 128,
        "correct_dht": 128,  # hopper/cp_fwd.py:302-304
    },
    "blackwell": {
        "fused_fwd": _BLACKWELL_FWD,
        "correct_h0": 128,
        "correct_dht": 128,  # blackwell/cp_fwd.py:301-303
    },
    "blackwell_sm120": {
        "fused_fwd": 128,  # blackwell_sm120/fused_fwd.py:726
        "correct_h0": 32,
        "correct_dht": 32,  # blackwell_sm120/cp_fwd.py:282
    },
}

BACKENDS = tuple(BLOCK_DV)

DV_TILED = tuple(dict.fromkeys(k for rules in BLOCK_DV.values() for k in rules))


def _rule(backend: str, kernel: str) -> Rule | int:
    if backend not in BLOCK_DV:
        raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")
    return BLOCK_DV[backend].get(kernel, DV)


def widths(backend: str, kernel: str) -> tuple[int, ...]:
    rule = _rule(backend, kernel)
    return rule.widths if isinstance(rule, Rule) else (rule,)


def launch_config(backend: str, kernel: str, shape: LaunchShape) -> LaunchConfig:
    rule = _rule(backend, kernel)
    if not isinstance(rule, Rule):
        return LaunchConfig(kernel, rule)
    block_DV = rule.pick(shape)
    if block_DV not in rule.widths:
        raise ValueError(
            f"{backend}/{kernel} picked block_DV={block_DV} for {shape}, which is "
            f"outside its declared widths {rule.widths}; no coefficient row was "
            f"fitted for it"
        )
    return LaunchConfig(kernel, block_DV)


def coef_rows(backend: str, kernels) -> tuple[str, ...]:
    out: list[str] = []
    for kernel in kernels:
        for width in widths(backend, kernel):
            row = coef_row(kernel, width)
            if row not in out:
                out.append(row)
    return tuple(out)


@functools.lru_cache(maxsize=1)
def current_arch() -> str:
    import torch

    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("This CuTe forward port targets SM120 only")
    return "sm120"
