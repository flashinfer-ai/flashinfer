# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""SM107 backend of :func:`flashinfer.qsa.qsa_topk_indexer`: the loader
cache, the torch custom-op registration, and the backend support check.
Shared validation and backend selection stay in the core entry point."""

import functools
import os
from types import SimpleNamespace
from typing import Optional

import torch

from ...utils import (
    get_compute_capability,
    register_custom_op,
    register_fake_op,
)
from .jit import gen_qsa_topk_indexer_module

# Kill switch: set FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1 to force the
# non-fused path. For this op that path does not exist, so the call raises
# NotImplementedError in the core entry point; the switch still guarantees
# the specialized kernel is never dispatched.
_SPECIALIZED_KILL_SWITCH = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"


@functools.cache
def _get_qsa_topk_indexer_module():
    module = gen_qsa_topk_indexer_module().build_and_load()

    @register_custom_op("flashinfer::qsa_topk_indexer", mutates_args=("packed",))
    def _qsa_topk_indexer(
        packed: torch.Tensor,
        qi: torch.Tensor,
        ck_cache: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cpage_base: torch.Tensor,
    ) -> None:
        module.run(qi, ck_cache, cu_seqlens, cpage_base, packed)

    @register_fake_op("flashinfer::qsa_topk_indexer")
    def _fake_qsa_topk_indexer(
        packed: torch.Tensor,
        qi: torch.Tensor,
        ck_cache: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cpage_base: torch.Tensor,
    ) -> None:
        pass

    return SimpleNamespace(run=_qsa_topk_indexer)


def unservable_reason(qi: torch.Tensor) -> Optional[str]:
    """Why the "specialized" backend cannot serve this call, or None.

    GPU-free guards first (the kill switch), then the device guard. Contract
    violations (dtype / geometry / contiguity) are caller errors and raise
    ValueError in the core entry point's shared validation instead.
    """
    if os.environ.get(_SPECIALIZED_KILL_SWITCH) == "1":
        return f"{_SPECIALIZED_KILL_SWITCH}=1 disables the specialized kernel"
    cc = get_compute_capability(qi.device)
    if cc != (10, 7):
        return f"specialized kernel targets SM107 only; device is sm{cc[0]}{cc[1]}"
    return None


def run(
    packed: torch.Tensor,
    qi: torch.Tensor,
    ck_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cpage_base: torch.Tensor,
) -> None:
    _get_qsa_topk_indexer_module().run(packed, qi, ck_cache, cu_seqlens, cpage_base)
