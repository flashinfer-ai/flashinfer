# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant fp8-block MoE backend of the unified MoE layer (``BatchInvariantFp8BlockConfig`` ->
``BatchInvariantFp8BlockRunner``): the fp8 formats with block scales under one backend key, grouped as TRTLLM-Gen groups
them (``TrtllmFp8BlockConfig`` / ``trtllm_fp8_block``). The quantization pair of the layer's ``QuantConfig`` selects the
class family and its runner: MXFP8 x MXFP8 runs the classes ``moe.mxfp8.<act>`` (:mod:`.runner_mxfp8`), the pair served.
A pair keeps its own kernels, classes, fingerprints and pack conventions: the grouping is the surface alone.

The pair runners take the shared pack / plan / launch flow of :class:`.runner.BatchInvariantMoeRunnerBase` (its reviewed
rules: the bounds at every entry, the deferred from-logits refusal, the routed scaling factor built at construction, the
packed mode's weights rounded by the combine kernel, the memoised ranking) and set the format hooks; the backend's own
rules here are the dense expert placement (an expert-parallel shard is refused by name) and ``do_finalize=True``.

The layer admits the backend for the declared pairs only (``supports_quant``); constructing the runner for another pair
raises the mode's error naming the pairs served, as the config's ``prepare_weights`` / ``prepare_activations`` do.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Type

import torch

from ...batch_invariant.support import BatchInvariantUnsupportedError, unsupported
from ..api import MoEActivationPack, MoEConfig, MoEWeightPack, QuantConfig, QuantFormat
from .runner import BatchInvariantMoeRunnerBase

#: The quantization pair served: MXFP8 experts and MXFP8 activations.
PAIR_MXFP8 = (QuantFormat.MXFP8, QuantFormat.MXFP8)


def unsupported_pair(quant: QuantConfig) -> BatchInvariantUnsupportedError:
    """The refusal of a quantization pair the backend does not serve (the layer filters it before construction; a runner
    constructed for it or a ``prepare_*`` call with it raises this)."""
    served = " or ".join(
        f"QuantConfig(weight={w.name}, activation={a.name})"
        for w, a in BatchInvariantFp8BlockRunner.supported_quant_variants
    )
    return BatchInvariantUnsupportedError(
        f"batch_invariant=True: the fp8-block MoE backend (`{BatchInvariantFp8BlockRunner.backend_key}`) serves {served}, "
        f"not weight={quant.weight.name}, activation={quant.activation.name}. The batch-invariant mode never falls back "
        "to another kernel."
    )


class BatchInvariantFp8BlockRunner(BatchInvariantMoeRunnerBase):
    """The unified layer's runner for the batch-invariant fp8-block MoE classes: constructing it for a served pair gives
    the pair's runner (a subclass, ``implementation``), which carries the pair's inputs, tactics and launch on the shared
    flow; a bare instance of this class serves no pair."""

    backend_key = "batch_invariant_fp8_block"
    supported_quant_variants = (PAIR_MXFP8,)
    # The classes' EP form is not part of this backend: the layer's shard hint lists the runners by this flag, the shared
    # assertion refuses a shard, and ``_check_experts`` states the refusal in the backend's terms first.
    supports_expert_parallelism = False
    CLASS = "moe.fp8_block"

    @classmethod
    def implementation(
        cls, pair: Tuple[QuantFormat, QuantFormat]
    ) -> Optional[Type["BatchInvariantFp8BlockRunner"]]:
        """The runner of a quantization pair; ``None`` for a pair the backend does not serve."""
        if pair == PAIR_MXFP8:
            from .runner_mxfp8 import BatchInvariantMxfp8Runner

            return BatchInvariantMxfp8Runner
        return None

    def __new__(
        cls, config: Optional[MoEConfig] = None, device: Any = None
    ) -> "BatchInvariantFp8BlockRunner":
        # the backend's class constructs the pair's runner; a pair without one is refused here (the layer never asks:
        # it admits the declared pairs only); without a config (the registry's contract tests) a bare instance of the class
        if cls is BatchInvariantFp8BlockRunner and config is not None:
            impl = cls.implementation(config.quant.pair)
            if impl is None:
                raise unsupported_pair(config.quant)
            cls = impl
        return super().__new__(cls)

    def _bare(self) -> bool:
        """A bare instance of the backend's class: no pair's runner."""
        return type(self) is BatchInvariantFp8BlockRunner

    # the runners' lifecycle and pack-contract rules on the backend's class itself: a served pair constructs its
    # implementation runner, which takes the shared flow; a bare instance refuses after the shared checks
    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        if self._bare():
            self._require_built()
            self._validate_pack_contract(act)
            raise unsupported_pair(self.config.quant)
        return super().pack_inputs(act, weights)

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: Any) -> List[Any]:
        if self._bare():
            self._require_built()
            raise unsupported_pair(self.config.quant)
        return super().get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._bare():
            self._require_built()
            raise unsupported_pair(self.config.quant)
        return super().forward(inputs, tactic, do_preparation, **kwargs)

    # ---- the backend's rules ---------------------------------------------------------------------------------------
    def _check_support(self) -> None:
        self._check_experts()
        super()._check_support()

    def _check_experts(self) -> None:
        experts, routing = self.config.experts, self.config.routing
        if experts.local_expert_offset or (
            experts.local_num_experts not in (None, 0, routing.num_experts)
        ):
            raise unsupported(
                self.CLASS,
                "expert-parallel shards (local_expert_offset / local_num_experts)",
                "the full expert set (the classes' EP form is not part of this backend)",
            )

    def _check_support_extra(self) -> None:
        if not self.config.finalize.do_finalize:
            raise unsupported(
                self.CLASS,
                "do_finalize=False",
                "do_finalize=True (the slot-order combine is part of the class)",
            )


__all__ = [
    "PAIR_MXFP8",
    "BatchInvariantFp8BlockRunner",
    "unsupported_pair",
]
