from typing import Any, Optional, Tuple

import torch

from ..autotuner import (
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
    is_in_profile_measurement,
)
from ..utils import get_compute_capability


def prepare_uniform_m_indptr(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    a, b, _, _, m_indptr = inputs
    num_experts = b.shape[0]
    rows_per_expert, remainder = divmod(a.shape[0], num_experts)
    counts = torch.full(
        (num_experts,), rows_per_expert, dtype=m_indptr.dtype, device=m_indptr.device
    )
    if remainder > 0:
        counts[:remainder] += 1
    uniform_m_indptr = torch.empty_like(m_indptr)
    uniform_m_indptr[0] = 0
    torch.cumsum(counts, dim=0, out=uniform_m_indptr[1:])
    return [a, b, inputs[2], inputs[3], uniform_m_indptr]


SM120_MOE_TUNING_CONFIG = TuningConfig(
    use_cold_l2_cache=True,
    inputs_pre_hook=prepare_uniform_m_indptr,
)


class Sm120MoeTunableRunner(TunableRunner):
    def __init__(
        self,
        out: torch.Tensor,
        is_gated: bool,
        scale_granularity_mnk: Tuple[int, int, int],
        scale_major_mode: str,
        tactics: tuple,
        tactic_schema_version: int,
        scale_contract: str,
    ) -> None:
        self._out = out
        self._is_gated = is_gated
        self._scale_granularity_mnk = scale_granularity_mnk
        self._scale_major_mode = scale_major_mode
        self._tactics = tactics
        self._tactic_schema_version = tactic_schema_version
        self._scale_contract = scale_contract
        self._profile_out: Optional[torch.Tensor] = None

    def __hash__(self) -> int:
        return hash((type(self), self._is_gated))

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile
    ) -> list[Any]:
        return list(self._tactics)

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple:
        a = inputs[0]
        device_index = (
            a.device.index if a.device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(device_index)
        return (
            self._scale_contract,
            self._tactic_schema_version,
            self._is_gated,
            self._scale_granularity_mnk,
            device_index,
            properties.name,
            get_compute_capability(a.device),
            tuple(str(tensor.dtype) for tensor in inputs),
        )

    def is_valid_tactic(
        self, tactic: Any, inputs: Optional[list[torch.Tensor]] = None
    ) -> bool:
        if type(tactic) is not tuple or len(tactic) != 3:
            return False
        if any(type(value) is not int for value in tactic):
            return False
        return tactic in self._tactics

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        a, b, _, _, _ = inputs
        if do_preparation:
            out_n = b.shape[1] // 2 if self._is_gated else b.shape[1]
            self._profile_out = torch.empty(
                (a.shape[0], out_n), dtype=torch.bfloat16, device=a.device
            )
            return self._profile_out

        profiling = is_in_profile_measurement()
        out = self._profile_out if profiling else self._out
        if out is None:
            raise RuntimeError("SM120 MoE autotuner profiling output was not prepared")
        is_gated = self._is_gated
        self._launch(
            inputs, out, is_gated, tactic if self.is_valid_tactic(tactic) else -1
        )
        return out

    def _launch(
        self,
        inputs: list[torch.Tensor],
        out: torch.Tensor,
        is_gated: bool,
        tactic: Any,
    ) -> None:
        raise NotImplementedError
