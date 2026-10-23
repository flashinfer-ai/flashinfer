"""Cross-backend SVDQuant tuning with lazy CUTLASS candidates and exact shapes."""

from collections.abc import Callable
from functools import cache
from typing import Final

import torch

from ..autotuner import AutoTuner, OptimizationProfile, TunableRunner, TuningConfig
from ..jit.core import logger
from ..quantization.nvfp4_quantization_utils import (
    current_nvfp4_4over6_config,
    env_flag_enabled,
)
from ..utils import BackendSupportedError, device_support_pdl
from . import gemm_svdquant as gemm
from . import svdquant_sm120_cutlass as cutlass
from .svdquant_sm120_inputs import _check_linear_gemm_inputs, _linear_gemm_inputs


# Composite runners profile eagerly, like the existing fused/unfused comparison.
# Every dimension is static: producer eligibility and workspace are shape-specific.
_TUNING_CONFIG: Final = TuningConfig(use_cuda_graph=False, use_cold_l2_cache=True)


def _cutlass_compatible(inputs: list[torch.Tensor], linear: bool) -> bool:
    """Check caller storage, which aligned autotuner synthesis cannot validate."""
    indices = (1, 2, 5, 6, 8, 9) if linear else (0, 1, 2, 3, 5, 6, 7, 8)
    if any(
        t is not None and (not t.is_contiguous() or t.data_ptr() % 16)
        for t in (inputs[i] for i in indices)
    ):
        return False
    if linear:
        scale = inputs[7]
        return (
            scale.dtype == torch.float32
            and scale.numel() == 1
            and scale.device == inputs[0].device
            and scale.is_contiguous()
        )
    return True


class _CutlassAutoRunner(TunableRunner):
    """Load native kernels and provision scratch only when this candidate is used."""

    def __init__(
        self, enable_pdl: bool, linear: bool = False, fused: bool = False
    ) -> None:
        self.enable_pdl = enable_pdl
        self.linear = linear
        self.fused = fused
        # TunableRunner excludes *_cache fields from its hash.
        self._runner_cache: cutlass._SvdquantRunner | None = None

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple:
        return (self.enable_pdl, self.linear, self.fused)

    def _runner(self, inputs: list[torch.Tensor]) -> cutlass._SvdquantRunner:
        if self._runner_cache is None:
            rank = inputs[5].shape[1]
            factory: Callable[[bool, torch.device, int], cutlass._SvdquantRunner] = (
                cutlass._nvfp4_svdquant_gemm_runner
            )
            if self.linear:
                factory = (
                    cutlass._sm120_fused_linear_runner
                    if self.fused
                    else cutlass._sm120_cutlass_linear_runner
                )
            self._runner_cache = factory(self.enable_pdl, inputs[0].device, rank)
        return self._runner_cache

    def _native_inputs(
        self, inputs: list[torch.Tensor], *, prepare: bool = False
    ) -> list[torch.Tensor]:
        a, b, _, _, _, down = inputs[:6]
        k = a.shape[1] if self.linear else a.shape[1] * 2
        size = max(
            cutlass.DEFAULT_WORKSPACE_SIZE,
            cutlass._sm120_max_workspace_bytes(
                a.device, a.shape[0], b.shape[0], k, down.shape[1]
            ),
        )
        workspace = cutlass._get_cache_buf(
            "nvfp4_svdquant_gemm_workspace", size, a.device
        )
        if self.linear:
            # Native tactic queries read original operands and workspace only.
            # Keep the ABI slots, allocating scratch only for an actual launch.
            scratch = [None, None, None]
            if prepare:
                values = _linear_gemm_inputs(inputs)
                scratch = [values[0], values[2], values[5]]
            return inputs[:9] + scratch + [inputs[9], workspace]
        return [*inputs, workspace]

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile
    ) -> list[int]:
        if not _cutlass_compatible(inputs, self.linear):
            return []
        try:
            runner = self._runner(inputs)
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, ImportError, OSError) as exc:
            # Only module loading is optional; kernel/validation errors still propagate.
            logger.warning(
                "Skipping unavailable SVDQuant CUTLASS auto candidate: %s", exc
            )
            return []
        return runner.get_valid_tactics(self._native_inputs(inputs), profile)

    def validate_tactic(self, inputs: list[torch.Tensor], tactic: int) -> bool:
        return _cutlass_compatible(inputs, self.linear) and self._runner(
            inputs
        ).validate_tactic(self._native_inputs(inputs), tactic)

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        runner = self._runner(inputs)
        values = self._native_inputs(inputs, prepare=True)
        tactic = cutlass._svdquant_workspace_safe_tactic(runner, values, tactic)
        return runner(values, tactic=tactic, do_preparation=do_preparation)


class _CuteLinearRunner(TunableRunner):
    """Time quantization, LoRA-down and a fixed GEMM tactic as one complete chain."""

    def __init__(self, enable_pdl: bool, unfused: bool = False) -> None:
        self.enable_pdl = enable_pdl
        self.unfused = unfused
        factory = (
            gemm._sm120_nvfp4_svdquant_unfused_runner
            if unfused
            else gemm._sm120_nvfp4_svdquant_runner
        )
        self._gemm_cache = factory(enable_pdl)

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple:
        return (self.enable_pdl, self.unfused)

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile
    ) -> list:
        return self._gemm_cache.get_valid_tactics(_linear_gemm_inputs(inputs), profile)

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic=-1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        x, weight, weight_sf, alpha, pqs, l2t, l1, scale, bias, out = inputs
        xq, sf = gemm.nvfp4_quantize_smooth(
            x, pqs, scale, backend="cute-dsl", enable_pdl=self.enable_pdl
        )
        down = torch.mm(x, l2t)
        return self._gemm_cache(
            [xq, weight, sf, weight_sf, alpha, down, l1, bias, out],
            tactic=tactic,
            do_preparation=do_preparation,
        )


@cache
def _auto_runners(
    enable_pdl: bool,
    device: torch.device,
    rank: int,
    *,
    cute: bool,
    native: bool,
    linear: bool = False,
    fused: bool = False,
) -> list[TunableRunner]:
    """Reuse candidates per device/rank without caching inputs or chosen tactics."""
    runners: list[TunableRunner] = []
    if cute:
        runners.extend(
            [_CuteLinearRunner(enable_pdl), _CuteLinearRunner(enable_pdl, unfused=True)]
            if linear
            else [
                gemm._sm120_nvfp4_svdquant_runner(enable_pdl),
                gemm._sm120_nvfp4_svdquant_unfused_runner(enable_pdl),
            ]
        )
    if native:
        if fused:
            runners.append(_CutlassAutoRunner(enable_pdl, linear=True, fused=True))
        runners.append(_CutlassAutoRunner(enable_pdl, linear=linear))
    return runners


def mm_auto(
    inputs: list[torch.Tensor], enable_pdl: bool, backends: list[str]
) -> torch.Tensor:
    if inputs[0].shape[0] == 0:
        return inputs[-1]
    native = "cutlass-sm120" in backends and _cutlass_compatible(inputs, False)
    runners = _auto_runners(
        enable_pdl,
        inputs[0].device,
        inputs[5].shape[1],
        cute="cute-dsl" in backends,
        native=native,
    )
    if not runners:
        raise BackendSupportedError(
            "No SVDQuant auto backend supports the input storage"
        )
    name = f"{cutlass._svdquant_op_name('sm120', inputs[5].shape[1])}_auto_v1_pdl{int(enable_pdl)}_cute{int('cute-dsl' in backends)}_cutlass{int(native)}"
    runner, tactic = AutoTuner.get().choose_one(name, runners, _TUNING_CONFIG, inputs)
    return runner(inputs, tactic=tactic)


def linear_auto(inputs: list[torch.Tensor], enable_pdl: bool | None) -> torch.Tensor:
    """Select across native and CuTe complete-linear implementations on SM120."""
    x, weight, _, alpha, pqs, l2t, _, scale, _, _ = inputs
    gemm._check_nvfp4_quantize_smooth_problem(x, pqs, scale)
    if weight.ndim != 2:
        raise ValueError("weight_fp4 must be a 2-D packed-e2m1 tensor")
    if l2t.shape[0] != x.shape[1] or l2t.dtype != x.dtype or l2t.device != x.device:
        raise ValueError("l2t_smoothed must match x's K dimension, dtype and device")
    x, pqs = gemm._normalize_nvfp4_quantize_smooth_inputs(x, pqs)
    inputs[0], inputs[4] = x, pqs
    inputs[9] = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )
    _check_linear_gemm_inputs(inputs)
    inputs[3] = alpha.reshape(-1)[:1]
    if x.shape[0] == 0:
        return inputs[9]
    if enable_pdl is None:
        enable_pdl = device_support_pdl(x.device)
    try:
        cute = gemm._cute_dsl_nvfp4_svdquant_requirement()
    except ValueError:
        cute = False
    policy = (
        current_nvfp4_4over6_config(),
        env_flag_enabled("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"),
    )
    native = policy == (None, False) and _cutlass_compatible(inputs, True)
    runners = _auto_runners(
        enable_pdl,
        x.device,
        l2t.shape[1],
        cute=cute,
        native=native,
        linear=True,
        fused=native
        and cutlass._sm120_fused_linear_supported(x.shape[0], x.shape[1], l2t.shape[1]),
    )
    if not runners:
        raise BackendSupportedError(
            "No SVDQuant linear auto backend supports the quantization policy and input storage"
        )
    name = f"{cutlass._sm120_linear_op_name(x.shape[0], x.shape[1], l2t.shape[1], enable_pdl=enable_pdl)}_auto_v1_cute{int(cute)}_cutlass{int(native)}_{policy}"
    runner, tactic = AutoTuner.get().choose_one(name, runners, _TUNING_CONFIG, inputs)
    return runner(inputs, tactic=tactic)
