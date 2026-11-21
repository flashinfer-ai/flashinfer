"""Unified GEMM runner adapters.

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

Each adapter wraps one of the ``TunableRunner`` objects the legacy
``mm_fp4`` / ``bmm_fp8`` functions already build, and only translates
``(GemmActivationPack, weight view)`` into that runner's native input list.
The autotuner sees the inner runner, the legacy op name (``"fp4_gemm"``,
``"fp8_gemm"``) and the legacy tuning config, so tuning caches, ``skip_ops``
and cache keys are shared with the old names by construction. This is the
same delegation the unified MoE adapters use (``TrtllmFp4RoutedRunner`` ->
``core.MoERunner``), chosen because the one MoE MVP blocker was an adapter
that had re-implemented its inner call.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor

from ..autotuner import TuningConfig
from ..utils import _get_stream_cache_buf, get_compute_capability
from ..tllm_enums import SfLayout
from .api import (
    BackendUnsupported,
    CublasGemmConfig,
    CudnnGemmConfig,
    CuteDslGemmConfig,
    CutlassGemmConfig,
    GemmActivationPack,
    GemmConfig,
    GemmProblem,
    Identity,
    QuantFormat,
    TrtllmGemmConfig,
    output_torch_dtype,
)

_NVFP4 = (QuantFormat.NVFP4, QuantFormat.NVFP4)
_FP8 = (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _unsupported(key: str, exc: BaseException) -> BackendUnsupported:
    return BackendUnsupported(f"{key}: {exc}")


class GemmRunner:
    """Base adapter. Subclasses set the class attributes and the four hooks.

    Lifecycle (mirrors ``MoERunner``): ``check_support()`` at construction
    (config-level, no tensors), ``build()`` once (loads the kernel module and
    constructs the inner runner), then per selection ``validate(act, view, out)``
    (the legacy requirement checks on real tensors) and per call
    ``pack_inputs`` + ``forward``.
    """

    backend_key: ClassVar[str] = ""
    op_name: ClassVar[str] = ""
    pair: ClassVar[Tuple[QuantFormat, QuantFormat]] = _NVFP4
    #: Whether the kernel applies ``GemmWeightPack.bias`` itself (MVP: none do).
    fused_bias: ClassVar[bool] = False
    workspace_name: ClassVar[str] = ""

    def __init__(self, backend_config: Any, config: GemmConfig, device: torch.device):
        self.backend_config = backend_config
        self.config = config
        self.device = device
        major, minor = get_compute_capability(device)
        self.sm_major, self.sm_minor = major, minor
        self.arch = major * 10 + minor
        self.out_dtype = output_torch_dtype(config.quant)
        self.enable_pdl = (
            True if config.execution.enable_pdl is None else config.execution.enable_pdl
        )
        self._inner: Any = None
        self._built = False

    # --- config-level admission ------------------------------------------

    def check_support(self) -> None:
        """Raise :class:`BackendUnsupported` with a reason, or return."""
        key = self.backend_key
        if self.config.quant.pair != self.pair:
            raise BackendUnsupported(
                f"{key}: runner serves {self.pair!r}, config is {self.config.quant.pair!r}."
            )
        if not self.backend_config.supports_pair_on(self.pair, self.arch):
            raise BackendUnsupported(
                f"{key}: quant pair {self.pair!r} is not supported on sm{self.arch}."
            )
        ep = self.config.epilogue
        if not isinstance(ep.activation, Identity):
            raise BackendUnsupported(
                f"{key}: fused activation {ep.activation!r} is not in the unified GEMM "
                "MVP (Identity only)."
            )
        if ep.per_row_alpha:
            raise BackendUnsupported(
                f"{key}: per_row_alpha is not in the unified GEMM MVP."
            )
        if self.config.execution.batch_invariant:
            raise BackendUnsupported(
                f"{key}: batch_invariant is not in the unified GEMM MVP "
                "(use mm_bf16(batch_invariant=True))."
            )
        q = self.config.quant
        if q.swizzled_scale_factors is not None or q.per_token_scale:
            raise BackendUnsupported(
                f"{key}: QuantConfig.swizzled_scale_factors / per_token_scale are not "
                "in the unified GEMM MVP."
            )
        self._check_support()

    def _check_support(self) -> None:  # pragma: no cover - overridden
        pass

    def check_problem(self, problem: GemmProblem) -> None:
        """Shape-level admission without tensors (for ``supported_backends``)."""
        if problem.form != "dense":
            raise BackendUnsupported(
                f"{self.backend_key}: form={problem.form!r} is not in the unified GEMM "
                "MVP (dense only)."
            )
        self._check_problem(problem)

    def _check_problem(self, problem: GemmProblem) -> None:  # pragma: no cover
        pass

    # --- build -----------------------------------------------------------

    def build(self) -> None:
        if not self._built:
            self._inner = self._build()
            self._built = True

    def _build(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def inner(self) -> Any:
        if not self._built:
            raise RuntimeError(f"{self.backend_key}: build() has not been called.")
        return self._inner

    # --- per call --------------------------------------------------------

    def accepts(self, act: GemmActivationPack, view: Optional[Dict[str, Any]]) -> bool:
        """Cheap per-call eligibility: the right pack shape for this pair and a view."""
        if view is None or act.form != "dense":
            return False
        return self._accepts(act, view)

    def _accepts(self, act: GemmActivationPack, view: Dict[str, Any]) -> bool:
        return True

    def validate(
        self, act: GemmActivationPack, view: Dict[str, Any], out: Tensor
    ) -> None:
        """The legacy requirement checks on the real tensors; raises BackendUnsupported."""
        try:
            ok = self._validate(act, view, out)
        except ValueError as exc:  # LibraryError is not a ValueError: it propagates
            raise _unsupported(self.backend_key, exc) from exc
        if ok is False:
            raise BackendUnsupported(
                f"{self.backend_key}: the legacy requirement check declined this problem."
            )

    def _validate(self, act, view, out) -> bool:  # pragma: no cover - overridden
        return True

    def workspace(self) -> Tensor:
        from .gemm_base import DEFAULT_WORKSPACE_SIZE

        return _get_stream_cache_buf(
            self.workspace_name, DEFAULT_WORKSPACE_SIZE, self.device
        )

    def pack_inputs(
        self, act: GemmActivationPack, view: Dict[str, Any], out: Tensor
    ) -> List[Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def tuning_config(self) -> TuningConfig:  # pragma: no cover - overridden
        raise NotImplementedError

    def forward(self, inputs: List[Any], tactic: Any = -1) -> Tensor:
        self.inner(inputs=inputs, tactic=tactic)
        return inputs[self.out_index]

    out_index: ClassVar[int] = 0


# ---------------------------------------------------------------------------
# NVFP4 (mm_fp4 kernels)
# ---------------------------------------------------------------------------


class _Nvfp4Runner(GemmRunner):
    op_name = "fp4_gemm"
    pair = _NVFP4
    workspace_name = "mm_fp4_workspace"
    out_index = 6
    sf_layout: SfLayout = SfLayout.layout_128x4

    @property
    def use_8x4(self) -> bool:
        return self.sf_layout == SfLayout.layout_8x4

    @property
    def tuning_config(self) -> TuningConfig:
        from . import gemm_base as gb

        return (
            gb._MM_FP4_TUNING_CONFIG_8x4
            if self.use_8x4
            else gb._MM_FP4_TUNING_CONFIG_128x4
        )

    def _check_support(self) -> None:
        if self.out_dtype not in (torch.bfloat16, torch.float16):
            raise BackendUnsupported(
                f"{self.backend_key}: NVFP4 GEMM emits bf16 or fp16 only."
            )

    def _check_problem(self, problem: GemmProblem) -> None:
        if problem.k % 16 != 0:
            raise BackendUnsupported(
                f"{self.backend_key}: NVFP4 needs K % 16 == 0, got K={problem.k}."
            )

    def _accepts(self, act: GemmActivationPack, view: Dict[str, Any]) -> bool:
        return (
            act.a.dim() == 2
            and act.a_scale is not None
            and act.sf_layout == self.sf_layout
            and view.get("recipe") in ("nvfp4_128x4", "nvfp4_128x4_shuffled")
        )

    def _alpha(self, act: GemmActivationPack, view: Dict[str, Any]) -> Optional[Tensor]:
        """``alpha = 1 / (a_global_scale * b_global_scale)`` as the kernels take it.

        Static (``prepare_weight_pack(activation_global_scale=...)``): read from
        the view, no kernel. Dynamic: one division of the view's precomputed
        ``1 / b_global_scale`` by the activation's scale, formed once per
        (activation pack, view) and memoized on the pack, so selection
        (validate + pack_inputs over every candidate) and the serving call each
        launch it at most once per weight.
        """
        alpha = view.get("alpha")
        if alpha is not None:
            return alpha
        a_gs = act.a_global_scale
        if a_gs is None:
            return None
        memo = act._alpha_by_view.get(id(view))
        if memo is not None and memo[0] is view:
            return memo[1]
        b_inv = view.get("b_global_scale_inv")
        if b_inv is not None:
            alpha = torch.div(b_inv, a_gs)
        else:  # a hand-built view without the reciprocal: the legacy arithmetic
            alpha = 1.0 / (a_gs * view["b_global_scale"])
        alpha = alpha.reshape(1).to(torch.float32)
        act._alpha_by_view[id(view)] = (view, alpha)
        return alpha

    def pack_inputs(self, act, view, out) -> List[Any]:
        # Exactly mm_fp4's input list, so the inner runner and its cache keys
        # are untouched.
        return [
            act.a,
            view["b_t"],
            act.a_scale,
            view["b_descale_t"],
            self._alpha(act, view),
            self.out_dtype,
            out,
            16,  # block_size
            True,  # use_nvfp4
            self.workspace(),
        ]

    def _legacy_args(self, act, view, out) -> Tuple[Any, ...]:
        return (
            act.a,
            view["b_t"],
            act.a_scale,
            view["b_descale_t"],
            self._alpha(act, view),
            self.out_dtype,
            out,
            16,
            self.use_8x4,
            self.backend_key,
            True,
            self.enable_pdl,
        )

    def _validate(self, act, view, out) -> bool:
        from . import gemm_base as gb

        args = self._legacy_args(act, view, out)
        return bool(gb._check_mm_fp4_problem_size(*args)) and bool(
            self._legacy_requirement()(*args)
        )

    def _legacy_requirement(self):  # pragma: no cover - overridden
        raise NotImplementedError


class CutlassNvfp4GemmRunner(_Nvfp4Runner):
    backend_key = "cutlass"

    def _legacy_requirement(self):
        from . import gemm_base as gb

        return gb._cutlass_gemm_fp4_requirement

    def _build(self):
        from .gemm_base import get_cutlass_fp4_gemm_module

        return get_cutlass_fp4_gemm_module(
            self.sm_major, self.sm_minor
        ).cutlass_fp4_gemm_runner()


class CudnnNvfp4GemmRunner(_Nvfp4Runner):
    backend_key = "cudnn"

    def _check_support(self) -> None:
        super()._check_support()
        from . import gemm_base as gb

        if not gb.CUDNN_AVAILABLE:
            raise BackendUnsupported("cudnn: nvidia-cudnn-frontend is not installed.")
        try:
            gb._check_cudnn_fp4_availability()
        except (RuntimeError, ValueError) as exc:
            raise _unsupported("cudnn", exc) from exc

    def _legacy_requirement(self):
        from . import gemm_base as gb

        return gb._cudnn_gemm_fp4_requirement

    def _build(self):
        from .gemm_base import _cudnn_gemm_fp4_runner

        return _cudnn_gemm_fp4_runner(self.tuning_config)


class TrtllmNvfp4GemmRunner(_Nvfp4Runner):
    backend_key = "trtllm"

    def __init__(self, backend_config, config, device):
        super().__init__(backend_config, config, device)
        self.sf_layout = (
            SfLayout.layout_8x4
            if backend_config.use_8x4_sf_layout
            else SfLayout.layout_128x4
        )

    def _check_support(self) -> None:
        super()._check_support()
        if self.out_dtype != torch.bfloat16:
            raise BackendUnsupported("trtllm: NVFP4 GEMM emits bf16 only.")

    def _accepts(self, act, view) -> bool:
        return (
            super()._accepts(act, view) and view.get("recipe") == "nvfp4_128x4_shuffled"
        )

    def _legacy_requirement(self):
        from . import gemm_base as gb

        return gb._trtllm_gemm_fp4_requirement

    def _build(self):
        from .gemm_base import get_trtllm_gemm_module

        return get_trtllm_gemm_module().trtllm_fp4_gemm_runner(self.use_8x4)


class CuteDslNvfp4GemmRunner(_Nvfp4Runner):
    backend_key = "cute-dsl"

    def _check_support(self) -> None:
        super()._check_support()
        from . import gemm_base as gb

        if self.arch not in (100, 103, 107):
            raise BackendUnsupported(
                f"cute-dsl: the unified GEMM MVP serves SM100/SM103/SM107 only, got sm{self.arch}."
            )
        try:
            gb._check_cute_dsl_availability()
            gb._check_cute_dsl_arch(self.device)
        except ValueError as exc:
            raise _unsupported("cute-dsl", exc) from exc

    def _check_problem(self, problem: GemmProblem) -> None:
        super()._check_problem(problem)
        if problem.n % 8 != 0:
            raise BackendUnsupported(
                f"cute-dsl: NVFP4 GEMM needs N % 8 == 0, got N={problem.n}."
            )

    def _legacy_requirement(self):
        from . import gemm_base as gb

        return gb._cute_dsl_gemm_fp4_requirement

    def _build(self):
        from .gemm_base import _cute_dsl_gemm_fp4_runner

        return _cute_dsl_gemm_fp4_runner(
            self.sm_major, self.sm_minor, self.enable_pdl, self.out_dtype, True
        )


# ---------------------------------------------------------------------------
# Per-tensor FP8 (bmm_fp8 kernels on a batch of one, as mm_fp8 does)
# ---------------------------------------------------------------------------


class _Fp8Runner(GemmRunner):
    op_name = "fp8_gemm"
    pair = _FP8
    workspace_name = "bmm_fp8_workspace"
    out_index = 4

    @property
    def tuning_config(self) -> TuningConfig:
        from . import gemm_base as gb

        return gb._FP8_GEMM_SM100_TUNING_CONFIG

    def _check_support(self) -> None:
        if self.out_dtype not in (torch.bfloat16, torch.float16):
            raise BackendUnsupported(
                f"{self.backend_key}: FP8 GEMM emits bf16 or fp16 only."
            )

    def _accepts(self, act, view) -> bool:
        return (
            act.a.dim() == 2
            and act.a.dtype in _FP8_DTYPES
            and view.get("recipe") == "fp8_per_tensor"
        )

    def _a_scale(self, act: GemmActivationPack) -> Tensor:
        if act.a_scale is not None:
            return act.a_scale
        from .gemm_base import _fp8_unit_scale

        return _fp8_unit_scale(self.device)

    def pack_inputs(self, act, view, out) -> List[Any]:
        # bmm_fp8's input list on a batch of one: (1, M, K) x (1, K, N)^T.
        return [
            act.a.unsqueeze(0),
            view["b3"],
            self._a_scale(act),
            view["b_scale"],
            out.unsqueeze(0),
            self.workspace(),
        ]

    def forward(self, inputs, tactic=-1) -> Tensor:
        self.inner(inputs=inputs, tactic=tactic)
        return inputs[self.out_index][0]

    def _legacy_args(self, act, view, out):
        return (
            act.a.unsqueeze(0),
            view["b3"],
            self._a_scale(act),
            view["b_scale"],
            self.out_dtype,
            out.unsqueeze(0),
            self.backend_key,
        )

    def _validate(self, act, view, out) -> bool:
        from . import gemm_base as gb

        args = self._legacy_args(act, view, out)
        if act.a.shape[1] != view["b"].shape[1]:
            raise BackendUnsupported(
                f"{self.backend_key}: K mismatch, activation K={act.a.shape[1]} vs weight "
                f"K={view['b'].shape[1]}."
            )
        gb._validate_fp8_output_dtype(self.out_dtype)  # raises ValueError
        return bool(self._legacy_requirement()(*args))

    def _legacy_requirement(self):  # pragma: no cover - overridden
        raise NotImplementedError


class CublasFp8GemmRunner(_Fp8Runner):
    backend_key = "cublas"

    def _legacy_requirement(self):
        from . import gemm_base as gb

        return gb._cublas_bmm_fp8_requirement

    def _build(self):
        from .gemm_base import get_gemm_module

        return get_gemm_module().cublas_fp8_gemm_runner()


class CudnnFp8GemmRunner(_Fp8Runner):
    backend_key = "cudnn"

    def _check_support(self) -> None:
        super()._check_support()
        from . import gemm_base as gb

        if not gb.CUDNN_AVAILABLE:
            raise BackendUnsupported("cudnn: nvidia-cudnn-frontend is not installed.")
        # bmm_fp8's heuristic keeps cuDNN out of the SM12x candidates when the
        # per-shape graph build is unbounded (no override_shape); same rule here.
        if self.arch in (120, 121) and not gb._is_cudnn_override_shape_available():
            raise BackendUnsupported(
                "cudnn: FP8 GEMM on SM12x needs cuDNN override_shape (backend >= 9.23.1)."
            )

    def _legacy_requirement(self):
        from . import gemm_base as gb

        return gb._cudnn_bmm_fp8_requirement

    def _build(self):
        from .gemm_base import _cudnn_gemm_fp8_runner

        return _cudnn_gemm_fp8_runner()


class CutlassFp8GemmRunner(_Fp8Runner):
    backend_key = "cutlass"

    def _accepts(self, act, view) -> bool:
        # CUTLASS has no e5m2 FP8 GEMM; the legacy heuristic drops it too.
        return (
            super()._accepts(act, view)
            and act.a.dtype != torch.float8_e5m2
            and view["b"].dtype != torch.float8_e5m2
        )

    def _legacy_requirement(self):
        from . import gemm_base as gb

        return gb._cutlass_bmm_fp8_requirement

    def _build(self):
        from . import gemm_base as gb

        if self.arch in (120, 121):
            return gb.get_gemm_sm120_module_cutlass_fp8().cutlass_fp8_gemm_runner()
        return gb.get_gemm_sm100_module_cutlass_fp8().cutlass_fp8_gemm_runner()


#: (backend config type, quant pair) -> runner adapter.
_BACKEND_RUNNERS: Dict[
    Tuple[type, Tuple[QuantFormat, QuantFormat]], Type[GemmRunner]
] = {
    (CutlassGemmConfig, _NVFP4): CutlassNvfp4GemmRunner,
    (CudnnGemmConfig, _NVFP4): CudnnNvfp4GemmRunner,
    (TrtllmGemmConfig, _NVFP4): TrtllmNvfp4GemmRunner,
    (CuteDslGemmConfig, _NVFP4): CuteDslNvfp4GemmRunner,
    (CublasGemmConfig, _FP8): CublasFp8GemmRunner,
    (CudnnGemmConfig, _FP8): CudnnFp8GemmRunner,
    (CutlassGemmConfig, _FP8): CutlassFp8GemmRunner,
}


def runner_class_for(
    backend_config: Any, config: GemmConfig
) -> Optional[Type[GemmRunner]]:
    return _BACKEND_RUNNERS.get((type(backend_config), config.quant.pair))
