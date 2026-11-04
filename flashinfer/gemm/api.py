"""Unified GEMM API: configuration dataclasses, backend configs and packs.

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

This module is the GEMM counterpart of ``flashinfer/fused_moe/api.py`` and
deliberately reuses its vocabulary (``QuantFormat``, ``QuantConfig``,
``BackendOptions``, the typed activation values, the ``prepare_for`` /
``get_view`` pack protocol) so that a model author configures a FlashInfer
linear layer and a FlashInfer MoE layer the same way. See
``docs/design_docs/flashinfer_gemm_api.md``.

MVP scope (this file accepts the full config space; the runners reject what
they do not implement with a named reason):

* dense ``[M, K] x [N, K]^T`` GEMMs;
* ``(NVFP4, NVFP4)`` over the ``cutlass`` / ``cute-dsl`` / ``trtllm`` / ``cudnn``
  kernels behind :func:`flashinfer.mm_fp4`;
* ``(FP8PerTensor, FP8PerTensor)`` over the ``cublas`` / ``cudnn`` / ``cutlass``
  kernels behind :func:`flashinfer.bmm_fp8`;
* BF16 / FP16 output; a bias applied after the GEMM.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    Literal,
    Optional,
    Tuple,
    cast,
)

import torch
from torch import Tensor

# Shared vocabulary. The MoE API owns these today; GEMM reuses them unchanged so
# the two configure identically. Moving them to a neutral module is a review
# decision tracked in the design doc, not something this file decides.
from ..fused_moe.api import (
    ActivationConfig,
    BackendOptions,
    Identity,
    QuantConfig,
    QuantFormat,
)
from ..tllm_enums import SfLayout

__all__ = [
    "ActivationConfig",
    "BackendOptions",
    "BackendReport",
    "BackendUnsupported",
    "CublasGemmConfig",
    "CudnnGemmConfig",
    "CuteDslGemmConfig",
    "CutlassGemmConfig",
    "DEFAULT_GEMM_BACKENDS",
    "EpilogueConfig",
    "ExecutionConfig",
    "GemmActivationPack",
    "GemmConfig",
    "GemmProblem",
    "GemmWeightPack",
    "Identity",
    "QuantConfig",
    "QuantFormat",
    "TrtllmGemmConfig",
    "prepare_activations",
    "prepare_weight_pack",
]

Form = Literal["dense", "batched", "grouped"]

# NVFP4 global-scale convention shared with ``tests/gemm/test_mm_fp4.py`` and
# the frameworks: ``global_scale = (448 * 6) / amax`` is the *quantization*
# scale handed to ``nvfp4_quantize``; the GEMM epilogue multiplies by
# ``alpha = 1 / (a_global_scale * b_global_scale)``.
_NVFP4_GLOBAL_SCALE_NUMERATOR = 448.0 * 6.0


class BackendUnsupported(ValueError):
    """A backend cannot run this configuration or problem here.

    Subclasses :class:`ValueError` so framework probes that already catch
    ``ValueError`` keep working. Candidate filtering (``auto``) skips a backend
    only on this exception; any other exception propagates, so a real bug is
    never scored as "slow tactic" or hidden behind "no usable backend".
    """


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpilogueConfig:
    """Extra work done on the GEMM output.

    Parameters
    ----------
    bias : bool
        Add a ``[N]`` bias (``GemmWeightPack.bias``). MVP: no runner fuses it;
        :class:`~flashinfer.gemm.layer.GemmLayer` adds it after the GEMM and
        reports ``fused_bias=False``.
    activation : ActivationConfig
        Fused activation. Only :class:`Identity` is implemented in the MVP; a
        gated activation halves ``N`` and is declared by the backend.
    per_row_alpha : bool
        One output scale per row (``mm_fp4``'s per-token ``alpha``). Not in the
        MVP.
    """

    bias: bool = False
    activation: ActivationConfig = field(default_factory=Identity)
    per_row_alpha: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.activation, ActivationConfig):
            raise TypeError(
                "EpilogueConfig.activation must be an ActivationConfig value "
                f"(SwiGLU(), SiLU(), Identity(), ...), got {self.activation!r}."
            )

    def __repr__(self) -> str:
        return (
            f"EpilogueConfig(bias={self.bias!r}, activation={self.activation!r}, "
            f"per_row_alpha={self.per_row_alpha!r})"
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """Launch knobs that do not change the math.

    Parameters
    ----------
    enable_pdl : bool or None
        Programmatic dependent launch (the next kernel's prologue overlaps this
        kernel's tail). ``None`` keeps each backend's default.
    tune_max_m : int
        Ceiling of the M (row-count) buckets the layer caches winners for;
        the GEMM analogue of ``tune_max_num_tokens``.
    batch_invariant : bool
        Fixed reduction order regardless of M (``mm_bf16(batch_invariant=True)``).
        Not in the MVP.
    """

    enable_pdl: Optional[bool] = None
    tune_max_m: int = 8192
    batch_invariant: bool = False

    def __post_init__(self) -> None:
        if self.tune_max_m <= 0:
            raise ValueError(f"tune_max_m must be positive, got {self.tune_max_m}.")

    def __repr__(self) -> str:
        return (
            f"ExecutionConfig(enable_pdl={self.enable_pdl!r}, "
            f"tune_max_m={self.tune_max_m!r}, batch_invariant={self.batch_invariant!r})"
        )


@dataclass(frozen=True)
class GemmProblem:
    """What the layer asks a backend about before any tensor exists.

    ``m`` may be ``None`` (unknown until the call); ``n`` and ``k`` are the
    weight's logical ``out_features`` and ``in_features``.
    """

    n: int
    k: int
    m: Optional[int] = None
    form: Form = "dense"

    def __post_init__(self) -> None:
        if self.n <= 0 or self.k <= 0:
            raise ValueError(f"n and k must be positive, got n={self.n}, k={self.k}.")
        if self.m is not None and self.m < 0:
            raise ValueError(f"m must be non-negative, got {self.m}.")


@dataclass(frozen=True)
class BackendReport:
    """One row of :func:`flashinfer.gemm.supported_backends`."""

    backend_key: str
    ok: bool
    reason: Optional[str]
    auto_rank: int
    needs_prepare: bool
    fused_bias: bool


# ---------------------------------------------------------------------------
# Backend configs: the capability records
# ---------------------------------------------------------------------------

_NVFP4 = (QuantFormat.NVFP4, QuantFormat.NVFP4)
_FP8 = (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor)

_OUTPUT_DTYPES: Dict[QuantFormat, torch.dtype] = {
    QuantFormat.BF16: torch.bfloat16,
    QuantFormat.FP16: torch.float16,
}


def output_torch_dtype(quant: QuantConfig) -> torch.dtype:
    """The torch dtype of ``quant.output``; raises for formats no runner emits."""
    try:
        return _OUTPUT_DTYPES[quant.output]
    except KeyError:
        raise BackendUnsupported(
            f"output={quant.output!r} is not an output format of the unified GEMM "
            "MVP (BF16 or FP16)."
        ) from None


def _legacy_api_for_pair(pair: Tuple[QuantFormat, QuantFormat]):
    # Lazy: gemm_base is large and already imported by flashinfer.gemm; this
    # module stays importable without it (CPU config tests).
    from . import gemm_base

    if pair == _NVFP4:
        return gemm_base.mm_fp4
    if pair == _FP8:
        return gemm_base.bmm_fp8
    return None


class _GemmBackendConfig:
    """Common shape of a GEMM backend config (see the concrete classes).

    ``backend_key`` equals the legacy ``backend=`` string of the ``mm_*`` /
    ``bmm_*`` functions, so ``BackendOptions((CutlassGemmConfig(),))`` is the
    config spelling of ``backend="cutlass"``.
    """

    backend_key: ClassVar[str] = ""
    supported_quant_pairs: ClassVar[FrozenSet[Tuple[QuantFormat, QuantFormat]]] = (
        frozenset()
    )
    supported_forms: ClassVar[FrozenSet[str]] = frozenset({"dense"})
    #: Epilogue features the kernels fuse. Empty in the MVP: bias is applied
    #: by the layer after the GEMM.
    supported_epilogues: ClassVar[FrozenSet[str]] = frozenset()
    #: Architectures (major*10+minor) on which at least one supported pair runs.
    _archs: ClassVar[FrozenSet[int]] = frozenset()

    @classmethod
    def supported(cls, arch: int) -> bool:
        """Whether any supported quant pair runs on ``arch`` (``major*10+minor``)."""
        return arch in cls._archs

    @classmethod
    def supports_quant(cls, quant: QuantConfig) -> bool:
        return quant.pair in cls.supported_quant_pairs

    @classmethod
    def supports_pair_on(cls, pair: Tuple[QuantFormat, QuantFormat], arch: int) -> bool:
        """Pair-level arch gate, read from the legacy API's ``@backend_requirement``
        tables so the two surfaces cannot drift."""
        if pair not in cls.supported_quant_pairs:
            return False
        api = _legacy_api_for_pair(pair)
        if api is None:
            return False
        return bool(api.is_backend_supported(cls.backend_key, arch))

    @classmethod
    def weight_recipe(cls, quant: QuantConfig) -> str:
        """Name of the weight preparation this backend consumes.

        Backends sharing a recipe share one prepared view (no extra memory):
        cutlass / cudnn / cute-dsl all read 128x4-swizzled NVFP4 weights,
        trtllm reads the shuffled layout, every FP8 backend reads a row-major
        e4m3 weight with one scale.
        """
        if quant.pair not in cls.supported_quant_pairs:
            raise BackendUnsupported(
                f"{cls.backend_key}: quant pair {quant.pair!r} is not supported."
            )
        if quant.pair == _NVFP4:
            return "nvfp4_128x4"
        if quant.pair == _FP8:
            return "fp8_per_tensor"
        raise BackendUnsupported(
            f"{cls.backend_key}: no weight recipe for quant pair {quant.pair!r}."
        )

    @classmethod
    def prepare_weights(
        cls,
        weight: Tensor,
        *,
        quant: QuantConfig,
        activation_global_scale: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Canonical ``[N, K]`` high-precision weight -> this backend's native view."""
        return _prepare_weight_view(
            cls.weight_recipe(quant),
            weight,
            quant=quant,
            activation_global_scale=activation_global_scale,
        )

    @staticmethod
    def prepare_activations(
        a: Tensor,
        *,
        quant: QuantConfig,
        global_scale: Optional[Tensor] = None,
    ) -> "GemmActivationPack":
        """Canonical activation pack for ``quant.activation`` (shared by every backend)."""
        return prepare_activations(a, quant=quant, global_scale=global_scale)

    def __repr__(self) -> str:
        # Every concrete backend config is a dataclass; the base class is not.
        fields = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in dataclasses.fields(cast(Any, self))
        )
        return f"{type(self).__name__}({fields})"


@dataclass(frozen=True)
class CutlassGemmConfig(_GemmBackendConfig):
    """CUTLASS dense GEMMs: NVFP4 (``mm_fp4``) and per-tensor FP8 (``bmm_fp8``)."""

    backend_key: ClassVar[str] = "cutlass"
    supported_quant_pairs: ClassVar[FrozenSet] = frozenset({_NVFP4, _FP8})
    _archs: ClassVar[FrozenSet[int]] = frozenset({100, 103, 107, 110, 120, 121})


@dataclass(frozen=True)
class CuteDslGemmConfig(_GemmBackendConfig):
    """CuTe-DSL dense NVFP4 GEMM (SM100 / SM103 / SM107; ``N % 8 == 0``)."""

    backend_key: ClassVar[str] = "cute-dsl"
    supported_quant_pairs: ClassVar[FrozenSet] = frozenset({_NVFP4})
    _archs: ClassVar[FrozenSet[int]] = frozenset({100, 103, 107})


@dataclass(frozen=True)
class TrtllmGemmConfig(_GemmBackendConfig):
    """trtllm-gen dense NVFP4 GEMM. Needs the shuffled weight layout and BF16 output.

    Parameters
    ----------
    use_8x4_sf_layout : bool
        Consume activations whose block scales use the small-M 8x4 layout
        instead of 128x4. The activation pack must be quantized to match.
    """

    backend_key: ClassVar[str] = "trtllm"
    supported_quant_pairs: ClassVar[FrozenSet] = frozenset({_NVFP4})
    _archs: ClassVar[FrozenSet[int]] = frozenset({100, 103, 107})

    use_8x4_sf_layout: bool = False

    @classmethod
    def weight_recipe(cls, quant: QuantConfig) -> str:
        if quant.pair == _NVFP4:
            return "nvfp4_128x4_shuffled"
        return super().weight_recipe(quant)


@dataclass(frozen=True)
class CudnnGemmConfig(_GemmBackendConfig):
    """cuDNN dense GEMMs: NVFP4 and per-tensor FP8 (needs ``nvidia-cudnn-frontend``)."""

    backend_key: ClassVar[str] = "cudnn"
    supported_quant_pairs: ClassVar[FrozenSet] = frozenset({_NVFP4, _FP8})
    _archs: ClassVar[FrozenSet[int]] = frozenset({89, 90, 100, 103, 107, 110, 120, 121})


@dataclass(frozen=True)
class CublasGemmConfig(_GemmBackendConfig):
    """cuBLASLt dense per-tensor FP8 GEMM (SM89+)."""

    backend_key: ClassVar[str] = "cublas"
    supported_quant_pairs: ClassVar[FrozenSet] = frozenset({_FP8})
    _archs: ClassVar[FrozenSet[int]] = frozenset({89, 90, 100, 103, 107, 110, 120, 121})


#: Deterministic ``auto`` order. Mirrors what the legacy functions pick untuned:
#: CUTLASS first for FP8 on Blackwell (``bmm_fp8``), cuBLAS where CUTLASS is
#: absent (SM89 / SM90), cuDNN last. trtllm is not a default candidate because
#: it needs a second (shuffled) copy of the weights; cute-dsl is not one
#: because its autotune compiles every tactic. Name either explicitly.
DEFAULT_GEMM_BACKENDS = BackendOptions(
    candidates=(CutlassGemmConfig(), CublasGemmConfig(), CudnnGemmConfig())
)

GemmBackendConfigType = _GemmBackendConfig


@dataclass(frozen=True)
class GemmConfig:
    """Top-level GEMM configuration; hashable, ``repr`` round-trips, ``**config``.

    Example
    -------
    >>> cfg = GemmConfig(
    ...     quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
    ...     epilogue=EpilogueConfig(bias=True),
    ...     backend=BackendOptions((CutlassGemmConfig(), CudnnGemmConfig())),
    ... )
    >>> fp8 = dataclasses.replace(cfg, quant=QuantConfig(
    ...     weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor))
    """

    quant: QuantConfig = field(default_factory=QuantConfig)
    epilogue: EpilogueConfig = field(default_factory=EpilogueConfig)
    backend: BackendOptions = field(default_factory=lambda: DEFAULT_GEMM_BACKENDS)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.quant, QuantConfig):
            raise TypeError(
                f"GemmConfig.quant must be a QuantConfig, got {self.quant!r}."
            )
        if not isinstance(self.backend, BackendOptions):
            raise TypeError(
                "GemmConfig.backend must be a BackendOptions, got "
                f"{self.backend!r}. Wrap backend configs: "
                "BackendOptions((CutlassGemmConfig(),))."
            )
        for cand in self.backend:
            if not isinstance(cand, _GemmBackendConfig):
                raise TypeError(
                    f"GemmConfig.backend candidate {cand!r} is not a GEMM backend "
                    "config (CutlassGemmConfig, CuteDslGemmConfig, TrtllmGemmConfig, "
                    "CudnnGemmConfig, CublasGemmConfig)."
                )
        if len(self.backend) == 0:
            raise ValueError("GemmConfig.backend has no candidates.")

    # --- Dict-unpacking protocol: enables ``**config`` at call sites ---
    def keys(self):
        return (f.name for f in dataclasses.fields(self))

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __repr__(self) -> str:
        return (
            f"GemmConfig(quant={self.quant!r}, epilogue={self.epilogue!r}, "
            f"backend={self.backend!r}, execution={self.execution!r})"
        )


# ---------------------------------------------------------------------------
# Packs: two lifetimes
# ---------------------------------------------------------------------------


def _tensor_summary(value: Any) -> str:
    if value is None:
        return "None"
    if not isinstance(value, Tensor):
        return repr(value)
    return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"


class GemmWeightPack:
    """Long-lived weights: one backend-native view per candidate backend.

    Built once after the weights are loaded (``Linear.prepare_weights()``,
    :func:`prepare_weight_pack`) and read on every call. Same protocol as
    ``MoEWeightPack``: ``prepare_for(backend_key, view)`` / ``get_view(key)``.
    Backends with the same weight recipe share one view, so the default
    candidate set costs one quantized copy of the weight.
    """

    def __init__(
        self,
        logical_shape: Tuple[int, int],
        *,
        bias: Optional[Tensor] = None,
    ) -> None:
        n, k = logical_shape
        self.logical_shape: Tuple[int, int] = (int(n), int(k))
        self.bias: Optional[Tensor] = bias
        self.native_views: Dict[str, Dict[str, Any]] = {}
        self._recipe_views: Dict[str, Dict[str, Any]] = {}

    @property
    def out_features(self) -> int:
        return self.logical_shape[0]

    @property
    def in_features(self) -> int:
        return self.logical_shape[1]

    def prepare_for(self, backend_key: str, view: Dict[str, Any]) -> None:
        """Register ``view`` (the output of a backend's ``prepare_weights``)."""
        if "recipe" not in view:
            raise ValueError("a weight view must carry its 'recipe' name")
        self.native_views[backend_key] = view
        self._recipe_views.setdefault(view["recipe"], view)

    def get_view(self, backend_key: str) -> Dict[str, Any]:
        try:
            return self.native_views[backend_key]
        except KeyError:
            raise KeyError(
                f"GemmWeightPack has no view for backend {backend_key!r}; prepared: "
                f"{sorted(self.native_views)}."
            ) from None

    def view_for_recipe(self, recipe: str) -> Optional[Dict[str, Any]]:
        return self._recipe_views.get(recipe)

    def shapes(self, backend_key: str) -> Dict[str, Tuple[int, ...]]:
        """Shapes of the view's tensors (padded as the kernel sees them), for fakes."""
        return {
            name: tuple(t.shape)
            for name, t in self.get_view(backend_key).items()
            if isinstance(t, Tensor)
        }

    def __repr__(self) -> str:
        views = ", ".join(
            f"{k}={{recipe={v['recipe']!r}}}" for k, v in self.native_views.items()
        )
        return (
            f"GemmWeightPack(logical_shape={self.logical_shape!r}, "
            f"bias={_tensor_summary(self.bias)}, views=[{views}])"
        )


@dataclass
class GemmActivationPack:
    """Per-call activations in the canonical layout of one quant pair.

    ========================  =======================  ==========================
    pair (weight x act)       ``a``                    ``a_scale``
    ========================  =======================  ==========================
    NVFP4 x NVFP4             ``[M, K/2]`` packed e2m1  block scales (e4m3 / uint8)
                                                       in ``sf_layout``
    FP8PerTensor x FP8Per..   ``[M, K]`` e4m3 / e5m2   one fp32 scale
    ========================  =======================  ==========================

    Parameters
    ----------
    a_global_scale : Tensor or None
        NVFP4 only: the per-tensor quantization scale ``(448*6)/amax`` handed to
        ``nvfp4_quantize``. Needed when the weights were prepared without a
        static ``activation_global_scale`` (then ``alpha`` is formed per call).
    sf_layout : SfLayout
        Layout of the NVFP4 block scales (``layout_128x4`` default; ``layout_8x4``
        only for ``TrtllmGemmConfig(use_8x4_sf_layout=True)``).
    """

    a: Tensor
    a_scale: Optional[Tensor] = None
    a_global_scale: Optional[Tensor] = None
    per_row_alpha: Optional[Tensor] = None
    sf_layout: SfLayout = SfLayout.layout_128x4
    # Grouped form (post-MVP). Typed loosely so this file has no GroupLayout yet.
    groups: Any = None

    @property
    def num_rows(self) -> int:
        return int(self.a.shape[0]) if self.a.dim() >= 2 else 0

    @property
    def form(self) -> str:
        if self.groups is not None:
            return "grouped"
        return "batched" if self.a.dim() == 3 else "dense"

    def __repr__(self) -> str:
        return (
            f"GemmActivationPack(a={_tensor_summary(self.a)}, "
            f"a_scale={_tensor_summary(self.a_scale)}, "
            f"a_global_scale={_tensor_summary(self.a_global_scale)}, "
            f"per_row_alpha={_tensor_summary(self.per_row_alpha)}, "
            f"sf_layout={self.sf_layout!r}, form={self.form!r})"
        )


# ---------------------------------------------------------------------------
# Preparation helpers (one per recipe; backends delegate to these)
# ---------------------------------------------------------------------------


def nvfp4_global_scale(x: Tensor) -> Tensor:
    """``(448 * 6) / amax(x)`` as an fp32 device scalar (the ``nvfp4_quantize`` input)."""
    amax = x.detach().float().abs().nan_to_num().max().clamp(min=1e-12)
    return (_NVFP4_GLOBAL_SCALE_NUMERATOR / amax).to(torch.float32)


def _as_f32_scalar(value: Any, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().to(device=device, dtype=torch.float32).reshape(())
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _fp8_per_tensor_quantize(
    x: Tensor, scale_inv: Optional[Tensor], dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[Tensor, Tensor]:
    """Per-tensor FP8 quantization; returns ``(x_q, scale_inv)`` as ``bmm_fp8`` wants.

    ``scale_inv`` is the dequantization scale (``bmm_fp8``'s ``A_scale`` /
    ``B_scale``). ``None`` means dynamic: derive it from ``amax(x)``.
    """
    finfo = torch.finfo(dtype)
    x = x.detach()
    if scale_inv is None:
        amax = x.float().abs().nan_to_num().max().clamp(min=1e-12)
        scale_inv = (amax / finfo.max).to(torch.float32)
    else:
        scale_inv = _as_f32_scalar(scale_inv, x.device)
    x_q = (x.float() / scale_inv).clamp(min=finfo.min, max=finfo.max).to(dtype)
    return x_q, scale_inv


def _prepare_weight_view(
    recipe: str,
    weight: Tensor,
    *,
    quant: QuantConfig,
    activation_global_scale: Optional[Tensor] = None,
) -> Dict[str, Any]:
    """Build the native view of ``recipe`` from a canonical ``[N, K]`` weight.

    The view also holds the derived tensors the legacy kernels take (``b_t``:
    the ``(K, N)`` column-major view, ``b3``: batch-of-one), so the per-call
    path is dictionary reads only.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be [N, K], got shape {tuple(weight.shape)}.")
    if weight.device.type != "cuda":
        raise ValueError("weights must be on a CUDA device before preparation.")
    n, k = (int(s) for s in weight.shape)
    weight = weight.detach()
    view: Dict[str, Any] = {"recipe": recipe, "logical_shape": (n, k)}

    if recipe in ("nvfp4_128x4", "nvfp4_128x4_shuffled"):
        from ..quantization.fp4_quantization import nvfp4_quantize

        if weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"NVFP4 weight preparation takes a bf16/fp16 weight, got {weight.dtype}."
            )
        if k % 16 != 0:
            raise BackendUnsupported(f"NVFP4 needs K % 16 == 0, got K={k}.")
        b_global_scale = nvfp4_global_scale(weight)
        b_fp4, b_sf = nvfp4_quantize(
            weight.contiguous(),
            b_global_scale,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=recipe.endswith("_shuffled"),
        )
        view.update(
            b=b_fp4,
            b_descale=b_sf,
            b_global_scale=b_global_scale,
            # (K/2, N) column-major view of the packed weight and its scales,
            # the layout mm_fp4 documents for ``b`` / ``b_descale``.
            b_t=b_fp4.T,
            b_descale_t=b_sf.T,
            alpha=None,
        )
        if activation_global_scale is not None:
            a_gs = _as_f32_scalar(activation_global_scale, weight.device)
            view["activation_global_scale"] = a_gs
            view["alpha"] = (1.0 / (a_gs * b_global_scale)).reshape(1).to(torch.float32)
        return view

    if recipe == "fp8_per_tensor":
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError(
                "fp8_per_tensor preparation takes the bf16/fp16/fp32 weight and "
                "quantizes it; pass the dequantization scale through "
                "GemmWeightPack views directly if the weight is already FP8."
            )
        b_fp8, b_scale = _fp8_per_tensor_quantize(weight, None)
        b_fp8 = b_fp8.contiguous()
        view.update(
            b=b_fp8,
            b_scale=b_scale,
            # bmm_fp8 takes B as (batch, K, N) column-major: (N, K) row-major
            # transposed and given a batch axis of one. Pure views, no copy.
            b3=b_fp8.t().unsqueeze(0),
        )
        return view

    raise BackendUnsupported(f"unknown weight recipe {recipe!r}.")


def prepare_weight_pack(
    weight: Tensor,
    config: GemmConfig,
    *,
    bias: Optional[Tensor] = None,
    for_backends: Optional[Iterable[str]] = None,
    activation_global_scale: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
) -> GemmWeightPack:
    """Prepare one ``[N, K]`` weight for every candidate backend of ``config``.

    Backends that share a weight recipe share one view. ``for_backends`` limits
    preparation to the named backend keys (memory control).
    ``activation_global_scale`` (NVFP4 only) folds a static activation scale
    into ``alpha`` at load time, as the frameworks do with calibrated scales.
    """
    device = device or weight.device
    arch = None
    if device.type == "cuda":
        from ..utils import get_compute_capability

        major, minor = get_compute_capability(device)
        arch = major * 10 + minor
    wanted = None if for_backends is None else set(for_backends)
    n_out, k_in = weight.shape
    pack = GemmWeightPack((int(n_out), int(k_in)), bias=bias)
    for cand in config.backend:
        key = cand.backend_key
        if wanted is not None and key not in wanted:
            continue
        if not cand.supports_quant(config.quant):
            continue
        if arch is not None and not cand.supports_pair_on(config.quant.pair, arch):
            continue
        recipe = cand.weight_recipe(config.quant)
        view = pack.view_for_recipe(recipe)
        if view is None:
            view = _prepare_weight_view(
                recipe,
                weight,
                quant=config.quant,
                activation_global_scale=activation_global_scale,
            )
        pack.prepare_for(key, view)
    if not pack.native_views:
        raise BackendUnsupported(
            "prepare_weight_pack: no candidate backend of "
            f"{[type(c).__name__ for c in config.backend]} supports quant pair "
            f"{config.quant.pair!r}"
            + (f" on sm{arch}" if arch is not None else "")
            + (f" within for_backends={sorted(wanted)}" if wanted else "")
            + "."
        )
    return pack


def prepare_activations(
    a: Tensor,
    *,
    quant: QuantConfig,
    global_scale: Optional[Tensor] = None,
    sf_layout: SfLayout = SfLayout.layout_128x4,
) -> GemmActivationPack:
    """Quantize a high-precision ``[M, K]`` activation into the pair's canonical pack.

    ``global_scale`` is the static per-tensor scale when known (NVFP4: the
    ``nvfp4_quantize`` global scale; FP8: the dequantization scale); ``None``
    derives a dynamic one from ``amax(a)`` (one extra reduction per call).
    """
    if a.dim() != 2:
        raise ValueError(f"activation must be [M, K], got shape {tuple(a.shape)}.")
    act = quant.activation
    if act == QuantFormat.NVFP4:
        from ..quantization.fp4_quantization import nvfp4_quantize

        if a.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"NVFP4 activation quantization takes bf16/fp16, got {a.dtype}."
            )
        a_gs = (
            nvfp4_global_scale(a)
            if global_scale is None
            else _as_f32_scalar(global_scale, a.device)
        )
        a_fp4, a_sf = nvfp4_quantize(
            a.contiguous(), a_gs, sfLayout=sf_layout, do_shuffle=False
        )
        return GemmActivationPack(
            a=a_fp4, a_scale=a_sf, a_global_scale=a_gs, sf_layout=sf_layout
        )
    if act == QuantFormat.FP8PerTensor:
        a_q, a_scale = _fp8_per_tensor_quantize(a, global_scale)
        return GemmActivationPack(a=a_q.contiguous(), a_scale=a_scale)
    raise BackendUnsupported(
        f"prepare_activations: activation format {act!r} is not in the unified GEMM "
        "MVP (NVFP4, FP8PerTensor)."
    )
