"""Activations of the gate/up GEMM's epilogue, as FlashInfer's MoE API names them
(``flashinfer.fused_moe.api``: ``SwiGLU``, ``SwiGLUStep``, ``GeGLU``, ``GeGLUTanh``, ``SiTU``, ``ReLU2``; ``CuteMoE(activation=)``
takes these classes or FlashInfer's own objects, see ``as_activation``). ``gate`` and ``up`` are the dequantized GEMM outputs of a
gate row and its up row (the weights' gate rows come first):

- ``SwiGLU(alpha=1, beta=0, limit=inf)``: ``g * sigmoid(alpha * g) * (u + beta)`` with ``g = min(gate, limit)`` and
  ``u = clamp(up, -limit, limit)`` (no limit at inf or FLT_MAX, FlashInfer's default; FlashInfer's SwiGLU/OA; DeepSeek V4 and GLM-5 use ``limit=10``, GPT-OSS
  ``alpha=1.702, beta=1, limit=7``). The default is the plain SwiGLU, for which the epilogues keep their own
  instruction sequences (``plain``).
- ``SwiGLUStep(limit=7)``: ``min(silu(gate), limit) * clamp(up, -limit, limit)``.
- ``GeGLU()``: ``gelu(gate) * up`` (erf GELU); ``GeGLUTanh()``: the tanh approximation of GELU.
- ``SiTU(gate_scale=4, linear_scale=25)`` (Kimi-K3's SiTU v2): ``gate_scale * tanh(gate / gate_scale) *
  sigmoid(gate) * linear_scale * tanh(up / linear_scale)``, the linear branch unclamped (``up``) with
  ``linear_scale=None``; FlashInfer's ``clamp_limit`` is not supported (FlashInfer's CuTe-DSL backend rejects it too).

Non-gated (``is_gated`` False; the gate/up GEMM has only the I "up" rows; ``torch_ref(x)`` of one value):

- ``ReLU2()``: ``relu(x) ** 2`` (Nemotron-H's experts: Nemotron 3 Nano / Super / Ultra).

FlashInfer's other non-gated activations (GELU, SiLU, ReLU, Identity) are not supported. ``torch_ref`` gives each activation in FP32 PyTorch for the reference.
"""

import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32

LOG2E = 1.4426950408889634
FLT_MAX = 3.4028234663852886e38  # FlashInfer's DEFAULT_SWIGLU_LIMIT: no limit


def _limited(limit):
    return not (math.isinf(limit) or limit >= FLT_MAX)


def _div_full(a, b):
    """``div.full.f32`` (as gemm.div_full)."""
    return cute.math.div(a, b, full=True)


def _sigmoid_mul(g, scale_in=None):
    """g / (1 + 2^(-g' log2 e)) with g' = scale_in * g (or g): the weight-major SwiGLU's instruction sequence
    (``div.full``), also used in the token-major epilogue."""
    gi = g if scale_in is None else g * Float32(scale_in)
    e = cute.math.exp2((Float32(0.0) - gi) * Float32(LOG2E))
    return _div_full(g, e + Float32(1.0))


@dataclass(frozen=True)
class SwiGLU:
    alpha: float = 1.0
    beta: float = 0.0
    limit: float = math.inf

    @property
    def plain(self):
        """The plain SwiGLU (alpha 1, beta 0, no limit): the kernels keep their own instruction sequences."""
        return self.alpha == 1.0 and self.beta == 0.0 and not _limited(self.limit)

    def apply(self, gate, up):
        """Kernel side (traced; Float32 scaled values)."""
        g, u = gate, up
        if cutlass.const_expr(_limited(self.limit)):
            g = cute.arch.fmin(g, Float32(self.limit))
            u = cute.arch.fmax(
                cute.arch.fmin(u, Float32(self.limit)), Float32(-self.limit)
            )
        if cutlass.const_expr(self.beta != 0.0):
            u = u + Float32(self.beta)
        return u * _sigmoid_mul(g, None if self.alpha == 1.0 else self.alpha)

    def torch_ref(self, gate, up):
        import torch

        g, u = gate, up
        if _limited(self.limit):
            g, u = g.clamp(max=self.limit), u.clamp(-self.limit, self.limit)
        return g * torch.sigmoid(self.alpha * g) * (u + self.beta)


@dataclass(frozen=True)
class SwiGLUStep:
    limit: float = 7.0

    plain = False

    def apply(self, gate, up):
        s = cute.arch.fmin(_sigmoid_mul(gate), Float32(self.limit))
        return s * cute.arch.fmax(
            cute.arch.fmin(up, Float32(self.limit)), Float32(-self.limit)
        )

    def torch_ref(self, gate, up):
        import torch.nn.functional as F

        return F.silu(gate).clamp(max=self.limit) * up.clamp(-self.limit, self.limit)


@dataclass(frozen=True)
class GeGLU:
    plain = False

    def apply(self, gate, up):
        half_g = gate * Float32(0.5)
        return (
            half_g * (Float32(1.0) + cute.math.erf(gate * Float32(math.sqrt(0.5)))) * up
        )

    def torch_ref(self, gate, up):
        import torch.nn.functional as F

        return F.gelu(gate) * up


@dataclass(frozen=True)
class GeGLUTanh:
    plain = False

    def apply(self, gate, up):
        inner = (
            gate
            * Float32(math.sqrt(2.0 / math.pi))
            * (Float32(1.0) + Float32(0.044715) * gate * gate)
        )
        half_g = gate * Float32(0.5)
        return half_g * (Float32(1.0) + cute.math.tanh(inner)) * up

    def torch_ref(self, gate, up):
        import torch.nn.functional as F

        return F.gelu(gate, approximate="tanh") * up


@dataclass(frozen=True)
class SiTU:
    gate_scale: float = 4.0
    linear_scale: Optional[float] = 25.0

    plain = False

    def apply(self, gate, up):
        gs = Float32(self.gate_scale)
        sig = _div_full(
            Float32(1.0),
            cute.math.exp2((Float32(0.0) - gate) * Float32(LOG2E)) + Float32(1.0),
        )
        g = gs * cute.math.tanh(gate * Float32(1.0 / self.gate_scale)) * sig
        if cutlass.const_expr(self.linear_scale is None):
            return g * up
        return g * (
            Float32(self.linear_scale)
            * cute.math.tanh(up * Float32(1.0 / self.linear_scale))
        )

    def torch_ref(self, gate, up):
        import torch

        g = self.gate_scale * torch.tanh(gate / self.gate_scale) * torch.sigmoid(gate)
        u = (
            up
            if self.linear_scale is None
            else self.linear_scale * torch.tanh(up / self.linear_scale)
        )
        return g * u


@dataclass(frozen=True)
class ReLU2:
    """The kernels compute it themselves (``gemm.relu2_bf16_pair``; outputs >= 0: the block max is a plain max)."""

    plain = False
    is_gated = False

    def torch_ref(self, x):
        import torch

        return torch.relu(x).square()


ACTIVATIONS = {
    c.__name__: c for c in (SwiGLU, SwiGLUStep, GeGLU, GeGLUTanh, SiTU, ReLU2)
}


def is_gated(a):
    """Whether the activation takes a gate and an up row (all but ``ReLU2``)."""
    return getattr(a, "is_gated", True)


def as_activation(a=None):
    """``None`` (plain SwiGLU), one of this module's activations, or FlashInfer's ``ActivationConfig`` of the same name
    (``flashinfer.fused_moe.api.SwiGLU(alpha=, beta=, limit=)`` etc.; read by class name and fields, without importing
    FlashInfer). Raises for activations the kernels do not implement."""
    if a is None:
        return SwiGLU()
    cls = ACTIVATIONS.get(type(a).__name__)
    if cls is None:
        raise NotImplementedError(
            f"activation {a!r}: supported are {', '.join(ACTIVATIONS)}"
        )
    if isinstance(a, cls):
        return a
    ours = {f.name for f in fields(cls)}
    if is_dataclass(
        a
    ):  # FlashInfer's fields beyond ours must be unset (e.g. SiTU's clamp_limit)
        extra = [
            f.name
            for f in fields(a)
            if f.name not in ours and getattr(a, f.name) is not None
        ]
        if extra:
            raise NotImplementedError(
                f"activation {a!r}: {', '.join(extra)} not supported"
            )
    return cls(
        **{n: (None if getattr(a, n) is None else float(getattr(a, n))) for n in ours}
    )


def parse(spec):
    """``"swiglu"``, ``"swiglu:alpha=1.702,beta=1,limit=7"``, ``"swiglu_step:limit=7"``, ``"geglu"``, ``"geglu_tanh"``,
    ``"situ"``, ``"situ:gate_scale=4,linear_scale=none"``, ``"relu2"``
    (case and underscores ignored) -> the activation."""
    name, _, args = spec.partition(":")
    key = name.replace("_", "").lower()
    cls = {c.lower(): v for c, v in ACTIVATIONS.items()}.get(key)
    if cls is None:
        raise ValueError(f"unknown activation {name!r}: {', '.join(ACTIVATIONS)}")
    kw = dict(kv.split("=") for kv in args.split(",") if kv)
    return cls(
        **{k: (None if v.lower() == "none" else float(v)) for k, v in kw.items()}
    )
