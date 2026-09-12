"""On-the-fly weight/activation quantization for the FlashInfer LLM example.

Weights are quantized once at load time from a plain BF16 Hugging Face
checkpoint; activations are quantized per forward. No pre-quantized checkpoint
is needed, and BF16 remains the default *and* the numerical reference.

    bf16   torch.nn.functional.linear (default, the reference path)
    fp8    W8A8 per-tensor e4m3, via ``flashinfer.bmm_fp8``
    nvfp4  W4A4 block-16 e2m1 + e4m3 block scales, via ``flashinfer.mm_fp4``

Scope. Only the dense projections (q/k/v/o, gate/up/down) are quantized. The
MoE router stays in model dtype (top-k is discrete — a quantization
perturbation flips experts and produces a step change no tolerance can
distinguish from a real bug), the MoE expert GEMMs keep ``quant_scales=None``,
and ``lm_head`` is left alone because it is literally the same tensor as the
embedding table under ``tie_word_embeddings``.

This is a verification harness, not a serving stack: the point is to exercise
FlashInfer's quantized GEMM kernels end to end, not to produce a well-behaved
quantized model. Expect FP4 output quality on a small model to be poor.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import torch

import flashinfer
from flashinfer.utils import get_compute_capability

QUANT_MODES = ("bf16", "fp8", "nvfp4")

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0
# e4m3 max * e2m1 max — the global scale-factor convention mm_fp4 expects.
NVFP4_SF_NUM = 448.0 * 6.0


class Linear:
    """BF16 reference path — numerically identical to the pre-quantization code."""

    kind = "bf16"

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        self.weight = weight
        self.bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class Fp8Linear:
    """W8A8 per-tensor e4m3 via ``bmm_fp8``.

    Both scales are *dequant multipliers* (``out = (A*a_s) @ (B*b_s)``), which
    is what ``tests/utils_fp8.py:to_float8`` returns as its second value — do
    not invert them again.
    """

    kind = "fp8"

    def __init__(self, weight, bias=None, backend: str = "auto"):
        self.bias = bias
        self.backend = backend
        w_fp8, self.b_scale = _to_fp8_per_tensor(weight)
        # bmm_fp8 wants B as (b, k, n) column-major. An (n, k) row-major fp8
        # tensor *is* that memory, so the transpose is a free view — quantize
        # first, then transpose, never .contiguous() the result.
        self.b = w_fp8.unsqueeze(0).transpose(-2, -1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_fp8, a_scale = _to_fp8_per_tensor(x)
        out = flashinfer.bmm_fp8(
            x_fp8.unsqueeze(0),
            self.b,
            a_scale,
            self.b_scale,
            torch.bfloat16,
            backend=self.backend,
        ).squeeze(0)
        return out if self.bias is None else out + self.bias


class Nvfp4Linear:
    """W4A4 block-16 e2m1 with e4m3 block scales via ``mm_fp4``.

    Layout follows ``tests/gemm/test_mm_fp4.py``: quantize the (n, k) weight as
    given, then pass ``b`` and ``b_descale`` as plain ``.T`` views. The block
    scales are a swizzled 128x4 blob whose 2-D shape is a fiction — never
    ``.contiguous()`` them, and never re-swizzle.
    """

    kind = "nvfp4"

    def __init__(self, weight, bias=None, backend: str = "auto"):
        self.bias = bias
        self.backend = backend
        self.w_sf_global = _global_sf(weight)
        w_fp4, w_sf = flashinfer.nvfp4_quantize(
            weight,
            self.w_sf_global,
            sfLayout=flashinfer.SfLayout.layout_128x4,
            # Only the TRT-LLM backend wants shuffled B scales, and it is not
            # reachable from this example's --quant-backend choices.
            do_shuffle=False,
        )
        self.b = w_fp4.T
        self.b_sf = w_sf.T

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        x_sf_global = _global_sf(x)
        x_fp4, x_sf = flashinfer.nvfp4_quantize(
            x,
            x_sf_global,
            sfLayout=flashinfer.SfLayout.layout_128x4,
            do_shuffle=False,
        )
        out = flashinfer.mm_fp4(
            x_fp4,
            self.b,
            x_sf,
            self.b_sf,
            1.0 / (x_sf_global * self.w_sf_global),
            torch.bfloat16,
            block_size=16,
            use_8x4_sf_layout=False,
            backend=self.backend,
            use_nvfp4=True,
        )
        return out if self.bias is None else out + self.bias


def _to_fp8_per_tensor(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize to e4m3 with one scale for the whole tensor.

    Stays device-side throughout (no ``.item()``), so the decode loop does not
    pick up a host sync per projection.
    """
    amax = t.float().abs().amax().clamp(min=1e-12)
    scale = FP8_MAX / amax
    q = (t * scale).clamp(min=-FP8_MAX, max=FP8_MAX).to(FP8_DTYPE)
    return q, scale.reciprocal()


def _global_sf(t: torch.Tensor) -> torch.Tensor:
    return NVFP4_SF_NUM / t.float().abs().nan_to_num().amax().clamp(min=1e-12)


def _shape_ok(mode: str, weight: torch.Tensor) -> bool:
    n, k = weight.shape
    if mode == "fp8":
        return n % 16 == 0 and k % 16 == 0
    # nvfp4_quantize itself only needs k % 16, but the CUTLASS SM100/103 kernel
    # hard-checks k % 32 and n % 32. Use the strict bound so which weights fall
    # back is a property of the checkpoint, not of the installed cuDNN version.
    return n % 32 == 0 and k % 32 == 0


def unsupported_reason(mode: str, backend: str, device) -> Optional[str]:
    """``None`` if this mode can run here, else a message for a clean skip."""
    if mode == "bf16":
        return None
    if mode not in QUANT_MODES:
        return f"unknown --quant mode {mode!r}"
    api = flashinfer.bmm_fp8 if mode == "fp8" else flashinfer.mm_fp4
    major, minor = get_compute_capability(device)
    cc = major * 10 + minor
    if not api.is_compute_capability_supported(cc):
        return f"--quant {mode} is not supported on sm{cc}"
    if backend != "auto" and not api.is_backend_supported(backend, cc):
        return f"--quant-backend {backend!r} is not supported for {mode} on sm{cc}"
    return None


@dataclasses.dataclass
class QuantConfig:
    """Factory for the per-projection linear, plus the accounting we report."""

    mode: str = "bf16"
    backend: str = "auto"
    quantized: int = 0
    fallbacks: List[Tuple[str, Tuple[int, ...]]] = dataclasses.field(
        default_factory=list
    )

    def linear(self, weight, bias=None, name: str = ""):
        if weight is None:
            return None
        if self.mode == "bf16":
            return Linear(weight, bias)
        if not _shape_ok(self.mode, weight):
            self.fallbacks.append((name, tuple(weight.shape)))
            return Linear(weight, bias)
        self.quantized += 1
        cls = Fp8Linear if self.mode == "fp8" else Nvfp4Linear
        return cls(weight, bias, self.backend)

    def resolved_backends(self) -> str:
        """Which backend ``auto`` actually picked, set after the first call."""
        if self.mode == "bf16":
            return ""
        api = flashinfer.bmm_fp8 if self.mode == "fp8" else flashinfer.mm_fp4
        return ",".join(getattr(api, "suitable_auto_backends", []) or [])

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "quantized_linears": self.quantized,
            "bf16_fallbacks": len(self.fallbacks),
            "resolved_backends": self.resolved_backends(),
        }


def self_check(
    mode: str, backend: str, device, m: int = 64, k: int = 512, n: int = 256
) -> Tuple[float, float]:
    """One random GEMM through the quantized path vs BF16 ``F.linear``.

    Returns ``(cosine_similarity, bar)``. The bars mirror the kernel tests
    (``tests/gemm/test_bmm_fp8.py`` > 0.99, ``tests/gemm/test_mm_fp4.py`` >
    0.97). This exists to separate "the wrapper is wired wrong" from "this
    model is small and FP4 is lossy": a transposed descale or a re-swizzled
    scale blob lands near cos=0, while honest quantization error does not.
    """
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w).float().reshape(-1)
    got = QuantConfig(mode, backend).linear(w)(x).float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
    return cos, (0.99 if mode == "fp8" else 0.97)
