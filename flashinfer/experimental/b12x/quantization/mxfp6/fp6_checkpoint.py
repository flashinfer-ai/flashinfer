"""ModelOpt-mirror HF safetensors schema for b12x MX-FP6 (W6A6) checkpoints.

This module defines the *on-disk* contract for a b12x FP6 checkpoint and
the low-level primitive that quantizes a single linear weight into that
contract. It is deliberately framework-agnostic: it produces plain CPU tensors
+ a ``quantization_config`` dict, so the model-level exporter
(:mod:`b12x.quantization.mxfp6.model_fp6`) and any serving adapter
(vLLM / SGLang) share one source of truth.

Per-tensor layout (one quantized linear = one dense ``nn.Linear`` *or* one MoE
expert projection), mirroring ModelOpt NVFP4 key conventions:

* ``<name>.weight``         uint8   packed FP6 codes ``(out_features, 3*in_features//4)``
* ``<name>.weight_scale``   uint8   UE8M0 per-32 block scales, **unswizzled**,
                                    ``(out_features, in_features//32)``
* ``<name>.weight_scale_2`` f32     scalar weight global scale (1.0 for pure MX)
* ``<name>.input_scale``    f32     scalar activation global scale (1.0; activations
                                    are dynamically FP6-quantized in-kernel)

Block scales are stored **unswizzled** (row-major ``[row, block]``), exactly like
ModelOpt NVFP4 stores its E4M3 K/16 grid; the cutlass swizzle is applied at load
time, so the artifact stays layout-portable. The scales are the raw UE8M0
exponent bytes kept as ``uint8`` (safetensors does not portably serialize
``float8_e8m0fnu``); the loader bitcasts them back.

``config.json`` carries a ``quantization_config`` with ``quant_method="modelopt"``
and ``quant_algo="W6A6"`` so a framework can route FP6 weights to the b12x
kernel.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from typing import Optional

import torch

from b12x._lib.fp6 import (
    FLOAT6_E2M3_MAX,
    FLOAT6_E3M2_MAX,
    SF_VEC_SIZE_FP6,
    _ue8m0_scale_from_block_max,
)
from b12x._lib.utils import mxfp6_packed_k_bytes

# --- schema constants --------------------------------------------------------

SCHEMA_VERSION = "b12x_fp6_safetensors_v1"  # historical schema id; do not rename
QUANT_METHOD = "modelopt"
QUANT_ALGO = "W6A6"
GROUP_SIZE = SF_VEC_SIZE_FP6  # 32-element MX block

WEIGHT_SUFFIX = ".weight"
WEIGHT_SCALE_SUFFIX = ".weight_scale"
WEIGHT_SCALE_2_SUFFIX = ".weight_scale_2"
INPUT_SCALE_SUFFIX = ".input_scale"

# UE8M0 bytes kept as uint8 on disk (see module docstring).
SCALE_DTYPE_TAG = "uint8_ue8m0"

# Offline weight UE8M0 exponent selection (runtime activation quant stays ceil).
_BLOCK_SCALE_RULES = frozenset({"ceil", "mse"})


def _validate_block_scale_rule(rule: str) -> str:
    """Normalize + validate ``block_scale_rule`` (shared by config + quantizer)."""
    normalized = str(rule).lower().strip()
    if normalized not in _BLOCK_SCALE_RULES:
        raise ValueError(
            f"block_scale_rule must be one of {sorted(_BLOCK_SCALE_RULES)}, got "
            f"{rule!r}"
        )
    return normalized

_TILE = 128


_SOURCE_FORMATS_E3M2 = frozenset({"mxfp6_e3m2", "e3m2", "float6_e3m2fn"})
_SOURCE_FORMATS_E2M3 = frozenset({"mxfp6_e2m3", "e2m3", "float6_e2m3fn"})
_SOURCE_FORMATS_GENERIC = frozenset({"mxfp6_default", "mxfp6_mixed", "mxfp6_w6a8"})
_SOURCE_FORMATS = _SOURCE_FORMATS_E3M2 | _SOURCE_FORMATS_E2M3 | _SOURCE_FORMATS_GENERIC


def _validate_source_format(source_format: str) -> str:
    """Normalize + validate ``source_format`` (fail fast on CLI typos)."""
    sf = str(source_format).lower().strip()
    if sf not in _SOURCE_FORMATS:
        raise ValueError(
            f"unsupported source_format {source_format!r}; expected one of "
            f"{sorted(_SOURCE_FORMATS)}"
        )
    return sf


def weight_format_for_source(source_format: str) -> str:
    """FP6 sub-format (``e2m3``/``e3m2``) for the *weight* operand."""
    sf = _validate_source_format(source_format)
    if sf in _SOURCE_FORMATS_E3M2:
        return "e3m2"
    # mxfp6_e2m3 aliases and mxfp6_default / mxfp6_mixed / mxfp6_w6a8: E2M3.
    return "e2m3"


def activation_format_for_source(source_format: str) -> str:
    """Activation operand sub-format (kernel quantizes these live)."""
    sf = _validate_source_format(source_format)
    if sf in _SOURCE_FORMATS_E2M3:
        return "e2m3"
    if sf == "mxfp6_w6a8":
        # W6A8: FP8 E4M3 activations (same mxf8f6f4 MMA kind / UE8M0 scales).
        return "e4m3"
    # e3m2 aliases and mxfp6_default / mxfp6_mixed: E3M2 activations
    # (more range for dynamic data).
    return "e3m2"


def build_quantization_config(
    *,
    source_format: str = "mxfp6_w6a8",
    exclude_modules: Optional[list[str]] = None,
    activation_format_overrides: Optional[dict[str, str]] = None,
    block_scale_rule: str = "mse",
) -> dict:
    """The ``quantization_config`` block to inject into ``config.json``.

    Mirrors ModelOpt's structure (``quant_method``/``quant_algo``) so frameworks
    can detect the scheme, while carrying the extra MX-FP6 specifics b12x
    needs.

    ``activation_format_overrides`` is an optional map of fnmatch patterns
    (matched against the normalized module path) to an activation sub-format
    (``e2m3`` / ``e3m2`` / ``e4m3``). Model-agnostic: any dense architecture
    can tune recurrent / linear-attn / MLP groups without re-exporting weights.
    Example::

        {"*.linear_attn.*": "e3m2", "*.mlp.*": "e4m3"}

    ``block_scale_rule`` records how offline weight UE8M0 exponents were chosen
    (``mse`` = ceil vs ceil-1 per block; ``ceil`` = amax containment only).
    Runtime does not read this field; it is provenance for the checkpoint.
    """
    rule = _validate_block_scale_rule(block_scale_rule)
    cfg: dict = {
        "quant_method": QUANT_METHOD,
        "quant_algo": QUANT_ALGO,
        "kv_cache_scheme": None,
        "group_size": GROUP_SIZE,
        "weight_format": weight_format_for_source(source_format),
        "activation_format": activation_format_for_source(source_format),
        "block_scale_rule": rule,
        "scale_dtype": SCALE_DTYPE_TAG,
        "exclude_modules": list(exclude_modules) if exclude_modules else [],
        # historical producer name; do not rename (existing checkpoints carry it)
        "producer": {"name": "b12x", "schema": SCHEMA_VERSION},
    }
    if activation_format_overrides:
        cfg["activation_format_overrides"] = dict(activation_format_overrides)
    return cfg


_ACT_FMT_ALIASES = {
    "e2m3": "e2m3",
    "e3m2": "e3m2",
    "e4m3": "e4m3",
    "float6_e2m3fn": "e2m3",
    "float6_e3m2fn": "e3m2",
    "float8_e4m3fn": "e4m3",
}


def normalize_activation_format(fmt: str) -> str:
    """Canonical activation sub-format (``e2m3`` / ``e3m2`` / ``e4m3``)."""
    key = str(fmt).lower().strip()
    if key not in _ACT_FMT_ALIASES:
        raise ValueError(
            f"unsupported activation format {fmt!r}; "
            f"expected one of {sorted(set(_ACT_FMT_ALIASES.values()))}"
        )
    return _ACT_FMT_ALIASES[key]


def parse_act_fmt_overrides(spec: str | dict | None) -> list[tuple[str, str]]:
    """Parse pattern->act_fmt overrides from a dict or ``pat=fmt,pat=fmt`` string.

    Order is preserved: first matching pattern wins. Empty / None -> ``[]``.
    """
    if spec is None or spec == "":
        return []
    if isinstance(spec, dict):
        items = list(spec.items())
    else:
        items = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    f"act-fmt override {part!r} must be pattern=fmt "
                    f"(e.g. '*.linear_attn.*=e3m2')"
                )
            pat, fmt = part.split("=", 1)
            items.append((pat.strip(), fmt.strip()))
    out: list[tuple[str, str]] = []
    for pat, fmt in items:
        if not pat:
            raise ValueError("empty pattern in activation_format_overrides")
        out.append((pat, normalize_activation_format(fmt)))
    return out


def resolve_activation_format(
    module_name: str,
    *,
    default_fmt: str,
    overrides: list[tuple[str, str]] | None = None,
) -> str:
    """Pick the activation format for ``module_name`` (first matching override).

    Patterns use :mod:`fnmatch` against the *normalized* module path (same
    convention as the vLLM plugin: no leading ``model.``). This is the single
    model-agnostic hook for per-group activation formats (linear_attn vs MLP,
    vision vs language, etc.) without architecture-specific branches.
    """
    default = normalize_activation_format(default_fmt)
    if not overrides:
        return default
    name = module_name.lstrip(".")
    if name.startswith("model."):
        name = name[len("model.") :]
    for pat, fmt in overrides:
        if fnmatch.fnmatch(name, pat):
            return fmt
    return default


def load_act_fmt_overrides(
    quant_config: dict | None = None,
) -> list[tuple[str, str]]:
    """Merge env + config overrides (env wins on conflict by coming first).

    * ``B12X_FP6_ACT_FMT_OVERRIDES`` — ``pat=fmt,pat=fmt`` (serve-time, no
      config edit).
    * ``quantization_config.activation_format_overrides`` — dict in
      ``config.json``.
    """
    env_spec = os.environ.get("B12X_FP6_ACT_FMT_OVERRIDES", "").strip()
    merged: list[tuple[str, str]] = []
    if env_spec:
        merged.extend(parse_act_fmt_overrides(env_spec))
    if quant_config:
        cfg_spec = quant_config.get("activation_format_overrides")
        if cfg_spec:
            merged.extend(parse_act_fmt_overrides(cfg_spec))
    return merged


# --- low-level single-linear quantizer --------------------------------------


@dataclass
class FP6LinearTensors:
    """On-disk FP6 tensors for one linear (ModelOpt-mirror key set)."""

    weight: torch.Tensor  # uint8 packed codes (out, 3*in//4)
    weight_scale: torch.Tensor  # uint8 UE8M0, unswizzled (out, in//32)
    weight_scale_2: torch.Tensor  # f32 scalar
    input_scale: torch.Tensor  # f32 scalar
    out_features: int
    in_features: int
    fmt: str

    def to_state_dict(self, name: str) -> dict[str, torch.Tensor]:
        """Map to the four on-disk keys under ``name`` (no ``.weight`` in name)."""
        return {
            name + WEIGHT_SUFFIX: self.weight,
            name + WEIGHT_SCALE_SUFFIX: self.weight_scale,
            name + WEIGHT_SCALE_2_SUFFIX: self.weight_scale_2,
            name + INPUT_SCALE_SUFFIX: self.input_scale,
        }


def _pack_codes(mat_bf16: torch.Tensor, fmt: str, *, use_gpu: bool) -> torch.Tensor:
    """Pack ``(out, in)`` bf16 to ``(out, 3*in//4)`` uint8 FP6 codes (unit gscale)."""
    m, k = mat_bf16.shape
    packed_k = mxfp6_packed_k_bytes(k)
    if use_gpu:
        if m % _TILE != 0 or k % _TILE != 0:
            raise ValueError(
                f"GPU FP6 quantizer requires out%{_TILE}==0 and in%{_TILE}==0, "
                f"got out={m} in={k}. Pass use_gpu=False for the torch reference."
            )
        # Lazy import to avoid an import cycle with the package __init__.
        from b12x.quantization.mxfp6 import (
            allocate_bf16_to_fp6_tma_outputs,
            compile_bf16_to_fp6_tma,
        )

        launch = compile_bf16_to_fp6_tma(m, k, fmt=fmt)
        gs_one = torch.ones(1, dtype=torch.float32, device=mat_bf16.device)
        out = allocate_bf16_to_fp6_tma_outputs(m, k, device=mat_bf16.device)
        launch(mat_bf16.contiguous(), gs_one, out.packed_a_flat, out.scale_flat)
        torch.cuda.synchronize()
        return out.packed_a_storage.view(m, packed_k).clone()

    from b12x._lib.fp6 import quantize_grouped_mxfp6_torch

    row_counts = torch.full((1,), m, dtype=torch.int32, device=mat_bf16.device)
    gs = torch.ones(1, dtype=torch.float32, device=mat_bf16.device)
    packed_grp, _ = quantize_grouped_mxfp6_torch(
        mat_bf16.unsqueeze(0), row_counts, gs, fmt=fmt  # type: ignore[arg-type]
    )
    return packed_grp.permute(2, 0, 1).contiguous()[0]


def _unswizzled_block_scales(mat_bf16: torch.Tensor, fmt: str) -> torch.Tensor:
    """Row-major UE8M0 per-32-block scale bytes ``(out, in//32)`` (no swizzle).

    Bit-identical to the swizzled MoE/dense path minus the cutlass permute, so a
    load-time ``swizzle_block_scale`` reproduces the exact kernel-ready scales.
    """
    fmt_max = FLOAT6_E3M2_MAX if fmt == "e3m2" else FLOAT6_E2M3_MAX
    rows, k = mat_bf16.shape
    if k % SF_VEC_SIZE_FP6 != 0:
        raise ValueError(f"in_features must be divisible by {SF_VEC_SIZE_FP6}, got {k}")
    blocks = k // SF_VEC_SIZE_FP6
    block_max = mat_bf16.float().view(rows, blocks, SF_VEC_SIZE_FP6).abs().amax(dim=-1)
    return _ue8m0_scale_from_block_max(block_max, fmt_max)  # (rows, blocks) uint8


# Cache: (fmt, device_str) -> (sorted unique values, bucketize bounds, codes)
_FP6_ROUND_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _fp6_round_tables(
    fmt: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sorted unique FP6 values, midpoints, and a representative code per value."""
    from b12x._lib.fp6 import _decode_fp6_e2m3, _decode_fp6_e3m2

    key = (fmt, str(device))
    cached = _FP6_ROUND_CACHE.get(key)
    if cached is not None:
        return cached
    decode = _decode_fp6_e3m2 if fmt == "e3m2" else _decode_fp6_e2m3
    best: dict[float, int] = {}
    for code in range(64):
        val = decode(code)
        prev = best.get(val)
        # Prefer even mantissa LSB on value collisions (matches encode tie-break).
        if prev is None or ((code & 1) == 0 and (prev & 1) == 1):
            best[val] = code
    vals_sorted = sorted(best.keys())
    lut = torch.tensor(vals_sorted, dtype=torch.float32, device=device)
    codes = torch.tensor([best[v] for v in vals_sorted], dtype=torch.uint8, device=device)
    bounds = (lut[1:] + lut[:-1]) * 0.5
    _FP6_ROUND_CACHE[key] = (lut, bounds, codes)
    return lut, bounds, codes


def _round_encode_fp6(
    scaled: torch.Tensor, fmt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest FP6 value + code for already-scaled block elements."""
    fmt_max = FLOAT6_E3M2_MAX if fmt == "e3m2" else FLOAT6_E2M3_MAX
    lut, bounds, code_lut = _fp6_round_tables(fmt, scaled.device)
    clipped = scaled.clamp(-fmt_max, fmt_max).contiguous()
    idx = torch.bucketize(clipped, bounds)
    return lut[idx], code_lut[idx]


def _pack_codes_and_scales_mse(
    mat_bf16: torch.Tensor, fmt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint codes + UE8M0 scales with MSE-optimal per-block exponent.

    Per 32-element block: try the amax-ceil containment scale and the one-finer
    exponent (ceil-1, may clip the block max). Keep whichever reconstruction has
    lower block MSE. Never blind-floors — ceil remains the safe default candidate.
    Model-agnostic; offline weights only (runtime activation quant stays ceil).
    """
    from b12x._lib.fp6 import pack_fp6_codes_tensor

    fmt_max = FLOAT6_E3M2_MAX if fmt == "e3m2" else FLOAT6_E2M3_MAX
    rows, k = mat_bf16.shape
    if k % SF_VEC_SIZE_FP6 != 0:
        raise ValueError(f"in_features must be divisible by {SF_VEC_SIZE_FP6}, got {k}")
    if k % 4 != 0:
        raise ValueError(f"in_features must be divisible by 4 for FP6 packing, got {k}")
    blocks = k // SF_VEC_SIZE_FP6
    v = mat_bf16.float().view(rows, blocks, SF_VEC_SIZE_FP6)
    amax = v.abs().amax(dim=-1, keepdim=True)
    e = torch.ceil(torch.log2((amax / fmt_max).clamp(min=1e-38)))
    e = e.clamp(-127.0, 128.0)
    scale = torch.exp2(e)
    scale_lo = scale * 0.5

    q_hi_vals, codes_hi = _round_encode_fp6(v / scale, fmt)
    q_lo_vals, codes_lo = _round_encode_fp6(v / scale_lo, fmt)
    q_hi = q_hi_vals * scale
    q_lo = q_lo_vals * scale_lo
    mse_hi = ((q_hi - v) ** 2).sum(dim=-1, keepdim=True)
    mse_lo = ((q_lo - v) ** 2).sum(dim=-1, keepdim=True)
    nonzero = amax > 0
    # ue==0 is reserved as the "all-zero block" sentinel (see
    # dequantize_linear_from_fp6), so the finer exponent is only eligible when
    # e-1 still maps to ue >= 1 (e-1 >= -126). Gating the CANDIDATE (rather
    # than clamping ue afterwards) keeps codes and stored scale consistent —
    # a post-hoc ue clamp would pair e-1-encoded codes with an e-exponent
    # scale and silently dequantize the block 2x too large.
    lo_representable = e > -126.0
    use_lo = (mse_lo < mse_hi) & nonzero & lo_representable

    codes = torch.where(use_lo, codes_lo, codes_hi)
    codes = torch.where(nonzero, codes, torch.zeros_like(codes))
    e_chosen = torch.where(use_lo, e - 1.0, e)
    ue = (e_chosen.squeeze(-1).to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
    ue = torch.where(amax.squeeze(-1) <= 0, torch.zeros_like(ue), ue)

    packed = pack_fp6_codes_tensor(codes.reshape(rows, k))
    return packed, ue


def quantize_linear_to_fp6(
    weight_bf16: torch.Tensor,
    *,
    source_format: str = "mxfp6_w6a8",
    use_gpu: bool = True,
    block_scale_rule: str = "mse",
) -> FP6LinearTensors:
    """Quantize one ``(out_features, in_features)`` bf16 weight to FP6 disk tensors.

    Uses pure MX scaling: unit global scale (``weight_scale_2 = 1.0``) with UE8M0
    per-32-block scales carrying the dynamic range, matching the validated W6A6
    MoE path. Activations are quantized in-kernel, so ``input_scale = 1.0``.

    ``block_scale_rule``:
      * ``"mse"`` (default): per block, try ceil vs ceil-1 exponent and keep the
        lower reconstruction MSE (Phase 1.2). Codes and scales are chosen jointly.
      * ``"ceil"``: classic amax-containment UE8M0 (TMA/GPU path when ``use_gpu``).
    """
    if weight_bf16.ndim != 2:
        raise ValueError(
            f"weight must be rank-2 (out, in), got {tuple(weight_bf16.shape)}"
        )
    rule = _validate_block_scale_rule(block_scale_rule)
    out_features, in_features = weight_bf16.shape
    fmt = weight_format_for_source(source_format)
    w = weight_bf16.to(torch.bfloat16)
    if rule == "mse":
        # Joint torch path (CUDA tensors OK); TMA ceil encode cannot pick ceil-1.
        packed, block_scale = _pack_codes_and_scales_mse(w, fmt)
    else:
        packed = _pack_codes(w, fmt, use_gpu=use_gpu)
        block_scale = _unswizzled_block_scales(w, fmt)
    one = torch.ones(1, dtype=torch.float32)
    return FP6LinearTensors(
        weight=packed.cpu(),
        weight_scale=block_scale.cpu(),
        weight_scale_2=one.clone(),
        input_scale=one.clone(),
        out_features=out_features,
        in_features=in_features,
        fmt=fmt,
    )


# --- vectorized dequantizer (error reporting / BF16 round-trip export) -------


def fp6_decode_lut(fmt: str, device: torch.device | str = "cpu") -> torch.Tensor:
    """64-entry float32 LUT mapping an FP6 code (0..63) to its decoded value."""
    from b12x._lib.fp6 import _decode_fp6_e2m3, _decode_fp6_e3m2

    decode = _decode_fp6_e3m2 if fmt == "e3m2" else _decode_fp6_e2m3
    return torch.tensor(
        [decode(c) for c in range(64)], dtype=torch.float32, device=device
    )


def dequantize_linear_from_fp6(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    *,
    fmt: str,
    weight_scale_2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reconstruct a ``(out, in)`` float32 weight from on-disk FP6 tensors.

    Vectorized inverse of :func:`quantize_linear_to_fp6`:
    ``x_hat = decode(code) * 2^(ue8m0 - 127) / gs``. ``packed`` is
    ``(out, 3*in//4)`` uint8 codes; ``block_scale`` is the **unswizzled**
    ``(out, in//32)`` UE8M0 bytes as stored on disk (ue==0 means an all-zero
    block).

    ``weight_scale_2`` is the on-disk global weight scale: quantization stores
    ``code ~= x * gs / block_scale`` (and derives the block scale from
    ``block_max * gs``), so a non-unit ``gs`` divides back out here. ``None``
    or an all-ones tensor keeps the pure-MX fast path unchanged. Accepts a
    scalar or a per-row ``(out,)`` tensor (future per-row global scales).
    """
    from b12x._lib.fp6 import expand_mxfp6_packed_to_bytes

    rows, packed_cols = packed.shape
    k = packed_cols * 4 // 3
    codes = expand_mxfp6_packed_to_bytes(packed, k)  # (out, in) uint8
    lut = fp6_decode_lut(fmt, device=packed.device)
    values = lut[codes.long()]  # (out, in) f32
    ue = block_scale.to(torch.int32)
    scale = torch.where(
        ue == 0,
        torch.zeros((), dtype=torch.float32, device=packed.device),
        torch.exp2((ue - 127).to(torch.float32)),
    )  # (out, in//32)
    if scale.shape != (rows, k // GROUP_SIZE):
        raise ValueError(
            f"block_scale shape {tuple(scale.shape)} does not match packed "
            f"weight ({rows}, {k // GROUP_SIZE})"
        )
    w_hat = values * scale.repeat_interleave(GROUP_SIZE, dim=1)
    if weight_scale_2 is not None:
        gs = weight_scale_2.to(device=w_hat.device, dtype=torch.float32).reshape(-1)
        if not bool(torch.all(gs == 1.0)):
            if gs.numel() == 1:
                w_hat = w_hat / gs
            elif gs.numel() == rows:
                w_hat = w_hat / gs.view(rows, 1)
            else:
                raise ValueError(
                    f"weight_scale_2 has {gs.numel()} elements; expected a "
                    f"scalar or one per output row ({rows})"
                )
    return w_hat
