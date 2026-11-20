"""Host side of the UMMA output-only KDA decode: availability check, compile cache
and launch. The caller (``kda_decode_wy_output_only``) validates the contract and
builds the argument list shared with the WY kernel."""

from __future__ import annotations

from typing import Optional

import torch

# Number of UMMA launches issued (tests read it to confirm the backend ran).
_LAUNCHES = 0
_KERNELS: dict = {}
_T_MAX = 16


def kda_oo_umma_unavailable_reason(device: torch.device, T: int) -> Optional[str]:
    """``None`` when the UMMA backend can run this configuration, else why not."""
    try:
        from . import kda_output_only_umma  # noqa: F401  (CuTe DSL with tcgen05)
    except (ImportError, RuntimeError) as e:  # pragma: no cover
        return f"UMMA module import failed ({type(e).__name__}: {e})"
    cc = torch.cuda.get_device_capability(device)
    if cc[0] != 10:
        return f"needs SM10x (tcgen05), got SM{cc[0]}{cc[1]}"
    if not 1 <= T <= _T_MAX:
        return f"T={T} outside [1, {_T_MAX}]"
    return None


# backend="auto" (with FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1) picks this kernel
# up to this many tokens: B200 / H=12 (ncu, B 1-2048) measured it 1.03-1.25x faster
# than the WY kernel at T <= 8 (dense) and 1.14-2.02x (drop-in); at T = 16 it is
# about even (dense 0.96-0.98x at B >= 512, drop-in 0.94-1.06x), so not selected.
_AUTO_T_MAX = 8


def kda_oo_umma_auto_ok(device: torch.device, T: int) -> bool:
    """Whether ``backend="auto"`` (with the experimental opt-in) should pick this
    kernel: where it is measured faster than the WY kernel and can run."""
    return T <= _AUTO_T_MAX and kda_oo_umma_unavailable_reason(device, T) is None


# Persistent grid (CTAs stride over the work items) where it measured faster on B200
# (H=12, ncu, B 16-2048): T > 8 (dense 1.04-1.10x, drop-in 1.09-1.12x) and T <= 4
# (dense T=1 up to 1.05x, drop-in 1.03-1.05x, else even); at 5 <= T <= 8 one item
# per CTA (persistent 0-2 % slower dense, up to 10 % drop-in). Tests set _PERSISTENT
# to force either grid.
_PERSISTENT: Optional[bool] = None


def _use_persistent(T: int) -> bool:
    if _PERSISTENT is not None:
        return _PERSISTENT
    return T <= 4 or T > 8


def run_kda_oo_umma(
    args: list,
    *,
    device: torch.device,
    T: int,
    gate_mode: int,
    has_dt_bias: bool,
    beta_is_logit: bool,
    emit_corrections: bool,
    vllm_dropin: bool,
    null_min: int,
    HV: int,
    H: int,
    kg_v4: bool = True,
    g_v4: bool = True,
) -> None:
    """Compile (once per specialisation) and launch on ``args`` -- the
    ``KdaDecodeWyOutputOnlyKernel`` argument list. ``kg_v4`` / ``g_v4``: the kg
    cache strides / the gate token stride are multiples of 8 elements (16-B
    accesses; else the 2-byte builds)."""
    global _LAUNCHES
    import cutlass.cute as cute

    from flashinfer.gdn_kernels.device_target import (
        gdn_compile_options,
        gdn_device_target,
    )

    from .kda_output_only_umma import KdaOutputOnlyUmma

    # SM107 (Rubin): min_blocks_per_mp > 1 makes the DSL auto-derive an SMEM carveout
    # that faults (#4957); the WY kernel forces 1 there too. Other SM10x parts keep
    # the build's own occupancy bound (ctas_per_sm).
    min_blocks = (
        1 if tuple(torch.cuda.get_device_capability(device)) == (10, 7) else None
    )
    persistent = _use_persistent(T)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    key = (
        gdn_device_target(device).compile_key,
        persistent,
        num_sms,
        bool(g_v4),
        min_blocks,
        T,
        int(gate_mode),
        bool(has_dt_bias),
        bool(beta_is_logit),
        bool(emit_corrections),
        bool(vllm_dropin),
        int(null_min),
        HV,
        H,
        bool(kg_v4),
    )
    fn = _KERNELS.get(key)
    if fn is None:
        kernel = KdaOutputOnlyUmma(
            T,
            gate_mode=gate_mode,
            has_dt_bias=has_dt_bias,
            beta_is_logit=beta_is_logit,
            emit_corrections=emit_corrections,
            vllm_dropin=vllm_dropin,
            null_min=null_min,
            kg_v4=kg_v4,
            min_blocks=min_blocks,
            g_v4=g_v4,
            hv_heads=HV,
            q_heads=H,
            num_sms=num_sms,
            persistent=persistent,
        )
        fn = cute.compile[gdn_compile_options(device)](kernel, *args)
        _KERNELS[key] = fn
    fn(*args)
    _LAUNCHES += 1
