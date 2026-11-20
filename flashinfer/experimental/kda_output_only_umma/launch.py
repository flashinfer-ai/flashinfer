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
# (H=12, ncu, B 128 / 2048): T > 8 (dense 1.01-1.11x with the next-item input
# prefetch, drop-in 1.07-1.12x) and the drop-in at T <= 4 (1.00-1.03x); else one item
# per CTA (dense T <= 8: the persistent grid 1-8 % slower; drop-in T 5-8 even to 9 %
# slower). The persistent grid is sized num_sms * ctas_per_sm(T), the occupancy the
# build's min_blocks_per_mp bound guarantees; where that bound is forced to 1 (SM107,
# below) the residency is not known (registers may then limit it), so those parts keep
# one item per CTA, which lets the hardware fill each SM with however many CTAs fit.
# The next-item input prefetch (registers loaded during the current item) is on for
# the dense persistent grid only (drop-in T = 16 0.98x, mixed elsewhere). Tests set
# _PERSISTENT / _PREFETCH_NEXT to force either choice.
_PERSISTENT: Optional[bool] = None
_PREFETCH_NEXT: Optional[bool] = None


def _use_persistent(T: int, min_blocks: Optional[int], dropin: bool) -> bool:
    if _PERSISTENT is not None:
        return _PERSISTENT
    if min_blocks is not None:
        return False
    return T > 8 or (dropin and T <= 4)


def _use_prefetch_next(dropin: bool) -> bool:
    if _PREFETCH_NEXT is not None:
        return _PREFETCH_NEXT
    return not dropin


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
    persistent = _use_persistent(T, min_blocks, vllm_dropin)
    prefetch_next = persistent and _use_prefetch_next(vllm_dropin)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    key = (
        gdn_device_target(device).compile_key,
        persistent,
        prefetch_next,
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
            prefetch_next=prefetch_next,
        )
        fn = cute.compile[gdn_compile_options(device)](kernel, *args)
        _KERNELS[key] = fn
    fn(*args)
    _LAUNCHES += 1
