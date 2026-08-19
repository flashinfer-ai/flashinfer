"""Offline knob tuner for the SM107 mxfp8 block-scaled mega kernel.

Invoked through the :mod:`flashinfer.moe_ep.tune` CLI shim (``--arch sm107
--dtype mxfp8_e4m3|mxfp8_e5m2``).  Thin quant-kind binding over the shared
SM107 driver in ``backends/mega/kernel/sm107/tuning.py``; the candidate space
and the rebuild-per-candidate collective sweep live in
``kernel_src/sm107/next_cutedsl_megamoe/shim/autotune.py``.
"""

from __future__ import annotations

from ..tuning import run_tuning as _run_tuning


def run_tuning(args) -> int:
    return _run_tuning(args, args.dtype)
