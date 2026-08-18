"""Offline knob tuner for the SM107 nvfp4 block-scaled mega kernel.

Invoked through the :mod:`flashinfer.moe_ep.tune` CLI shim (``--arch sm107
--dtype nvfp4``).  Thin quant-kind binding over the shared SM107 driver in
``backends/mega/kernel/sm107/tuning.py``; the candidate space and the
rebuild-per-candidate collective sweep live in
``kernel_src/next_cutedsl_megamoe/shim/autotune.py``.
"""

from __future__ import annotations

from ..tuning import run_tuning as _run_tuning


def run_tuning(args) -> int:
    return _run_tuning(args, "nvfp4")
