"""Mega-path compute backends (fused comm + local MoE).

Organized by taxonomy: ``sm<arch>/<act_dtype>_<weight_dtype>_<out_dtype>_<kernel_style>``.
Adapters live in ``moe_ep/shims/<sm>/``; sources live in ``moe_ep/kernels/``,
grouped by the project that supplies their updates.
``sm103`` wraps the experimental backend under
``flashinfer/experimental/cake_w8a8_megamoe_ep16/`` instead.
"""

from . import sm90, sm100, sm103, sm107, sm120

__all__ = ["sm90", "sm100", "sm103", "sm107", "sm120"]
