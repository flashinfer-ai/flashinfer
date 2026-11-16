"""Resolve the two supported combine formats."""

from ..src.quant_def import CombineFormat


def get_combine_format(name):
    if name not in ("bf16", "mxfp8"):
        raise ValueError("Kimi combine must be bf16 or mxfp8")
    return CombineFormat.parse("32e4m3xe8m0" if name == "mxfp8" else name)
