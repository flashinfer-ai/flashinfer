"""Compiled FC1 weight extent of the dynamic MoE kernel.

The dynamic kernel addresses each expert's FC1 weights with the compiled
expert stride, so the compiled row extent must match the prepared layout.
w4a8_nvfp4 prepares FC1 with each gated half padded to the 128-row level
tile; compiling the unpadded 2n reads every expert after the first at the
wrong offset when n % 128 != 0 (n=320: 640 compiled rows vs 768 prepared).
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe.fused_moe import _impl


class _Compiled(Exception):
    pass


def _compiled_fc1_shape(
    monkeypatch, quant_mode: str, n: int, k: int = 2048, E: int = 8
):
    """Return (rows, k_extent, experts) of the FC1 weight operand compiled for n."""
    shapes = []
    make_fake = _impl.cute.runtime.make_fake_compact_tensor

    def record(dtype, shape, *args, **kwargs):
        shapes.append((tuple(shape), kwargs.get("stride_order")))
        return make_fake(dtype, shape, *args, **kwargs)

    def stop(*args, **kwargs):
        raise _Compiled

    monkeypatch.setattr(_impl.cute.runtime, "make_fake_compact_tensor", record)
    monkeypatch.setattr(_impl, "b12x_compile", stop)
    monkeypatch.setattr(_impl, "_DYNAMIC_KERNEL_CACHE", {})
    with pytest.raises(_Compiled):
        _impl._get_dynamic_kernel(
            E,
            64,
            k,
            n,
            2,
            128,
            topk_ids_dtype=torch.int32,
            fast_math=True,
            mac_override=148,
            activation="silu",
            quant_mode=quant_mode,
        )
    # FC1 is the first expert-major weight operand: (rows, K extent, E).
    return next(
        shape
        for shape, order in shapes
        if order == (1, 0, 2) and len(shape) == 3 and shape[2] == E
    )


@pytest.mark.parametrize("n,rows", [(320, 768), (384, 768), (1024, 2048)])
def test_w4a8_nvfp4_fc1_covers_padded_gate_and_up_halves(monkeypatch, n, rows):
    assert _compiled_fc1_shape(monkeypatch, "w4a8_nvfp4", n)[0] == rows
