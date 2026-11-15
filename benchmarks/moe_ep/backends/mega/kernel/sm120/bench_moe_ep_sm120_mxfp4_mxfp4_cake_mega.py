"""Latency benchmark of the Cake SM120 MXFP4 x MXFP4 (W4A4) MegaMoE backend (``MoEEpLayer``).

The harness, columns and options of ``bench_moe_ep_sm120_w4a8_cake_mega.py`` (same shapes,
variants, return codecs, tables and the same random MXFP4 weight pack) with
``sm120_mxfp4_mxfp4_bf16_cake`` layers: BF16 inputs quantized to MXFP4 by the backend
(``quantize_input=True``).  Shapes outside the FP4-activation region (``hidden_size % 1024 != 0``,
or ``intermediate_size % 256 != 0`` with the shared expert) are skipped.  Correctness is covered
by ``tests/moe_ep/test_moe_ep_sm120_mxfp4a4_cake_mega.py``.

    torchrun --standalone --nproc_per_node=4 \\
        benchmarks/moe_ep/backends/mega/kernel/sm120/bench_moe_ep_sm120_mxfp4_mxfp4_cake_mega.py \\
        --return-codecs bf16 int8
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_moe_ep_sm120_w4a8_cake_mega as w4a8_bench  # noqa: E402


def _mxfp4_layer(
    rank,
    world,
    geometry,
    capacity,
    shared,
    table,
    limit,
    activation_fields,
    codec,
    seed,
):
    """The MXFP4 W4A4 layer of one (shape, variant, codec); ``None`` outside the region."""
    if geometry.hidden % 1024 or (shared and geometry.intermediate % 256):
        return None
    from flashinfer.moe_ep import Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_synthetic import (
        make_layer,
        random_weights,
    )

    return make_layer(
        rank,
        world,
        random_weights(rank, world, geometry, shared=shared, seed=seed),
        geometry,
        max_tokens_per_rank=capacity,
        shared=shared,
        transport_table=table,
        swiglu_limit=limit,
        activation_fields=activation_fields,
        return_codec=codec,
        config_type=Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig,
    )


if __name__ == "__main__":
    w4a8_bench.main(
        build=_mxfp4_layer, kernel="mxfp4", description=__doc__.splitlines()[0]
    )
