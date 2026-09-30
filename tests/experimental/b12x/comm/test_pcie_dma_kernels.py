"""CPU-only dispatch tests for the CuTe PCIe-DMA kernel facade."""

from __future__ import annotations

import pytest

from b12x.comm.pcie import _dma_kernels as dma_kernels
from b12x.comm.pcie import pcie_dma


@pytest.mark.parametrize("world_size", [1, 3, 5, 7, 9, 11, 12])
def test_dma_rejects_unreviewed_world_sizes_before_backend_setup(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
) -> None:
    monkeypatch.setattr(pcie_dma.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        pcie_dma.dist, "get_world_size", lambda group: world_size
    )

    def unexpected_backend_setup():
        raise AssertionError("backend setup must not run for an unreviewed world size")

    monkeypatch.setattr(pcie_dma, "_load_kernels", unexpected_backend_setup)

    with pytest.raises(ValueError, match="reviewed world sizes"):
        pcie_dma.PCIeDmaAllReduce(
            exchange_group=object(),
            device="cuda:0",
            max_bytes=4096,
        )


def test_dma_reviewed_world_sizes_match_tp10_accumulator_limit() -> None:
    assert pcie_dma.SUPPORTED_WORLD_SIZES == (2, 4, 6, 8, 10)
    assert max(pcie_dma.SUPPORTED_WORLD_SIZES) - 1 == 9


def test_dequant_accum_nine_sources_uses_a_separate_abi(monkeypatch) -> None:
    compiled: list[tuple[str, tuple[object, ...], object, tuple[object, ...]]] = []

    def fake_compile(kernel_id, key, launch, *args):
        compiled.append((kernel_id, key, launch, args))
        return object()

    monkeypatch.setattr(dma_kernels, "_compile", fake_compile)
    dma_kernels._compiled_dequant_accum.cache_clear()
    try:
        dma_kernels._compiled_dequant_accum("e4m3", 8)
        dma_kernels._compiled_dequant_accum("e4m3", 9)
    finally:
        dma_kernels._compiled_dequant_accum.cache_clear()

    legacy_id, legacy_key, legacy_launch, legacy_args = compiled[0]
    tp10_id, tp10_key, tp10_launch, tp10_args = compiled[1]
    assert legacy_id == tp10_id == "comm.pcie_dma.dequant_accum"
    assert legacy_key == ("e4m3", 8)
    assert tp10_key == ("e4m3", 9)
    assert type(legacy_launch) is dma_kernels._DequantAccumLaunch
    assert type(tp10_launch) is dma_kernels._DequantAccum9Launch
    # out + input + N payloads + N scales + blocks + grid_x
    assert len(legacy_args) == 20
    assert len(tp10_args) == 22


def test_dequant_accum_dispatch_preserves_legacy_padding_and_accepts_nine(
    monkeypatch,
) -> None:
    launches: list[tuple[str, int, tuple[object, ...]]] = []

    def fake_legacy(*args):
        launches.append(("e4m3", 7, args))

    def fake_tp10(*args):
        launches.append(("mx", 9, args))

    monkeypatch.setattr(
        dma_kernels, "_ptr", lambda _dtype, address, _align: int(address)
    )
    monkeypatch.setattr(dma_kernels, "current_cuda_stream", lambda: 99)
    legacy = dma_kernels.DmaKernels(ipc=None)
    legacy.install(
        dma_kernels.DmaLaunchers(
            set_flag=object(), wait_flag=object(), add=(), dequant_accum=fake_legacy
        )
    )
    tp10 = dma_kernels.DmaKernels(ipc=None)
    tp10.install(
        dma_kernels.DmaLaunchers(
            set_flag=object(), wait_flag=object(), add=(), dequant_accum=fake_tp10
        )
    )
    legacy._dequant_accum(
        "e4m3",
        1,
        2,
        list(range(100, 107)),
        list(range(200, 207)),
        256,
    )
    tp10._dequant_accum(
        "mx",
        3,
        4,
        list(range(300, 309)),
        list(range(400, 409)),
        256,
    )

    legacy_codec, legacy_nsrc, legacy_args = launches[0]
    assert (legacy_codec, legacy_nsrc) == ("e4m3", 7)
    assert legacy_args[2:10] == (*range(100, 107), 100)
    assert legacy_args[10:18] == (*range(200, 207), 200)
    assert legacy_args[18:] == (2, 1, 99)

    tp10_codec, tp10_nsrc, tp10_args = launches[1]
    assert (tp10_codec, tp10_nsrc) == ("mx", 9)
    assert tp10_args[2:11] == tuple(range(300, 309))
    assert tp10_args[11:20] == tuple(range(400, 409))
    assert tp10_args[20:] == (2, 1, 99)


@pytest.mark.parametrize("nsrc", [0, 10])
def test_dequant_accum_rejects_source_counts_outside_tp10(nsrc: int) -> None:
    payloads = list(range(100, 100 + nsrc))
    scales = list(range(200, 200 + nsrc))
    with pytest.raises(ValueError, match="requires 1..9 sources"):
        dma_kernels.DmaKernels(ipc=None)._dequant_accum(
            "e4m3", 1, 2, payloads, scales, 128
        )


def test_dequant_accum_rejects_mismatched_pointer_counts() -> None:
    with pytest.raises(ValueError, match="source counts must match"):
        dma_kernels.DmaKernels(ipc=None)._dequant_accum(
            "e4m3", 1, 2, [100] * 9, [200] * 8, 128
        )
