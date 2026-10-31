from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import b12x.attention.dsa_indexer.contiguous_kernel as contiguous_kernel
import b12x.attention.dsa_indexer.kernel as paged_kernel


def _reject_fake_data_ptr(self):
    raise AssertionError("FakeTensor has no runtime data pointer")


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "buffers", ["separate", "disjoint_views", "alias_values", "alias_indices"]
)
def test_tiled_topk_carry_alias_validation_without_fake_pointers(
    monkeypatch, fake, buffers
):
    from b12x._lib.compile_plan import compile_only_launches
    from b12x.attention.dsa_indexer import tiled_topk

    monkeypatch.setattr(FakeTensor, "data_ptr", _reject_fake_data_ptr)
    monkeypatch.setattr(tiled_topk, "current_cuda_stream", lambda: None)
    launches = []
    monkeypatch.setattr(tiled_topk, "b12x_launch", lambda *a, **k: launches.append(k))
    with FakeTensorMode() if fake else nullcontext(), compile_only_launches():
        values = torch.empty((2, 512), dtype=torch.float32)
        indices = torch.empty((2, 512), dtype=torch.int32)
        carry_values = torch.empty_like(values)
        carry_indices = torch.empty_like(indices)
        if buffers == "disjoint_views":
            values, carry_values = torch.empty((2, 2, 512)).unbind(0)
            indices, carry_indices = torch.empty((2, 2, 512), dtype=torch.int32).unbind(
                0
            )
        elif buffers == "alias_values":
            carry_values = values.view_as(values)
        elif buffers == "alias_indices":
            carry_indices = indices.view_as(indices)
        error = (
            pytest.raises(ValueError, match="must not alias")
            if buffers.startswith("alias")
            else nullcontext()
        )
        with error:
            result = tiled_topk.run_tiled_topk(
                tile_logits=torch.empty((16 * 256,)),
                k_start=None,
                lengths=torch.empty((2,), dtype=torch.int32),
                topk=512,
                block_q=16,
                block_k=256,
                zero_row_start=True,
                carry_values=carry_values,
                carry_indices=carry_indices,
                output_values=values,
                output_indices=indices,
                is_first=False,
            )
            assert result[0] is values and result[1] is indices
        assert len(launches) == (0 if buffers.startswith("alias") else 1)


@pytest.mark.parametrize("layout", ["paged", "mla"])
@pytest.mark.parametrize("offset", [0, 4, 16])
@pytest.mark.parametrize("fake", [False, True])
def test_fused_indexer_preparation_preserves_query_alignment(
    monkeypatch, layout, offset, fake
):
    from b12x._lib.compile_plan import compile_only_launches
    from b12x.attention.dsa_indexer import _preparation, fused_indexer

    monkeypatch.setattr(FakeTensor, "data_ptr", _reject_fake_data_ptr)
    monkeypatch.setattr(fused_indexer, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: type("Props", (), {"multi_processor_count": 188})(),
    )
    kernels = []
    monkeypatch.setattr(
        fused_indexer, "_launch_fused", lambda kernel, *a, **k: kernels.append(kernel)
    )
    real_q = torch.empty((2 * 2 * 128 + offset,), dtype=torch.uint8)[offset:].view(
        2, 2, 128
    )
    descriptor = dict(
        shape=tuple(real_q.shape),
        strides=tuple(real_q.stride()),
        dtype="uint8",
        alignment=_preparation._alignment(real_q),
    )
    with FakeTensorMode() if fake else nullcontext(), compile_only_launches():
        q = _preparation._fake(descriptor, torch.device("cpu")) if fake else real_q
        common = dict(q_bytes=q, weights=torch.empty((2, 2)), num_heads=2, topk=512)
        if layout == "paged":
            fused_indexer.run_fused_paged_indexer(
                **common,
                k_quant_bytes=torch.empty((4, 64, 128), dtype=torch.uint8),
                k_scales=torch.empty((4, 64)),
                real_page_table=torch.empty((2, 4), dtype=torch.int32),
                seqlens=torch.empty((2,), dtype=torch.int32),
                ctas_per_group=1,
                merge_threshold=0,
            )
        else:
            fused_indexer.run_fused_indexer_mla(
                **common,
                k_quant_bytes=torch.empty((256, 128), dtype=torch.uint8),
                k_scales=torch.empty((256,)),
                k_start=torch.empty((2,), dtype=torch.int32),
                k_end=torch.empty((2,), dtype=torch.int32),
            )
    assert len(kernels) == 1
    assert kernels[0].vectorized_q_load == (offset % 16 == 0)
    assert kernels[0].q_row_stride_bytes == (256 if offset % 16 == 0 else 0)


def _paged_tensors():
    q_fp8 = torch.empty((2, 2, 128), dtype=torch.float8_e4m3fn)
    weights = torch.empty((2, 2), dtype=torch.float32)
    index_k_cache = torch.empty((4, 64 * (128 + 4)), dtype=torch.uint8)
    real_page_table = torch.empty((2, 4), dtype=torch.int32)
    seqlens_per_query = torch.empty((2,), dtype=torch.int32)
    active_width = torch.empty((1,), dtype=torch.int32)
    tile_logits = torch.empty((32 * 512,), dtype=torch.float32)
    schedule_metadata = torch.empty((2, 2), dtype=torch.int32)
    return (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        active_width,
        tile_logits,
        schedule_metadata,
    )


def _contiguous_tensors():
    q_fp8 = torch.empty((2, 2, 128), dtype=torch.float8_e4m3fn)
    weights = torch.empty((2, 2), dtype=torch.float32)
    k_quant = torch.empty((64, 128), dtype=torch.float8_e4m3fn)
    k_scale = torch.empty((64,), dtype=torch.float32)
    k_start = torch.empty((2,), dtype=torch.int32)
    k_end = torch.empty((2,), dtype=torch.int32)
    tile_logits = torch.empty((32 * 256,), dtype=torch.float32)
    return q_fp8, weights, k_quant, k_scale, k_start, k_end, tile_logits


def test_paged_logits_kernel_binding_run_uses_binding_argument(monkeypatch) -> None:
    (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        active_width,
        _tile_logits,
        schedule_metadata,
    ) = _paged_tensors()
    binding = paged_kernel.build_indexer_paged_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        schedule_metadata=schedule_metadata,
        preinitialize_invalid_logits=False,
    )
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return "logits"

    monkeypatch.setattr(paged_kernel, "run_paged_logits_kernel", fake_run)

    assert binding.run() == "logits"
    assert calls["binding"] is binding


def test_paged_tiled_logits_kernel_binding_supplies_common_call(monkeypatch) -> None:
    (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        active_width,
        tile_logits,
        _schedule_metadata,
    ) = _paged_tensors()
    binding = paged_kernel.build_indexer_paged_tiled_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        tile_block_q=16,
        preinitialize_tile_logits=False,
    )
    calls = {}

    def fake_common(**kwargs):
        calls.update(kwargs)
        return "tile-logits"

    monkeypatch.setattr(
        paged_kernel, "_run_paged_tiled_logits_kernel_common", fake_common
    )

    assert paged_kernel.run_paged_tiled_logits_kernel(binding=binding) == "tile-logits"
    assert calls["q_fp8"] is q_fp8
    assert calls["weights"] is weights
    assert calls["index_k_cache"] is index_k_cache
    assert calls["real_page_table"] is real_page_table
    assert calls["seqlens_per_query"] is seqlens_per_query
    assert calls["active_width"] is active_width
    assert calls["tile_logits"] is tile_logits
    assert calls["tile_block_q"] == 16
    assert calls["preinitialize_tile_logits"] is False
    assert calls["source_page_offset"] == 0
    assert calls["output_width_tokens"] is None
    assert calls["supertile"] is False


def test_paged_supertile_logits_kernel_binding_supplies_common_call(
    monkeypatch,
) -> None:
    (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        active_width,
        tile_logits,
        _schedule_metadata,
    ) = _paged_tensors()
    binding = paged_kernel.build_indexer_paged_supertile_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
        active_width=active_width,
        tile_logits=tile_logits,
        source_page_offset=3,
        output_width_tokens=1024,
    )
    calls = {}

    def fake_common(**kwargs):
        calls.update(kwargs)
        return "supertile-logits"

    monkeypatch.setattr(
        paged_kernel, "_run_paged_tiled_logits_kernel_common", fake_common
    )

    assert (
        paged_kernel.run_paged_supertile_logits_kernel(binding=binding)
        == "supertile-logits"
    )
    assert calls["q_fp8"] is q_fp8
    assert calls["tile_logits"] is tile_logits
    assert calls["source_page_offset"] == 3
    assert calls["output_width_tokens"] == 1024
    assert calls["supertile"] is True


def test_paged_logits_kernel_rejects_binding_plus_runtime_tensors() -> None:
    (
        q_fp8,
        weights,
        index_k_cache,
        real_page_table,
        seqlens_per_query,
        *_rest,
    ) = _paged_tensors()
    binding = paged_kernel.build_indexer_paged_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seqlens_per_query=seqlens_per_query,
    )

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        paged_kernel.run_paged_logits_kernel(binding=binding, q_fp8=q_fp8)


def test_contiguous_logits_kernel_binding_run_uses_binding_argument(
    monkeypatch,
) -> None:
    q_fp8, weights, k_quant, k_scale, k_start, k_end, tile_logits = (
        _contiguous_tensors()
    )
    binding = contiguous_kernel.build_indexer_contiguous_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
        preinitialize_invalid_logits=False,
        tile_logits=tile_logits,
        tile_k_offset=1,
        tile_num_k_tiles=2,
    )
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return "contiguous-logits"

    monkeypatch.setattr(contiguous_kernel, "run_contiguous_logits_kernel", fake_run)

    assert binding.run() == "contiguous-logits"
    assert calls["binding"] is binding


def test_contiguous_logits_kernel_rejects_binding_plus_runtime_tensors() -> None:
    q_fp8, weights, k_quant, k_scale, k_start, k_end, _tile_logits = (
        _contiguous_tensors()
    )
    binding = contiguous_kernel.build_indexer_contiguous_logits_kernel_binding(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    )

    with pytest.raises(ValueError, match="binding owns runtime tensors"):
        contiguous_kernel.run_contiguous_logits_kernel(binding=binding, weights=weights)


def test_contiguous_logits_kernel_without_binding_reports_missing_argument() -> None:
    with pytest.raises(TypeError, match="requires q_fp8 or binding"):
        contiguous_kernel.run_contiguous_logits_kernel()
