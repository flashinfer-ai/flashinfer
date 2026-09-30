"""Prove O_DIRECT input, final destinations, bounded alignment and ownership."""

import os
import subprocess
import sys
import textwrap
from array import array

import pytest
import torch
from safetensors.torch import save_file

from b12x.loader._checkpoint import DirectWeightSession



@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_tp_column_slices_preserve_padding_and_expand_bf16(
    tmp_path, rank, padded, dtype
):
    """TP rows must read the selected columns without overwriting row padding."""
    expected = torch.arange(2 * 128 * 256, dtype=torch.float32).to(torch.bfloat16)
    expected = expected.reshape(2, 128, 256)
    path = tmp_path / "tp.safetensors"
    save_file({"weight": expected}, path)
    with DirectWeightSession() as session:
        source = dict(session.weights([path]))["weight"][
            :, :, rank * 128 : (rank + 1) * 128
        ]
        backing = torch.full(
            (2, 128, 130 if padded else 128), -17.0, dtype=dtype, device="cuda"
        )
        target = backing[:, :, 1:-1] if padded else backing
        assert session(target, source)
        stats = session.stats()
        if not padded:
            assert stats["strided_copy_bytes"] == source.nbytes
            assert stats["reads"] < 8
        assert stats["descriptors"] == 2
    torch.testing.assert_close(
        target.cpu(),
        expected[:, :, rank * 128 : (rank + 1) * 128].to(dtype),
        rtol=0,
        atol=0,
    )
    if padded:
        assert torch.all(backing[:, :, 0] == -17)
        assert torch.all(backing[:, :, -1] == -17)


def test_transform_materialization_owns_values_after_session_close(tmp_path):
    expected = torch.arange(4096, dtype=torch.float32).reshape(32, 128)
    path = tmp_path / "transform.safetensors"
    save_file({"weight": expected}, path)
    with (

        DirectWeightSession() as session,
    ):
        source = dict(session.weights([path]))["weight"][:, 64:]
        actual = session.materialize(source)
    path.unlink()
    torch.testing.assert_close(actual.cpu(), expected[:, 64:], rtol=0, atol=0)


@pytest.mark.parametrize("offset", [0, 4103, 2**32 + 4103])
def test_ring_reads_exact_bytes_with_large_file_offsets(
    tmp_path, offset
):
    path = tmp_path / "payload"
    expected = bytes(range(256)) * 65537
    with path.open("wb") as output:
        output.seek(offset)
        output.write(expected)
    with DirectWeightSession() as session:
        backing = torch.full(
            (len(expected) + 514,), 199, device="cuda", dtype=torch.uint8
        )
        target = backing[257:-257]
        with (
            path.open("rb") as buffered,
            pytest.raises(RuntimeError, match="requires O_DIRECT"),
        ):
            session._execute(array("Q", (buffered.fileno(), offset, len(expected), target.data_ptr(), 0, 1, 0, 0)))
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            session._execute(array("Q", (fd, offset, len(expected), target.data_ptr(), 0, 1, 0, 0)))
        finally:
            os.close(fd)
        stats = session.stats()
        assert stats["scratch_bytes"] == (16 << 20) + (64 << 10)
        assert stats["bounced_bytes"] == len(expected)
    path.unlink()
    assert target.cpu().numpy().tobytes() == expected
    assert torch.all(backing[:257] == 199)
    assert torch.all(backing[-257:] == 199)


def test_descriptor_views_read_the_selected_source_into_fused_parameters(tmp_path):
    path = tmp_path / "model.safetensors"
    expected = torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64)
    save_file({"layer.weight": expected, "layer.scale": torch.tensor(2.0)}, path)
    with DirectWeightSession() as session:
        sources = dict(
            session.weights([path], needs_values=lambda entry: entry.shape == ())
        )
        assert sources["layer.weight"].device.type == "meta"
        assert sources["layer.scale"].item() == 2
        destination = torch.full((128, 64), -1.0, device="cuda")
        assert session(destination[64:], sources["layer.weight"][32:96])
        with pytest.raises(
            NotImplementedError, match="unsupported data transformation"
        ):
            session(destination, sources["layer.weight"] + 1)
        sources["layer.weight"].add_(1)
        with pytest.raises(NotImplementedError, match="explicit transform"):
            session(destination, sources["layer.weight"])
    torch.testing.assert_close(destination[64:].cpu(), expected[32:96])
    torch.testing.assert_close(destination[:64].cpu(), torch.full((64, 64), -1.0))


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e8m0fnu])
def test_mxfp4_tp_shards_preserve_packed_bytes_and_exponents(tmp_path, tp_size, dtype):
    """Signed FP4 payloads and E8M0 views must not undergo numeric FP8 casts."""
    bits = torch.arange(256, dtype=torch.uint8).repeat(64).reshape(256, 64)
    path = tmp_path / "packed.safetensors"
    save_file({"weight": bits.view(dtype)}, path)
    with DirectWeightSession() as session:
        source = dict(session.weights([path]))["weight"]
        if dtype == torch.float8_e8m0fnu:
            source = source.view(torch.uint8)
        width = bits.shape[1] // tp_size
        targets = []
        for rank in range(tp_size):
            shard = source[:, rank * width : (rank + 1) * width]
            target = torch.empty(shard.shape, dtype=torch.uint8, device="cuda")
            session(target, shard)
            targets.append(target)
        with pytest.raises(
            NotImplementedError, match="unsupported data transformation"
        ):
            session(targets[0], source[:, :width].to(torch.float32))
        session.flush()
        assert session.stats()["transform_scratch_bytes"] == 0
    assert torch.equal(torch.cat([t.cpu() for t in targets], dim=1), bits)


def test_truncated_direct_input_fails_without_buffered_retry(tmp_path):
    path = tmp_path / "short"
    path.write_bytes(bytes(4096))
    with DirectWeightSession() as session:
        target = torch.empty(8192, device="cuda", dtype=torch.uint8)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            with pytest.raises(RuntimeError, match="file range"):
                session._execute(array("Q", (fd, 0, 8192, target.data_ptr(), 0, 1, 0, 0)))
        finally:
            os.close(fd)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_dtype_conversion_reuses_bounded_gpu_scratch(tmp_path, dtype):
    path = tmp_path / "model.safetensors"
    expected = torch.arange((8 << 20) + 128, dtype=torch.float32) / 1024
    save_file({"weight": expected}, path)
    with DirectWeightSession() as session:
        source = dict(session.weights([path]))["weight"]
        target = torch.empty(source.shape, dtype=dtype, device="cuda")
        pointer = target.data_ptr()
        assert session(target, source)
        assert target.data_ptr() == pointer
        stats = session.stats()
        assert stats["transform_scratch_bytes"] == 8 << 20
        assert stats["transform_bytes"] == expected.nbytes
    torch.testing.assert_close(target.cpu(), expected.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("count", [0, 1, 48, 65536, (8 << 20) + 2])
def test_bf16_expands_in_ring_without_extra_transform_scratch(
    tmp_path, count
):
    path = tmp_path / "model.safetensors"
    # Include every BF16 bit pattern, including signed zeros, subnormals and NaNs.
    bits = torch.arange(count, dtype=torch.int32).to(torch.int16)
    expected = bits.view(torch.bfloat16)
    save_file({"A_log": expected}, path)
    with DirectWeightSession() as session:
        source = dict(session.weights([path]))["A_log"]
        backing = torch.full((count + 130,), -17.0, device="cuda")
        target = backing[65 : 65 + count]
        pointer = target.data_ptr()
        assert session(target, source)
        assert target.data_ptr() == pointer
        stats = session.stats()
        assert stats["transform_scratch_bytes"] == 0
        assert stats["transform_bytes"] == 0
        assert stats["bf16_expansion_bytes"] == expected.nbytes
    actual = backing.cpu()
    torch.testing.assert_close(actual[:65], torch.full((65,), -17.0))
    torch.testing.assert_close(actual[-65:], torch.full((65,), -17.0))
    assert torch.equal(
        actual[65 : 65 + count].view(torch.int32), bits.to(torch.int32) << 16
    )


def test_batch_reorders_disjoint_destinations_and_retains_views_until_completion(
    tmp_path,
):
    path = tmp_path / "weights.safetensors"
    expected = torch.arange(1024 * 1024, dtype=torch.float32).reshape(256, -1)
    save_file({"weight": expected}, path)
    with (

        DirectWeightSession(io_threads=4) as session,
    ):
        source = dict(session.weights([path]))["weight"]
        target = torch.full_like(expected, -1, device="cuda")
        before = session.native.direct_stats(session.reader)["reads"]
        for row in reversed(range(256)):
            session(target[row], source[row])
        assert session.native.direct_stats(session.reader)["reads"] == before
        session.flush()
        stats = session.stats()
        assert stats["batches"] == 1
        assert stats["descriptors"] == 256
        assert stats["scratch_bytes"] == (16 << 20) + (64 << 10)
        assert stats["bounced_bytes"] == expected.nbytes
    torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)


def test_batch_rejects_overlaps_before_writing_any_destination(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_file({"weight": torch.ones(128)}, path)
    with DirectWeightSession() as session:
        source = dict(session.weights([path]))["weight"]
        target = torch.zeros(128, device="cuda")
        session(target, source)
        session(target[64:], source[:64])
        with pytest.raises(RuntimeError, match="overlapping batch destinations"):
            session.flush()
    assert torch.count_nonzero(target) == 0


def test_failed_batch_drains_workers_and_allows_an_independent_next_batch(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_file({"weight": torch.arange(4096, dtype=torch.float32)}, path)
    with DirectWeightSession(io_threads=4) as session:
        source = dict(session.weights([path]))["weight"]
        target = torch.zeros(4096, device="cuda")
        session(target, source)
        with path.open("r+b") as file:
            file.truncate(512)
        with pytest.raises(RuntimeError, match="file range"):
            session.flush()
        replacement = tmp_path / "replacement.safetensors"
        expected = torch.arange(4096, dtype=torch.float32) + 10
        save_file({"weight": expected}, replacement)
        source = dict(session.weights([replacement]))["weight"]
        session(target, source)
        session.flush()
    torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)


def test_scalar_metadata_reads_are_coalesced_and_keep_independent_values(tmp_path):
    path = tmp_path / "scales.safetensors"
    values = {f"scale_{i:04d}": torch.tensor(float(i)) for i in range(2048)}
    save_file(values, path)
    with DirectWeightSession() as session:
        loaded = dict(session.weights([path], needs_values=lambda entry: True))
        assert session.stats()["reads"] < 10
    for name, expected in values.items():
        assert loaded[name].item() == expected.item()


def test_cpu_scale_descriptors_keep_sources_alive_and_write_in_one_batch():
    with DirectWeightSession(io_threads=4) as session:
        destination = torch.zeros(2048, device="cuda")
        for index in range(2048):
            assert session(destination[index], torch.tensor(float(index)))
        session.flush()
        stats = session.stats()
        assert stats["batches"] == 1
        assert stats["reads"] == 0
        assert stats["metadata_host_copy_bytes"] == 2048 * 4
    torch.testing.assert_close(
        destination.cpu(), torch.arange(2048, dtype=torch.float32)
    )


@pytest.mark.parametrize("io_threads", [1, 8, 16])
def test_bounce_ring_recycles_slots_without_changing_final_weights(tmp_path, io_threads):
    """A payload larger than both the ring and batch chunk must survive reuse."""
    path = tmp_path / "large.safetensors"
    expected = torch.arange((18 << 20) + 129, dtype=torch.int32)
    save_file({"weight": expected}, path)
    with DirectWeightSession(
        io_threads=io_threads
    ) as session:
        source = dict(session.weights([path]))["weight"]
        backing = torch.full((expected.numel() + 2,), -1, device="cuda", dtype=expected.dtype)
        target = backing[1:-1]
        session(target, source)
        session.flush()
        stats = session.stats(flush=False)
        assert stats["bounce_buffer_bytes"] == 16 << 20
        assert stats["scratch_bytes"] == (16 << 20) + (64 << 10)
        assert stats["bounced_bytes"] == expected.nbytes
        assert stats["io_uring_submits"] > 0
        assert stats["max_inflight_reads"] == io_threads
        # Flush is the publication boundary, before destroying the executor.
        torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)
    assert backing[0].item() == backing[-1].item() == -1


def test_preexisting_cuda_storage_survives_session_and_orders_loading_stream(tmp_path):
    """The loader must accept ordinary preexisting tensors and preserve their lifetime."""
    expected = torch.arange(1 << 20, dtype=torch.int32)
    path = tmp_path / "ordinary.safetensors"
    save_file({"weight": expected}, path)
    target = torch.empty(expected.shape, dtype=expected.dtype, device="cuda")
    pointer = target.data_ptr()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), DirectWeightSession() as session:
        target.fill_(-1)
        source = dict(session.weights([path]))["weight"]
        session(target, source)
        session.flush()
        output = target.clone()
    stream.synchronize()
    path.unlink()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output.copy_(target)
    output.zero_()
    graph.replay()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert target.data_ptr() == pointer


def test_expandable_weights_span_cuda_mappings_and_reject_unmapped_tail(tmp_path):
    """Large ordinary weights cross mapping boundaries; reserved VA is not storage."""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import ctypes as c
            import sys
            from array import array
            import torch
            from safetensors.torch import save_file
            from b12x.loader._checkpoint import DirectWeightSession

            expected = torch.arange(40 << 20, dtype=torch.uint8)
            path = sys.argv[1]
            save_file({"weight": expected}, path)
            target = torch.empty_like(expected, device="cuda")
            driver = c.CDLL("libcuda.so.1")
            base, size = c.c_uint64(), c.c_size_t()
            assert driver.cuMemGetAddressRange_v2(
                c.byref(base), c.byref(size), c.c_uint64(target.data_ptr())
            ) == 0
            assert target.nbytes > size.value - (target.data_ptr() - base.value)
            with DirectWeightSession(read_mode="bounce") as session:
                source = dict(session.weights([path]))["weight"]
                session(target, source)
                session.flush()
                torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)
                assert driver.cuMemGetAddressRange_v2(
                    c.byref(base), c.byref(size),
                    c.c_uint64(target.data_ptr() + target.nbytes - 1)
                ) == 0
                extent = base.value + size.value - target.data_ptr() + 1
                try:
                    session._execute(array("Q", (
                        session.files[0], 0, extent, target.data_ptr(), 0, 1, 0, 0
                    )))
                except RuntimeError as error:
                    assert "CUDA device allocation" in str(error), error
                else:
                    raise AssertionError("accepted an unmapped destination tail")
        """), str(tmp_path / "expandable.safetensors")],
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
             "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        check=True,
        timeout=60,
    )
