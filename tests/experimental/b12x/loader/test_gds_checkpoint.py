"""Qualify checkpoint transfers into discrete-GPU weight storage on a GDS filesystem."""

import json
from array import array

import pytest
import torch
from safetensors.torch import save_file

from b12x.loader import capabilities
from b12x.loader._checkpoint import DirectWeightSession



@pytest.fixture
def session():
    if capabilities()["host_page_tables"]:
        pytest.skip("discrete-GPU checkpoint transport required")
    with (

        DirectWeightSession(io_threads=2, read_mode="gds") as session,

    ):
        yield session


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_tp_bytes_and_bf16_bits_preserve_destination_padding(
    tmp_path, session, expanded, padded
):
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(256, 256)
    path = tmp_path / "weight.safetensors"
    save_file({"weight": bits.view(torch.bfloat16)}, path)
    source = dict(session.weights([path]))["weight"]
    targets = []
    for rank in range(2):
        backing = torch.full(
            (256, 130 if padded else 128),
            -17,
            dtype=torch.float32 if expanded else torch.bfloat16,
            device="cuda",
        )
        target = backing[:, 1:-1] if padded else backing
        session(target, source[:, rank * 128 : (rank + 1) * 128])
        targets.append((backing, target))
    stats = session.stats()
    assert stats["gds_enabled"] == 1
    assert stats["gpu_scratch_bytes"] == 2 * ((8 << 20) + 65536)
    assert stats["transform_scratch_bytes"] == stats["bf16_expansion_bytes"] == 0
    for rank, (backing, target) in enumerate(targets):
        expected = bits[:, rank * 128 : (rank + 1) * 128]
        actual = target.cpu().view(torch.int32 if expanded else torch.int16)
        assert torch.equal(
            actual, expected.to(torch.int32) << 16 if expanded else expected
        )
        if padded:
            assert torch.all(backing[:, 0] == -17) and torch.all(backing[:, -1] == -17)


@pytest.mark.parametrize("offset", [4096, 2**32 + 4103])
def test_file_and_device_offsets_with_partial_eof_and_graph_lifetime(
    tmp_path, session, offset
):
    expected = torch.arange(256, dtype=torch.uint8).repeat(65537)
    header = (
        json.dumps(
            {
                "padding": {
                    "dtype": "U8",
                    "shape": [offset - 4096],
                    "data_offsets": [0, offset - 4096],
                },
                "weight": {
                    "dtype": "U8",
                    "shape": [expected.numel()],
                    "data_offsets": [offset - 4096, offset - 4096 + expected.numel()],
                },
            }
        )
        .encode()
        .ljust(4088, b" ")
    )
    path = tmp_path / "offset.safetensors"
    with path.open("wb") as file:
        file.write(len(header).to_bytes(8, "little") + header)
        file.seek(offset)
        file.write(expected.numpy().tobytes())
    source = dict(session.weights([path], prefixes=("weight",)))["weight"]
    prefix = (1 << 32) + 65537 if offset > 2**32 else 65536
    backing = torch.empty(
        prefix + expected.numel() + 1, device="cuda", dtype=torch.uint8
    )
    target = backing[prefix:-1]
    backing[prefix - 1 : prefix].fill_(199)
    backing[-1:].fill_(199)
    session(target, source)
    stats = session.stats()
    if offset == 4096:
        assert stats["destination_bytes"] >= 16 << 20
    assert torch.equal(target.cpu(), expected)
    assert backing[prefix - 1].item() == backing[-1].item() == 199
    pointer = target.data_ptr()
    session._close()
    path.unlink()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output = target[:65536].clone()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output.copy_(target[:65536])
    for _ in range(3):
        output.zero_()
        graph.replay()
        assert torch.equal(output.cpu(), expected[:65536])
    assert target.data_ptr() == pointer


def test_contiguous_cast_reuses_gpu_scratch_and_explicit_metadata(tmp_path, session):
    expected = torch.arange((8 << 20) + 128, dtype=torch.float32) / 1024
    path = tmp_path / "cast.safetensors"
    save_file({"weight": expected, "scale": torch.tensor(3.25)}, path)
    source = dict(
        session.weights([path], needs_values=lambda entry: entry.name == "scale")
    )
    target = torch.empty_like(expected, dtype=torch.bfloat16, device="cuda")
    scalar = torch.empty((), device="cuda")
    session(target, source["weight"])
    session(scalar, source["scale"])
    stats = session.stats()
    assert stats["transform_scratch_bytes"] == 8 << 20
    assert stats["metadata_h2d_bytes"] == 4
    assert stats["torch_copy_bytes"] == 0
    torch.testing.assert_close(target.cpu(), expected.bfloat16(), rtol=0, atol=0)
    assert scalar.item() == 3.25


def test_invalid_destinations_and_truncation_fail_before_writes(tmp_path, session):
    path = tmp_path / "weight.safetensors"
    save_file({"weight": torch.ones(128)}, path)
    source = dict(session.weights([path]))["weight"]
    target = torch.zeros(128, device="cuda")
    session(target, source)
    session(target[64:], source[:64])
    with pytest.raises(RuntimeError, match="overlapping batch destinations"):
        session.flush()
    assert torch.count_nonzero(target) == 0
    entry = session.sources[source.untyped_storage()._cdata][1]
    with pytest.raises(RuntimeError, match="CUDA device allocation"):
        session._execute(array("Q", (entry.fd, entry.offset, 4, 1, 0, 1, 0, 0)))
    session(target, source)
    with path.open("r+b") as file:
        file.truncate(128)
    with pytest.raises(RuntimeError, match="file size changed"):
        session.flush()
    assert torch.count_nonzero(target) == 0


def test_cufile_reads_use_gpu_transport_without_posix_compatibility(tmp_path, session):
    if not hasattr(session._gds, "synchronous_transport_stats"):
        pytest.skip("cuFile statistics require development headers >= 1.15")
    expected = torch.arange(256, dtype=torch.uint8).repeat(65537)
    path = tmp_path / "transport.safetensors"
    save_file({"weight": expected}, path)
    source = dict(session.weights([path]))["weight"]
    target = torch.empty_like(expected, device="cuda")
    session._execute(array("Q"))
    session._gds.start_stats(3)
    before = session._gds.synchronous_transport_stats()
    session(target, source)
    session.flush()
    after = session._gds.synchronous_transport_stats()
    assert (
        after["nvfs_reads"] + after["p2p_reads"]
        > before["nvfs_reads"] + before["p2p_reads"]
    )
    assert after["posix_reads"] == before["posix_reads"]
    assert after["read_errors"] == before["read_errors"]
    assert torch.equal(target.cpu(), expected)
