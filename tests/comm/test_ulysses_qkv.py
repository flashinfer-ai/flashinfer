"""Real collectives and lifecycle checks for the prepared Sage2 QKV path."""

import contextlib
from dataclasses import replace
import importlib
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator, UlyssesQKV, UlyssesQKVWorkspace
import flashinfer.comm._ulysses_lowp as lowp
from flashinfer.comm.ulysses import missing_ulysses_pcie_dependencies
from flashinfer.comm.ulysses_topology import UlyssesBackendError
from tests.comm.test_ulysses_communicator import (
    _make_mock_pcie_comm,
    _ref_gather_heads,
    _run_multi_rank,
)
from tests.comm.test_ulysses_lowp_boundary import inputs


def _require_gpus(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    if any(
        torch.cuda.get_device_capability(i) not in ((9, 0), (12, 0))
        for i in range(world_size)
    ):
        pytest.skip("requires SM90 or SM120")
    if len({torch.cuda.get_device_capability(i) for i in range(world_size)}) != 1:
        pytest.skip("requires one homogeneous Ulysses group")


def _shards(rank, world, batch, length, dim, dtype, used, *, contiguous=False):
    tensors = inputs(batch, world * length, world, dtype, used, head_dim=dim)
    shard = tuple(x[:, rank * length : (rank + 1) * length] for x in tensors)
    return tuple(t.contiguous() for t in shard) if contiguous else shard


def _payload_bytes(shape, world):
    cls = (
        lowp.UlyssesLowpSageLayoutSM90
        if torch.cuda.get_device_capability() == (9, 0)
        else lowp.UlyssesLowpSageLayout
    )
    batch, length, heads, dim = shape
    return int(
        cls(head_dim=dim).payload_spec(
            batch_size=batch,
            local_sequence=length,
            num_heads=heads,
            world_size=world,
        )["payload_bytes"]
    )


def _standalone(shard, rank, world, group, used):
    """Retain the pre-communicator execution chain as the byte-level oracle."""
    cls = (
        lowp.UlyssesLowpSageLayoutSM90
        if torch.cuda.get_device_capability() == (9, 0)
        else lowp.UlyssesLowpSageLayout
    )
    layout = cls(head_dim=shard[0].shape[-1])
    send, ctx = layout.local_stats(
        *shard, rank=rank, world_size=world, used_sequence=used, enable_pdl=False
    )
    gathered = torch.empty(world * send.numel(), device=send.device, dtype=send.dtype)
    dist.all_gather_into_tensor(gathered, send, group=group)
    stats = layout.finalize_stats(gathered, ctx, shard[1], enable_pdl=False)
    payload = layout.quant_and_pack(*shard, stats, enable_pdl=False)
    recv = torch.empty_like(payload)
    dist.all_to_all_single(recv, payload, group=group)
    batch, length, heads, _ = shard[0].shape
    result = layout.unpack_for_sage(
        recv,
        batch_size=batch,
        local_sequence=length,
        local_heads=heads // world,
        world_size=world,
        scale_sequence=used,
        enable_pdl=False,
    )
    h = heads // world
    v_scale = stats.v_scale_global[:, rank * h : (rank + 1) * h].contiguous()
    return (*result, v_scale), payload, recv


def _assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def _assert_result(actual, expected, *, used, total, dtype):
    assert isinstance(actual, UlyssesQKV)
    for got, want in zip(actual[:6], expected, strict=True):
        assert got.is_contiguous()
        _assert_bytes(got, want)
    assert actual.used_sequence == used
    assert actual.logical_sequence == total
    assert actual.input_dtype == dtype
    assert actual.layout == (
        "sage2_sm90"
        if torch.cuda.get_device_capability() == (9, 0)
        else "sage2_sm89_sm120"
    )


def _correctness_body(rank, world, group, dim):
    # Includes nonaligned shards, a group spanning several ranks, and whole
    # padding-only ranks; both contiguous tensors and fused projection views.
    cases = (
        (1, 129, torch.bfloat16, world * 129 - 1, False, False),
        (2, 65, torch.float16, min(world * 65, 129), False, False),
        (2, 17, torch.bfloat16, 1, False, True),
        (1, 64, torch.float16, world * 64, True, False),
    )
    for batch, length, dtype, used, contiguous, zero_v in cases:
        shard = _shards(
            rank, world, batch, length, dim, dtype, used, contiguous=contiguous
        )
        if zero_v:
            shard[2].zero_()
        expected, payload, recv = _standalone(shard, rank, world, group, used)
        with UlyssesCommunicator(
            group,
            max_bytes=_payload_bytes(shard[0].shape, world),
            dtype=dtype,
            backend="nccl",
        ) as comm:
            workspace = comm.prepare_qkv(shard[0].shape, used_sequence=used)
            assert workspace.transport == "nccl"
            assert comm.backend == "nccl"
            exchange = comm.exchange_chunks
            calls = []

            def recording_exchange(x, *, out=None, dtype=None):
                assert x.shape == (1, 1, world, payload.shape[-1])
                assert x.data_ptr() == workspace._send_buffer.data_ptr()
                assert out.data_ptr() == workspace._recv_buffer.data_ptr()
                assert dtype == torch.uint8
                calls.append(x.shape)
                return exchange(x, out=out, dtype=dtype)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(comm, "exchange_chunks", recording_exchange)
                actual = comm.scatter_qkv(*shard, workspace=workspace)
            assert len(calls) == 1
            _assert_result(
                actual, expected, used=used, total=world * length, dtype=dtype
            )
            _assert_bytes(workspace._send_buffer, payload)
            _assert_bytes(workspace._recv_buffer, recv)
            # The budget covers the joint wire payload, including scales/padding.
            assert workspace._send_buffer.nbytes == comm.max_bytes
            assert comm.max_bytes > shard[0].nbytes
            reused = comm.scatter_qkv(*shard, workspace=workspace, out=actual)
            assert reused is actual
            _assert_result(
                reused, expected, used=used, total=world * length, dtype=dtype
            )
        # Final tensors remain usable after the owner and its staging close.
        _assert_result(actual, expected, used=used, total=world * length, dtype=dtype)
    return "ok", f"D{dim}, four input cases"


@pytest.mark.parametrize(
    "world_size,head_dim",
    [(p, d) for p in (2, 4, 8) for d in (64, 128)],
    ids=[f"p{p}-d{d}" for p in (2, 4, 8) for d in (64, 128)],
)
def test_qkv_matches_standalone(world_size, head_dim):
    _require_gpus(world_size)
    _run_multi_rank(_correctness_body, world_size, head_dim, timeout=600)


def _payload_capacity_body(rank, world, group, _arg):
    shard = _shards(rank, world, 1, 65, 64, torch.bfloat16, 129)
    payload_bytes = _payload_bytes(shard[0].shape, world)
    assert payload_bytes > shard[0].nbytes
    with UlyssesCommunicator(
        group, max_bytes=shard[0].nbytes, dtype=shard[0].dtype, backend="nccl"
    ) as comm:
        with pytest.raises((ValueError, RuntimeError), match="max_bytes|capacity"):
            comm.prepare_qkv(shard[0].shape, used_sequence=129)
        # Rejecting the Lowp wire size does not invalidate ordinary communication.
        assert comm.scatter_heads(shard[0].contiguous()).numel() == shard[0].numel()
    with UlyssesCommunicator(
        group, max_bytes=payload_bytes, dtype=shard[0].dtype, backend="nccl"
    ) as comm:
        workspace = comm.prepare_qkv(shard[0].shape, used_sequence=129)
        assert workspace._send_buffer.nbytes == payload_bytes
        expected, _, _ = _standalone(shard, rank, world, group, 129)
        actual = comm.scatter_qkv(*shard, workspace=workspace)
        _assert_result(
            actual, expected, used=129, total=world * 65, dtype=shard[0].dtype
        )
    return "ok", "joint QKV wire capacity includes scales and alignment"


def test_qkv_capacity_covers_joint_payload():
    _require_gpus(2)
    _run_multi_rank(_payload_capacity_body, 2, None, timeout=600)


def _ownership_body(rank, world, group, _arg):
    shape = (2, 128, world, 64)
    dtype = torch.bfloat16
    with UlyssesCommunicator(
        group, max_bytes=_payload_bytes(shape, world), dtype=dtype, backend="nccl"
    ) as comm:
        workspace = comm.prepare_qkv(shape, used_sequence=129)
        shard = _shards(rank, world, 2, 128, 64, dtype, 129)
        first = comm.scatter_qkv(*shard, workspace=workspace)
        saved = tuple(t.clone() for t in first[:6])
        changed = tuple(t * 2 for t in shard)
        second = comm.scatter_qkv(*changed, workspace=workspace)
        _assert_result(first, saved, used=129, total=world * 128, dtype=dtype)
        assert all(
            a.data_ptr() != b.data_ptr()
            for a, b in zip(first[:6], second[:6], strict=True)
        )
        assert not torch.equal(first.v_scale, second.v_scale)
        assert comm.scatter_qkv(*changed, workspace=workspace, out=first) is first
        _assert_result(first, second[:6], used=129, total=world * 128, dtype=dtype)

        other_length = comm.prepare_qkv(shape, used_sequence=130)
        shard130 = _shards(rank, world, 2, 128, 64, dtype, 130)
        for workspace_arg, output in (
            (other_length, first),
            (workspace, first._replace(input_dtype=torch.float16)),
            (workspace, first._replace(logical_sequence=world * 128 + 1)),
            (workspace, first._replace(layout="wrong-layout")),
            (workspace, first._replace(k=first.q)),
        ):
            with pytest.raises((ValueError, TypeError)):
                comm.scatter_qkv(*shard130, workspace=workspace_arg, out=output)

        # Outputs are contiguous; an input can still be a projection view.
        alias = workspace._send_buffer.reshape(-1)[: first.q.numel()]
        alias = alias.view(torch.int8).view_as(first.q)
        with pytest.raises((ValueError, TypeError)):
            comm.scatter_qkv(*shard, workspace=workspace, out=first._replace(q=alias))
        q_dense = shard[0].contiguous()
        input_alias = q_dense.view(torch.uint8).reshape(-1)[: first.q.numel()]
        input_alias = input_alias.view(torch.int8).view_as(first.q)
        with pytest.raises((ValueError, TypeError)):
            comm.scatter_qkv(
                q_dense,
                *shard[1:],
                workspace=workspace,
                out=first._replace(q=input_alias),
            )
        misaligned = torch.empty(
            q_dense.numel() + 1, dtype=dtype, device=q_dense.device
        )[1:].view_as(q_dense)
        with pytest.raises((ValueError, TypeError)):
            comm.scatter_qkv(misaligned, *shard[1:], workspace=workspace)
        with (
            UlyssesCommunicator(
                group,
                max_bytes=_payload_bytes(shard[0].shape, world),
                dtype=dtype,
                backend="nccl",
            ) as other,
            pytest.raises((ValueError, TypeError)),
        ):
            other.scatter_qkv(*shard, workspace=workspace)
        with pytest.raises((ValueError, TypeError)):
            comm.scatter_qkv(*shard, workspace=comm.create_workspace())
        # Hot validation failures must not consume or invalidate the workspace.
        comm.scatter_qkv(*shard, workspace=workspace, out=first)
    with pytest.raises(RuntimeError):
        comm.scatter_qkv(*shard, workspace=workspace)
    return "ok", "output ownership, metadata, aliases and closed owner"


def test_qkv_output_ownership_and_validation():
    _require_gpus(2)
    _run_multi_rank(_ownership_body, 2, None, timeout=600)


def _mutated_tensor_body(rank, world, group, _arg):
    shard = _shards(rank, world, 1, 65, 64, torch.bfloat16, 129)
    with UlyssesCommunicator(
        group,
        max_bytes=_payload_bytes(shard[0].shape, world),
        dtype=shard[0].dtype,
        backend="nccl",
    ) as comm:
        workspace = comm.prepare_qkv(shard[0].shape, used_sequence=129)
        output = comm.scatter_qkv(*shard, workspace=workspace)
        expected = tuple(t.clone() for t in output[:6])
        misaligned = torch.empty(
            shard[0].numel() + 1, dtype=shard[0].dtype, device=shard[0].device
        )[1:].view_as(shard[0])

        def reached_collective(*_args, **_kwargs):
            raise AssertionError("invalid reused tensors reached data communication")

        # Preserve Python tensor identities. A metadata/pointer cache populated
        # by the successful call above must not hide later resize_/set_ changes.
        mutations = (
            (shard[0], lambda t: t.resize_(1, 64, world, 64)),
            (shard[0], lambda t: t.set_(misaligned)),
            (output.q_scale, lambda t: t.resize_(t.numel() - 1)),
            (output.k, lambda t: t.set_(output.q)),
            (workspace._recv_buffer, lambda t: t.set_(workspace._send_buffer)),
            (workspace._stats_gather, lambda t: t.resize_(t.numel() - 1)),
        )
        for tensor, mutate in mutations:
            original = tensor.detach()
            try:
                mutate(tensor)
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(dist, "all_gather_into_tensor", reached_collective)
                    patch.setattr(dist, "all_to_all_single", reached_collective)
                    with pytest.raises((TypeError, ValueError)):
                        comm.scatter_qkv(*shard, workspace=workspace, out=output)
            finally:
                tensor.set_(original)
            assert comm.scatter_qkv(*shard, workspace=workspace, out=output) is output
            _assert_result(
                output, expected, used=129, total=world * 65, dtype=shard[0].dtype
            )
    return "ok", "tensor mutations revalidated before communication"


def test_qkv_revalidates_mutated_tensors_before_communication():
    _require_gpus(2)
    _run_multi_rank(_mutated_tensor_body, 2, None, timeout=600)


def _public_validation_body(rank, world, group, _arg):
    shard = _shards(rank, world, 1, 65, 64, torch.bfloat16, 129)
    cls = (
        lowp.UlyssesLowpSageLayoutSM90
        if torch.cuda.get_device_capability() == (9, 0)
        else lowp.UlyssesLowpSageLayout
    )
    layout = cls(head_dim=64)
    with pytest.raises(ValueError, match="identical"):
        layout.local_stats(
            shard[0], shard[1][:, :-1], shard[2], rank=rank, world_size=world
        )
    send, context = layout.local_stats(
        *shard, rank=rank, world_size=world, used_sequence=129
    )
    gathered = torch.empty(world * send.numel(), dtype=send.dtype, device=send.device)
    dist.all_gather_into_tensor(gathered, send, group=group)
    with pytest.raises(TypeError, match="fp32"):
        layout.finalize_stats(gathered.to(torch.float16), context, shard[1])
    stats = layout.finalize_stats(gathered, context, shard[1])
    wrong_scale = replace(stats, v_scale_global=stats.v_scale_global[..., :-1])
    with pytest.raises(ValueError, match="v_scale_global"):
        layout.quant_and_pack(*shard, wrong_scale)
    payload = layout.quant_and_pack(*shard, stats)
    received = torch.empty_like(payload)
    dist.all_to_all_single(received, payload, group=group)
    unpack_args = dict(
        batch_size=1,
        local_sequence=65,
        local_heads=1,
        world_size=world,
        scale_sequence=129,
    )
    result = layout.unpack_for_sage(received, **unpack_args)
    bad_out = (*result[:3], result[3].to(torch.float16), result[4])
    with pytest.raises(ValueError, match="q_scale"):
        layout.unpack_for_sage(received, out=bad_out, **unpack_args)
    return "ok", "standalone public operations retain argument validation"


def test_qkv_standalone_primitives_keep_public_validation():
    _require_gpus(2)
    _run_multi_rank(_public_validation_body, 2, None, timeout=600)


def _prepare_failure_body(rank, world, group, _arg):
    shape = (1, 128, world, 64)
    with UlyssesCommunicator(
        group,
        max_bytes=_payload_bytes((2, 128, world, 128), world),
        dtype=torch.bfloat16,
        backend="nccl",
    ) as comm:
        workspace = comm.prepare_qkv(shape, used_sequence=129)
        shard = _shards(rank, world, 1, 128, 64, torch.bfloat16, 129)
        expected = comm.scatter_qkv(*shard, workspace=workspace)
        for bad in (
            {"qkv_shape": None},
            {"qkv_shape": (1, 128, world, 128)},
            {"used_sequence": 130},
            {"quantization": "unsupported"},
            {"qkv_shape": (1, 0, world, 64)},
        ):
            kwargs = dict(qkv_shape=shape, used_sequence=129)
            if rank == 0:
                kwargs.update(bad)
            with pytest.raises((ValueError, RuntimeError)):
                comm.prepare_qkv(**kwargs)
            got = comm.scatter_qkv(*shard, workspace=workspace)
            _assert_result(
                got, expected[:6], used=129, total=world * 128, dtype=shard[0].dtype
            )

        def fail(*_args, **_kwargs):
            raise RuntimeError("injected QKV prepare failure")

        loader = (
            "get_ulysses_lowp_sm90_module"
            if torch.cuda.get_device_capability() == (9, 0)
            else "get_ulysses_lowp_module"
        )
        for target, name in ((lowp, loader), (UlyssesQKVWorkspace, "_allocate")):
            with pytest.MonkeyPatch.context() as patch:
                if rank == 0:
                    patch.setattr(target, name, fail)
                with pytest.raises((ValueError, RuntimeError), match="prepare_qkv"):
                    comm.prepare_qkv(shape, used_sequence=130)
            got = comm.scatter_qkv(*shard, workspace=workspace)
            _assert_result(
                got, expected[:6], used=129, total=world * 128, dtype=shard[0].dtype
            )
        recovered = comm.prepare_qkv(shape, used_sequence=129)
        comm.scatter_qkv(*shard, workspace=recovered)
    return "ok", "collective prepare errors and rollback"


def test_qkv_prepare_failures_are_collective_and_recoverable():
    _require_gpus(2)
    _run_multi_rank(_prepare_failure_body, 2, None, timeout=600)


def _mixed_backend_body(rank, world, group, _arg):
    shard = _shards(rank, world, 1, 65, 64, torch.bfloat16, 129)
    with UlyssesCommunicator(
        group,
        max_bytes=_payload_bytes(shard[0].shape, world),
        dtype=shard[0].dtype,
        backend="nvlink",
    ) as comm:
        workspace = comm.prepare_qkv(shard[0].shape, used_sequence=129)
        assert comm.backend == "nvlink" and workspace.transport == "nccl"
        encoded = comm.scatter_qkv(*shard, workspace=workspace)
        output = encoded.q.to(encoded.input_dtype)
        expected = _ref_gather_heads(output, world, rank, group)
        actual = comm.gather_heads(output)
        _assert_bytes(actual, expected)
    return "ok", "NCCL input followed by NVLink gather"


def test_qkv_nccl_input_nvlink_output():
    _require_gpus(2)
    _run_multi_rank(_mixed_backend_body, 2, None, timeout=600, allow_skip=True)


def _single_rank_body(rank, world, group, _arg):
    x = torch.randn(1, 8, 2, 64, dtype=torch.bfloat16, device="cuda")
    with UlyssesCommunicator(
        group, max_bytes=x.nbytes, dtype=x.dtype, backend="nccl"
    ) as comm:
        with pytest.raises((ValueError, RuntimeError)):
            comm.prepare_qkv(x.shape)
        assert comm.scatter_heads(x) is x
        assert comm.gather_heads(x) is x
    return "ok", "P1 rejects quantization and retains plain identity"


def test_qkv_rejects_single_rank_without_changing_plain_identity():
    _require_gpus(1)
    _run_multi_rank(_single_rank_body, 1, None)


def _non_nccl_group_body(rank, world, group, backend):
    metadata_group = dist.new_group(backend="gloo")
    try:
        x = torch.full((1, 4, world, 64), rank + 1, dtype=torch.bfloat16, device="cuda")
        with pytest.raises((ValueError, RuntimeError)) as rejected:
            UlyssesCommunicator(
                metadata_group,
                max_bytes=x.nbytes,
                dtype=x.dtype,
                backend=backend,
            )
        if isinstance(rejected.value, UlyssesBackendError):
            # Without NVLink the forced topology check rejects first; the
            # runner reports that as a skip.
            raise rejected.value
        assert "nccl" in str(rejected.value).lower(), rejected.value
    finally:
        dist.destroy_process_group(metadata_group)
    return "ok", f"Gloo-only groups are rejected before {backend} resources are armed"


@pytest.mark.parametrize("backend", ["nccl", "nvlink"])
def test_qkv_rejects_group_without_nccl_at_construction(backend):
    _require_gpus(2)
    _run_multi_rank(
        _non_nccl_group_body,
        2,
        backend,
        timeout=600,
        allow_skip=backend == "nvlink",
    )


def test_qkv_workspace_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        UlyssesQKVWorkspace()


# ---- experimental PCIe backend ---------------------------------------------------


def _mock_pcie_qkv_comm(monkeypatch, transport, *, max_bytes=1 << 20):
    """Two-rank host-only PCIe communicator with faked native transport and JIT.

    Every rank-local outcome is mirrored by the fake peer, so a call either
    succeeds or fails on both ranks, as under the SPMD contract.
    """
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm = _make_mock_pcie_comm()
    comm.transport = transport
    comm.dtype = torch.bfloat16
    comm.max_bytes = max_bytes
    comm._qkv_pcie_slot = None
    comm._group_supports_cuda_alltoall = lambda: (True, "nccl")
    native = SimpleNamespace(
        calls=[],
        storage=torch.zeros(max_bytes, dtype=torch.uint8),
        landing=torch.zeros(max_bytes, dtype=torch.uint8),
    )

    def allocate_output(_handle, x, mode, capacity):
        native.calls.append(("allocate", tuple(x.shape), mode, capacity))
        return native.storage, [1, 2, 3]

    def input_landing(_handle, out):
        assert transport != "p2p", "the all-P2P route has no landing buffer"
        assert out.data_ptr() == native.storage.data_ptr()
        return native.landing

    native.module = SimpleNamespace(
        allocate_output=allocate_output,
        connect_output=lambda *_args: None,
        input_landing=input_landing,
        exchange=lambda *_args: pytest.fail("native exchange must not run"),
        teardown_safe=lambda *_args: True,
        disconnect_output_ptr=lambda *_args: None,
        dispose_output_ptr=lambda *_args: None,
        dispose=lambda *_args: None,
    )
    monkeypatch.setattr(ulysses_mod, "get_ulysses_pcie_module", lambda: native.module)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: "caller")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(
        torch.cuda.nvtx, "range", lambda _name: contextlib.nullcontext()
    )
    native.stream_waits = []
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda device=None: SimpleNamespace(wait_stream=native.stream_waits.append),
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: contextlib.nullcontext())
    monkeypatch.setattr(
        lowp,
        "capability",
        lambda _device: {
            "device_capability": (12, 0),
            "supported": True,
            "supported_head_dims": (64, 128),
            "layout_class": "UlyssesLowpSageLayout",
        },
    )
    monkeypatch.setattr(lowp, "get_ulysses_lowp_module", lambda: "sage2-module")
    monkeypatch.setattr(lowp, "device_support_pdl", lambda _device: False)
    return comm, native


def _registrations(native):
    return [call for call in native.calls if call[0] == "allocate"]


@pytest.mark.parametrize("transport", ["hybrid", "rdma", "p2p"])
def test_qkv_pcie_prepares_every_geometry_on_one_registered_slot(
    monkeypatch, transport
):
    """A registration holds max_bytes until close, so each preparation must
    not add one: shapes and live lengths change between calls."""
    comm, native = _mock_pcie_qkv_comm(monkeypatch, transport)
    try:
        first = comm.prepare_qkv((1, 64, 2, 64))
        second = comm.prepare_qkv((2, 33, 4, 128), used_sequence=40)
        assert _registrations(native) == [("allocate", (1, 1, 2, 1), 2, comm.max_bytes)]
        assert list(comm._pcie_outputs) == [native.storage.data_ptr()]
        assert first._wire_shape != second._wire_shape
        for workspace in (first, second):
            assert workspace.transport == "pcie"
            assert workspace._recv_buffer.data_ptr() == native.storage.data_ptr()
            assert tuple(workspace._send_buffer.shape) == workspace._wire_shape
            assert tuple(workspace._recv_buffer.shape) == workspace._wire_shape
            # The RDMA routes pack where the NIC reads; all-P2P reads in place.
            landed = workspace._send_buffer.data_ptr() == native.landing.data_ptr()
            assert landed == (transport != "p2p")
        # Each RDMA workspace orders its vote stream once after the flag's
        # zero fill on the caller's stream; all-P2P has no vote.
        assert native.stream_waits == (["caller"] * 2 if transport != "p2p" else [])

        comm.close()
        assert comm._qkv_pcie_slot is None
        assert comm._pcie_outputs == {}
    finally:
        comm._state = "closed"


def test_qkv_pcie_rejections_before_registration_register_nothing(monkeypatch):
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm, native = _mock_pcie_qkv_comm(monkeypatch, "hybrid", max_bytes=4096)
    try:
        # The payload is checked against max_bytes before the slot exists.
        with pytest.raises(RuntimeError, match="prepare_qkv failed.*max_bytes"):
            comm.prepare_qkv((1, 64, 2, 64))
        # Statistics stay on NCCL, so a PCIe group must still provide it.
        comm._group_supports_cuda_alltoall = lambda: (False, "gloo")
        with pytest.raises(ValueError, match="CUDA NCCL process group"):
            comm.prepare_qkv((1, 8, 2, 64))
        assert native.calls == []
        assert comm._qkv_pcie_slot is None
        assert comm._state == ulysses_mod._OPEN
    finally:
        comm._state = "closed"


def test_qkv_pcie_allocation_failure_keeps_the_registered_slot(monkeypatch):
    """Allocation after registration stays recoverable; the slot is reused."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm, native = _mock_pcie_qkv_comm(monkeypatch, "rdma")

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected QKV allocation failure")

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(lowp, "get_ulysses_lowp_module", fail)
            with pytest.raises(RuntimeError, match="prepare_qkv failed"):
                comm.prepare_qkv((1, 64, 2, 64))
        assert comm._state == ulysses_mod._OPEN
        assert comm._qkv_pcie_slot is not None
        workspace = comm.prepare_qkv((1, 64, 2, 64))
        assert len(_registrations(native)) == 1
        assert workspace._recv_buffer.data_ptr() == native.storage.data_ptr()
    finally:
        comm._state = "closed"


def test_qkv_pcie_registration_failure_is_fail_stop(monkeypatch):
    """The slot is an ordinary PCIe registration and keeps its contract."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm, native = _mock_pcie_qkv_comm(monkeypatch, "hybrid")

    def fail(*_args):
        raise RuntimeError("injected registration failure")

    native.module.allocate_output = fail
    try:
        with pytest.raises(RuntimeError, match="BROKEN state"):
            comm.prepare_qkv((1, 64, 2, 64))
        assert comm._state == ulysses_mod._BROKEN
        assert comm._qkv_pcie_slot is None
    finally:
        comm._state = "closed"


def _patch_qkv_stages(monkeypatch, native, order):
    """Fake the Sage2 kernels and the statistics AllGather, in call order."""
    monkeypatch.setattr(lowp, "_validate_qkv", lambda *_args: None)  # host tensors

    def local_stats(*_args):
        order.append("stats")
        return torch.zeros(1), "ctx"

    def finalize(*_args):
        order.append("finalize")
        # [B, H, D] for the (1, 64, 2, 64) geometry these tests prepare.
        return SimpleNamespace(v_scale_global=torch.ones(1, 2, 64))

    monkeypatch.setattr(lowp, "_local_stats_impl", local_stats)
    monkeypatch.setattr(
        dist, "all_gather_into_tensor", lambda *_args, **_kw: order.append("gather")
    )
    monkeypatch.setattr(lowp, "_finalize_stats_impl", finalize)
    monkeypatch.setattr(
        lowp, "_quant_and_pack_impl", lambda *_args: order.append("pack")
    )
    monkeypatch.setattr(
        lowp, "_unpack_for_sage_impl", lambda *_args: order.append("unpack")
    )
    native.module.exchange = lambda _handle, x, out, mode, *shape: order.append(
        ("exchange", x.data_ptr(), out.data_ptr(), mode, shape)
    )
    # The fake peer's vote: set native.peer_vote = 1 to model a peer failure.
    native.peer_vote = 0

    def all_reduce(tensor, op=None, group=None):
        order.append("vote")
        tensor.fill_(max(int(tensor.item()), native.peer_vote))

    monkeypatch.setattr(dist, "all_reduce", all_reduce)


@pytest.mark.parametrize("transport", ["hybrid", "p2p"])
def test_scatter_qkv_pcie_exchanges_into_the_registered_slot(monkeypatch, transport):
    comm, native = _mock_pcie_qkv_comm(monkeypatch, transport)
    order = []
    _patch_qkv_stages(monkeypatch, native, order)
    try:
        workspace = comm.prepare_qkv((1, 64, 2, 64))
        q, k, v = (torch.zeros(1, 64, 2, 64, dtype=torch.bfloat16) for _ in range(3))
        result = comm.scatter_qkv(q, k, v, workspace=workspace)
        chunk = workspace._wire_shape[1]
        # Only the RDMA routes, whose barrier has no deadline, vote first.
        vote = ["vote"] if transport != "p2p" else []
        assert order == [
            "stats",
            "gather",
            "finalize",
            "pack",
            *vote,
            (
                "exchange",
                workspace._send_buffer.data_ptr(),
                native.storage.data_ptr(),
                2,
                (1, 1, 2, chunk),
            ),
            "unpack",
        ]
        assert (workspace._exchange_vote is None) == (transport == "p2p")
        assert torch.equal(result.v_scale, torch.ones(1, 1, 64))
        assert comm._pcie_stream == "caller"  # bound by the exchange
    finally:
        comm._state = "closed"


@pytest.mark.parametrize(
    "transport,failure",
    [
        (transport, failure)
        for transport in ("hybrid", "p2p")
        for failure in ("capture", "stream", "workspace", "statistics")
        # The all-P2P route is capturable.
        if (transport, failure) != ("p2p", "capture")
    ],
)
def test_scatter_qkv_pcie_rejects_before_enqueueing(monkeypatch, transport, failure):
    """Mirrors the PCIe collective envelope: anything that stops this rank
    before its exchange is enqueued is an ordinary error on the RDMA routes
    and fail-stop on the all-P2P route, whose barrier has no abort protocol.
    Capture and stream admission are decided before the statistics
    AllGather, so a refused call has enqueued no collective work. A later
    rank-local failure on an RDMA route still joins the AllGather it skipped
    and the vote, so that no peer enters the exchange alone."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm, native = _mock_pcie_qkv_comm(monkeypatch, transport)
    order = []
    _patch_qkv_stages(monkeypatch, native, order)
    try:
        workspace = comm.prepare_qkv((1, 64, 2, 64))
        q, k, v = (torch.zeros(1, 64, 2, 64, dtype=torch.bfloat16) for _ in range(3))
        expected_order = []
        if failure == "capture":
            monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        elif failure == "stream":
            comm._pcie_stream = "bound elsewhere"
        elif failure == "workspace":
            workspace = None
        else:

            def fail(*_args):
                order.append("stats")
                raise RuntimeError("injected statistics failure")

            monkeypatch.setattr(lowp, "_local_stats_impl", fail)
            expected_order = (
                ["stats"] if transport == "p2p" else ["stats", "gather", "vote"]
            )

        with pytest.raises((ValueError, RuntimeError)) as info:
            comm.scatter_qkv(q, k, v, workspace=workspace)
        assert order == expected_order
        if transport == "p2p":
            assert "BROKEN state" in str(info.value)
            assert comm._state == ulysses_mod._BROKEN
            assert comm._pcie_python_teardown_safe is False
        else:
            assert "BROKEN" not in str(info.value)
            assert comm._state == ulysses_mod._OPEN
            assert comm._pcie_python_teardown_safe is True
        assert comm._pcie_stream in (None, "bound elsewhere")
    finally:
        comm._state = "closed"


@pytest.mark.parametrize("transport", ["hybrid", "rdma"])
@pytest.mark.parametrize("failing", ["finalize", "peer"])
def test_scatter_qkv_rdma_vote_keeps_peers_out_of_the_barrier(
    monkeypatch, transport, failing
):
    """A rank-local failure between admission and the exchange raises on
    every rank before any of them enters the unbounded RDMA barrier."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm, native = _mock_pcie_qkv_comm(monkeypatch, transport)
    order = []
    _patch_qkv_stages(monkeypatch, native, order)
    try:
        workspace = comm.prepare_qkv((1, 64, 2, 64))
        q, k, v = (torch.zeros(1, 64, 2, 64, dtype=torch.bfloat16) for _ in range(3))
        with pytest.MonkeyPatch.context() as patch:
            if failing == "finalize":

                def fail(*_args):
                    order.append("finalize")
                    raise RuntimeError("injected finalize failure")

                patch.setattr(lowp, "_finalize_stats_impl", fail)
                expected = ["stats", "gather", "finalize", "vote"]
                message = "injected"
            else:
                patch.setattr(native, "peer_vote", 1)
                expected = ["stats", "gather", "finalize", "pack", "vote"]
                message = "peer"
            with pytest.raises(RuntimeError, match=message):
                comm.scatter_qkv(q, k, v, workspace=workspace)
        assert order == expected  # no exchange on any rank
        assert comm._state == ulysses_mod._OPEN
        assert comm._pcie_stream is None

        # Nobody entered the barrier, so the same communicator and workspace
        # proceed normally on the next call.
        order.clear()
        comm.scatter_qkv(q, k, v, workspace=workspace)
        assert [step for step in order if not isinstance(step, tuple)] == [
            "stats",
            "gather",
            "finalize",
            "pack",
            "vote",
            "unpack",
        ]
        assert any(isinstance(step, tuple) and step[0] == "exchange" for step in order)
        assert comm._pcie_stream == "caller"  # bound by that exchange
    finally:
        comm._state = "closed"


@pytest.mark.parametrize("transport", ["hybrid", "p2p"])
def test_scatter_qkv_pcie_output_allocation_failure_joins_the_envelope(
    monkeypatch, transport
):
    """With ``out=None`` the outputs are allocated after admission, where a
    rank-local OOM is covered like any other: on the RDMA routes it joins the
    statistics AllGather in place and the vote, on all-P2P it is fail-stop."""
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    comm, native = _mock_pcie_qkv_comm(monkeypatch, transport)
    order = []
    _patch_qkv_stages(monkeypatch, native, order)
    try:
        workspace = comm.prepare_qkv((1, 64, 2, 64))
        q, k, v = (torch.zeros(1, 64, 2, 64, dtype=torch.bfloat16) for _ in range(3))

        def fail(*_args):
            raise torch.OutOfMemoryError("injected output allocation failure")

        monkeypatch.setattr(comm, "_prepare_qkv_out", fail)
        with pytest.raises((RuntimeError, torch.OutOfMemoryError)) as info:
            comm.scatter_qkv(q, k, v, workspace=workspace)
        if transport == "p2p":
            assert order == []
            assert comm._state == ulysses_mod._BROKEN
        else:
            assert "injected output allocation failure" in str(info.value)
            assert order == ["gather", "vote"]  # joined in place, then voted
            assert comm._state == ulysses_mod._OPEN
    finally:
        comm._state = "closed"


def test_scatter_qkv_rdma_local_failure_survives_a_zero_vote(monkeypatch):
    """A rank that failed raises its own error even if the reduced vote it
    reads is zero, rather than entering the exchange."""
    comm, native = _mock_pcie_qkv_comm(monkeypatch, "rdma")
    order = []
    _patch_qkv_stages(monkeypatch, native, order)

    def fail(*_args):
        order.append("finalize")
        raise RuntimeError("injected finalize failure")

    monkeypatch.setattr(lowp, "_finalize_stats_impl", fail)
    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda tensor, op=None, group=None: (order.append("vote"), tensor.fill_(0)),
    )
    try:
        workspace = comm.prepare_qkv((1, 64, 2, 64))
        q, k, v = (torch.zeros(1, 64, 2, 64, dtype=torch.bfloat16) for _ in range(3))
        with pytest.raises(RuntimeError, match="injected finalize failure"):
            comm.scatter_qkv(q, k, v, workspace=workspace)
        assert order == ["stats", "gather", "finalize", "vote"]
    finally:
        comm._state = "closed"


def test_scatter_qkv_rdma_failure_before_gather_joins_it_in_place(monkeypatch):
    """Peers are already in the statistics AllGather; the failing rank joins
    it from its own slice of the gather buffer, allocating nothing."""
    comm, native = _mock_pcie_qkv_comm(monkeypatch, "rdma")
    comm.rank = 1
    order = []
    _patch_qkv_stages(monkeypatch, native, order)
    gathers = []
    monkeypatch.setattr(
        dist,
        "all_gather_into_tensor",
        lambda out, inp, group=None: gathers.append(
            (out.data_ptr(), inp.data_ptr(), inp.numel())
        ),
    )

    def fail(*_args):
        raise RuntimeError("injected statistics failure")

    monkeypatch.setattr(lowp, "_local_stats_impl", fail)
    try:
        workspace = comm.prepare_qkv((1, 64, 2, 64))
        q, k, v = (torch.zeros(1, 64, 2, 64, dtype=torch.bfloat16) for _ in range(3))
        with pytest.raises(RuntimeError, match="injected"):
            comm.scatter_qkv(q, k, v, workspace=workspace)
        chunk = workspace._stats_numel
        base = workspace._stats_gather.data_ptr()
        assert gathers == [(base, base + chunk * 4, chunk)]
        assert order == ["vote"]
    finally:
        comm._state = "closed"


def _force_pcie_route(route):
    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise UlyssesBackendError(
            f"PCIe transport needs {', '.join(missing)}, missing on this machine"
        )
    os.environ["FLASHINFER_ULYSSES_PCIE_ROUTE"] = route


def _check_pcie_route(comm, route):
    if route == "p2p":
        # A forced p2p route never falls back, so a mismatch is a test bug.
        assert comm.transport == "p2p", comm.transport
    elif comm.transport != route:
        raise UlyssesBackendError(f"forced {route} fell back to {comm.transport}")


def _pcie_correctness_body(rank, world, group, arg):
    route, dim = arg
    _force_pcie_route(route)
    # Two geometries per dtype share one registration. B=2 crosses the RDMA
    # routes too: the payload is one [1, 1, P, chunk] operand whatever B is.
    cases = {
        torch.bfloat16: ((1, 129, world * 129 - 1, False), (2, 17, 1, True)),
        torch.float16: (
            (2, 65, min(world * 65, 129), False),
            (1, 64, world * 64, True),
        ),
    }
    for dtype, geometries in cases.items():
        shards = [
            _shards(rank, world, batch, length, dim, dtype, used, contiguous=contiguous)
            for batch, length, used, contiguous in geometries
        ]
        expected = [
            _standalone(shard, rank, world, group, geometry[2])
            for shard, geometry in zip(shards, geometries, strict=True)
        ]
        capacity = max(_payload_bytes(shard[0].shape, world) for shard in shards)
        with UlyssesCommunicator(
            group, max_bytes=capacity, dtype=dtype, backend="pcie"
        ) as comm:
            _check_pcie_route(comm, route)
            workspaces = [
                comm.prepare_qkv(shard[0].shape, used_sequence=geometry[2])
                for shard, geometry in zip(shards, geometries, strict=True)
            ]
            assert len(comm._pcie_outputs) == 1
            slot = workspaces[0]._recv_buffer.data_ptr()
            for workspace in workspaces:
                assert workspace.transport == "pcie"
                assert workspace._recv_buffer.data_ptr() == slot
            # Alternate the geometries on the shared slot.
            for _ in range(2):
                for shard, geometry, workspace, (want, payload, recv) in zip(
                    shards, geometries, workspaces, expected, strict=True
                ):
                    _batch, length, used, _ = geometry
                    got = comm.scatter_qkv(*shard, workspace=workspace)
                    _assert_result(
                        got, want, used=used, total=world * length, dtype=dtype
                    )
                    _assert_bytes(workspace._send_buffer, payload)
                    _assert_bytes(workspace._recv_buffer, recv)
                    reused = comm.scatter_qkv(*shard, workspace=workspace, out=got)
                    assert reused is got
                    _assert_result(
                        reused, want, used=used, total=world * length, dtype=dtype
                    )
    return "ok", f"{route} D{dim}"


_PCIE_ROUTES = [(r, p) for r in ("p2p", "rdma") for p in (2, 4, 8)] + [("hybrid", 8)]


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    "route,world_size", _PCIE_ROUTES, ids=[f"{r}-p{p}" for r, p in _PCIE_ROUTES]
)
def test_qkv_pcie_matches_nccl(route, world_size, head_dim):
    """The PCIe routes only move bytes: the result equals the NCCL chain."""
    _require_gpus(world_size)
    _run_multi_rank(
        _pcie_correctness_body,
        world_size,
        (route, head_dim),
        timeout=900,
        allow_skip=True,
    )


def _pcie_split_direction_body(rank, world, group, route):
    _force_pcie_route(route)
    shard = _shards(rank, world, 1, 65, 64, torch.bfloat16, 129, contiguous=True)
    heads = shard[0].shape[2]
    expected, _, _ = _standalone(shard, rank, world, group, 129)
    attention = torch.empty(
        (1, world * 65, heads // world, 64), dtype=torch.bfloat16, device="cuda"
    )
    input_stream, output_stream = torch.cuda.Stream(), torch.cuda.Stream()
    with (
        UlyssesCommunicator(
            group,
            max_bytes=_payload_bytes(shard[0].shape, world),
            dtype=torch.bfloat16,
            backend="pcie",
        ) as input_comm,
        UlyssesCommunicator(
            group, max_bytes=attention.nbytes, dtype=torch.bfloat16, backend="pcie"
        ) as output_comm,
    ):
        _check_pcie_route(input_comm, route)
        _check_pcie_route(output_comm, route)
        workspace = input_comm.prepare_qkv(shard[0].shape, used_sequence=129)
        gathered = output_comm.allocate_output(attention, "gather_heads")
        torch.cuda.synchronize()
        for _ in range(3):
            input_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(input_stream):
                encoded = input_comm.scatter_qkv(*shard, workspace=workspace)
                # Any floating [B, P*L, H/P, D] tensor stands in for attention.
                attention.copy_(encoded.q)
                ready = torch.cuda.Event()
                ready.record(input_stream)
            output_stream.wait_event(ready)
            with torch.cuda.stream(output_stream):
                output_comm.gather_heads(attention, out=gathered)
            torch.cuda.synchronize()
            _assert_result(
                encoded, expected, used=129, total=world * 65, dtype=torch.bfloat16
            )
            _assert_bytes(gathered, _ref_gather_heads(attention, world, rank, group))
        assert input_comm._pcie_stream == input_stream
        assert output_comm._pcie_stream == output_stream
    return "ok", f"{route} input and output on separate communicators and streams"


@pytest.mark.parametrize("route", ["p2p", "rdma"])
def test_qkv_pcie_split_input_output_communicators(route):
    """The head-chunk pipeline's composition: input and output communication
    on separate PCIe communicators, each bound to its own stream."""
    _require_gpus(2)
    _run_multi_rank(_pcie_split_direction_body, 2, route, timeout=600, allow_skip=True)


def _pcie_rank_local_failure_body(rank, world, group, route):
    _force_pcie_route(route)
    shard = _shards(rank, world, 1, 65, 64, torch.bfloat16, 129)
    expected, _, _ = _standalone(shard, rank, world, group, 129)

    def fail(*_args):
        raise RuntimeError("injected rank-local failure")

    with UlyssesCommunicator(
        group,
        max_bytes=_payload_bytes(shard[0].shape, world),
        dtype=torch.bfloat16,
        backend="pcie",
    ) as comm:
        _check_pcie_route(comm, route)
        workspace = comm.prepare_qkv(shard[0].shape, used_sequence=129)
        # Rank 0 fails allocating the outputs, then in the statistics, then
        # after the statistics AllGather. Peers that entered the RDMA barrier
        # alone would hang until the timeout.
        for target, stage in (
            (comm, "_prepare_qkv_out"),
            (lowp, "_local_stats_impl"),
            (lowp, "_finalize_stats_impl"),
        ):
            with pytest.MonkeyPatch.context() as patch:
                if rank == 0:
                    patch.setattr(target, stage, fail)
                with pytest.raises(RuntimeError, match="injected|peer rank"):
                    comm.scatter_qkv(*shard, workspace=workspace)
            got = comm.scatter_qkv(*shard, workspace=workspace)
            _assert_result(
                got, expected, used=129, total=world * 65, dtype=torch.bfloat16
            )
    return "ok", f"{route}: three failures raised on every rank, then recovered"


@pytest.mark.parametrize("route,world_size", [("rdma", 2), ("rdma", 8), ("hybrid", 8)])
def test_qkv_pcie_rank_local_failure_releases_peers(route, world_size):
    """The RDMA routes vote before their unbounded barrier, so a failure on
    one rank raises everywhere and the communicator stays usable."""
    _require_gpus(world_size)
    _run_multi_rank(
        _pcie_rank_local_failure_body,
        world_size,
        route,
        timeout=600,
        allow_skip=True,
    )
