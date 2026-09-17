"""Real collectives and lifecycle checks for the prepared Sage2 QKV path."""

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator, UlyssesQKV, UlyssesQKVWorkspace
import flashinfer.comm._ulysses_lowp as lowp
from tests.comm.test_ulysses_communicator import (
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
            group, max_elems=shard[0].numel(), dtype=dtype, backend="nccl"
        ) as comm:
            workspace = comm.prepare_qkv(shard[0].shape, used_sequence=used)
            assert workspace.transport == "nccl"
            assert comm.backend == "nccl"
            actual = comm.scatter_qkv(*shard, workspace=workspace)
            _assert_result(
                actual, expected, used=used, total=world * length, dtype=dtype
            )
            _assert_bytes(workspace._send_buffer, payload)
            _assert_bytes(workspace._recv_buffer, recv)
            # Byte workspace can exceed the original floating-operand budget.
            assert workspace._send_buffer.numel() > shard[0].numel() * dtype.itemsize
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


def _ownership_body(rank, world, group, _arg):
    shape = (2, 128, world, 64)
    dtype = torch.bfloat16
    with UlyssesCommunicator(
        group, max_elems=2 * 128 * world * 64, dtype=dtype, backend="nccl"
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
                group, max_elems=shard[0].numel(), dtype=dtype, backend="nccl"
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
        group, max_elems=shard[0].numel(), dtype=shard[0].dtype, backend="nccl"
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
        group, max_elems=2 * 128 * world * 128, dtype=torch.bfloat16, backend="nccl"
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
        group, max_elems=shard[0].numel(), dtype=shard[0].dtype, backend="nvlink"
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
        group, max_elems=x.numel(), dtype=x.dtype, backend="nccl"
    ) as comm:
        with pytest.raises((ValueError, RuntimeError)):
            comm.prepare_qkv(x.shape)
        assert comm.scatter_heads(x) is x
        assert comm.gather_heads(x) is x
    return "ok", "P1 rejects quantization and retains plain identity"


def test_qkv_rejects_single_rank_without_changing_plain_identity():
    _require_gpus(1)
    _run_multi_rank(_single_rank_body, 1, None)


def _non_nccl_group_body(rank, world, group, _arg):
    metadata_group = dist.new_group(backend="gloo")
    try:
        x = torch.full((1, 4, world, 64), rank + 1, dtype=torch.bfloat16, device="cuda")
        with UlyssesCommunicator(
            metadata_group,
            max_elems=x.numel(),
            dtype=x.dtype,
            backend="nvlink",
        ) as comm:
            with pytest.raises((ValueError, RuntimeError), match="NCCL|nccl"):
                comm.prepare_qkv(x.shape)
            output = comm.scatter_heads(x)
            expected = torch.cat(
                [torch.full_like(output[:, :4], i + 1) for i in range(world)], dim=1
            )
            _assert_bytes(output, expected)
    finally:
        dist.destroy_process_group(metadata_group)
    return "ok", "Gloo metadata does not enable NCCL QKV"


def test_qkv_rejects_group_without_nccl_but_retains_nvlink():
    _require_gpus(2)
    _run_multi_rank(_non_nccl_group_body, 2, None, timeout=600, allow_skip=True)


def test_qkv_workspace_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        UlyssesQKVWorkspace()
