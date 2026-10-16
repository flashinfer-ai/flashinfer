"""Opaque destination-chunk exchange through the public communicator API."""

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator
from tests.comm.test_ulysses_communicator import (
    _forbid_ipc_and_jit,
    _run_multi_rank,
    gloo_pg as _gloo_pg,
    requires_cuda,
)

gloo_pg = _gloo_pg


def _exchange_body(rank, world, group, backend):
    # The communicator's floating default remains suitable for attention output;
    # byte chunks opt into uint8 per call without altering that default.
    with UlyssesCommunicator(
        group, max_bytes=world * 259 * 4, dtype=torch.bfloat16, backend=backend
    ) as comm:
        assert comm.backend == backend
        for dtype in (torch.uint8, torch.float16, torch.bfloat16, torch.float32):
            for count in (1, 259):
                # uint8 wraps naturally and includes all byte patterns at C=259.
                x = (
                    torch.arange(world * count, device="cuda").reshape(
                        1, 1, world, count
                    )
                    + rank * 17
                ).to(dtype)
                gathered = [torch.empty_like(x) for _ in range(world)]
                dist.all_gather(gathered, x, group=group)
                expected = torch.cat(
                    [source[:, :, rank : rank + 1] for source in gathered], dim=2
                )
                output = comm.allocate_output(x, "exchange_chunks", dtype=dtype)
                with pytest.raises(ValueError, match="alias|overlap"):
                    comm.exchange_chunks(x, out=x, dtype=dtype)
                assert comm.exchange_chunks(x, out=output, dtype=dtype) is output
                assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                allocated = comm.exchange_chunks(x, dtype=dtype)
                assert allocated.data_ptr() != x.data_ptr()
                assert torch.equal(
                    allocated.view(torch.uint8), expected.view(torch.uint8)
                )
                assert comm.dtype == torch.bfloat16
        return "ok", f"{backend}: four dtypes, independent all-gather oracle"


@pytest.mark.parametrize(
    "world_size,backend", [(2, "nccl"), (4, "nccl"), (8, "nccl"), (2, "nvlink")]
)
def test_exchange_chunks_matches_independent_oracle(world_size, backend):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    _run_multi_rank(
        _exchange_body,
        world_size,
        backend,
        timeout=300,
        allow_skip=backend == "nvlink",
    )


@requires_cuda
def test_exchange_chunks_single_rank_identity_and_destination(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    with UlyssesCommunicator(
        gloo_pg, max_bytes=32, dtype=torch.bfloat16, backend="nccl"
    ) as comm:
        for dtype in (torch.bfloat16, torch.uint8):
            x = torch.arange(8, device="cuda").to(dtype).view(1, 1, 1, 8)
            kwargs = {} if dtype == comm.dtype else {"dtype": dtype}
            assert comm.exchange_chunks(x, **kwargs) is x
            assert comm.exchange_chunks(x, out=x, **kwargs) is x
            output = comm.allocate_output(x, "exchange_chunks", **kwargs)
            assert output.dtype == dtype and output.shape == x.shape
            assert comm.exchange_chunks(x, out=output, **kwargs) is output
            assert torch.equal(output, x)
    with pytest.raises(RuntimeError, match="use-after-close"):
        comm.exchange_chunks(x, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="use-after-close"):
        comm.allocate_output(x, "exchange_chunks", dtype=torch.uint8)


@requires_cuda
def test_exchange_chunks_validation(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    with UlyssesCommunicator(
        gloo_pg, max_bytes=8, dtype=torch.bfloat16, backend="nccl"
    ) as comm:
        x = torch.arange(8, dtype=torch.uint8, device="cuda").view(1, 1, 1, 8)
        bad_inputs = (
            ([1, 2], (TypeError, ValueError)),
            (x[0], ValueError),
            (x.view(1, 1, 2, 4), ValueError),
            (x.view(2, 1, 1, 4), ValueError),
            (x.cpu(), ValueError),
            (x[..., ::2], ValueError),
            (x[..., :0], ValueError),
            (torch.zeros(1, 1, 1, 9, dtype=x.dtype, device=x.device), ValueError),
        )
        for invalid, error in bad_inputs:
            with pytest.raises(error):
                comm.exchange_chunks(invalid, dtype=torch.uint8)
        for dtype in (torch.int16, "uint8", torch.bfloat16, None):
            with pytest.raises((TypeError, ValueError)):
                comm.exchange_chunks(x, dtype=dtype)
        bad_outputs = (
            x[0],
            x.cpu(),
            x.to(torch.bfloat16),
            torch.empty(1, 1, 1, 16, dtype=x.dtype, device=x.device)[..., ::2],
        )
        for invalid in bad_outputs:
            with pytest.raises((TypeError, ValueError)):
                comm.exchange_chunks(x, out=invalid, dtype=torch.uint8)
        backing = torch.empty(9, dtype=x.dtype, device=x.device)
        with pytest.raises(ValueError, match="alias|overlap"):
            comm.exchange_chunks(
                backing[:8].view_as(x),
                out=backing[1:].view_as(x),
                dtype=torch.uint8,
            )
        with pytest.raises(ValueError):
            comm.allocate_output(x, "unsupported", dtype=torch.uint8)
        # Per-call byte dtype is restricted to chunks; ordinary head exchange
        # retains the dtype configured at communicator construction.
        for operation in (comm.scatter_heads, comm.gather_heads):
            with pytest.raises((TypeError, ValueError)):
                operation(x, dtype=torch.uint8)
        assert comm.exchange_chunks(x, dtype=torch.uint8) is x


@requires_cuda
def test_exchange_chunks_uses_bytes_for_capacity(gloo_pg, monkeypatch):
    _forbid_ipc_and_jit(monkeypatch)
    # Odd capacities are useful for raw bytes and need not be divisible by the
    # communicator's default floating item size.
    with UlyssesCommunicator(
        gloo_pg, max_bytes=9, dtype=torch.float32, backend="nccl"
    ) as comm:
        x = torch.arange(9, dtype=torch.uint8, device="cuda").view(1, 1, 1, 9)
        assert comm.max_bytes == 9
        assert comm.exchange_chunks(x, dtype=torch.uint8) is x
        floating = torch.ones(1, 1, 1, 3, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="max_bytes|capacity"):
            comm.exchange_chunks(floating)
        with pytest.raises(ValueError, match="max_bytes|capacity"):
            comm.allocate_output(floating, "exchange_chunks")
        with pytest.raises(ValueError, match="max_bytes|capacity"):
            comm.scatter_heads(floating)
