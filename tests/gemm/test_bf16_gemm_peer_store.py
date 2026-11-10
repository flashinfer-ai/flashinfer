"""Peer-store epilogues of the low-M BF16 GEMMs on one GPU.

``world_size`` local buffers stand in for the peers (unicast ``peer_ptrs``). The
destination range of each buffer starts as zeros, the sentinel, and must end up
holding the plain GEMM output, canonicalized, at the documented offset of the half
the epoch selects; every other byte holds a nonzero canary that must stay untouched,
and so must ``out``. Multicast needs several GPUs and is not covered.
"""

import pytest
import torch

from flashinfer.utils import get_compute_capability


def _require_splitk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("The split-K BF16 GEMMs require SM100/SM103/SM107.")
    from flashinfer.cute_dsl.utils import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is not available.")


_CANARY = 0xA5


def _destination(m, n, layout, rank, world_size, slot_bytes, epoch):
    if layout == "gather":
        offset = (epoch & 1) * world_size * slot_bytes + rank * m * n * 2
    else:
        offset = ((epoch & 1) * world_size + rank) * slot_bytes
    return slice(offset, offset + m * n * 2)


def _fill_canaries(buffers, destination):
    """Canaries everywhere, the zero sentinel in the destination range."""
    for buffer in buffers:
        buffer.fill_(_CANARY)
        buffer[destination].zero_()


def _expected(ref, destination, world_size, slot_bytes):
    bits = ref.view(torch.int32)
    bits = torch.where(bits == 0, torch.full_like(bits, 0x8000), bits)
    expected = torch.full(
        (2 * world_size * slot_bytes,), _CANARY, dtype=torch.uint8, device="cuda"
    )
    expected[destination] = bits.view(torch.uint8).flatten()
    return expected


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize(
    "kernel,layout",
    [
        ("warp", "broadcast"),
        ("warp", "gather"),
        ("cluster", "broadcast"),
        ("cluster", "gather"),
        ("tgv", "broadcast"),
        ("tgv", "gather"),
    ],
)
@pytest.mark.parametrize("m,n,k", [(5, 2048, 4096), (32, 1024, 2048)])
def test_peer_store(m, n, k, kernel, layout, world_size):
    _require_splitk()
    from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as cluster
    from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk as warp
    from flashinfer.gemm.kernels import tgv_gemm_cute_ext as tgv

    generator = torch.Generator(device="cuda").manual_seed(m + world_size)
    a = torch.randn((m, k), device="cuda", generator=generator).bfloat16()
    w = (torch.randn((n, k), device="cuda", generator=generator) / k**0.5).bfloat16()
    a[0].zero_()  # whole zero words
    w[3].zero_()  # isolated +0.0 elements, kept in their nonzero words
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    if kernel == "warp":
        tactic = warp.default_tactic(m, n, k)
        warp.run_warp_splitk_dense(a, w.T, ref, True, tactic)
        run = warp.run_warp_splitk_dense_peer_store
    elif kernel == "cluster":
        tactic = cluster.default_tactic(m, n, k)
        cluster.run_splitk_dense(a, w.T, None, ref, True, tactic)
        run = cluster.run_splitk_dense_peer_store
    else:
        tactic = 18  # 2-CTA 64 x 16 tile
        tgv.run_tgv_cute_ext(a, w.T, None, ref, True, tactic)
        run = tgv.run_tgv_cute_ext_peer_store

    rank, slot_bytes = world_size - 1, m * n * 2 + 64
    buffers = [
        torch.zeros(2 * world_size * slot_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(world_size)
    ]
    epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
    for value in (0, 1, 6, 3):
        epoch.fill_(value)
        destination = _destination(m, n, layout, rank, world_size, slot_bytes, value)
        _fill_canaries(buffers, destination)
        out = torch.full_like(ref, 1)
        run(
            a,
            w.T,
            out,
            tactic,
            peer_ptrs=[buffer.data_ptr() for buffer in buffers],
            epoch_ptr=epoch.data_ptr(),
            slot_bytes=slot_bytes,
            rank=rank,
            world_size=world_size,
            layout=layout,
        )
        torch.testing.assert_close(out, torch.full_like(ref, 1), rtol=0, atol=0)
        expected = _expected(ref, destination, world_size, slot_bytes)
        for buffer in buffers:
            torch.testing.assert_close(buffer, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kernel", ["cluster", "tgv"])
def test_peer_store_partial_feature_tile(kernel):
    """A partial last feature tile, and a 2-CTA grid rounded up past N: no peer store may land beyond N."""
    _require_splitk()
    from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as cluster
    from flashinfer.gemm.kernels import tgv_gemm_cute_ext as tgv

    # N = 2600 is 40.6 64-feature tiles; the 2-CTA TGV tactic 18 (64 x 16) launches 42.
    m, n, k, world_size = 16, 2600, 4096, 2
    a = torch.randn((m, k), device="cuda").bfloat16()
    w = (torch.randn((n, k), device="cuda") / k**0.5).bfloat16()
    ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    if kernel == "cluster":
        tactic = cluster.default_tactic(m, n, k)
        cluster.run_splitk_dense(a, w.T, None, ref, True, tactic)
        run = cluster.run_splitk_dense_peer_store
    else:
        tactic = 18
        tgv.run_tgv_cute_ext(a, w.T, None, ref, True, tactic)
        run = tgv.run_tgv_cute_ext_peer_store
    rank, slot_bytes = 0, m * n * 2 + 64
    buffers = [
        torch.zeros(2 * world_size * slot_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(world_size)
    ]
    epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
    destination = _destination(m, n, "broadcast", rank, world_size, slot_bytes, 0)
    _fill_canaries(buffers, destination)
    out = torch.full_like(ref, 1)
    run(
        a,
        w.T,
        out,
        tactic,
        peer_ptrs=[buffer.data_ptr() for buffer in buffers],
        epoch_ptr=epoch.data_ptr(),
        slot_bytes=slot_bytes,
        rank=rank,
        world_size=world_size,
    )
    torch.testing.assert_close(out, torch.full_like(ref, 1), rtol=0, atol=0)
    expected = _expected(ref, destination, world_size, slot_bytes)
    for buffer in buffers:
        torch.testing.assert_close(buffer, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kernel", ["warp", "cluster", "tgv"])
def test_peer_store_compiles_once_for_new_buffers(kernel, monkeypatch):
    """The addresses are kernel arguments: new peer buffers reuse the kernel."""
    _require_splitk()
    from cutlass.cute import experimental as cute_ext
    from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as cluster
    from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk as warp
    from flashinfer.gemm.kernels import tgv_gemm_cute_ext as tgv

    m, n, k, slot_bytes = 8, 1536, 2048, 8 * 1536 * 2
    a = torch.randn((m, k), device="cuda").bfloat16()
    w = (torch.randn((n, k), device="cuda") / k**0.5).bfloat16()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ref = (a.float() @ w.float().T).expand(2, m, n)
    run, tactic = {
        "warp": (warp.run_warp_splitk_dense_peer_store, warp.default_tactic),
        "cluster": (cluster.run_splitk_dense_peer_store, cluster.default_tactic),
        "tgv": (tgv.run_tgv_cute_ext_peer_store, lambda *_: 18),  # 2-CTA 64 x 16 tile
    }[kernel]
    count, fn = [], cute_ext.compile
    monkeypatch.setattr(cute_ext, "compile", lambda *x: count.append(x) or fn(*x))
    epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
    rank = 1
    dest = dict(epoch_ptr=epoch.data_ptr(), slot_bytes=slot_bytes, world_size=2)
    counts = []
    for _ in range(2):  # a new allocation (two peers) for each launch
        bufs = torch.zeros(2, 4 * slot_bytes, dtype=torch.uint8, device="cuda")
        ptrs = [buf.data_ptr() for buf in bufs]
        run(a, w.T, out, tactic(m, n, k), peer_ptrs=ptrs, rank=rank, **dest)
        tiles = bufs[:, rank * slot_bytes : (rank + 1) * slot_bytes]
        got = tiles.view(torch.bfloat16).reshape(2, m, n).float()
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
        counts.append(len(count))
    # The first launch compiles at most once (none if cached); the second never.
    assert counts[0] <= 1 and counts[1] == counts[0], counts


def test_peer_store_rejects_bad_arguments():
    _require_splitk()
    from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as cluster
    from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk as warp

    m, n, k = 8, 1024, 2048
    a = torch.randn((m, k), device="cuda").bfloat16()
    w = torch.randn((n, k), device="cuda").bfloat16()
    buffers = [torch.zeros(4 * m * n * 2, device="cuda") for _ in range(2)]
    peers = [buffer.data_ptr() for buffer in buffers]
    epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
    args = dict(epoch_ptr=epoch.data_ptr(), slot_bytes=m * n * 2, rank=0, world_size=2)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    tactic = warp.default_tactic(m, n, k)
    with pytest.raises(ValueError, match="exactly one"):
        warp.run_warp_splitk_dense_peer_store(a, w.T, out, tactic, **args)
    with pytest.raises(ValueError, match="peer_ptrs"):
        warp.run_warp_splitk_dense_peer_store(
            a, w.T, out, tactic, peer_ptrs=peers[:1], **args
        )
    with pytest.raises(ValueError, match="slot_bytes"):
        warp.run_warp_splitk_dense_peer_store(
            a, w.T, out, tactic, peer_ptrs=peers, **{**args, "slot_bytes": 1024}
        )
    with pytest.raises(ValueError, match="layout"):
        cluster.run_splitk_dense_peer_store(
            a,
            w.T,
            out,
            cluster.default_tactic(m, n, k),
            peer_ptrs=peers,
            layout="scatter",
            **args,
        )
    # Peer-store addressing has no batch index: batched TGV inputs are rejected.
    from flashinfer.gemm.kernels import tgv_gemm_cute_ext as tgv

    a3, b3, out3 = a[None], w.T[None], out[None]
    with pytest.raises(ValueError, match="2D"):
        tgv.run_tgv_cute_ext_peer_store(a3, b3, out3, 18, peer_ptrs=peers, **args)
    config = warp.make_lamport_peer_store_config(
        peer_ptrs=peers, multicast_ptr=None, layout="broadcast", m=m, n=n, **args
    )
    with pytest.raises(ValueError, match="2D"):
        tgv.run_tgv_cute_ext(a3, b3, None, out3, True, 18, peer_store=config)
    with pytest.raises(ValueError, match="out"):
        tgv.run_tgv_cute_ext_peer_store(a, w.T, out[:, :8], 18, peer_ptrs=peers, **args)
    # A config made for a smaller GEMM must not be reused for a larger one.
    small = warp.make_lamport_peer_store_config(
        peer_ptrs=peers,
        multicast_ptr=None,
        layout="broadcast",
        m=m // 2,
        n=n,
        **{**args, "slot_bytes": m * n},
    )
    with pytest.raises(ValueError, match="slot_bytes"):
        tgv.run_tgv_cute_ext(a, w.T, None, out, True, 18, peer_store=small)
    # The epilogue stages a row-major output tile.
    out_cm = torch.empty((n, m), device="cuda", dtype=torch.bfloat16).T
    with pytest.raises(ValueError, match="row-major"):
        tgv.run_tgv_cute_ext_peer_store(a, w.T, out_cm, 18, peer_ptrs=peers, **args)
    with pytest.raises(ValueError, match="row-major"):
        cluster.run_splitk_dense_peer_store(
            a,
            w.T,
            out_cm,
            cluster.default_tactic(m, n, k),
            peer_ptrs=peers,
            **args,
        )


if __name__ == "__main__":
    pytest.main([__file__])
