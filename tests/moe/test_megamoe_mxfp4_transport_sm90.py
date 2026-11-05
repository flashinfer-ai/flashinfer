"""NCCL-window transport of the SM90 MXFP4 MegaMoE kernel.

The same inputs go through a torch-symmetric-memory buffer and through an
NCCL-window buffer; inside one NVLink domain both must produce the same output,
with plain resident experts and with redundant expert slots.

Run with plain Python (workers are spawned here):
    python tests/moe/test_megamoe_mxfp4_transport_sm90.py --ep 8
"""

import argparse
import datetime
import os
import sys

import pytest
import torch
import torch.distributed as dist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.moe.test_megamoe_fp4_sm90 import _weights  # noqa: E402
from tests.moe.test_megamoe_mxfp4_eplb_sm90 import (  # noqa: E402
    BITWISE_REPEAT_MAX_TOKENS,
    _assert_close,
    _skewed_routing,
)


def _same(y, expected, m, label):
    """Both transports must agree: bitwise at small ``m``, within tolerance above."""
    torch.cuda.synchronize()
    if m <= BITWISE_REPEAT_MAX_TOKENS:
        assert torch.equal(y, expected), label
    else:
        _assert_close(y, expected, label)


def _worker(rank, args):
    """Per-rank body: build both buffers, run identical inputs through each and compare."""
    from flashinfer.megamoe import (
        MegaMoEExpertSlots,
        MegaMoENcclTransport,
        MegaMoESymmBuffer,
        mxfp4_mega_moe,
        transform_mxfp4_weights_for_mega_moe,
    )

    ep = args.ep
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=ep,
        timeout=datetime.timedelta(minutes=10),
        device_id=torch.device("cuda", rank),
    )
    group = dist.group.WORLD
    h, ih, E, topk, S = (
        args.hidden,
        args.intermediate,
        args.experts,
        args.topk,
        args.slots,
    )
    H = E // ep
    generator = torch.Generator(device="cuda").manual_seed(4099 + rank)
    raw1, _ = _weights(H, ih * 2, h, 32, generator)
    raw2, _ = _weights(H, h, ih, 32, generator)
    w1, w2 = transform_mxfp4_weights_for_mega_moe(raw1, raw2)
    gs1 = torch.linspace(0.73, 1.37, H, device="cuda")
    gs2 = torch.linspace(1.31, 0.67, H, device="cuda")
    hot = torch.randperm(
        E, device="cuda", generator=torch.Generator(device="cuda").manual_seed(31)
    )[:S].tolist()

    transport = MegaMoENcclTransport(group, gin_contexts=args.gin_contexts)
    # Plain resident experts: one buffer per transport.
    plain = [
        MegaMoESymmBuffer(group, E, args.capacity, topk, h, ih),
        MegaMoESymmBuffer(group, E, args.capacity, topk, h, ih, transport=transport),
    ]
    # Redundant expert slots: the buffer geometry counts the slots as experts.
    slotted = [
        MegaMoESymmBuffer(group, E + ep * S, args.capacity, topk, h, ih),
        MegaMoESymmBuffer(
            group, E + ep * S, args.capacity, topk, h, ih, transport=transport
        ),
    ]
    slots = [MegaMoEExpertSlots(b, E, S) for b in slotted]
    for s in slots:
        s.set_resident_weights(w1, w2, gs1, gs2)
        assert s.set_replica_plan(torch.full((ep, S), -1)) == 0
    peers = transport.peer_pointers(plain[1].window)
    assert len(peers) == ep and all(p != 0 for p in peers), peers
    assert peers[rank] == plain[1].buffer.data_ptr()

    for m in args.tokens:
        x = (torch.randn(m, h, device="cuda", generator=generator) * 32).to(
            torch.float8_e4m3fn
        )
        sf = torch.full((m, h // 128), 1 / 128, device="cuda")
        idx, routing = _skewed_routing(m, E, topk, hot, 0.5, generator)

        outputs, counts = [], []
        for b in plain:
            b.stage(x, sf, idx, routing)
            y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
            stats = torch.zeros(H, dtype=torch.int32, device="cuda")
            mxfp4_mega_moe(
                y,
                w1,
                w2,
                b,
                cumulative_local_expert_recv_stats=stats,
                l1_global_scales=gs1,
                l2_global_scales=gs2,
            )
            outputs.append(y)
            counts.append(stats)
        _same(outputs[1], outputs[0], m, ("plain", m))
        assert torch.equal(counts[0], counts[1]), ("plain counts", m)

        outputs, counts = [], []
        for b, s in zip(slotted, slots):
            b.stage(x, sf, idx, routing)
            s.remap(m, s.block_m(m))
            y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
            stats = torch.zeros(H + S, dtype=torch.int32, device="cuda")
            mxfp4_mega_moe(
                y,
                s.l1_weights,
                s.l2_weights,
                b,
                cumulative_local_expert_recv_stats=stats,
                l1_global_scales=s.l1_global_scales,
                l2_global_scales=s.l2_global_scales,
                expert_slots=s,
            )
            outputs.append(y)
            counts.append(stats)
        _same(outputs[1], outputs[0], m, ("slots", m))
        assert torch.equal(counts[0], counts[1]), ("slot counts", m)
        if rank == 0:
            print(f"transport check passed at {m} tokens per rank", flush=True)

    torch.cuda.synchronize()
    dist.barrier()
    for s in slots:
        s.destroy()
    for b in plain + slotted:
        b.destroy()
    transport.close()
    dist.barrier()
    dist.destroy_process_group()


def _run(args):
    """Spawn ``args.ep`` workers on as many GPUs."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.ep)


def _namespace(ep, **overrides):
    """Default test arguments for ``ep`` ranks, with ``overrides`` applied."""
    args = dict(
        ep=ep,
        hidden=512,
        intermediate=256,
        experts=64,
        topk=4,
        slots=2,
        capacity=384,
        tokens=[24, 96, 384],
        gin_contexts=0,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


@pytest.mark.parametrize("ep", [4, 8])
def test_mxfp4_nccl_transport(ep):
    """Torch symmetric memory and NCCL windows give the same kernel output."""
    if torch.cuda.device_count() < ep:
        pytest.skip(f"needs {ep} GPUs")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 only")
    _run(_namespace(ep))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=384)
    parser.add_argument("--tokens", type=int, nargs="+", default=[24, 96, 384])
    parser.add_argument("--gin-contexts", type=int, default=0)
    _run(parser.parse_args())
