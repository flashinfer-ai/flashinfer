"""Redundant expert slots of the SM90 MXFP4 MegaMoE kernel: a caller-provided replica plan and the in-kernel planner.

Run with plain Python (workers are spawned here):
    python tests/moe/test_megamoe_mxfp4_eplb_sm90.py --ep 8
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

from tests.moe.test_megamoe_fp4_sm90 import _gather, _reference, _weights  # noqa: E402

RELATIVE_L2_TOLERANCE = 0.015


def _skewed_routing(m, experts, topk, hot, hot_share, generator):
    """Top-k routing where a share of all slots lands on a few hot experts."""
    scores = torch.rand(m, experts, device="cuda", generator=generator)
    mask = torch.rand(m, len(hot), device="cuda", generator=generator) < hot_share
    scores[:, torch.tensor(hot, device="cuda")] += mask.float() * 10.0
    idx = scores.topk(topk, dim=-1).indices
    idx[::3, -1] = -1
    weights = torch.randn(m, topk, device="cuda", generator=generator) * 0.25
    return idx, weights


def _histograms(idx, experts):
    """``counts[r][e]``: tokens rank ``r`` routes to expert ``e``, all-gathered."""
    return _gather(torch.bincount(idx[idx >= 0], minlength=experts)).view(-1, experts)


def _assert_close(y, expected, label):
    """``y`` is finite and within the relative L2 tolerance of ``expected``."""
    assert torch.isfinite(y).all(), label
    error = torch.linalg.vector_norm(y.float() - expected.float())
    relative_l2 = error / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    assert relative_l2 < RELATIVE_L2_TOLERANCE, (label, relative_l2.item())


def _run_graph(buffer, idx, prepare, run, replays=2):
    """Capture the routing restore, ``prepare`` and ``run`` into one CUDA graph,
    then yield the replay index after each synchronized replay."""
    m = idx.shape[0]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        buffer.topk_idx[:m].copy_(idx)
        prepare()
        run()
    for replay in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        yield replay


def _replicated_plan(counts, hot, block_m, R, E, H, S):
    """Replicate each hot expert onto one slot of every other rank, resident
    rank first with one block, then even shares in rank order."""
    slot_expert = torch.full((R, S), -1, dtype=torch.int32)
    copies = torch.full((E, R, 2), -1, dtype=torch.int32)
    copies[:, 0, 0] = torch.arange(E) // H
    copies[:, 0, 1] = torch.iinfo(torch.int32).max
    for s, e in enumerate(hot):
        resident = e // H
        others = [r for r in range(R) if r != resident]
        for r in others:
            slot_expert[r, s] = e
        blocks = -(-int(counts[:, e].sum()) // block_m)
        shares = [min(1, blocks)] + [0] * len(others)
        for i in range(blocks - shares[0]):
            shares[1 + i % len(others)] += 1
        cum = 0
        for k, (r, share) in enumerate(zip([resident] + others, shares, strict=True)):
            cum += share
            copies[e, k] = torch.tensor([r, cum], dtype=torch.int32)
    return slot_expert, copies


def _plane_checksums(slots):
    sums = []
    for v in (
        slots.l1_weights,
        slots.l2_weights,
        slots.l1_global_scales,
        slots.l2_global_scales,
    ):
        flat = v.reshape(v.shape[0], -1)
        if flat.dtype == torch.float32:
            flat = flat.view(torch.int32)
        sums.append(flat.to(torch.int64).sum(1))
    return torch.stack(sums)


def _assert_slots_hold_experts(slots):
    """Every assigned redundant slot is byte-identical to its expert's resident slot."""
    local = _plane_checksums(slots)
    gathered = [torch.empty_like(local) for _ in range(slots.num_ranks)]
    dist.all_gather(gathered, local)
    table = slots.slot_expert.cpu()
    H = slots.num_resident_experts
    for r in range(slots.num_ranks):
        for s in range(slots.num_redundant_slots):
            e = int(table[r, s])
            if e >= 0:
                assert torch.equal(
                    gathered[r][:, H + s], gathered[e // H][:, e - e // H * H]
                ), (r, s, e)


def _setup(rank, args, seed, hot_seed, num_hot):
    """Process group, symmetric buffer and slots holding this rank's resident
    experts. Returns the geometry ``(ep, h, E, topk, S, H)``, this rank's
    generator, the hot experts shared by every rank, and closures that run the
    kernel and the logical reference."""
    from flashinfer.megamoe import (
        MegaMoEExpertSlots,
        MegaMoESymmBuffer,
        mxfp4_mega_moe,
        transform_mxfp4_weights_for_mega_moe,
    )

    ep = args.ep
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
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
    generator = torch.Generator(device="cuda").manual_seed(seed + rank)
    buffer = MegaMoESymmBuffer(group, E + ep * S, args.capacity, topk, h, ih)
    slots = MegaMoEExpertSlots(buffer, E, S)
    raw1, ref1 = _weights(H, ih * 2, h, 32, generator)
    raw2, ref2 = _weights(H, h, ih, 32, generator)
    w1, w2 = transform_mxfp4_weights_for_mega_moe(raw1, raw2)
    gs1 = torch.linspace(0.73, 1.37, H, device="cuda")
    gs2 = torch.linspace(1.31, 0.67, H, device="cuda")
    slots.set_resident_weights(w1, w2, gs1, gs2)
    # The same shuffled hot experts on every rank.
    hot_generator = torch.Generator(device="cuda").manual_seed(hot_seed)
    hot = torch.randperm(E, device="cuda", generator=hot_generator)[:num_hot].tolist()

    def run(y, stats, in_kernel_copy=True):
        return mxfp4_mega_moe(
            y,
            slots.l1_weights,
            slots.l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=stats,
            l1_global_scales=slots.l1_global_scales,
            l2_global_scales=slots.l2_global_scales,
            expert_slots=slots if in_kernel_copy else None,
        )

    def reference(x, idx, routing):
        return _reference(x.float() / 128, idx, routing, ref1, ref2, gs1, gs2, rank, H)

    return (ep, h, E, topk, S, H), buffer, slots, generator, hot, run, reference


def _worker(rank, args):
    geometry, buffer, slots, generator, hot, run, reference = _setup(
        rank, args, seed=7919, hot_seed=29, num_hot=args.slots
    )
    ep, h, E, topk, S, H = geometry

    def check(y, expected, stats, expected_stats, label):
        torch.cuda.synchronize()
        _assert_close(y, expected, label)
        assert torch.equal(stats.long(), expected_stats), (label, stats, expected_stats)

    for m in args.tokens:
        x = (torch.randn(m, h, device="cuda", generator=generator) * 32).to(
            torch.float8_e4m3fn
        )
        sf = torch.full((m, h // 128), 1 / 128, device="cuda")
        idx, routing = _skewed_routing(m, E, topk, hot, 0.5, generator)
        expected, counts_logical = reference(x, idx, routing)
        counts = _histograms(idx, E)
        block_m = slots.block_m(m)
        y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)

        # (a) Empty placement: every expert stays resident; the physical kernel
        # must agree with the logical reference and the counts land on the
        # resident slots.
        assert slots.set_replica_plan(torch.full((ep, S), -1)) == 0
        buffer.stage(x, sf, idx, routing)
        slots.remap(m, block_m)
        stats = torch.zeros(H + S, dtype=torch.int32, device="cuda")
        run(y, stats, True)
        expected_stats = torch.cat(
            [counts_logical, torch.zeros(S, dtype=torch.int64, device="cuda")]
        )
        check(y, expected, stats, expected_stats, ("empty", m))
        assert int(slots.copy_count.item()) == 0

        # (b) The hot experts replicated onto one slot of every other rank with
        # a block split chosen here; the separate copy kernel fills the slots.
        slot_expert, copies = _replicated_plan(counts.cpu(), hot, block_m, ep, E, H, S)
        copy_table = _device_copies(copies)
        base = counts[:rank].sum(0).to(torch.int32)
        changed = slots.set_replica_plan(slot_expert, copies, base)
        assert changed == sum(1 for e in hot if e // H != rank)
        slots.copy_weights()
        torch.cuda.synchronize()
        dist.barrier()
        _assert_slots_hold_experts(slots)
        assert int(slots.copy_count.item()) == 0
        buffer.stage(x, sf, idx, routing)
        slots.remap(m, block_m)
        expected_stats = _expected_stats(
            idx, counts, copy_table, slot_expert.tolist(), block_m, rank
        )
        routed = expected_stats[H:].sum()
        dist.all_reduce(routed)
        assert routed > 0, "the plan must route tokens to redundant slots"
        stats.zero_()
        run(y, stats, False)
        check(y, expected, stats, expected_stats, ("replicated, separate copy", m))
        first = y.clone()
        run(y, stats, True)
        check(
            y,
            expected,
            stats,
            expected_stats * 2,
            ("replicated, in-kernel copy of nothing", m),
        )
        assert torch.equal(y, first), (
            "in-kernel copy of an empty list changed the output"
        )

        # (c) Remap-only on changed routing under the same tables.
        idx2, routing2 = _skewed_routing(m, E, topk, hot, 0.7, generator)
        expected2, _ = reference(x, idx2, routing2)
        buffer.stage(x, sf, idx2, routing2)
        slots.remap(m, block_m)
        expected_stats2 = _expected_stats(
            idx2, counts, copy_table, slot_expert.tolist(), block_m, rank
        )
        stats.zero_()
        run(y, stats, True)
        check(y, expected2, stats, expected_stats2, ("remap only", m))

        # (d, e) The slots swap experts between calls: the in-kernel copy must
        # refill every listed slot (copy_list, copy_done) before its tasks.
        swapped = slot_expert.clone()
        swapped[:, : len(hot)] = slot_expert[:, : len(hot)].flip(1)
        changed = slots.set_replica_plan(swapped, copies, base)
        assert changed == sum(1 for e in hot if e // H != rank)
        listed = slots.copy_list.cpu()[:changed, 0].tolist()
        buffer.stage(x, sf, idx, routing)
        slots.remap(m, block_m)
        expected_stats = _expected_stats(
            idx, counts, copy_table, swapped.tolist(), block_m, rank
        )
        stats.zero_()
        run(y, stats, True)
        check(y, expected, stats, expected_stats, ("swapped slots, in-kernel copy", m))
        torch.cuda.synchronize()
        dist.barrier()
        _assert_slots_hold_experts(slots)
        done = slots.copy_done.cpu()
        for s in listed:
            assert done[s, 0].item() == done[s, 1].item() == buffer.num_sms, (
                s,
                done[s],
            )
        assert int(slots.copy_count.item()) == 0

        # (f) Bitwise repeat, then remap + kernel under CUDA graph replay.
        first = y.clone()
        buffer.stage(x, sf, idx, routing)
        slots.remap(m, block_m)
        run(y, stats, True)
        torch.cuda.synchronize()
        assert torch.equal(y, first), "repeat changed the output"
        dist.barrier()
        torch.cuda.synchronize()
        for replay in _run_graph(
            buffer, idx, lambda: slots.remap(m, block_m), lambda: run(y, stats, True)
        ):
            assert torch.equal(y, first), "CUDA graph replay changed the output"
            assert torch.equal(stats.long(), expected_stats * (3 + replay))
        if rank == 0:
            print(f"PASS mxfp4 slots ep={ep} S={S} M={m} block_m={block_m}", flush=True)
    slots.destroy()
    buffer.destroy()
    dist.destroy_process_group()


def _run(args, worker=_worker):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29652")
    torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.ep)


def _namespace(ep, **overrides):
    args = dict(
        ep=ep,
        hidden=512,
        intermediate=256,
        experts=64,
        topk=4,
        slots=2,
        capacity=384,
        tokens=[24, 96, 384],
    )
    args.update(overrides)
    return argparse.Namespace(**args)


@pytest.mark.parametrize("ep", [4, 8])
def test_mxfp4_external_replica_plan(ep):
    if torch.cuda.device_count() < ep or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip(f"requires {ep} Hopper GPUs")
    _run(_namespace(ep))


# ---------------------------------------------------------------------------
# In-kernel planner: invariants and a serial reference of the same decisions
# ---------------------------------------------------------------------------


def _ceil_div(a, b):
    return -(-a // b)


def _plan_reference(
    counts, num_redundant_slots, block_m, prev_slot_expert, open_gate_blocks=0
):
    """Test-only serial oracle of `eplb_plan_from_histogram` (MegaFlux's
    load-only block-granular greedy, arXiv 2610.00671 App. A.1), torch-free.

    ``counts[r][e]`` tokens rank ``r`` routes to expert ``e``;
    ``prev_slot_expert[r][s]`` the expert a redundant slot currently holds
    (reusing it costs no weight copy, so such ranks are preferred receivers and
    exempt from ``open_gate_blocks``). Returns ``(blocks[e][r], slot_expert,
    copies[e] = [(rank, cumulative blocks), ...], loads)``.
    """
    R, E, S = len(counts), len(counts[0]), num_redundant_slots
    H = E // R
    resident = [e // H for e in range(E)]
    cached = [set(x for x in prev_slot_expert[r] if x >= 0) for r in range(R)]
    b = [_ceil_div(sum(counts[r][e] for r in range(R)), block_m) for e in range(E)]
    q = [[0] * R for _ in range(E)]
    for e in range(E):
        q[e][resident[e]] = b[e]
    load = [sum(q[e][r] for e in range(E)) for r in range(R)]
    tau = _ceil_div(sum(b), R)

    def replicas_on(r):
        return sum(1 for e in range(E) if resident[e] != r and q[e][r] > 0)

    def holds(r, e):
        return q[e][r] > 0 or e in cached[r]

    for _ in range(S + 2):
        moved = False
        donors = sorted(
            (r for r in range(R) if load[r] > tau), key=lambda r: (-(load[r] - tau), r)
        )
        for d in donors:
            experts = sorted(
                (e for e in range(d * H, (d + 1) * H) if q[e][d] > 0),
                key=lambda e: (-q[e][d], e),
            )
            for e in experts:
                if load[d] <= tau:
                    break
                chosen = None
                for r in sorted(
                    (r for r in range(R) if r != d and holds(r, e)),
                    key=lambda r: (load[r], r),
                ):
                    v = min(q[e][d], max(0, tau - load[r]))
                    if v > 0 and (q[e][r] > 0 or replicas_on(r) < S):
                        chosen = (r, v)
                        break
                if chosen is None:
                    fresh = sorted(
                        (
                            r
                            for r in range(R)
                            if r != d
                            and not holds(r, e)
                            and load[r] < tau
                            and replicas_on(r) < S
                        ),
                        key=lambda r: (load[r], r),
                    )
                    for r in fresh:
                        v = min(q[e][d], max(0, tau - load[r]))
                        if v >= max(1, open_gate_blocks):
                            chosen = (r, v)
                            break
                if chosen is None:
                    continue
                r, v = chosen
                q[e][d] -= v
                q[e][r] += v
                load[d] -= v
                load[r] += v
                moved = True
        if not moved:
            break
    # Fold back replicas whose allocation fits elsewhere under the maximum load.
    max_load = max(load)
    for v, e, r in sorted(
        (q[e][r], e, r)
        for e in range(E)
        for r in range(R)
        if resident[e] != r and q[e][r] > 0
    ):
        if q[e][r] != v or v == 0:
            continue
        targets = [resident[e]] + sorted(
            (t for t in range(R) if t != r and t != resident[e] and q[e][t] > 0),
            key=lambda t: (load[t], t),
        )
        for t in targets:
            if load[t] + v <= max_load:
                q[e][t] += v
                q[e][r] = 0
                load[t] += v
                load[r] -= v
                break
    # Single-block moves between the extremes while they share an expert.
    for _ in range(E * R):
        hi = max(range(R), key=lambda r: (load[r], -r))
        lo = min(range(R), key=lambda r: (load[r], r))
        if load[hi] - load[lo] <= 1:
            break
        shared = [e for e in range(E) if q[e][hi] > 0 and q[e][lo] > 0]
        if not shared:
            break
        q[shared[0]][hi] -= 1
        q[shared[0]][lo] += 1
        load[hi] -= 1
        load[lo] += 1
    slot_expert = [list(prev_slot_expert[r]) for r in range(R)]
    for r in range(R):
        needed = [e for e in range(E) if resident[e] != r and q[e][r] > 0]
        assert len(needed) <= S
        for s in range(S):
            if slot_expert[r][s] not in needed:
                slot_expert[r][s] = -1
        free = [s for s in range(S) if slot_expert[r][s] < 0]
        for e in needed:
            if e not in slot_expert[r]:
                slot_expert[r][free.pop(0)] = e
    copies = []
    for e in range(E):
        acc, cum = 0, []
        for r in [resident[e]] + [
            r for r in range(R) if r != resident[e] and q[e][r] > 0
        ]:
            acc += q[e][r]
            cum.append((r, acc))
        copies.append(cum)
    return q, slot_expert, copies, load


def _remap_reference(copies, slot_expert, counts, block_m, topk_idx_per_rank):
    """Test-only serial oracle of the remap: physical slot of every routing
    entry. Tokens of an expert are ordered by source rank then position; the
    ordinal's block selects the first copy whose bound exceeds it."""
    R, E = len(counts), len(counts[0])
    H, S = E // R, len(slot_expert[0])
    out = []
    for r in range(R):
        base = [sum(counts[rr][e] for rr in range(r)) for e in range(E)]
        seen = [0] * E
        rows = []
        for row in topk_idx_per_rank[r]:
            new_row = []
            for e in row:
                if e < 0:
                    new_row.append(e)
                    continue
                block = (base[e] + seen[e]) // block_m
                seen[e] += 1
                dest = copies[e][-1][0]
                for rank, cum in copies[e]:
                    if block < cum:
                        dest = rank
                        break
                local = (
                    e - dest * H if dest == e // H else H + slot_expert[dest].index(e)
                )
                new_row.append(dest * (H + S) + local)
            rows.append(new_row)
        out.append(rows)
    return out


def _device_copies(copies):
    """Device table ``[E, R, 2]`` to the reference's list form."""
    out = []
    for row in copies.tolist():
        entries = []
        for rank, cum in row:
            if rank < 0:
                break
            entries.append((rank, cum))
        out.append(entries)
    return out


def _expected_stats(idx, counts, copies, slot_expert, block_m, rank):
    """Receive count per local physical slot of ``rank`` that the tables
    prescribe, from the remap oracle over every rank's routing."""
    R, E = counts.shape
    H, S = E // R, len(slot_expert[0])
    rows = _gather(idx).view(R, *idx.shape).cpu().tolist()
    physical = _remap_reference(copies, slot_expert, counts.tolist(), block_m, rows)
    received = torch.zeros(R * (H + S), dtype=torch.int64)
    for r in range(R):
        for row in physical[r]:
            for p in row:
                if p >= 0:
                    received[p] += 1
    return received[rank * (H + S) : (rank + 1) * (H + S)].cuda()


def _planner_worker(rank, args):
    geometry, buffer, slots, generator, hot, run, reference = _setup(
        rank, args, seed=104729, hot_seed=31, num_hot=args.slots + 1
    )
    ep, h, E, topk, S, H = geometry

    def tables():
        torch.cuda.synchronize()
        return (
            slots.slot_expert.cpu().tolist(),
            _device_copies(slots.copies.cpu()),
            slots.expert_base.cpu().tolist(),
            int(slots.copy_count.item()),
            slots.copy_list.cpu().tolist(),
        )

    def check_plan(prev, counts, block_m, open_gate_blocks, label):
        slot_expert, copies, expert_base, copy_count, copy_list = tables()
        # Identical tables on every rank.
        for mine in (slots.slot_expert, slots.copies):
            everyone = _gather(mine).view(ep, *mine.shape)
            assert torch.equal(everyone, mine.expand_as(everyone)), (
                label,
                "tables differ across ranks",
            )
        # Structural invariants.
        fixed_load = [0] * ep
        planned_load = [0] * ep
        for e in range(E):
            blocks = _ceil_div(int(counts[:, e].sum()), block_m)
            fixed_load[e // H] += blocks
            assert copies[e][0][0] == e // H, (label, e, copies[e])
            prev_cum = 0
            for r, cum in copies[e]:
                assert cum >= prev_cum, (label, e, copies[e])
                planned_load[r] += cum - prev_cum
                prev_cum = cum
                assert r == e // H or e in slot_expert[r], (label, e, r, slot_expert[r])
            assert prev_cum == blocks, (label, e, copies[e], blocks)
        for r in range(ep):
            held = [e for e in slot_expert[r] if e >= 0]
            assert len(held) == len(set(held)) <= S and all(
                e // H != r for e in held
            ), (label, r, held)
        assert expert_base == counts[:rank].sum(0).tolist(), label
        assert not slots.local_counter.cpu().any(), (
            label,
            "ordinal counters not cleared",
        )
        assert max(planned_load) <= max(fixed_load), (label, planned_load, fixed_load)
        # The copy list is exactly the slots of this rank whose expert changed.
        changed = sorted(
            (s, e)
            for s, e in enumerate(slot_expert[rank])
            if e >= 0 and e != prev[rank][s]
        )
        assert sorted(tuple(x) for x in copy_list[:copy_count]) == changed, (
            label,
            copy_list,
            changed,
        )
        # Bit for bit against the serial reference.
        ref_blocks, ref_slots, ref_copies, ref_loads = _plan_reference(
            counts.tolist(), S, block_m, prev, open_gate_blocks
        )
        assert slot_expert == ref_slots, (label, slot_expert, ref_slots)
        assert copies == ref_copies, (
            label,
            [
                (e, copies[e], ref_copies[e])
                for e in range(E)
                if copies[e] != ref_copies[e]
            ][:4],
        )
        assert planned_load == ref_loads, label
        return slot_expert, copies, max(planned_load), max(fixed_load)

    for m in args.tokens:
        x = (torch.randn(m, h, device="cuda", generator=generator) * 32).to(
            torch.float8_e4m3fn
        )
        sf = torch.full((m, h // 128), 1 / 128, device="cuda")
        idx, routing = _skewed_routing(m, E, topk, hot, 0.6, generator)
        expected, _ = reference(x, idx, routing)
        counts = _histograms(idx, E)
        block_m = slots.block_m(m)
        y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
        stats = torch.zeros(H + S, dtype=torch.int32, device="cuda")

        # Warm plan from whatever the previous token count left in the slots.
        prev = slots.slot_expert.cpu().tolist()
        generation = int(slots.plan_generation.item())
        buffer.stage(x, sf, idx, routing)
        slots.plan_and_remap(m, block_m)
        slot_expert, copies, planned, fixed = check_plan(
            prev, counts, block_m, 0, ("warm", m)
        )
        assert int(slots.plan_generation.item()) == generation + 1
        stats_expected = _expected_stats(
            idx, counts, copies, slot_expert, block_m, rank
        )
        run(y, stats)
        torch.cuda.synchronize()
        dist.barrier()
        _assert_close(y, expected, ("warm", m))
        assert torch.equal(stats.long(), stats_expected), (
            "warm",
            m,
            stats,
            stats_expected,
        )
        _assert_slots_hold_experts(slots)
        assert int(slots.copy_count.item()) == 0
        first = y.clone()

        # The same routing planned again: identical tables, nothing to copy,
        # bitwise-equal output.
        buffer.stage(x, sf, idx, routing)
        slots.plan_and_remap(m, block_m)
        again = check_plan(slot_expert, counts, block_m, 0, ("repeat", m))
        assert again[:2] == (slot_expert, copies) and int(slots.copy_count.item()) == 0
        run(y, stats)
        torch.cuda.synchronize()
        assert torch.equal(y, first), "replanning the same routing changed the output"

        # Cold plan: every slot forgotten, so every assigned slot is listed and
        # the kernel refills all of them before their tasks. The split may
        # differ from the warm plan, so the counts follow the cold tables.
        slots.slot_expert.fill_(-1)
        prev = [[-1] * S for _ in range(ep)]
        buffer.stage(x, sf, idx, routing)
        slots.plan_and_remap(m, block_m)
        cold = check_plan(prev, counts, block_m, 0, ("cold", m))
        assert int(slots.copy_count.item()) == sum(1 for e in cold[0][rank] if e >= 0)
        stats_expected = 2 * stats_expected + _expected_stats(
            idx, counts, cold[1], cold[0], block_m, rank
        )
        run(y, stats)
        torch.cuda.synchronize()
        dist.barrier()
        _assert_slots_hold_experts(slots)
        assert torch.equal(stats.long(), stats_expected), ("cold", m)
        _assert_close(y, expected, ("cold", m))

        # Gate every fresh slot closed: no replica, every expert resident.
        slots.slot_expert.fill_(-1)
        buffer.stage(x, sf, idx, routing)
        slots.plan_and_remap(m, block_m, open_gate_blocks=1 << 20)
        gated = check_plan(prev, counts, block_m, 1 << 20, ("gated", m))
        assert all(e < 0 for row in gated[0] for e in row) and gated[2] == gated[3]

        # Plan + kernel under CUDA graph replay: the generation advances on the
        # device, the tables stay put and the output is bitwise stable.
        buffer.stage(x, sf, idx, routing)
        slots.plan_and_remap(m, block_m)
        steady = check_plan(prev, counts, block_m, 0, ("steady", m))[:2]
        steady_expected = _expected_stats(
            idx, counts, steady[1], steady[0], block_m, rank
        )
        run(y, stats)
        torch.cuda.synchronize()
        dist.barrier()
        stats_expected += steady_expected
        steady_y = y.clone()
        generation = int(slots.plan_generation.item())
        for replay in _run_graph(
            buffer, idx, lambda: slots.plan_and_remap(m, block_m), lambda: run(y, stats)
        ):
            stats_expected += steady_expected
            assert torch.equal(y, steady_y), "CUDA graph replay changed the output"
            assert torch.equal(stats.long(), stats_expected), ("replay", m)
            assert int(slots.plan_generation.item()) == generation + 1 + replay
            assert tables()[:2] == steady
        if rank == 0:
            print(
                f"PASS mxfp4 planner ep={ep} S={S} M={m} block_m={block_m} "
                f"busiest rank blocks fixed={fixed} planned={planned}",
                flush=True,
            )
        if m == max(args.tokens):
            assert planned < fixed, "the planner did not lower the busiest rank's load"
    slots.destroy()
    buffer.destroy()
    dist.destroy_process_group()


@pytest.mark.parametrize("ep", [4, 8])
def test_mxfp4_in_kernel_planner(ep):
    if torch.cuda.device_count() < ep or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip(f"requires {ep} Hopper GPUs")
    _run(_namespace(ep), _planner_worker)


# ---------------------------------------------------------------------------
# CPU-only checks
# ---------------------------------------------------------------------------


def test_mxfp4_jit_cache_distinguishes_in_kernel_copy():
    from flashinfer.jit.megamoe import (
        get_megamoe_sm90_eplb_uri,
        get_megamoe_sm90_mxfp4_uri,
    )
    from flashinfer.megamoe import select_mxfp4_fused_plan

    plan = select_mxfp4_fused_plan(132, 8, 288, 6, 4096, 2048, 4096, 1024)
    args = (plan, 132, 8, 4096, 288, 6, 4096, 2048, 10.0, True)
    plain, copy = (
        get_megamoe_sm90_mxfp4_uri(*args),
        get_megamoe_sm90_mxfp4_uri(*args, True),
    )
    assert plain != copy and "mxfp4" in plain
    assert get_megamoe_sm90_eplb_uri(8, 256, 4) != get_megamoe_sm90_eplb_uri(8, 256, 2)
    assert get_megamoe_sm90_eplb_uri(8, 256, 4) != get_megamoe_sm90_eplb_uri(4, 256, 4)


@pytest.mark.parametrize(
    "ranks,experts,slots,physical",
    [
        (8, 255, 4, 287),
        (8, 256, 0, 256),
        (8, 256, 33, 520),
        (8, 256, 4, 256),
        (4, 256, 4, 288),
    ],
)
def test_slot_geometry_is_validated(ranks, experts, slots, physical):
    from flashinfer.megamoe.eplb import _check_geometry

    with pytest.raises(ValueError):
        _check_geometry(ranks, experts, slots, physical)


def test_slot_geometry_accepts_matching_buffer():
    from flashinfer.megamoe.eplb import _check_geometry

    _check_geometry(8, 256, 4, 288)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name, default in vars(_namespace(8)).items():
        if name == "ep":
            parser.add_argument("--ep", type=int, choices=[4, 8], default=default)
        elif isinstance(default, list):
            parser.add_argument(f"--{name}", type=int, nargs="+", default=default)
        else:
            parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument(
        "--planner", action="store_true", help="run the in-kernel planner checks"
    )
    parsed = parser.parse_args()
    _run(parsed, _planner_worker if parsed.planner else _worker)
