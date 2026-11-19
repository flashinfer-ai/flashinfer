# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Two-rank ping-pong of one dispatch-sized row: data-as-flag versus flag-after-data.

Rank 0 pushes a 4224-byte row into rank 1's landing slot; rank 1 waits for it and
pushes a row back; repeat. One warp per rank. The round trip divided by two bounds
the one-way publish-to-observe latency of each protocol:

- ``sentinel``: rows are pushed with ``st.volatile.v4``; the receiver polls every
  16-byte vector until no byte is the 0xFF sentinel, checks the payload, and resets
  the slot with 0xFF. Two slots alternate so a reset slot is rewritten only after
  the peer has seen the next reply.
- ``flag``: rows are pushed with weak stores, then ``fence.acq_rel.sys`` and a
  ``st.release.sys`` flag; the receiver polls the flag with ``ld.acquire.sys`` and then
  reads and checks the row.

Run with ``torchrun --nproc-per-node=2 -m flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.litmus.pingpong``.
"""

import os

import torch
import torch.distributed as dist

ROW_BYTES = 4224
SLOT_BYTES = 4352  # row rounded up to 256 B
FLAG_OFFSET = 2 * SLOT_BYTES
BUFFER_BYTES = FLAG_OFFSET + 256


def _build(mode: str):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint32

    from ..comm import ptx

    vectors = ROW_BYTES // 16
    sentinel = mode == "sentinel"

    @cute.jit
    def pattern(iteration: Int32, vector: Int32, word: Int32) -> Uint32:
        # Never contains a 0xFF byte.
        value = Uint32(iteration) * Uint32(0x01010101) + Uint32(
            vector * 4 + word
        ) * Uint32(0x00010001)
        return value & Uint32(0x7F7F7F7F)

    @cute.jit
    def send(peer_slot: Int64, iteration: Int32, lane: Int32):
        for vector in cutlass.range(lane, vectors, 32):
            address = peer_slot + Int64(vector) * Int64(16)
            if cutlass.const_expr(sentinel):
                ptx.st_volatile_v4_words(
                    address,
                    pattern(iteration, vector, Int32(0)),
                    pattern(iteration, vector, Int32(1)),
                    pattern(iteration, vector, Int32(2)),
                    pattern(iteration, vector, Int32(3)),
                )
            else:
                ptx.st_volatile_v4_words(
                    address,
                    pattern(iteration, vector, Int32(0)),
                    pattern(iteration, vector, Int32(1)),
                    pattern(iteration, vector, Int32(2)),
                    pattern(iteration, vector, Int32(3)),
                )

    per_lane = (vectors + 31) // 32

    @cute.jit
    def receive(own_slot: Int64, iteration: Int32, lane: Int32) -> Int32:
        # All of a lane's loads are issued before any is checked, then only vectors
        # still holding the sentinel are reloaded.
        frag = cute.make_rmem_tensor((per_lane * 4,), Uint32)
        for j in cutlass.range_constexpr(per_lane):
            vector = lane + Int32(j * 32)
            if vector < Int32(vectors):
                words = ptx.ld_volatile_v4(own_slot + Int64(vector) * Int64(16))
                for q in cutlass.range_constexpr(4):
                    frag[j * 4 + q] = words[q]
            else:
                for q in cutlass.range_constexpr(4):
                    frag[j * 4 + q] = Uint32(0)
        if cutlass.const_expr(sentinel):
            pending = Int32(1)
            while pending != Int32(0):
                pending = Int32(0)
                for j in cutlass.range_constexpr(per_lane):
                    missing = (
                        ptx.has_ff_byte(frag[j * 4])
                        | ptx.has_ff_byte(frag[j * 4 + 1])
                        | ptx.has_ff_byte(frag[j * 4 + 2])
                        | ptx.has_ff_byte(frag[j * 4 + 3])
                    )
                    if missing:
                        vector = lane + Int32(j * 32)
                        words = ptx.ld_volatile_v4(own_slot + Int64(vector) * Int64(16))
                        for q in cutlass.range_constexpr(4):
                            frag[j * 4 + q] = words[q]
                        pending = Int32(1)
        errors = Int32(0)
        for j in cutlass.range_constexpr(per_lane):
            vector = lane + Int32(j * 32)
            if vector < Int32(vectors):
                for q in cutlass.range_constexpr(4):
                    if frag[j * 4 + q] != pattern(iteration, vector, Int32(q)):
                        errors += Int32(1)
                if cutlass.const_expr(sentinel):
                    ptx.st_volatile_v4_words(
                        own_slot + Int64(vector) * Int64(16),
                        Uint32(0xFFFFFFFF),
                        Uint32(0xFFFFFFFF),
                        Uint32(0xFFFFFFFF),
                        Uint32(0xFFFFFFFF),
                    )
        return errors

    @cute.kernel
    def kernel(
        local: cute.Pointer,
        peer_offset: Int64,
        is_ping: Int32,
        iterations: Int32,
        out: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        base = local.toint()
        peer_base = base + peer_offset
        errors = Int32(0)
        start = ptx.globaltimer_ns()
        for iteration in cutlass.range(iterations):
            slot = Int64(iteration % 2) * Int64(SLOT_BYTES)
            if is_ping == Int32(1):
                send(peer_base + slot, iteration, lane)
                if cutlass.const_expr(not sentinel):
                    cute.arch.sync_warp()
                    if lane == Int32(0):
                        ptx.fence_acq_rel_sys()
                        ptx.st_release_sys_u32(
                            peer_base + Int64(FLAG_OFFSET), Uint32(iteration + 1)
                        )
                    while ptx.ld_acquire_sys_u32(base + Int64(FLAG_OFFSET)) != Uint32(
                        iteration + 1
                    ):
                        pass
                errors += receive(base + slot, iteration, lane)
            else:
                if cutlass.const_expr(not sentinel):
                    while ptx.ld_acquire_sys_u32(base + Int64(FLAG_OFFSET)) != Uint32(
                        iteration + 1
                    ):
                        pass
                errors += receive(base + slot, iteration, lane)
                cute.arch.sync_warp()
                send(peer_base + slot, iteration, lane)
                if cutlass.const_expr(not sentinel):
                    cute.arch.sync_warp()
                    if lane == Int32(0):
                        ptx.fence_acq_rel_sys()
                        ptx.st_release_sys_u32(
                            peer_base + Int64(FLAG_OFFSET), Uint32(iteration + 1)
                        )
            cute.arch.sync_warp()
        stop = ptx.globaltimer_ns()
        if lane == Int32(0):
            out[0] = stop - start
        out[1 + lane] = Int64(errors)

    @cute.jit
    def launch(
        local: cute.Pointer,
        peer_offset: Int64,
        is_ping: Int32,
        iterations: Int32,
        out: cute.Tensor,
    ):
        kernel(local, peer_offset, is_ping, iterations, out).launch(
            grid=[1, 1, 1], block=[32, 1, 1]
        )

    return launch


def main() -> None:
    import cutlass
    import cutlass.cute as cute
    import torch.distributed._symmetric_memory as symm_mem
    from cutlass.cute.runtime import from_dlpack, make_ptr
    from cutlass.cute.typing import AddressSpace

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    buffer = symm_mem.empty(BUFFER_BYTES, dtype=torch.uint8, device="cuda")
    handle = symm_mem.rendezvous(buffer, dist.group.WORLD)
    peer_offset = int(handle.buffer_ptrs[1 - rank]) - int(handle.buffer_ptrs[rank])
    out = torch.zeros(33, dtype=torch.int64, device="cuda")
    iterations = int(os.environ.get("ITERS", "2000"))
    for mode in ("sentinel", "flag"):
        launch = _build(mode)
        compiled = cute.compile(
            launch,
            make_ptr(
                cutlass.Uint8, buffer.data_ptr(), AddressSpace.gmem, assumed_align=256
            ),
            cutlass.Int64(peer_offset),
            cutlass.Int32(rank == 0),
            cutlass.Int32(iterations),
            from_dlpack(out),
        )
        for repeat in range(3):
            buffer.fill_(0xFF)
            buffer[FLAG_OFFSET:].zero_()
            torch.cuda.synchronize()
            dist.barrier()
            compiled(
                make_ptr(
                    cutlass.Uint8,
                    buffer.data_ptr(),
                    AddressSpace.gmem,
                    assumed_align=256,
                ),
                cutlass.Int64(peer_offset),
                cutlass.Int32(rank == 0),
                cutlass.Int32(iterations),
                from_dlpack(out),
            )
            torch.cuda.synchronize()
            if rank == 0:
                total_ns = int(out[0])
                errors = int(out[1:].sum())
                print(
                    f"{mode:8s} repeat {repeat}: one-way {total_ns / iterations / 2 / 1e3:.3f} us "
                    f"({iterations} round trips, {errors} payload errors)",
                    flush=True,
                )
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
