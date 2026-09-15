"""MSA proxy-score bf16 decode schedule for SM90 (Hopper).

The mma_fallback variant of the Kernel-Factory bf16 decode winner: 1.57x over
the shipped Triton decode on held-out shapes, and the only variant of that
campaign that held up under independent verification (the campaign's routed
variants failed held-out shapes the campaign itself had passed).

Used when the index cache is bf16; the fp8 cache path is faster and preferred
where available -- see proxy_score_decode_sm90.
"""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cute.runtime import from_dlpack

HD = 128   # head dim
PG = 128   # page size == kv block size
NT = 128   # threads per CTA (4 warps)
NNT = 4    # mma n-tiles (8 kv tokens each) per warp
NCH = 4    # 128-bit chunks along the head dim
TPW = 32   # kv tokens per warp


@cute.kernel
def _msa_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mCu: cute.Tensor,
    mPT: cute.Tensor,
    mSK: cute.Tensor,
    mO: cute.Tensor,
    KPS: cutlass.Constexpr,
    KHS: cutlass.Constexpr,
    G: cutlass.Constexpr,
    SQ: cutlass.Constexpr,
    TPB: cutlass.Constexpr,
):
    M = G * SQ

    tid, _, _ = cute.arch.thread_idx()
    bx, b, hkv = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = cute.arch.lane_idx()
    R = lane // 4
    j = lane % 4

    NEG = cutlass.Float32(-cutlass.Float32.inf)
    sk = mSK[b]
    lo = mCu[b]
    nb = (sk + (PG - 1)) // PG
    mkt = mPT.shape[1]

    # rA is filled on the first in-range tile only; declared here so it stays
    # live across the tile sweep.
    rA = cute.make_rmem_tensor((8, NCH, 2), cutlass.BFloat16)

    smem = cutlass_utils.SmemAllocator()
    sQb = smem.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((16, HD), stride=(HD, 1)), byte_alignment=16
    )
    sR = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((4, 16), stride=(16, 1)), byte_alignment=16
    )

    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
    atom = cute.make_mma_atom(op)

    # A CTA sweeps TPB consecutive kv blocks.  CTA launch throughput on Hopper
    # is only ~1-2 blocks per clock for the whole GPU, and up to 4.7x of the
    # (max_k_tiles, B) rectangle is past the end of a ragged sequence, so on the
    # most ragged shapes the launch of blocks that only store -inf was costing
    # ~20% of the runtime.  Sweeping folds them into predicated stores.
    for jj in cutlass.range_constexpr(TPB):
      tblk = bx * TPB + jj
      if cutlass.const_expr(TPB == 1) or tblk < mkt:
        # page read before the seqused_k-dependent branch: always in bounds, and
        # it overlaps the metadata reads instead of chaining behind them.
        page = mPT[b, tblk]
        if tblk < nb:
            # ---- K -> B fragments, straight from global memory (16 x LDG.128).
            pbase = mK.iterator.toint() + (
                cutlass.Int64(page) * cutlass.Int64(KPS)
                + cutlass.Int64(hkv) * cutlass.Int64(KHS)
                + cutlass.Int64(warp) * cutlass.Int64(TPW * HD)
            ) * 2
            kptr = cute.make_ptr(
                cutlass.BFloat16, pbase, cute.AddressSpace.gmem, assumed_align=16
            )
            tileK = cute.make_tensor(
                kptr, cute.make_layout((NNT, 8, NCH, 4, 8), stride=(8 * HD, HD, 32, 8, 1))
            )
            rB = cute.make_rmem_tensor((8, NCH, NNT), cutlass.BFloat16)
            for nt in cutlass.range_constexpr(NNT):
                for c in cutlass.range_constexpr(NCH):
                    # K is pure streaming -- no line is ever revisited -- so mark it
                    # evict-first and leave L1 for the page_table / seqused_k reads,
                    # which every CTA (including the ragged-tail ones that only
                    # store -inf) has to make before it can branch.
                    cute.autovec_copy(
                        tileK[nt, R, c, j, None], rB[None, c, nt],
                        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
                    )

            # ---- Q -> smem, once per CTA (Q is tile invariant).  Issued after the
            # K loads so their latency is already in flight.
            if cutlass.const_expr(jj == 0):
              for r in cutlass.range_constexpr(16):
                  if cutlass.const_expr(r < M):
                      g = r // SQ
                      i = r % SQ
                      sQb[r, tid] = mQ[lo + i, hkv * G + g, tid]
                  else:
                      sQb[r, tid] = cutlass.BFloat16(0.0)
              cute.arch.sync_threads()

              # A fragment: a0a1 = row R cols (p=0,1), a2a3 = row R+8 same, a4a5 =
              # row R cols (p=2,3), a6a7 = row R+8 same -- four 2-element shared
              # loads per k-tile.
              sA = cute.make_tensor(
                  sQb.iterator,
                  cute.make_layout((16, NCH, 4, 2, 2, 2), stride=(HD, 32, 8, 4, 2, 1)),
              )
              for c in cutlass.range_constexpr(NCH):
                for h in cutlass.range_constexpr(2):
                    base = 8 * c + 32 * h
                    for pr in cutlass.range_constexpr(2):
                        cute.autovec_copy(
                            sA[R, c, j, h, pr, None],
                            cute.make_tensor(
                                rA.iterator + (base + 4 * pr), cute.make_layout(2)
                            ),
                        )
                        cute.autovec_copy(
                            sA[R + 8, c, j, h, pr, None],
                            cute.make_tensor(
                                rA.iterator + (base + 4 * pr + 2), cute.make_layout(2)
                            ),
                        )

            acc = cute.make_rmem_tensor((4, NNT), cutlass.Float32)
            acc.fill(0.0)
            for nt in cutlass.range_constexpr(NNT):
                frgC = cute.make_tensor(acc.iterator + 4 * nt, cute.make_layout((4, 1, 1)))
                for c in cutlass.range_constexpr(NCH):
                    for h in cutlass.range_constexpr(2):
                        frgA = cute.make_tensor(
                            rA.iterator + (8 * c + 32 * h), cute.make_layout((8, 1, 1))
                        )
                        frgB = cute.make_tensor(
                            rB.iterator + (4 * h + 8 * c + 32 * nt),
                            cute.make_layout((4, 1, 1)),
                        )
                        cute.gemm(atom, frgC, frgA, frgB, frgC)

            # acc[.,nt] holds rows {R, R+8} x cols {2j, 2j+1} of n-tile nt.
            tok0 = tblk * PG + warp * TPW + 2 * j
            lim_lo = sk - SQ + (R % SQ)
            lim_hi = sk - SQ + ((R + 8) % SQ)
            best0 = NEG
            best1 = NEG
            for nt in cutlass.range_constexpr(NNT):
                tk = tok0 + nt * 8
                v0 = acc[0, nt] if tk <= lim_lo else NEG
                v1 = acc[1, nt] if tk + 1 <= lim_lo else NEG
                v2 = acc[2, nt] if tk <= lim_hi else NEG
                v3 = acc[3, nt] if tk + 1 <= lim_hi else NEG
                best0 = cute.arch.fmax(best0, cute.arch.fmax(v0, v1))
                best1 = cute.arch.fmax(best1, cute.arch.fmax(v2, v3))
            for e in cutlass.range_constexpr(2):
                best0 = cute.arch.fmax(
                    best0, cute.arch.shuffle_sync_bfly(best0, offset=(1 << e))
                )
                best1 = cute.arch.fmax(
                    best1, cute.arch.shuffle_sync_bfly(best1, offset=(1 << e))
                )
            sR[warp, R] = best0
            sR[warp, R + 8] = best1
            cute.arch.sync_threads()
            if tid < M:
                r = sR[0, tid]
                for w in cutlass.range_constexpr(1, 4):
                    r = cute.arch.fmax(r, sR[w, tid])
                mO[hkv * G + tid // SQ, tblk, lo + tid % SQ] = r
            if cutlass.const_expr(TPB > 1):
                cute.arch.sync_threads()
        else:
            if tid < M:
                mO[hkv * G + tid // SQ, tblk, lo + tid % SQ] = NEG


@cute.jit
def _msa_host(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mCu: cute.Tensor,
    mPT: cute.Tensor,
    mSK: cute.Tensor,
    mO: cute.Tensor,
    stream: cuda.CUstream,
    HKV: cutlass.Constexpr,
    KPS: cutlass.Constexpr,
    KHS: cutlass.Constexpr,
    G: cutlass.Constexpr,
    SQ: cutlass.Constexpr,
    TPB: cutlass.Constexpr,
):
    max_k_tiles = mPT.shape[1]
    batch = mCu.shape[0] - 1
    _msa_kernel(mQ, mK, mCu, mPT, mSK, mO, KPS, KHS, G, SQ, TPB).launch(
        grid=[cute.ceil_div(max_k_tiles, TPB), batch, HKV],
        block=[NT, 1, 1],
        stream=stream,
    )


_COMPILED = {}
_STREAM = {}


@torch.no_grad()
def run(q, k, cu_seqlens_q, page_table, seqused_k, max_score):
    total_q = q.shape[0]
    Hq = q.shape[1]
    Hkv = k.shape[1]
    B = cu_seqlens_q.shape[0] - 1
    SQ = total_q // B
    G = Hq // Hkv
    KPS = k.stride(0)
    KHS = k.stride(1)

    # How much of the (max_k_tiles, B) rectangle is past the end of a ragged
    # sequence is known here without touching the device: num_pages is k.shape[0]
    # and every page belongs to exactly one sequence.  Only sweep several tiles
    # per CTA when that ratio is high enough for the saved block launches to beat
    # the loss of cross-tile overlap.
    tiles = B * page_table.shape[1]
    waste = tiles / max(1, k.shape[0])
    tpb = 4 if (tiles >= 2048 and waste >= 2.5) else 1

    raw = torch.cuda.current_stream().cuda_stream
    stream = _STREAM.get(raw)
    if stream is None:
        stream = cuda.CUstream(raw)
        _STREAM[raw] = stream

    key = (Hkv, KPS, KHS, G, SQ, tpb)
    fn = _COMPILED.get(key)
    if fn is None:
        mQ = from_dlpack(q, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=2
        )
        mK = from_dlpack(k, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=3
        )
        mCu = from_dlpack(
            cu_seqlens_q, assumed_align=4, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=0)
        mPT = from_dlpack(
            page_table, assumed_align=4, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=1)
        mSK = from_dlpack(
            seqused_k, assumed_align=4, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=0)
        mO = from_dlpack(
            max_score, assumed_align=16, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=2)
        fn = cute.compile(
            _msa_host,
            mQ, mK, mCu, mPT, mSK, mO, stream,
            Hkv, KPS, KHS, G, SQ, tpb,
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    fn(q, k, cu_seqlens_q, page_table, seqused_k, max_score, stream)



