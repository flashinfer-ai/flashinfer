"""SM90 (Hopper) dispatch for the MSA ops.

Routes each operation / shape class to the backend that serves it: the Cake
programs of ``cake_hopper_sm90`` or the Hopper CuTe DSL backends in
``cute_dsl/*_sm90.py``. Selection is on capability, never on measured shape
thresholds: a Cake program is chosen where one exists for the class
(``*_route_available``), and the surface's raise rules apply before any
backend is reached.

Cake-served classes:

* sparse decode: every admitted coordinate;
* proxy score, decode regime (``max_seqlen_q <= 4``): fp8 and bf16 q with a
  one-head index cache, Hq in {1, 2, 4};
* proxy score, prefill regime: fp8 q with a one-head index cache, in both
  accumulation precisions (``use_fp32_acc``);
* top-k select: every (Hq, max_k_tiles, total_q) whose planner coordinate
  (columns per CTA, tile groups, index bits, mask form) is a delivered
  program;
* sparse prefill: GQA groups 4, 8 and 16 with at most 4096 pages per sequence.

Both sparse routes serve bf16 ``q`` only: the Cake programs reject other
dtypes and the CuTe DSL prefill kernels type their q / out pointers bf16
without checking, so fp16 raises at the dispatch instead of being misread.

The remaining classes keep the CuTe DSL kernels (proxy-score decode of a bf16
index cache for Hq outside {1, 2, 4} or with several index heads, top-k
planner coordinates without a program, sparse prefill of GQA groups 1 / 2 /
32 / 64 or above 4096 pages).  The fp8 proxy-score CuTe kernels re-view the
index cache with a one-head page stride and select no KV head, so an fp8
index cache with several heads raises in both regimes instead of scoring the
wrong pages. Both proxy-score prefill backends support f32 accumulation by
default and explicit f16 accumulation through ``use_fp32_acc=False``.
"""

import functools
from typing import Iterable, Optional

import torch

_BLK_KV = 128

# The SM90 sparse-prefill union schedule is correct only while a sequence's
# page list stays small: measured exact at max_pages_per_seq 64/320/560 and
# WRONG at 1408/2344/3520 (~8% of elements outside tolerance -- it does not
# fail, it silently mis-attends). Above the bound the one-token-per-CTA
# schedule is used instead: correct on all three, 1.70-1.98x over Triton.
# The campaign dispatcher keyed on TOTAL pages, which is why it routed these
# shapes into the bad path. 800 keeps margin below the measured cliff.
_MPG_UNION_MAX = 800
# Per-sequence metadata (page table, prefix, seqused, cu_seqlens) is read
# through fixed 16-entry views inside the kernels.
_MAX_BATCH = 16


@functools.lru_cache(maxsize=64)
def is_sm90a_device(device: torch.device) -> bool:
    """``flashinfer.utils.is_sm90a_supported`` resolved once per device.

    The helper re-parses the CUDA version string (two ``packaging.version.Version``
    constructions) on every call; the public MSA entries call it per op, i.e.
    three times per MiniMax-M3 layer on the decode path.

    A non-CUDA device is "not SM90" rather than an error, like
    ``is_blackwell_msa_device``: the surfaces consult this predicate before
    their own raise rules (``out=``, dtype), and ``get_compute_capability``
    raises ``ValueError`` for a host tensor, which would pre-empt them.
    """
    if device.type != "cuda":
        return False
    from ..utils import is_sm90a_supported

    return bool(is_sm90a_supported(device))


@functools.lru_cache(maxsize=64)
def _sm_count(device: torch.device) -> int:
    from ..utils import get_device_sm_count

    return int(get_device_sm_count(device))


def _i32(t: torch.Tensor) -> torch.Tensor:
    """``t.to(torch.int32).contiguous()`` without the two no-op dispatches when ``t`` already is (same object)."""
    if t.dtype == torch.int32 and t.is_contiguous():
        return t
    return t.to(torch.int32).contiguous()


# Interleaved-K/V admission memoized per geometry.  Keys are plain tuples (never tensors): shape, strides, the byte
# distance of v from k, element size and dtypes.  Equal geometry means the same contract outcome: with equal strides
# and v starting ``head_dim`` elements after k inside k's first row, the two views interleave element-wise, so they
# can only be halves of one allocation -- the storage identity ``_as_packed_kv`` probes is implied.
_PACKED_KV_ADMITTED: set = set()


def _check_packed_kv(k: torch.Tensor, v: torch.Tensor) -> None:
    """The layout contract of ``_as_packed_kv`` (raises identically), resolved once per K/V geometry."""
    key = (
        tuple(k.shape),
        tuple(v.shape),
        k.stride(),
        v.stride(),
        v.data_ptr() - k.data_ptr(),
        k.element_size(),
        k.dtype,
        v.dtype,
    )
    if key in _PACKED_KV_ADMITTED:
        return
    _as_packed_kv(k, v)
    if len(_PACKED_KV_ADMITTED) >= 64:
        _PACKED_KV_ADMITTED.clear()
    _PACKED_KV_ADMITTED.add(key)


def _prefix_lens(cu_seqlens_q: torch.Tensor, seqused_k: torch.Tensor) -> torch.Tensor:
    """Tokens already resident before this chunk, per sequence.

    The proxy kernels align the causal mask on it, which is what ``q_offset``
    means for a chunked prefill: seqused_k counts the whole sequence, so the
    query rows in this call sit at the end of it.
    """
    qlen = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
    return (seqused_k.to(torch.int32) - qlen).contiguous()


def _fold_scales(q, out_scale, softmax_scale, head_dim):
    """Fold vLLM's softmax/dequant scales into tensors the SM90 kernels accept.

    The sparse kernels hardcode 1/sqrt(D) and take no scale argument, so an
    fp8 KV cache -- where vLLM folds k_scale into softmax_scale and passes
    v_scale as v_global_scale -- cannot be served by passing them through.
    Pre-scaling q by softmax_scale*sqrt(D) makes the kernel's fixed 1/sqrt(D)
    reproduce the requested scale exactly; the v scale is applied to the output
    by the caller. Returns q unchanged when the requested scale already is the
    kernel's own, so the common bf16 path costs nothing.
    """
    if softmax_scale is None:
        return q
    default = head_dim**-0.5
    if abs(softmax_scale - default) <= 1e-9 * max(1.0, abs(default)):
        return q
    return q * (softmax_scale / default)


def preload_sm90_programs(kinds: Optional[Iterable[str]] = None) -> dict[str, int]:
    """Build and load the Cake Hopper MSA programs of the op kinds ``kinds``
    (default: all) now, outside CUDA graph capture.

    Explicit opt-in.  By default each program is JIT-built at the first eager
    use of its route, and a program first needed while a CUDA graph is being
    captured raises instead of building inside the capture; a per-geometry
    eager warm-up before capture (vLLM's capture loop) therefore needs no
    preloading.  Frameworks that capture geometries they never ran eagerly,
    or that want the build cost at model load, call this once per process.
    Kinds: ``sparse_decode``, ``proxy_decode``, ``proxy_prefill``,
    ``topk_select``, ``sparse_prefill``; about 4 s per program not yet in the
    JIT cache (89 programs in all), ~10 ms per prebuilt one.  Returns the
    programs loaded per kind.
    """
    from .cake_hopper_sm90 import preload_programs

    return preload_programs(kinds)


def proxy_score_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    per_head: torch.Tensor,
    *,
    max_seqlen_q: int,
    batch_size: int,
    kv_fp8: bool,
    q_offset: Optional[torch.Tensor] = None,
    use_fp32_acc: bool = True,
) -> torch.Tensor:
    """MSA proxy score on Hopper. Writes ``per_head`` (Hq, max_k_tiles, total_q).

    ``use_fp32_acc`` selects the accumulation precision of the prefill regime
    (exact-product f32 by default, f16 like the CuTe DSL kernel when False);
    the decode-regime programs accumulate in f32 whichever value is given.
    """
    total_q = q.shape[0]
    sq = total_q // batch_size if batch_size else 0
    decode = (
        max_seqlen_q <= 4
        and batch_size * max_seqlen_q == total_q
        and sq == max_seqlen_q
    )

    cu = _i32(cu_seqlens_q)
    pt = _i32(page_table)
    sk = _i32(seqused_k)

    # The decode/prefill schedules read the index cache with a fixed element type,
    # so a mismatched q or k dtype is read as the wrong bits rather than rejected.
    if kv_fp8:
        if k.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"SM90 proxy-score fp8 path requires an fp8 e4m3 k cache, got {k.dtype}"
            )
        if q.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                f"SM90 proxy-score fp8 path requires fp8 e4m3 q, got {q.dtype}"
            )
    else:
        if k.dtype != torch.bfloat16 or q.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"SM90 proxy-score bf16 path requires bf16 q and k, got "
                f"q={q.dtype} k={k.dtype}"
            )

    if decode:
        if kv_fp8:
            # The fp8 decode schedules index a log2 table keyed on Hq.
            if q.shape[1] not in (1, 2, 4):
                raise NotImplementedError(
                    f"SM90 fp8 proxy decode supports Hq in (1, 2, 4), got {q.shape[1]}"
                )
        from .cake_hopper_sm90 import (
            hopper_msa_proxy_score_decode,
            proxy_decode_route_available,
        )

        # Cake programs serve fp8 and bf16 decode scoring of a one-head index
        # cache for Hq in (1, 2, 4) and max_seqlen_q in (1, 2, 3, 4); bf16 Hq
        # outside that set and bf16 multi-head index caches keep the CuTe DSL
        # decode schedules.
        if not _cute_proxy_decode_preferred(
            q, k, kv_fp8
        ) and proxy_decode_route_available(
            q_dtype=q.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k.shape[1],
            max_seqlen_q=max_seqlen_q,
        ):
            return hopper_msa_proxy_score_decode(
                q,
                k,
                page_table=pt,
                seqused_k=sk,
                per_head=per_head,
                max_seqlen_q=max_seqlen_q,
                batch_size=batch_size,
                q_offset=(_i32(q_offset) if q_offset is not None else None),
            )
        if kv_fp8:
            if k.shape[1] != 1:
                # The fp8 CuTe decode schedule re-views the cache as consecutive
                # one-head pages ((64, 128, 2 * num_pages) with a fixed half-page
                # stride) and its grid has no KV-head axis: with several index
                # heads every head would score the wrong pages.
                raise NotImplementedError(
                    "SM90 fp8 proxy-score decode serves a one-head index cache; "
                    f"got {k.shape[1]} index heads"
                )
            from .cute_dsl.proxy_score_decode_sm90 import run as _decode
        else:
            from .cute_dsl.proxy_score_decode_bf16_sm90 import run as _decode
        _decode(q, k, cu, pt, sk, per_head)
        return per_head

    if not kv_fp8:
        raise NotImplementedError(
            "SM90 proxy-score prefill requires an fp8 e4m3 index cache; "
            "bf16 is supported for decode only"
        )
    from .cake_hopper_sm90 import (
        hopper_msa_proxy_score_prefill,
        proxy_prefill_route_available,
    )

    # Cake programs serve fp8 prefill scoring of a one-head index cache in
    # both accumulation precisions; the CuTe DSL kernel serves the same
    # one-head class in both precisions, so several index heads raise in both.
    if proxy_prefill_route_available(
        q_dtype=q.dtype, num_kv_heads=k.shape[1], use_fp32_acc=use_fp32_acc
    ):
        return hopper_msa_proxy_score_prefill(
            q,
            k,
            cu,
            page_table=pt,
            seqused_k=sk,
            per_head=per_head,
            max_seqlen_q=max_seqlen_q,
            batch_size=batch_size,
            q_offset=(_i32(q_offset) if q_offset is not None else None),
            use_fp32_acc=use_fp32_acc,
        )
    if k.shape[1] != 1:
        # The CuTe prefill kernel rebuilds K as (page, d, num_pages) with a
        # one-head page stride and its grid has no KV-head axis.
        raise NotImplementedError(
            "SM90 proxy-score prefill "
            f"serves a one-head index cache; got {k.shape[1]} index heads"
        )
    from .cute_dsl.proxy_score_prefill_sm90 import run as _prefill

    pfx = (
        q_offset.to(torch.int32).contiguous()
        if q_offset is not None
        else _prefix_lens(cu, sk)
    )
    _prefill(
        q,
        k,
        cu,
        pt,
        sk,
        pfx,
        per_head,
        max_seqlen_q=max_seqlen_q,
        use_fp32_acc=use_fp32_acc,
    )
    return per_head


def topk_select_sm90(
    max_score: torch.Tensor,
    topk: int,
    output: torch.Tensor,
    *,
    num_valid_pages=None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor:
    """MSA top-k block selection on Hopper. Writes sorted indices into ``output``.

    The kernels read validity from the ``-inf`` tiles ``msa_proxy_score`` already
    writes, so they need no sequence-length arguments.
    """
    if topk != 16:
        raise NotImplementedError(
            f"SM90 msa_topk_select supports topk=16 only, got {topk}"
        )
    # Forced blocks and per-token validity are applied inside the kernels at the
    # score load, so they cost no extra traffic and no extra launch. Biasing
    # max_score in a separate pass instead measured 431us against a 6.8us kernel.
    nvp = None
    if num_valid_pages is not None and not (
        isinstance(num_valid_pages, int) and num_valid_pages == max_score.shape[1]
    ):
        if not isinstance(num_valid_pages, torch.Tensor):
            # Neither the Cake programs nor the CuTe DSL kernel read a
            # batch-wide count: both take the per-token (total_q,) int32
            # tensor at the score load. A scalar equal to the score width is
            # the unclamped case and is accepted above.
            raise NotImplementedError(
                "SM90 msa_topk_select takes num_valid_pages as a per-token "
                f"(total_q,) int32 tensor; got the scalar {num_valid_pages} for "
                f"{max_score.shape[1]} score columns"
            )
        nvp = _i32(num_valid_pages)

    hq, tiles, total_q = (int(x) for x in max_score.shape)
    from .cake_hopper_sm90 import hopper_msa_topk_select, topk_route_available

    # The Cake program writes the op's (total_q, Hq, topk) layout directly, so
    # it needs neither the permuted view nor the staging copy below.
    if topk_route_available(
        num_heads=hq,
        tiles=tiles,
        total_q=total_q,
        num_sms=_sm_count(max_score.device),
        masked=force_begin_blocks > 0 or force_end_blocks > 0,
        nvp=nvp is not None,
    ):
        return hopper_msa_topk_select(
            max_score,
            topk,
            output,
            num_valid_pages=nvp,
            force_begin_blocks=force_begin_blocks,
            force_end_blocks=force_end_blocks,
        )

    from .cute_dsl.topk_select_sm90 import run as _topk

    # The CuTe kernel emits (Hq, total_q, topk); this op returns (total_q, Hq, topk).
    # Permuting the caller's buffer gives the kernel's view for free whenever that
    # view is contiguous -- always so for the MQA indexer (Hq == 1), which is the
    # MiniMax-M3 path. Staging through a temporary instead costs an allocation and
    # a copy launch, ~12us against a ~7us kernel, so only pay it when forced.
    view = output.permute(1, 0, 2)
    if view.is_contiguous():
        _topk(max_score, None, None, view, nvp, force_begin_blocks, force_end_blocks)
        return output
    tmp = torch.empty((hq, total_q, topk), dtype=torch.int32, device=max_score.device)
    _topk(max_score, None, None, tmp, nvp, force_begin_blocks, force_end_blocks)
    output.copy_(tmp.permute(1, 0, 2))
    return output


def _as_packed_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: Optional[int] = None,
    *,
    require_dense: bool = False,
) -> torch.Tensor:
    """Recover the (num_pages, Hkv, page, 2*D) cache the SM90 decode kernel reads.

    vLLM stores MSA K and V interleaved in one allocation and hands out halves;
    when that is what we were given the packed view is free. Materializing it
    otherwise would copy the whole cache every call, so refuse instead.
    ``require_dense`` applies the CuTe kernels' stricter stride contract; Cake
    reads the tensor strides and also admits interleaved pool views.
    """
    if k.dim() != 4:
        raise NotImplementedError(
            "SM90 MSA reads a 4D paged KV cache "
            f"(pages, kv_heads, {_BLK_KV}, head_dim), got shape {tuple(k.shape)}"
        )
    if int(k.shape[2]) != _BLK_KV:
        raise NotImplementedError(
            f"SM90 MSA is built around page size {_BLK_KV}, got {int(k.shape[2])}"
        )
    if head_dim is not None and int(k.shape[3]) != int(head_dim):
        raise NotImplementedError(
            f"SM90 MSA KV head_dim {int(k.shape[3])} does not match q head_dim "
            f"{int(head_dim)}"
        )
    if k.dtype != v.dtype or k.shape != v.shape or k.stride() != v.stride():
        raise NotImplementedError("SM90 sparse decode needs matching k/v layouts")
    d = k.shape[-1]
    same_buf = k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
    adjacent = v.data_ptr() - k.data_ptr() == d * k.element_size()
    if not (same_buf and adjacent and k.stride()[-1] == 1):
        raise NotImplementedError(
            "SM90 sparse decode requires K and V interleaved in one cache "
            "(v must be the second half of k's last dim); separate allocations "
            "would need a full-cache copy per call"
        )
    if require_dense:
        # CuTe _views() reconstructs every stride from the shape and ignores
        # the tensor's page/head strides, so its packed layout must be dense.
        want = ((k.shape[1] * _BLK_KV * 2 * d), (_BLK_KV * 2 * d), (2 * d), 1)
        got = tuple(k.stride())
        if got != want:
            raise NotImplementedError(
                "SM90 MSA reads the paged cache with a dense packed layout; needs "
                f"strides {want} for shape {tuple(k.shape)}, got {got}"
            )
    # Preserve k's storage offset: a per-layer KV cache is a view into a pool, so
    # hardcoding 0 would silently read the wrong memory rather than fail.
    base = k.as_strided(
        k.shape[:-1] + (2 * d,), k.stride()[:-1] + (1,), k.storage_offset()
    )
    if base.data_ptr() != k.data_ptr():
        raise NotImplementedError(
            "SM90 sparse attention could not rebuild the packed KV view "
            "(reconstructed base does not start at k)"
        )
    return base


def _cake_decode_available(q, k, v) -> bool:
    """Whether the Cake decode program admits this geometry.

    Mirrors its ``_admit_sparse_decode`` checks so an unadmitted shape picks the
    CuTe DSL schedule instead of raising: head_dim and page size are fixed, q is
    contiguous bf16 and the paged cache is fp8 e4m3 with matching k/v shapes.
    """
    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        return False
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        return False
    if k.ndim != 4 or k.shape != v.shape:
        return False
    return int(k.shape[2]) == _BLK_KV and int(k.shape[3]) == int(q.shape[-1])


def sparse_decode_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    *,
    seqlen_q: int,
    softmax_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
) -> torch.Tensor:
    """MSA sparse decode attention on Hopper. Writes ``out``.

    Served by the Cake Hopper decode program: it takes ``softmax_scale`` as a
    kernel parameter (no q pre-scaling launch) and reads K and V through their
    own strides, so the interleaved halves are passed as given once the
    surface's layout contract (``_as_packed_kv``) has admitted them.  Query
    ``i`` of a sequence attends at position ``seqused_k - seqlen_q + i``
    (right-aligned causal, the surface's decode semantics) with the caller's
    ``seqlen_q``, which the program checks against ``q``, ``seqused_k`` and
    ``page_table``; the public entry rejects ``causal=False`` and ``q_offset``
    and validates the paged metadata before reaching this function.  The
    program consumes bf16 ``q`` only, so any other ``q`` dtype raises here.

    The Cake program is the default: it is 2-10% faster than the CuTe DSL
    schedule on every measured shape and needs no dense-packed rebuild. The
    CuTe schedule is kept as the fallback for geometries Cake does not admit,
    so this op is not single-sourced.
    """
    from .cake_hopper_sm90 import hopper_msa_sparse_decode_attention

    if q.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"SM90 msa_sparse_decode_attention serves bf16 q; got {q.dtype}"
        )
    _check_packed_kv(k, v)
    if _cake_decode_available(q, k, v):
        hopper_msa_sparse_decode_attention(
            q,
            k,
            v,
            _i32(q2k_indices),
            page_table=_i32(page_table),
            seqused_k=_i32(seqused_k),
            seqlen_q=seqlen_q,
            softmax_scale=softmax_scale,
            out=out,
        )
    else:
        from .cute_dsl.sparse_decode_sm90 import run as _decode

        _decode(
            q,
            _as_packed_kv(k, v),
            _i32(q2k_indices),
            _i32(page_table),
            _i32(seqused_k),
            out,
            softmax_scale=softmax_scale,
        )
    if v_global_scale is not None and v_global_scale != 1.0:
        out.mul_(v_global_scale)
    return out


def _cute_proxy_decode_preferred(q, k, kv_fp8) -> bool:
    """Whether to serve proxy-score decode from the CuTe DSL schedule.

    Measured on H100 against the Cake program (kernel time, outputs agreeing):
    0.86-0.90x at batch 1-8 and 1.00x from batch 8 up, i.e. tie-or-better at
    every size measured. Only the fp8 one-head schedule was covered, so bf16
    and multi-head index caches stay on Cake.
    """
    return bool(kv_fp8) and int(k.shape[1]) == 1


def _cute_prefill_preferred(q, k, v, page_table) -> bool:
    """Whether the CuTe DSL sparse-prefill schedules can serve this call.

    Measured on H100 against the Cake program at the chunk sizes serving
    engines actually use (kernel time, outputs agreeing): 0.58x at 8192 tokens
    over a 16k context, 0.82x at 68k, 0.92x at 256k. At 2048 tokens the two are
    within 11% either way. Cake keeps everything these schedules cannot serve.
    """
    if int(page_table.shape[0]) > _MAX_BATCH:
        return False
    if int(q.shape[1]) // max(1, int(k.shape[1])) not in (8, 16):
        return False
    try:
        _as_packed_kv(k, v, q.shape[-1], require_dense=True)
    except NotImplementedError:
        return False
    return True


def sparse_prefill_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    *,
    q_offset: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
) -> torch.Tensor:
    """MSA sparse prefill attention on Hopper. Writes ``out``.

    The Cake programs take ``softmax_scale`` as a kernel parameter, derive the
    query positions from ``seqused_k`` in the kernel when no ``q_offset`` is
    given, and read K and V through their own strides, so they allocate
    nothing per call (CUDA-graph capturable as is).  Query ``i`` of sequence
    ``b`` attends the keys of its selected blocks at positions
    ``<= q_offset[b] + i``.  Every SM90 sparse-prefill schedule reads ``q``
    and writes ``out`` as bf16 -- the Cake programs reject other dtypes and
    the CuTe DSL kernels type the raw pointers bf16 without checking -- so
    fp16 raises here, before a route is chosen.
    """
    from .cake_hopper_sm90 import hopper_msa_sparse_attention, prefill_route_available

    if q.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"SM90 msa_sparse_attention serves bf16 q; got {q.dtype}"
        )
    if not _cute_prefill_preferred(q, k, v, page_table) and prefill_route_available(
        num_q_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        max_pages=int(page_table.shape[1]),
    ):
        _check_packed_kv(k, v)
        hopper_msa_sparse_attention(
            q,
            k,
            v,
            _i32(q2k_indices),
            _i32(cu_seqlens_q),
            page_table=_i32(page_table),
            seqused_k=_i32(seqused_k),
            out=out,
            q_offset=(_i32(q_offset) if q_offset is not None else None),
            softmax_scale=softmax_scale,
        )
        if v_global_scale is not None and v_global_scale != 1.0:
            out.mul_(v_global_scale)
        return out

    if int(page_table.shape[1]) > _MPG_UNION_MAX:
        # The single-token copy-out moves g * (D // 8) vectors with one
        # warpgroup, so g < 8 gives the loop zero iterations and never writes
        # out. The union kernel sizes that loop by MMA_M, so it takes any g.
        g = int(q.shape[1]) // int(k.shape[1])
        if g not in (8, 16):
            raise NotImplementedError(
                f"SM90 MSA single-token prefill (page_table width "
                f"{int(page_table.shape[1])} > {_MPG_UNION_MAX}) supports "
                f"Hq/Hkv of 8 or 16, got {g}"
            )
        from .cute_dsl.sparse_prefill_single_sm90 import run as _prefill
    else:
        from .cute_dsl.sparse_prefill_sm90 import run as _prefill

    if int(page_table.shape[0]) > _MAX_BATCH:
        raise NotImplementedError(
            f"SM90 MSA prefill indexes per-sequence metadata with a fixed extent "
            f"of {_MAX_BATCH} sequences, got {int(page_table.shape[0])}"
        )
    kv = _as_packed_kv(k, v, q.shape[-1], require_dense=True)
    cu = cu_seqlens_q.to(torch.int32).contiguous()
    sk = seqused_k.to(torch.int32).contiguous()
    pfx = (
        q_offset.to(torch.int32).contiguous()
        if q_offset is not None
        else _prefix_lens(cu, sk)
    )
    _prefill(
        q,
        kv,
        q2k_indices.to(torch.int32).contiguous(),
        cu,
        page_table.to(torch.int32).contiguous(),
        sk,
        pfx,
        out,
        softmax_scale=softmax_scale,
    )
    if v_global_scale is not None and v_global_scale != 1.0:
        out.mul_(v_global_scale)
    return out
