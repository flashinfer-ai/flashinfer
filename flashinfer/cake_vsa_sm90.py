"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Hopper (SM90) variable block-sparse attention on the generated Cake kernel.

``backend="cake"`` of :class:`flashinfer.sparse.VariableBlockSparseAttentionWrapper`
routes here on compute capability 9.0.  Supported scope: BF16 HND ``q/k/v`` with
shape ``(H, S, 128)``, equal query/KV head counts, noncausal 64-token query and
KV blocks, 1..64 selected KV blocks per query block (ragged counts allowed),
custom finite softmax scales (including 0 and negative), no LSE / PDL /
positional encoding / logits soft cap.

The plan is built on the host (:func:`plan_vsa_sm90`) and uploaded once: a per-CTA
list of *tiles* for a persistent kernel (``grid = min(tiles, SMs)``).  A tile is
either one 64-row query block whose KV list alternates between the two consumer
warpgroups (``split``) or two query blocks of one head that share the KV ring
(``pair``); the mode with the smaller modelled makespan wins.  The layout of the
int32 plan rows mirrors the Cake kernel (``loom/examples/weave/vsa_sm90_bf16.py``)
and is validated by the kernel's own tests; :func:`plan_vsa_sm90` is a
dependency-free port of the Cake planner and must stay byte-identical to it.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Optional

import numpy as np
import torch

BLOCK = 64
HEAD_DIM = 128
NUM_K_STAGES = 3
NUM_V_STAGES = 3
NUM_CONSUMER_WGS = 2
MAX_SEQ = 136
MAX_SELECTED = 64
TILE_FIXED_COST = 0.3
LOAD_COST = 1.0
PAIRING_WINDOW = 16
LOG2E = 1.4426950408889634
_FP32_MAX = 3.4028234663852886e38

META_NSEQ = 4
META_NOWN = 5
META_NTILES = 7
META_SEQ_OFF = 8
META_SEQ_WORDS = MAX_SEQ // 2
MAX_OWN = MAX_SEQ // 2
OWN_WORDS = MAX_OWN // 2
META_OWN_OFF = META_SEQ_OFF + META_SEQ_WORDS
META_WORDS = META_OWN_OFF + NUM_CONSUMER_WGS * OWN_WORDS  # 144


# ---------------------------------------------------------------------------
# Host planner (port of loom/examples/weave/vsa_sm90_bf16.py::plan_vsa_sm90)
# ---------------------------------------------------------------------------


def _order_union_rounds(a: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    sa, sb = set(a), set(b)
    i = j = 0
    c = [0, 0]
    emitted: set[int] = set()
    blocks: list[int] = []
    bits: list[int] = []
    while i < len(a) or j < len(b):
        while i < len(a) and a[i] in emitted:
            i += 1
        while j < len(b) and b[j] in emitted:
            j += 1
        if i >= len(a) and j >= len(b):
            break
        if (
            i < len(a)
            and j < len(b)
            and a[i] == b[j]
            or j >= len(b)
            or (i < len(a) and (c[0] < c[1] or (c[0] == c[1] and a[i] < b[j])))
        ):
            blk = a[i]
        else:
            blk = b[j]
        bt = (1 if blk in sa else 0) | (2 if blk in sb else 0)
        emitted.add(blk)
        blocks.append(blk)
        bits.append(bt)
        for w in (0, 1):
            if bt & (1 << w):
                c[w] += 1
    return blocks, bits


def _schedule_feasible(bits: list[int]) -> bool:
    """Exact simulation of the two consumer warpgroups and the two producer rings.

    ``bits[p]`` are the use bits of ring position ``p``.  Mirrors the kernel's
    program order: the K half of a position is released right after its QK
    completes (or, for a partner-only position, as its K tile lands), the V
    half after its PV completes, partner-only positions released in order,
    count-2 releases, and the two producer warps' in-order loads each gated on
    the previous use of its half of the stage.  False when the CTA would
    deadlock.
    """
    n_seq = len(bits)
    own = [
        [p for p, bt in enumerate(bits) if bt & (1 << w)]
        for w in range(NUM_CONSUMER_WGS)
    ]

    def program(w: int) -> list[tuple]:
        o = own[w]
        prog: list[tuple] = []
        prev = -1

        def visit(hi: int) -> None:
            nonlocal prev
            for p in range(prev + 1, hi):
                prog.append(("waitk", p))
                prog.append(("relk", p))
                prog.append(("waitv", p))
                prog.append(("relv", p))
            prev = max(prev, hi - 1)

        n = len(o)
        for j in range(n):
            visit(o[j])
            prog.append(("waitk", o[j]))
            if j >= 1:
                prog.append(("waitv", o[j - 1]))
            prog.append(("relk", o[j]))
            if j >= 1:
                prog.append(("relv", o[j - 1]))
            prev = o[j]
        if n >= 1:
            prog.append(("waitv", o[n - 1]))
            prog.append(("relv", o[n - 1]))
        visit(n_seq)
        prog.append(("done",))
        return prog

    progs = [program(w) for w in range(NUM_CONSUMER_WGS)]
    pc = [0] * NUM_CONSUMER_WGS
    relk = [0] * n_seq
    relv = [0] * n_seq
    loaded = {"k": 0, "v": 0}

    def step(w: int) -> bool:
        op = progs[w][pc[w]]
        if op[0] == "done":
            return False
        if op[0] == "waitk":
            if op[1] >= loaded["k"]:
                return False
        elif op[0] == "waitv":
            if op[1] >= loaded["v"]:
                return False
        elif op[0] == "relk":
            relk[op[1]] += 1
        elif op[0] == "relv":
            relv[op[1]] += 1
        pc[w] += 1
        return True

    while True:
        progressed = False
        for ring, rel in (("k", relk), ("v", relv)):
            depth = NUM_K_STAGES if ring == "k" else NUM_V_STAGES
            while loaded[ring] < n_seq and (
                loaded[ring] < depth or rel[loaded[ring] - depth] >= NUM_CONSUMER_WGS
            ):
                loaded[ring] += 1
                progressed = True
        for w in range(NUM_CONSUMER_WGS):
            while step(w):
                progressed = True
        if all(progs[w][pc[w]][0] == "done" for w in range(NUM_CONSUMER_WGS)):
            return True
        if not progressed:
            return False


def _tile_cost(owns_t) -> float:
    return max(len(o) for o in owns_t) + TILE_FIXED_COST


def _assign_tiles(
    costs: list[float], lens: list[int], sms: int
) -> tuple[list[list[int]], float]:
    """List-schedule the tiles onto ``min(T, sms)`` persistent CTAs.

    A CTA's key is the larger of its consumer time (sum of tile costs) and its
    load time (positions loaded x ``LOAD_COST`` x occupancy ``g / sms``).
    """
    n = len(costs)
    g = max(1, min(n, sms))
    load_unit = LOAD_COST * g / sms
    cons = [0.0] * g
    load = [0.0] * g
    heap = [(0.0, c) for c in range(g)]
    lists: list[list[int]] = [[] for _ in range(g)]
    for t in range(n):
        _, c = heapq.heappop(heap)
        lists[c].append(t)
        cons[c] += costs[t]
        load[c] += lens[t] * load_unit
        heapq.heappush(heap, (max(cons[c], load[c]), c))
    return lists, max(key for key, _ in heap)


def _pairs_for_head(
    hh: int, mode: str, rows, counts, mb: int, ragged: bool
) -> list[tuple[int, int]]:
    if mode != "pair":
        return [(qb, qb) for qb in range(mb)]
    qbs = (
        sorted(range(mb), key=lambda i: -int(counts[hh, i]))
        if ragged
        else list(range(mb))
    )
    if mb > 1:
        sets = [set(rows[hh][qb]) for qb in range(mb)]
        used = [False] * mb
        pairs = []
        for i in range(mb):
            if used[i]:
                continue
            used[i] = True
            best, best_score = -1, None
            for j in range(i + 1, min(mb, i + 1 + PAIRING_WINDOW)):
                if used[j]:
                    continue
                score = (
                    len(sets[qbs[i]] & sets[qbs[j]]),
                    -abs(int(counts[hh, qbs[i]]) - int(counts[hh, qbs[j]])),
                    -j,
                )
                if best_score is None or score > best_score:
                    best, best_score = j, score
            if best < 0:
                pairs.append((qbs[i], qbs[i]))
            else:
                used[best] = True
                pairs.append((qbs[i], qbs[best]))
        return pairs
    return [
        (qbs[i], qbs[i + 1]) if i + 1 < mb else (qbs[i], qbs[i])
        for i in range(0, mb, 2)
    ]


def _pairs_of(lst):
    return [
        (lst[k], lst[k + 1] if k + 1 < len(lst) else -1) for k in range(0, len(lst), 2)
    ]


def _emit_tile(
    hh: int, qb: int, partner: int, plan_mode: str, rows, infos, seqs, owns, lens
) -> None:
    """Tile info word 3: 0 = pair tile, 1 = split tile, 2 = lone block of a pair plan."""
    if partner == qb:
        sps = _pairs_of(rows[hh][qb])
        if plan_mode == "pair":
            order = [(sp, 1) for sp in sps]
            infos.append((hh, qb, qb, 2))
        else:
            order = [(sp, 1 << (k % 2)) for k, sp in enumerate(sps)]
            infos.append((hh, qb, qb, 1))
    else:
        la, lb = rows[hh][qb], rows[hh][partner]
        order = []
        sset = set(la) & set(lb)
        ss = _pairs_of([b for b in la if b in sset])
        sa = _pairs_of([b for b in la if b not in sset])
        sb = _pairs_of([b for b in lb if b not in sset])
        for k in range(max(len(ss), len(sa), len(sb))):
            if k < len(ss):
                order.append((ss[k], 3))
            if k < len(sa):
                order.append((sa[k], 1))
            if k < len(sb):
                order.append((sb[k], 2))
        if not _schedule_feasible([bt for _, bt in order]):
            order = []
        if not order:
            sa, sb = _pairs_of(la), _pairs_of(lb)
            for k in range(max(len(sa), len(sb))):
                if k < len(sa):
                    order.append((sa[k], 1))
                if k < len(sb):
                    order.append((sb[k], 2))
        infos.append((hh, qb, partner, 0))
    if 2 * len(order) > MAX_SEQ:
        raise ValueError("KV sequence exceeds MAX_SEQ")
    bits = [bt for _, bt in order]
    if not _schedule_feasible(bits):
        raise AssertionError("ring schedule infeasible (kernel protocol bug)")
    seqs.append([b for (a, b2), _ in order for b in (a, b2)])
    owns.append(
        [
            [
                (pos << 3) | ((1 if sp[1] >= 0 else 0) << 2) | bt
                for pos, (sp, bt) in enumerate(order)
                if bt & (1 << w)
            ]
            for w in range(NUM_CONSUMER_WGS)
        ]
    )
    lens.append(len(order))


def plan_vsa_sm90(
    block_mask: torch.Tensor, *, sms: int, mode: Optional[str] = None
) -> dict[str, Any]:
    """Turn a boolean ``[H, MB, NB]`` block mask into the kernel's per-CTA plan.

    Returns the CPU int32 plan ``meta[G * stride, META_WORDS]`` (row
    ``c * stride + i`` = CTA ``c``'s i-th tile; the first row of a CTA carries
    its tile count), the tile stride, the chosen mode, the persistent CTA count
    and the tile count.  ``sms`` is the multiprocessor count of the executing
    device (it sizes the persistent grid).
    """
    if block_mask.dtype != torch.bool or block_mask.ndim != 3:
        raise ValueError("block_mask must be a boolean [H, MB, NB] tensor")
    mask = block_mask.to(device="cpu").contiguous()
    h, mb, nb = mask.shape
    if h == 0 or mb == 0 or nb == 0:
        raise ValueError("block_mask must be non-empty")
    if nb > 0xFFFF:
        raise ValueError("at most 65535 KV blocks are supported")
    counts = mask.sum(dim=-1)
    if int(counts.min()) == 0:
        raise ValueError("empty sparse rows are not supported")
    if int(counts.max()) > MAX_SELECTED:
        raise ValueError(
            f"at most {MAX_SELECTED} KV blocks per query block are supported"
        )
    if mode is not None and mode not in {"pair", "split"}:
        raise ValueError(f"unknown mode {mode!r}")
    if sms < 1:
        raise ValueError("sms must be positive")
    rows = [[m.nonzero().flatten().tolist() for m in mask[hh]] for hh in range(h)]
    ragged = int(counts.min()) != int(counts.max())

    def build(mode: str):
        infos: list[tuple[int, int, int, int]] = []
        seqs: list[list[int]] = []
        owns: list[list[list[int]]] = []
        lens: list[int] = []
        for hh in range(h):
            pairs = _pairs_for_head(hh, mode, rows, counts, mb, ragged)
            for qb, partner in pairs:
                _emit_tile(hh, qb, partner, mode, rows, infos, seqs, owns, lens)
        costs = [_tile_cost(owns[t]) for t in range(len(infos))]
        order = (
            sorted(range(len(infos)), key=lambda i: (infos[i][0], -costs[i]))
            if ragged
            else list(range(len(infos)))
        )
        return tuple([x[i] for i in order] for x in (infos, seqs, owns, lens, costs))

    if mode is None:
        best = None
        for cand in ("split", "pair"):  # ties go to split
            built = build(cand)
            _, makespan = _assign_tiles(built[4], built[3], sms)
            if best is None or makespan < best[0]:
                best = (makespan, cand, built)
        _, mode, (infos, seqs, owns, lens, costs) = best
    else:
        infos, seqs, owns, lens, costs = build(mode)
    lists, makespan = _assign_tiles(costs, lens, sms)
    g = len(lists)
    stride = max(len(lst) for lst in lists)
    meta = np.zeros((g * stride, META_WORDS), dtype=np.uint32)
    for c, lst in enumerate(lists):
        for i, t in enumerate(lst):
            r = c * stride + i
            meta[r, 0:4] = infos[t]
            meta[r, META_NSEQ] = lens[t]
            sq = np.asarray(seqs[t], dtype=np.int64) & 0xFFFF
            meta[r, META_SEQ_OFF : META_SEQ_OFF + len(sq) // 2] = sq[0::2] | (
                sq[1::2] << 16
            )
            for w in range(NUM_CONSUMER_WGS):
                ol = owns[t][w]
                meta[r, META_NOWN + w] = len(ol)
                if len(ol) > MAX_OWN:
                    raise ValueError("own list exceeds MAX_OWN")
                ow = np.asarray(ol + [0] * (len(ol) & 1), dtype=np.int64)
                base = META_OWN_OFF + w * OWN_WORDS
                meta[r, base : base + len(ow) // 2] = ow[0::2] | (ow[1::2] << 16)
        meta[c * stride, META_NTILES] = len(lst)
    return {
        "meta": torch.from_numpy(meta.view(np.int32).copy()),
        "tile_stride": stride,
        "mode": mode,
        "makespan": makespan,
        "num_ctas": g,
        "num_tiles": len(infos),
        "tile_lists": lists,
        "H": h,
        "MB": mb,
        "NB": nb,
    }


# ---------------------------------------------------------------------------
# Wrapper-facing plan object
# ---------------------------------------------------------------------------


def _check_descriptors(block_mask_map, block_row_sz, block_col_sz, num_qo_heads):
    if (
        not isinstance(block_mask_map, torch.Tensor)
        or block_mask_map.dtype != torch.bool
    ):
        raise ValueError("cake (SM90) requires a boolean block_mask_map")
    if block_mask_map.ndim != 3:
        raise ValueError("block_mask_map must have shape (num_heads, MB, NB)")
    h, mb, nb = block_mask_map.shape
    if h != num_qo_heads:
        raise ValueError(
            "cake (SM90) requires one sparse pattern per head: block_mask_map.shape[0] "
            f"({h}) must equal num_qo_heads ({num_qo_heads})"
        )
    for name, sizes, count in (
        ("block_row_sz", block_row_sz, mb),
        ("block_col_sz", block_col_sz, nb),
    ):
        if not isinstance(sizes, torch.Tensor) or tuple(sizes.shape) != (h, count):
            raise ValueError(f"{name} must have shape ({h}, {count})")
        if not bool((sizes.to("cpu") == BLOCK).all()):
            raise ValueError(f"cake (SM90) requires {BLOCK}-token blocks in {name}")


class CakeVsaSm90Plan:
    """Wrapper-owned plan: uploaded tile metadata plus the launch geometry.

    ``run`` issues exactly one kernel launch on the current stream (after an
    event wait on the metadata upload), so a fixed plan is CUDA-graph
    capturable; ``plan()`` may not replace a plan that a graph has captured.
    """

    def __init__(
        self,
        device: torch.device,
        block_mask_map: torch.Tensor,
        block_row_sz: torch.Tensor,
        block_col_sz: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("cake (SM90) requires a CUDA device")
        if torch.cuda.get_device_capability(self.device) != (9, 0):
            raise ValueError("cake (SM90) VSA requires Hopper compute capability 9.0")
        if causal:
            raise ValueError("cake (SM90) VSA supports noncausal attention only")
        if pos_encoding_mode != "NONE":
            raise ValueError("cake (SM90) VSA does not apply positional encodings")
        if use_fp16_qk_reduction:
            raise ValueError("cake (SM90) VSA accumulates QK in FP32 only")
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise ValueError("cake (SM90) VSA does not support logits_soft_cap")
        if q_data_type != torch.bfloat16 or (
            kv_data_type is not None and kv_data_type != torch.bfloat16
        ):
            raise ValueError("cake (SM90) VSA supports BF16 q/k/v only")
        if num_qo_heads != num_kv_heads or num_qo_heads < 1:
            raise ValueError(
                "cake (SM90) VSA requires equal, positive query/KV head counts"
            )
        if head_dim != HEAD_DIM:
            raise ValueError(f"cake (SM90) VSA requires head_dim {HEAD_DIM}")
        _check_descriptors(block_mask_map, block_row_sz, block_col_sz, num_qo_heads)
        scale = 1.0 / math.sqrt(HEAD_DIM) if sm_scale is None else float(sm_scale)
        # The kernel consumes scale * log2(e) as FP32.
        if not math.isfinite(scale * LOG2E) or abs(scale * LOG2E) > _FP32_MAX:
            raise ValueError("sm_scale must be finite in FP32")
        self.sm_scale = scale
        self.scale_log2 = scale * LOG2E
        props = torch.cuda.get_device_properties(self.device)
        plan = plan_vsa_sm90(block_mask_map, sms=int(props.multi_processor_count))
        self.mode = plan["mode"]
        self.num_ctas = plan["num_ctas"]
        self.num_tiles = plan["num_tiles"]
        self.tile_stride = plan["tile_stride"]
        self.num_heads = plan["H"]
        self.qo_len = plan["MB"] * BLOCK
        self.kv_len = plan["NB"] * BLOCK
        self.captured = False
        with torch.cuda.device(self.device):
            host_meta = plan["meta"]
            if non_blocking:
                host_meta = host_meta.pin_memory()
            self.meta = host_meta.to(self.device, non_blocking=non_blocking)
            # The kernel's debug/timeline buffers are inert in the exported
            # build; they only have to exist on the device.
            self.dbg = torch.zeros((64,), dtype=torch.int32, device=self.device)
            self.tl = torch.zeros((8,), dtype=torch.uint64, device=self.device)
            # Capture must retain an external wait node for the metadata
            # upload, which was recorded outside the captured stream.
            self.ready = torch.cuda.Event(external=True)
            self.ready.record(torch.cuda.current_stream(self.device))
        self._host_meta = host_meta  # keeps pinned storage alive for the copy

    def check_replan(self) -> None:
        if self.captured:
            raise RuntimeError(
                "A captured VSA plan cannot be replaced; create a new wrapper and "
                "keep the original wrapper alive while its graph may replay"
            )

    def _check_tensor(self, name, tensor, shape) -> None:
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"{name} must have shape {tuple(shape)} and dtype bfloat16"
            )
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous on the planned device {self.device}"
            )

    @staticmethod
    def _aligned(tensor: torch.Tensor) -> torch.Tensor:
        # TMA needs 16-byte aligned bases; an offset view is copied on the
        # current stream (graph-capturable).
        if tensor.data_ptr() % 16 == 0:
            return tensor
        return tensor.clone(memory_format=torch.contiguous_format)

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        if return_lse or lse is not None:
            raise ValueError("cake (SM90) VSA does not support log-sum-exp outputs")
        if enable_pdl:
            raise ValueError("cake (SM90) VSA does not support PDL")
        for name, tensor, length in (
            ("q", q, self.qo_len),
            ("k", k, self.kv_len),
            ("v", v, self.kv_len),
        ):
            self._check_tensor(name, tensor, (self.num_heads, length, HEAD_DIM))
        # Preserve VariableBlockSparseAttentionWrapper's existing DPS ABI: the
        # provided output is [H*M, 1, D], while the return value is HND.
        if out is None:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "CUDA graph capture requires a preallocated out buffer"
                )
            result = torch.empty_like(q)
        else:
            self._check_tensor("out", out, (self.num_heads * self.qo_len, 1, HEAD_DIM))
            result = out.view(self.num_heads, self.qo_len, HEAD_DIM)
            begin = out.data_ptr()
            end = begin + out.numel() * out.element_size()
            for tensor in (q, k, v):
                tensor_begin = tensor.data_ptr()
                tensor_end = tensor_begin + tensor.numel() * tensor.element_size()
                if begin < tensor_end and tensor_begin < end:
                    raise ValueError("out must not overlap Q/K/V storage")

        import tvm_ffi

        from .jit.cake_vsa_sm90 import load_cake_vsa_sm90_module

        module, _record = load_cake_vsa_sm90_module()
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if torch.cuda.is_current_stream_capturing():
                self.captured = True
            # The plan may have been built on another stream: an event wait
            # enqueues the dependency without a CPU synchronization.
            stream.wait_event(self.ready)
            aligned_q = self._aligned(q)
            aligned_k = self._aligned(k)
            aligned_v = self._aligned(v)
            target = result if result.data_ptr() % 16 == 0 else torch.empty_like(result)
            with tvm_ffi.use_torch_stream():
                module.run(
                    aligned_q,
                    aligned_k,
                    aligned_v,
                    target,
                    self.meta,
                    int(self.tile_stride),
                    int(self.qo_len),
                    int(self.kv_len),
                    float(self.scale_log2),
                    self.dbg,
                    self.tl,
                    int(self.num_ctas),
                    1,
                    1,
                )
            if target is not result:
                result.copy_(target)
        return result


def create_plan(device, *args, **kwargs) -> CakeVsaSm90Plan:
    """Validate the request and build the uploaded plan (outside graph capture)."""
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("cake (SM90) VSA requires Hopper compute capability 9.0")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Call plan() outside CUDA graph capture")
        return CakeVsaSm90Plan(device, *args, **kwargs)


__all__ = ["CakeVsaSm90Plan", "create_plan", "plan_vsa_sm90"]
