# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# See LICENSE.kda-for-kda.txt for the full license.

"""Spread a complete recurrent chain over idle SMs through state linearity.

All tokens contribute to the affine composition. Handoff and final-state
storage are FP32. Final-state composition uses TF32 tensor-core products with
FP32 accumulation; tail-output correction rounds its state operand to BF16
and accumulates BF16 products in FP32. These choices follow the validated
relative-L2 and per-element tolerances rather than bitwise FP32 equivalence.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_DIM = 128
CHUNK = 32
FIX_PROGRAMS_PER_SM = 3
PREFIX_UNIFORM_HEADS = (64, 96)


@triton.jit
def _split_fix_kernel(
    w_ptr,
    zo_ptr,
    out_ptr,
    sa_ptr,
    p_ptr,
    z_ptr,
    final_ptr,
    tail_tokens,
    num_heads: tl.constexpr,
    IN_HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compose a complete tail using TF32 state and BF16 output products."""
    head = tl.program_id(0)
    part = tl.program_id(1)
    offs_d = tl.arange(0, 128)
    state_parts: tl.constexpr = 128 // BLOCK_V
    base = head * 16384
    first = part * BLOCKS_PER_PROGRAM
    if first * BLOCK_T < tail_tokens:
        s_a = tl.load(sa_ptr + base + offs_d[:, None] * 128 + offs_d[None, :])
        s_a = s_a.to(tl.bfloat16)
        for i in tl.static_range(BLOCKS_PER_PROGRAM):
            offs_t = (first + i) * BLOCK_T + tl.arange(0, BLOCK_T)
            mask = (offs_t < tail_tokens)[None, :]
            offs = (offs_t[None, :] * num_heads + head) * 128 + offs_d[:, None]
            offs_in = (offs_t[None, :] * IN_HEADS + head) * 128 + offs_d[:, None]
            w_t = tl.load(
                w_ptr + offs_in, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            corr = tl.dot(s_a, w_t)
            o = tl.load(
                zo_ptr + offs_in, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            o = (o.to(tl.float32) + corr).to(tl.bfloat16)
            tl.store(out_ptr + offs, o, mask=mask, cache_modifier=".cs")
    if part < state_parts:
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = part * BLOCK_V + tl.arange(0, BLOCK_V)
        rows = base + offs_v[:, None] * 128
        state = tl.load(z_ptr + rows + offs_d[None, :])
        for kk in tl.range(0, 128, BLOCK_K, num_stages=3):
            s_rows = tl.load(sa_ptr + rows + kk + offs_k[None, :])
            p = tl.load(p_ptr + base + (kk + offs_k[:, None]) * 128 + offs_d[None, :])
            state = tl.dot(s_rows, p, acc=state, input_precision="tf32")
        tl.store(final_ptr + rows + offs_d[None, :], state)


@triton.jit
def _split_fix2_kernel(
    w_ptr,
    zo_ptr,
    out_ptr,
    st_ptr,
    pa_ptr,
    za_ptr,
    pb_ptr,
    zb_ptr,
    final_ptr,
    cnt_ptr,
    tail_tokens,
    num_heads: tl.constexpr,
    IN_HEADS: tl.constexpr,
    NA: tl.constexpr,
    NB: tl.constexpr,
    PA: tl.constexpr,
    PB: tl.constexpr,
    MAXB: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fix for a tail cut into two dual segments A (NA token blocks) and B.

    Programs ``part < 128 / BLOCK_V`` (all in segment A) first form rows of
    S_b = S_a P_A + Z_A into ``st_ptr[2]`` and release a per-head counter;
    programs ``part < PA`` correct segment A's blocks with S_a
    (``st_ptr[0]``), programs ``PA <= part < PA + PB`` wait for the counter
    and correct segment B's blocks with S_b (the last of them resets the
    counters, so graph replays start from zero); the state programs finally
    form ``final = S_b P_B + Z_B`` for their rows. One wave of programs is
    required (the B programs spin).
    """
    head = tl.program_id(0)
    part = tl.program_id(1)
    offs_d = tl.arange(0, 128)
    offs_k = tl.arange(0, BLOCK_K)
    state_parts: tl.constexpr = 128 // BLOCK_V
    base = head * 16384
    sb_ptr = st_ptr + 2 * num_heads * 16384
    offs_v = part * BLOCK_V + tl.arange(0, BLOCK_V)
    rows = base + offs_v[:, None] * 128
    if part < state_parts:
        s_b = tl.load(za_ptr + rows + offs_d[None, :])
        for kk in tl.range(0, 128, BLOCK_K, num_stages=3):
            s_rows = tl.load(st_ptr + rows + kk + offs_k[None, :])
            p = tl.load(pa_ptr + base + (kk + offs_k[:, None]) * 128 + offs_d[None, :])
            s_b = tl.dot(s_rows, p, acc=s_b, input_precision="tf32")
        tl.store(sb_ptr + rows + offs_d[None, :], s_b)
        tl.debug_barrier()
        tl.atomic_add(cnt_ptr + head, 1, sem="release", scope="gpu")
        state = tl.load(zb_ptr + rows + offs_d[None, :])
        for kk in tl.range(0, 128, BLOCK_K, num_stages=3):
            s_rows = tl.load(sb_ptr + rows + kk + offs_k[None, :])
            p = tl.load(pb_ptr + base + (kk + offs_k[:, None]) * 128 + offs_d[None, :])
            state = tl.dot(s_rows, p, acc=state, input_precision="tf32")
        tl.store(final_ptr + rows + offs_d[None, :], state)
    in_b = part >= PA
    j = tl.where(in_b, part - PA, part)
    seg0 = tl.where(in_b, NA, 0)
    seg_n = tl.where(in_b, NB, NA)
    seg_p = tl.where(in_b, PB, PA)
    first = seg0 + j * seg_n // seg_p
    last = seg0 + (j + 1) * seg_n // seg_p
    if part < PA + PB:
        mat = st_ptr + base
        if part >= PA:
            while (
                tl.atomic_add(cnt_ptr + head, 0, sem="acquire", scope="gpu")
                < state_parts
            ):
                pass
            done = tl.atomic_add(
                cnt_ptr + num_heads + head, 1, sem="relaxed", scope="gpu"
            )
            if done == PB - 1:
                tl.atomic_xchg(cnt_ptr + head, 0, sem="relaxed", scope="gpu")
                tl.atomic_xchg(
                    cnt_ptr + num_heads + head, 0, sem="relaxed", scope="gpu"
                )
            mat = sb_ptr + base
        s_m = tl.load(mat + offs_d[:, None] * 128 + offs_d[None, :])
        s_m = s_m.to(tl.bfloat16)
        for i in tl.static_range(MAXB):
            blk = first + i
            offs_t = blk * BLOCK_T + tl.arange(0, BLOCK_T)
            mask = ((offs_t < tail_tokens) & (blk < last))[None, :]
            offs = (offs_t[None, :] * num_heads + head) * 128 + offs_d[:, None]
            offs_in = (offs_t[None, :] * IN_HEADS + head) * 128 + offs_d[:, None]
            w_t = tl.load(
                w_ptr + offs_in, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            corr = tl.dot(s_m, w_t)
            o = tl.load(
                zo_ptr + offs_in, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            o = (o.to(tl.float32) + corr).to(tl.bfloat16)
            tl.store(out_ptr + offs, o, mask=mask, cache_modifier=".cs")


def _handoff_pack(rows_in, n_ctas, tail_chunks):
    """Pack equal tail pieces on fewer CTAs through exact state handoffs.

    With ``P`` pieces of ``L`` chunks on ``C`` CTAs and ``w = P // C``, plain
    packing puts ``w + 1`` pieces on some CTAs. Instead, groups of ``n`` CTAs
    carry ``w n + n - 1`` pieces at ``(w + 1 - 1 / n) L`` each: ``n - 1``
    chains are cut at ``(n - 1) / n`` of their length, CTA ``i < n`` runs
    ``[first part i, w whole]`` and the last CTA ``[w whole, every second
    part]`` (H64: pairs ``[C1, A]`` / ``[B, C2]`` and triples ``[D1, A]`` /
    ``[B, D2, E2]`` / ``[E1, C]``; H96: groups of 3 and 4 CTAs). A first part
    stores its FP32 state into the launch's ``final_state`` and releases
    per-row flags (store mode 2); the second part waits for them (init mode
    2), so the recurrence stays exact. Every first part is the first piece on
    its CTA and every second part follows ``w >= 1`` whole pieces, so nothing
    waits in the model, and no CTA waits on a CTA that is not running a first
    part (no deadlock in a one-CTA-per-SM grid). ``(w + 1) C - P`` groups are
    needed; they must have at least two CTAs each.

    :return: (rows, cta_vals, n_split) or None when the shape does not fit.
    """
    pieces = len(rows_in)
    whole = pieces // n_ctas
    groups = (whole + 1) * n_ctas - pieces
    if whole < 1 or not 0 < groups <= n_ctas // 2:
        return None
    base, extra = divmod(n_ctas, groups)
    sizes = [base + 1] * extra + [base] * (groups - extra)
    while sizes.count(3) >= 2 and sizes.count(2) >= 2:
        for v in (3, 3, 2, 2):
            sizes.remove(v)
        sizes = [5, 5] + sizes
    rows_iter = iter(rows_in)
    bins = []
    n_split = 0

    def cut(row, chunks):
        """Split a tail job at a chunk boundary with an FP32 handoff slot."""
        nonlocal n_split
        at = row[0] + chunks * CHUNK
        flag_base = n_split * 128
        n_split += 1
        first = [row[0], at, row[2], row[3], row[4], 2, row[6], flag_base]
        second = [at, row[1], at, row[3], 2, row[5], row[6], flag_base]
        return (first, second)

    def cut3(row, c1, c2):
        """Partition one tail chain into three pieces linked by FP32 state slots."""
        nonlocal n_split
        at1 = row[0] + c1 * CHUNK
        at2 = at1 + c2 * CHUNK
        flag_base = n_split * 128
        n_split += 2
        p1 = [row[0], at1, row[2], row[3], row[4], 2, row[6], flag_base]
        p2 = [at1, at2, at1, row[3], 2, 3, row[6], flag_base]
        p3 = [at2, row[1], at2, row[3], 2, row[5], row[6], flag_base + 128]
        return (p1, p2, p3)

    for n in sizes:
        if n == 5:
            a = (3 * tail_chunks + 2) // 5
            b = (tail_chunks + 2) // 5
            whole_rows = [[next(rows_iter) for _ in range(whole)] for _ in range(5)]
            x1, x2 = cut(next(rows_iter), a)
            z1, z2 = cut(next(rows_iter), a)
            y1, y2, y3 = cut3(next(rows_iter), a, b)
            bins += [
                [x1, *whole_rows[0]],
                [z1, *whole_rows[1]],
                [y1, *whole_rows[2]],
                [*whole_rows[3], y2, x2],
                [*whole_rows[4], z2, y3],
            ]
            continue
        first_chunks = ((n - 1) * tail_chunks * 2 + n) // (2 * n)
        order = [0, n - 1, *range(1, n - 1)]
        group = [[] for _ in range(n)]
        for c in order:
            group[c].extend((next(rows_iter) for _ in range(whole)))
        for c in range(n - 1):
            head, tail = cut(next(rows_iter), first_chunks)
            group[c].insert(0, head)
            group[n - 1].append(tail)
        bins += [group[c] for c in order]
    if next(rows_iter, None) is not None or len(bins) != n_ctas:
        return None
    rows = []
    cta = []
    for rows_here in bins:
        cta.extend(
            [
                len(rows),
                len(rows_here),
                sum(((r[1] - r[0]) // CHUNK for r in rows_here)),
            ]
        )
        rows.extend(rows_here)
    return (rows, cta, n_split)


class FixedSplitLaunch:
    """Prefix chain plus zero-state and identity-state tail chains."""

    def __init__(
        self,
        module,
        tail_module,
        args,
        *,
        q,
        k,
        g,
        beta,
        initial_state,
        final_state,
        num_heads,
        tokens,
        tail_chunks,
        num_sms,
        block_t=64,
        blocks_per_program=7,
        block_v=32,
        block_k=32,
        num_warps=4,
        handoff=True,
        dual_module=None,
        seg_b=0,
    ):
        """Allocate private prefix, dual-tail, correction, and stream resources."""
        device = q.device
        heads = num_heads
        split = tokens - tail_chunks * CHUNK
        tail = tokens - split
        if split <= 0 or tail <= 0 or split % CHUNK:
            raise ValueError("invalid split")
        helpers = num_sms - heads
        if helpers < 2:
            raise ValueError("no idle SMs")
        self.module = tail_module if heads in PREFIX_UNIFORM_HEADS else module
        self.tail_module = tail_module
        self.num_sms = num_sms
        self.split = split
        self.tail = tail
        self.heads = heads
        f32 = torch.float32
        self.ws_state = torch.empty(2, heads, 128, 128, dtype=f32, device=device)
        self.ws_p = torch.empty(1, heads, 128, 128, dtype=f32, device=device)
        self.ident = (
            torch.eye(128, dtype=f32, device=device)
            .expand(1, heads, 128, 128)
            .contiguous()
        )
        self.v_zero = torch.zeros(16, heads, 128, dtype=torch.bfloat16, device=device)
        self.w = torch.empty(tail, heads, 128, dtype=torch.bfloat16, device=device)
        self.final_state = final_state
        self.out_tail = args["out"][split:]
        self.dual_module = dual_module
        self.seg_b = 0
        n_z = helpers // 2
        n_w = helpers - n_z
        self.flags = []

        def table(rows_in, n_ctas, tail_args=None):
            """Pack tail jobs and allocate the corresponding private CTA tables."""
            packed = (
                _handoff_pack(rows_in, n_ctas, tail_chunks)
                if handoff and tail_args is not None
                else None
            )
            if packed is not None:
                rows, cta, n_split = packed
                flags = torch.zeros(n_split * 128, dtype=torch.int32, device=device)
                self.flags.append(flags)
                tail_args["handoff_flags_addr"] = flags.data_ptr()
                tail_args["generation"] = 1
                return (
                    torch.tensor(rows, dtype=torch.int32, device=device),
                    torch.tensor(cta, dtype=torch.int32, device=device),
                    (len(cta) // 3, 1, 1),
                )
            bins = [[] for _ in range(min(n_ctas, len(rows_in)))]
            for i, row in enumerate(rows_in):
                bins[i % len(bins)].append(row)
            rows = []
            cta = []
            for rows_here in bins:
                cta.extend(
                    [
                        len(rows),
                        len(rows_here),
                        sum(((r[1] - r[0]) // CHUNK for r in rows_here)),
                    ]
                )
                rows.extend(rows_here)
            return (
                torch.tensor(rows, dtype=torch.int32, device=device),
                torch.tensor(cta, dtype=torch.int32, device=device),
                (len(cta) // 3, 1, 1),
            )

        prefix = [[0, split, 0, 0, 1, 1, h, 0] for h in range(heads)]
        zero_tail = [[split, tokens, split, 0, 0, 1, h, 0] for h in range(heads)]
        ident_tail = [[0, tail, 0, 0, 1, 1, h, 0] for h in range(heads)]
        self.args_p = dict(args)
        pt, ct, self.grid_p = table(prefix, heads)
        self.args_p["piece_table"] = pt
        self.args_p["cta_table"] = ct
        self.args_p["initial_state"] = initial_state
        self.args_p["final_state"] = self.ws_state
        self.args_p["handoff_flags_addr"] = 0
        self.args_z = dict(args)
        self.args_z["handoff_flags_addr"] = 0
        pt, ct, self.grid_z = table(zero_tail, n_z, self.args_z)
        self.args_z["piece_table"] = pt
        self.args_z["cta_table"] = ct
        self.args_z["initial_state"] = initial_state
        self.args_z["final_state"] = self.ws_state[1:]
        self.args_b = dict(args)
        self.args_b["handoff_flags_addr"] = 0
        pt, ct, self.grid_b = table(ident_tail, n_w, self.args_b)
        self.args_b["piece_table"] = pt
        self.args_b["cta_table"] = ct
        self.args_b["initial_state"] = self.ident
        self.args_b["final_state"] = self.ws_p
        self.args_b["v"] = self.v_zero
        self.args_b["v_tma"] = self.v_zero
        self.args_b["out"] = self.w
        self.args_b["out_tma"] = self.w
        if dual_module is not None:
            self.zw = torch.empty(
                tail, 2 * heads, 128, dtype=torch.bfloat16, device=device
            )
            self.seg_b = seg_b if helpers > heads else 0
            self.ws_zp = torch.empty(
                4 if self.seg_b else 2, heads, 128, 128, dtype=f32, device=device
            )
            self.args_d = dict(args)
            if self.seg_b:
                a_tok = tail - self.seg_b * CHUNK
                if a_tok <= 0 or a_tok % (2 * CHUNK):
                    raise ValueError("segment A must be a positive even chunk count")
                self.a_tok = a_tok
                spare = helpers - heads
                bins = [[[0, a_tok, 0, 0, 0, 1, h, 0]] for h in range(heads)]
                bins += [[] for _ in range(spare)]
                for h in range(heads):
                    bins[heads + h % spare].append([a_tok, tail, a_tok, 2, 0, 1, h, 0])
                rows = []
                cta = []
                for rows_here in bins:
                    cta.extend(
                        [
                            len(rows),
                            len(rows_here),
                            sum(((r[1] - r[0]) // CHUNK for r in rows_here)),
                        ]
                    )
                    rows.extend(rows_here)
                pt = torch.tensor(rows, dtype=torch.int32, device=device)
                ct = torch.tensor(cta, dtype=torch.int32, device=device)
                self.grid_d = (len(cta) // 3, 1, 1)
                self.ws_state = torch.empty(
                    3, heads, 128, 128, dtype=f32, device=device
                )
                self.args_p["final_state"] = self.ws_state
                self.seg_cnt = torch.zeros(2 * heads, dtype=torch.int32, device=device)
            else:
                dual_rows = [[0, tail, 0, 0, 0, 1, h, 0] for h in range(heads)]
                pt, ct, self.grid_d = table(dual_rows, helpers)
            self.args_d["piece_table"] = pt
            self.args_d["cta_table"] = ct
            self.args_d["initial_state"] = self.ws_zp
            self.args_d["final_state"] = self.ws_zp
            self.args_d["out"] = self.zw
            self.args_d["out_tma"] = self.zw
            self.args_d["v"] = args["v"][split:]
            self.args_d["v_tma"] = args["v_tma"][split:]
            # Stream completion orders dual writes before correction, so
            # no per-CTA device completion counter is needed.
            self.args_d["handoff_flags_addr"] = 0
            self.args_d["generation"] = 0
        self.bind_inputs(q, k, g, beta)
        token_blocks = -(-tail // block_t)

        def fix_parts(bpp):
            """Size the correction grid to cover output tiles and state rows."""
            return max(128 // block_v, -(-token_blocks // bpp))

        while (
            heads * fix_parts(blocks_per_program) > FIX_PROGRAMS_PER_SM * num_sms
            and blocks_per_program < token_blocks
        ):
            blocks_per_program += 1
        self.block_t = block_t
        self.blocks_per_program = blocks_per_program
        self.block_v = block_v
        self.block_k = block_k
        self.num_warps = num_warps
        self.fix_grid = (heads, fix_parts(blocks_per_program))
        self.side_z = torch.cuda.Stream(device=device)
        self.side_w = torch.cuda.Stream(device=device)

    def bind_inputs(self, q, k, g, beta):
        """Bind live gate and activation views for the split tail."""
        split = self.split
        q_tail = q[split:]
        k_tail = k[split:]
        g_tail = g[split:]
        beta_tail = beta[split:]
        for key, value in (
            ("q", q_tail),
            ("q_tma", q_tail),
            ("k", k_tail),
            ("k_tma", k_tail),
            ("g", g_tail),
            ("g_tma", g_tail),
            ("beta", beta_tail),
            ("beta_tma", beta_tail),
        ):
            self.args_b[key] = value
            if self.dual_module is not None:
                self.args_d[key] = value

    def set_g(self, g_flat):
        """Rebind one gate buffer consistently across prefix and tail phases."""
        for args in (self.args_p, self.args_z):
            args["g"] = g_flat
            args["g_tma"] = g_flat
        g_tail = g_flat[self.split :]
        self.args_b["g"] = g_tail
        self.args_b["g_tma"] = g_tail
        if self.dual_module is not None:
            self.args_d["g"] = g_tail
            self.args_d["g_tma"] = g_tail

    def launch(self):
        """Run prefix and tail concurrently, joining producers before correction."""
        main = torch.cuda.current_stream()
        if self.dual_module is not None:
            self.side_z.wait_stream(main)
            self.module.launch(grid=self.grid_p, **self.args_p)
            with torch.cuda.stream(self.side_z):
                self.dual_module.launch(grid=self.grid_d, **self.args_d)
            # Producer CTAs finish before correction CTAs can occupy SMs.
            main.wait_stream(self.side_z)
            self.fix()
            return
        self.side_z.wait_stream(main)
        self.side_w.wait_stream(main)
        self.module.launch(grid=self.grid_p, **self.args_p)
        with torch.cuda.stream(self.side_z):
            self.tail_module.launch(grid=self.grid_z, **self.args_z)
        with torch.cuda.stream(self.side_w):
            self.tail_module.launch(grid=self.grid_b, **self.args_b)
        main.wait_stream(self.side_z)
        main.wait_stream(self.side_w)
        self.fix()

    def fix(self):
        """Compose tail outputs and FP32 state after its producer kernels complete."""
        if self.dual_module is not None and self.seg_b:
            self._fix_two_segments()
            return
        if self.dual_module is not None:
            w, zo, p, z = (
                self.zw[:, self.heads :],
                self.zw,
                self.ws_zp[1],
                self.ws_zp[0],
            )
            in_heads = 2 * self.heads
        else:
            w, zo, p, z = (self.w, self.out_tail, self.ws_p[0], self.ws_state[1])
            in_heads = self.heads
        _split_fix_kernel[self.fix_grid](
            w,
            zo,
            self.out_tail,
            self.ws_state[0],
            p,
            z,
            self.final_state,
            self.tail,
            num_heads=self.heads,
            IN_HEADS=in_heads,
            BLOCK_T=self.block_t,
            BLOCKS_PER_PROGRAM=self.blocks_per_program,
            BLOCK_V=self.block_v,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
        )

    def _fix_two_segments(self):
        """Correct two consecutive tail segments and compose their final state."""
        heads = self.heads
        bt = self.block_t
        na = self.a_tok // bt
        nb = -(-(self.tail - self.a_tok) // bt)
        parts = max(128 // self.block_v, FIX_PROGRAMS_PER_SM * self.num_sms // heads)
        best = None
        for pb in range(1, parts):
            pa = parts - pb
            if pa < 128 // self.block_v:
                break
            maxb = max(-(-na // pa), -(-nb // pb))
            if best is None or maxb < best[2]:
                best = (pa, pb, maxb)
        pa, pb, maxb = best
        _split_fix2_kernel[heads, parts](
            self.zw[:, heads:],
            self.zw,
            self.out_tail,
            self.ws_state,
            self.ws_zp[1],
            self.ws_zp[0],
            self.ws_zp[3],
            self.ws_zp[2],
            self.final_state,
            self.seg_cnt,
            self.tail,
            num_heads=heads,
            IN_HEADS=2 * heads,
            NA=na,
            NB=nb,
            PA=pa,
            PB=pb,
            MAXB=maxb,
            BLOCK_T=bt,
            BLOCK_V=self.block_v,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
        )
