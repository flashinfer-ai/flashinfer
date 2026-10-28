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
"""

DESCRIPTION = """
Paged recurrent-state GDN prefill and decode (flashinfer.gdn_paged).

Prefill compares, on batches whose prefixes end on page boundaries (the only
case the packed API can serve):

- paged: ``chunk_gated_delta_rule_paged`` reads the prefix page and writes every
  page boundary and the final state into the pool in one call, given the rows
  of ``gdn_paged_prefill_slots`` (computed once per step and shared by all
  layers; "slots ms" is that one-time cost, not included in "paged ms").
- packed+copy: what a caller does with the packed checkpoint API: gather the
  prefix states (zero-filling sequences without a prefix), run
  ``chunk_gated_delta_rule`` with a packed initial state, packed final state
  and a packed ``state_checkpoints`` buffer
  (``checkpoint_every_n_tokens=page_size``), then scatter the checkpoints and
  the final states that do not end on a page boundary into their pages.
- no-ckpt: ``chunk_gated_delta_rule`` with ``state_indices`` and no
  checkpoints: each sequence reads and overwrites its last page. This is the
  cost of the recurrence alone, a timing reference, not a correct caching
  scheme.

Workloads whose prefixes are not multiples of ``--page-size`` are skipped.

One selected state per sequence (a chunk-aligned, sequence-local position: one
of the last three chunk ends of the sequence, chosen by its index modulo 3)
compares:

- selected: ``checkpoint_every_n_tokens=64`` with ``checkpoint_state_indices``
  that are ``-1`` except at the selected entry, which names the sequence's row
  of a ``[num_seqs, H, V, K]`` buffer (the index array is built on the device
  inside the timed region).
- all+gather: every 64-token checkpoint into a packed buffer, then a gather of
  the selected ones.
- no-ckpt: no checkpoints.

Decode compares ``gated_delta_rule_decode_paged``, given the rows of
``gdn_paged_decode_slots`` (computed once per step and shared by all layers;
"slots ms" is that one-time cost), with ``gated_delta_rule_decode_pretranspose``
given the same read/write rows built by the caller.

All prefill arms run the non-context-parallel kernels (``use_cp=False``).

Every arm is captured in a CUDA graph and timed by replay
(``bench_gpu_time(use_cuda_graph=True)``, warm L2), so the copies and index
arithmetic around the kernels are inside the number. ``extra MiB`` is the
memory an arm needs beyond the inputs, the output and the pool: the buffers it
owns plus the peak of its transient allocations in one eager call.

Usage:
  python benchmarks/bench_gdn_paged_state.py
  python benchmarks/bench_gdn_paged_state.py --heads 16 16 32 --page-size 1024
"""

import argparse
import math
import statistics

import torch
import torch.nn.functional as F

from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.gdn_paged import (
    chunk_gated_delta_rule_paged,
    gated_delta_rule_decode_paged,
    gdn_paged_decode_slots,
    gdn_paged_prefill_slots,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.testing import bench_gpu_time

D = 128

# (name, [(prefix_len, new_len), ...]) with page-aligned prefixes.
PREFILL_WORKLOADS = [
    ("1x new 8K", [(0, 8192)]),
    ("4x reuse 4K + 2K", [(4096, 2048)] * 4),
    ("16x reuse 8K + 1K", [(8192, 1024)] * 16),
    ("8x mixed", [(0, 3000), (1024, 4096), (2048, 777), (0, 1024)] * 2),
    ("64x reuse 16K + 256", [(16384, 256)] * 64),
]


def _cu(lens, device="cuda", dtype=torch.int32):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=dtype, device=device)


def _median_ms(fn):
    times = bench_gpu_time(fn, use_cuda_graph=True, cold_l2_cache=False)
    return statistics.median(times)


def _extra_mib(fn, owned_bytes=0):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (owned_bytes + torch.cuda.max_memory_allocated() - base) / 2**20


def _tokens(num_tokens, heads):
    hq, hk, hv = heads
    h = max(hq, hv)
    q = torch.randn(num_tokens, hq, D, dtype=torch.bfloat16, device="cuda") * 0.1
    k = F.normalize(torch.randn(num_tokens, hk, D, device="cuda"), dim=-1)
    v = torch.randn(num_tokens, hv, D, dtype=torch.bfloat16, device="cuda") * 0.1
    g = 0.9 + 0.1 * torch.rand(num_tokens, h, device="cuda")
    beta = torch.rand(num_tokens, h, device="cuda")
    return q, k.to(torch.bfloat16), v, g, beta


def _prefill_row(name, spans, heads, page_size, state_dtype):
    hq, hk, hv = heads
    h = max(hq, hv)
    scale = 1.0 / math.sqrt(D)
    assert all(p % page_size == 0 for p, _ in spans)
    prefix = [p for p, _ in spans]
    new = [n for _, n in spans]
    totals = [p + n for p, n in spans]
    num_pages = [max(1, math.ceil(t / page_size)) for t in totals]
    b = len(spans)
    t_new = sum(new)
    q, k, v, g, beta = _tokens(t_new, heads)
    cu = _cu(new)

    pool_pages = sum(num_pages)
    perm = torch.randperm(pool_pages, device="cuda").to(torch.int32)
    indptr = _cu(num_pages)
    last = torch.tensor(
        [t - (m - 1) * page_size for t, m in zip(totals, num_pages, strict=True)],
        dtype=torch.int32,
        device="cuda",
    )
    pool = torch.randn(pool_pages, h, D, D, device="cuda").to(state_dtype) * 0.01

    # Host-side rows for the packed+copy arm (a caller knows them anyway).
    starts = indptr.tolist()
    page_ids = perm.tolist()
    read_rows, final_rows, final_src, ckpt_rows = [], [], [], []
    ckpt_counts = []
    for i in range(b):
        pages = page_ids[starts[i] : starts[i + 1]]
        p, t = prefix[i], totals[i]
        read_rows.append(pages[p // page_size - 1] if p else -1)
        first = p // page_size
        count = t // page_size - first
        ckpt_counts.append(count)
        ckpt_rows += pages[first : first + count]
        if t % page_size:
            final_rows.append(pages[-1])
            final_src.append(i)
    no_prefix = [i for i, r in enumerate(read_rows) if r < 0]
    no_prefix_idx = torch.tensor(no_prefix, dtype=torch.long, device="cuda")
    read_idx = torch.tensor([max(r, 0) for r in read_rows], device="cuda")
    final_all = len(final_src) == b
    final_idx = torch.tensor(final_rows, dtype=torch.long, device="cuda")
    final_src_idx = torch.tensor(final_src, dtype=torch.long, device="cuda")
    ckpt_idx = torch.tensor(ckpt_rows, dtype=torch.long, device="cuda")
    ckpt_cu = _cu(ckpt_counts)
    n_ckpt = max(sum(ckpt_counts), 1)
    init_buf = torch.empty(b, h, D, D, dtype=state_dtype, device="cuda")
    final_buf = torch.empty_like(init_buf)
    ckpt_buf = torch.empty(n_ckpt, h, D, D, dtype=state_dtype, device="cuda")
    out = torch.empty(t_new, max(hq, hv), D, dtype=torch.bfloat16, device="cuda")
    owned = final_buf.numel() + ckpt_buf.numel()
    if len(no_prefix) < b:
        owned += init_buf.numel()
    owned_bytes = owned * init_buf.element_size()

    def slots_fn():
        return gdn_paged_prefill_slots(cu, perm, indptr, page_size, last)

    slots = slots_fn()

    def paged():
        chunk_gated_delta_rule_paged(
            q,
            k,
            v,
            g,
            beta,
            scale,
            cu_seqlens=cu,
            state_pool=pool,
            page_indices=perm,
            page_indptr=indptr,
            page_size=page_size,
            last_page_len=last,
            output=out,
            slots=slots,
        )

    def packed_copy():
        if len(no_prefix) < b:
            torch.index_select(pool, 0, read_idx, out=init_buf)
            if no_prefix:
                init_buf.index_fill_(0, no_prefix_idx, 0)
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state=init_buf if len(no_prefix) < b else None,
            output_final_state=True,
            cu_seqlens=cu,
            output=out,
            output_state=final_buf,
            state_checkpoints=ckpt_buf,
            checkpoint_cu_starts=ckpt_cu,
            checkpoint_every_n_tokens=page_size,
            use_cp=False,
        )
        if ckpt_rows:
            pool.index_copy_(0, ckpt_idx, ckpt_buf[: len(ckpt_rows)])
        if final_all:
            pool.index_copy_(0, final_idx, final_buf)
        elif final_rows:
            pool.index_copy_(0, final_idx, final_buf[final_src_idx])

    no_ckpt_rows = torch.tensor(
        [page_ids[starts[i + 1] - 1] for i in range(b)],
        dtype=torch.int32,
        device="cuda",
    )

    def no_ckpt():
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state=pool,
            output_final_state=True,
            cu_seqlens=cu,
            output=out,
            output_state=pool,
            state_indices=no_ckpt_rows,
            use_cp=False,
        )

    # Same pool contents after one call of either caching arm.
    start = pool.clone()
    paged()
    paged_pool = pool.clone()
    pool.copy_(start)
    packed_copy()
    torch.testing.assert_close(pool.float(), paged_pool.float(), atol=2e-2, rtol=2e-2)
    pool.copy_(start)
    mib_paged = _extra_mib(paged)
    mib_packed = _extra_mib(packed_copy, owned_bytes)
    no_ckpt()

    t_paged = _median_ms(paged)
    t_slots = _median_ms(slots_fn)
    t_packed = _median_ms(packed_copy)
    t_plain = _median_ms(no_ckpt)
    return (
        name,
        b,
        t_new,
        sum(ckpt_counts),
        t_slots,
        t_paged,
        t_packed,
        t_plain,
        mib_paged,
        mib_packed,
    )


def bench_prefill(heads, page_size, state_dtype, workloads):
    rows = []
    for name, spans in workloads:
        if any(p % page_size for p, _ in spans):
            print(f"# skipped {name}: a prefix is not a multiple of page_size")
            continue
        rows.append(_prefill_row(name, spans, heads, page_size, state_dtype))
        torch.cuda.empty_cache()
    return rows


# (name, sequence lengths); the selected position of sequence i is
# (len // 64 - i % 3) * 64 tokens.
SELECTED_WORKLOADS = [
    ("1x 32K", [32768]),
    ("8x 8K", [8192] * 8),
    ("32x 2K", [2048] * 32),
    ("64x 1K", [1024] * 64),
]


def _selected_row(name, lens, heads, state_dtype):
    h = max(heads[0], heads[2])
    scale = 1.0 / math.sqrt(D)
    b = len(lens)
    t = sum(lens)
    q, k, v, g, beta = _tokens(t, heads)
    cu = _cu(lens)
    counts = [n // 64 for n in lens]
    positions = torch.tensor(
        [(n // 64 - i % 3) * 64 for i, n in enumerate(lens)],
        dtype=torch.int32,
        device="cuda",
    )
    starts = _cu(counts)[:-1].contiguous()
    rows = torch.arange(b, dtype=torch.int32, device="cuda")
    selected_entries = (starts + positions // 64 - 1).long()
    entries = torch.empty(sum(counts), dtype=torch.int32, device="cuda")
    selected_buf = torch.empty(b, h, D, D, dtype=state_dtype, device="cuda")
    all_buf = torch.empty(sum(counts), h, D, D, dtype=state_dtype, device="cuda")
    all_cu = _cu(counts)
    out = torch.empty(t, h, D, dtype=torch.bfloat16, device="cuda")
    kwargs = dict(cu_seqlens=cu, output=out, use_cp=False)

    def selected():
        entries.fill_(-1)
        entries.index_copy_(0, selected_entries, rows)
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale,
            state_checkpoints=selected_buf,
            checkpoint_cu_starts=starts,
            checkpoint_every_n_tokens=64,
            checkpoint_state_indices=entries,
            **kwargs,
        )

    def all_gather():
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale,
            state_checkpoints=all_buf,
            checkpoint_cu_starts=all_cu,
            checkpoint_every_n_tokens=64,
            **kwargs,
        )
        torch.index_select(all_buf, 0, selected_entries, out=selected_buf)

    def no_ckpt():
        chunk_gated_delta_rule(q, k, v, g, beta, scale, **kwargs)

    all_gather()
    expected = selected_buf.clone()
    selected_buf.zero_()
    selected()
    torch.testing.assert_close(
        selected_buf.float(), expected.float(), atol=2e-2, rtol=2e-2
    )
    del expected
    elem = selected_buf.element_size()
    mib_sel = _extra_mib(
        selected, selected_buf.numel() * elem + entries.numel() * entries.element_size()
    )
    mib_all = _extra_mib(all_gather, (selected_buf.numel() + all_buf.numel()) * elem)
    no_ckpt()
    t_sel = _median_ms(selected)
    t_all = _median_ms(all_gather)
    t_plain = _median_ms(no_ckpt)
    return name, b, t, t_sel, t_all, t_plain, mib_sel, mib_all


def bench_selected(heads, state_dtype, workloads):
    rows = []
    for name, lens in workloads:
        rows.append(_selected_row(name, lens, heads, state_dtype))
        torch.cuda.empty_cache()
    return rows


def bench_decode(heads, state_dtype, batches):
    hq, hk, hv = heads
    rows = []
    for b in batches:
        q = torch.randn(b, 1, hq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(b, 1, hk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(b, 1, hv, D, dtype=torch.bfloat16, device="cuda")
        A_log = torch.rand(hv, device="cuda") - 2.0
        a = torch.randn(b, 1, hv, dtype=torch.bfloat16, device="cuda")
        dt_bias = torch.randn(hv, device="cuda")
        bb = torch.randn(b, 1, hv, dtype=torch.bfloat16, device="cuda")
        # Two pages per sequence; every other sequence opens its second page.
        pool = torch.randn(2 * b, hv, D, D, device="cuda").to(state_dtype) * 0.01
        indices = torch.randperm(2 * b, device="cuda").to(torch.int32)
        indptr = _cu([2] * b)
        last = torch.tensor(
            [1 if i % 2 else 77 for i in range(b)], dtype=torch.int32, device="cuda"
        )
        # clone(): a one-element view is already "contiguous" and would keep
        # its 4-byte offset, which the decode kernels' index tensors reject.
        write = indices[1::2].clone()
        read = torch.where(last > 1, write, indices[0::2]).contiguous()
        out = torch.empty(b, 1, hv, D, dtype=torch.bfloat16, device="cuda")

        def slots_fn():
            return gdn_paged_decode_slots(indices, indptr, last)

        slots = slots_fn()

        def paged():
            gated_delta_rule_decode_paged(
                q,
                k,
                v,
                A_log,
                a,
                dt_bias,
                bb,
                output=out,
                state_pool=pool,
                page_indices=indices,
                page_indptr=indptr,
                last_page_len=last,
                slots=slots,
            )

        def explicit_rows():
            gated_delta_rule_decode_pretranspose(
                q,
                k,
                v,
                None,
                A_log,
                a,
                dt_bias,
                bb,
                output=out,
                initial_state=pool,
                initial_state_indices=read,
                output_state_indices=write,
            )

        rows.append(
            (b, _median_ms(slots_fn), _median_ms(paged), _median_ms(explicit_rows))
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--heads", type=int, nargs=3, default=[16, 16, 32])
    parser.add_argument("--page-size", type=int, default=1024)
    parser.add_argument(
        "--state-dtype", choices=["float32", "bfloat16"], default="float32"
    )
    parser.add_argument("--decode-batches", type=int, nargs="*", default=[1, 32, 256])
    args = parser.parse_args()
    if args.page_size <= 0 or args.page_size % 64:
        parser.error("--page-size must be a positive multiple of 64")
    state_dtype = getattr(torch, args.state_dtype)
    dev = torch.cuda.get_device_properties(0)
    print(
        f"# {dev.name} (sm_{dev.major}{dev.minor}), torch {torch.__version__}, "
        f"CUDA {torch.version.cuda}, heads q/k/v={tuple(args.heads)}, "
        f"page_size={args.page_size}, state {args.state_dtype}"
    )
    print(
        "| workload | seqs | new tokens | page ckpts | slots ms | paged ms "
        "| packed+copy ms | speedup | no-ckpt ms | paged extra MiB "
        "| packed+copy extra MiB |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, b, t, n, ts, tp, tc, t0, mp, mc in bench_prefill(
        tuple(args.heads), args.page_size, state_dtype, PREFILL_WORKLOADS
    ):
        print(
            f"| {name} | {b} | {t} | {n} | {ts:.4f} | {tp:.4f} | {tc:.4f} "
            f"| {tc / tp:.3f} | {t0:.4f} | {mp:.1f} | {mc:.1f} |"
        )
    print(
        "\n| one state per seq | seqs | tokens | selected ms | all+gather ms "
        "| speedup | no-ckpt ms | selected extra MiB | all+gather extra MiB |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, b, t, ts, ta, t0, ms, ma in bench_selected(
        tuple(args.heads), state_dtype, SELECTED_WORKLOADS
    ):
        print(
            f"| {name} | {b} | {t} | {ts:.4f} | {ta:.4f} | {ta / ts:.3f} "
            f"| {t0:.4f} | {ms:.1f} | {ma:.1f} |"
        )
    if args.decode_batches and dev.major < 9:
        print("\n(decode skipped: GDN decode needs SM90 or newer)")
    elif args.decode_batches:
        hq, hk, hv = args.heads
        print("\n| decode batch | slots ms | paged ms | explicit rows ms | ratio |")
        print("|---:|---:|---:|---:|---:|")
        for b, ts, tp, te in bench_decode(
            (hq, hk, hv), state_dtype, args.decode_batches
        ):
            print(f"| {b} | {ts:.4f} | {tp:.4f} | {te:.4f} | {te / tp:.3f} |")


if __name__ == "__main__":
    main()
