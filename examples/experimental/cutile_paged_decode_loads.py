# SPDX-License-Identifier: Apache-2.0
"""Compare two KV access patterns in a split-KV cuTile decode on synthetic inputs.

Single-token GQA with BF16 Q/output, 64 query heads, 8 KV heads, head dim 64,
and page-16 HND FP8 E4M3 KV (unit scales). Requires cuda-tile and tileiras.
Pass --sms to run the same graphs on a PyTorch CUDA green-context stream.

    python examples/experimental/cutile_paged_decode_loads.py --length 131 --batch 5 --shuffled
    python examples/experimental/cutile_paged_decode_loads.py --length 32768 --batch 32 --sms 80
"""

import argparse
import math
import statistics

import cuda.tile as ct
import torch


@ct.function
def join_eight_pages(pages):
    pairs = ()
    for i in ct.static_iter(range(0, 8, 2)):
        pairs += (ct.cat((pages[i], pages[i + 1]), axis=0),)
    first = ct.cat((pairs[0], pairs[1]), axis=0)
    second = ct.cat((pairs[2], pairs[3]), axis=0)
    return ct.cat((first, second), axis=0)


@ct.kernel
def decode_part(
    q_in,
    kv,
    table,
    lengths,
    partial,
    lse,
    tile_n: ct.Constant[int],
    splits: ct.Constant[int],
    page_load: ct.Constant[bool],
):
    split = ct.bid(0)
    head = ct.bid(1)
    batch = ct.bid(2)
    length = ct.gather(lengths, batch)
    q = ct.reshape(ct.load(q_in, (batch, head, 0), (1, 8, 64)), (8, 64))
    q_t = ct.transpose(q)
    d = ct.arange(64, dtype=ct.int32)
    offsets = ct.arange(tile_n, dtype=ct.int32)
    maximum = ct.full((8,), -math.inf, ct.float32)
    total = ct.full((8,), 0.0, ct.float32)
    values = ct.full((64, 8), 0.0, ct.float32)
    for tile in range(split, ct.cdiv(length, tile_n), splits):
        token = tile * tile_n + offsets
        valid = token < length
        if page_load:
            keys = ()
            values_by_page = ()
            for i in ct.static_iter(range(tile_n // 16)):
                logical_page = tile * (tile_n // 16) + i
                page = ct.gather(
                    table,
                    (batch, logical_page),
                    mask=logical_page * 16 < length,
                    padding_value=0,
                )
                shape = (1, 1, 1, 16, 64)
                k_page = ct.reshape(ct.load(kv, (page, 0, head, 0, 0), shape), (16, 64))
                v_page = ct.reshape(ct.load(kv, (page, 1, head, 0, 0), shape), (16, 64))
                keys += (k_page,)
                values_by_page += (v_page,)
            key = join_eight_pages(keys)
            value = join_eight_pages(values_by_page)
            # Match masked gather on padding, including uninitialized FP8 values.
            value = ct.where(valid[:, None], value, ct.full((tile_n, 64), 0, ct.uint8))
        else:
            page = ct.gather(table, (batch, token // 16), mask=valid)
            key = ct.gather(
                kv,
                (page[:, None], 0, head, (token % 16)[:, None], d[None, :]),
                mask=valid[:, None],
            )
            value = ct.gather(
                kv,
                (page[:, None], 1, head, (token % 16)[:, None], d[None, :]),
                mask=valid[:, None],
            )
        key = ct.astype(ct.bitcast(key, ct.float8_e4m3fn), ct.bfloat16)
        value = ct.astype(ct.bitcast(value, ct.float8_e4m3fn), ct.bfloat16)
        # log2(e) / sqrt(head_dim) for base-2 online softmax.
        logits = (
            ct.mma(key, q_t, ct.full((tile_n, 8), 0.0, ct.float32))
            * 0.18033688011112042
        )
        logits = ct.where(valid[:, None], logits, -math.inf)
        new_max = ct.maximum(maximum, ct.max(logits, 0))
        rescale = ct.exp2(maximum - new_max)
        p = ct.exp2(logits - new_max[None, :])
        total = total * rescale + ct.sum(p, 0)
        values = ct.mma(
            ct.transpose(value), ct.astype(p, ct.bfloat16), values * rescale[None, :]
        )
        maximum = new_max
    denom = ct.where(total > 0, total, 1.0)
    normalized = ct.transpose(values / denom[None, :])
    ct.store(
        partial, (batch, head, split, 0, 0), ct.reshape(normalized, (1, 1, 1, 8, 64))
    )
    ct.store(
        lse, (batch, head, split, 0), ct.reshape(maximum + ct.log2(denom), (1, 1, 1, 8))
    )


@ct.kernel
def merge_splits(partial, lse, out, splits: ct.Constant[int]):
    q_head = ct.bid(0)
    batch = ct.bid(1)
    s = ct.arange(splits, dtype=ct.int32)
    d = ct.arange(64, dtype=ct.int32)
    log_norm = ct.gather(lse, (batch, q_head // 8, s, q_head % 8))
    weights = ct.exp2(log_norm - ct.max(log_norm, 0))
    partial_values = ct.gather(
        partial, (batch, q_head // 8, s[:, None], q_head % 8, d[None, :])
    )
    result = ct.sum(weights[:, None] * partial_values, 0) / ct.sum(weights, 0)
    ct.store(
        out, (batch, q_head, 0), ct.reshape(ct.astype(result, ct.bfloat16), (1, 1, 64))
    )


def capture_decode(q, kv, table, lengths, page_load, stream):
    batch = q.shape[0]
    splits = 16
    partial = torch.empty(
        (batch, 8, splits, 8, 64), device=q.device, dtype=torch.float32
    )
    lse = torch.empty(partial.shape[:-1], device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    raw_kv = kv.view(torch.uint8)

    def run():
        ct.launch(
            stream,
            (splits, 8, batch),
            decode_part,
            (q, raw_kv, table, lengths, partial, lse, 128, splits, page_load),
        )
        ct.launch(stream, (64, batch, 1), merge_splits, (partial, lse, out, splits))

    run()  # Compile both kernels before graph capture.
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    # Captured launches reference these preallocated tensors on every replay.
    return graph, out, (partial, lse)


def reference(q, kv, table, lengths):
    # Process one request at a time to bound FP32 scratch at long context.
    output = []
    for i, length in enumerate(lengths):
        pages = table[i, : (length + 15) // 16].long()
        k, v = kv.view(torch.uint8)[pages].view(kv.dtype).float().unbind(1)
        k = k.transpose(0, 1).flatten(1, 2)[:, :length]
        v = v.transpose(0, 1).flatten(1, 2)[:, :length]
        scores = q[i].float().reshape(8, 8, 64) @ k.transpose(1, 2) / 8
        output.append((scores.softmax(-1) @ v).reshape(64, 64))
    return torch.stack(output)


@torch.inference_mode()
def compare(args, stream):
    lens = [
        max(1, args.length - (i % 5 if args.ragged else 0)) for i in range(args.batch)
    ]
    lengths = torch.tensor(lens, device="cuda", dtype=torch.int32)
    pages_per_request = (max(lens) + 15) // 16
    pages = pages_per_request * args.batch
    page_ids = (
        torch.randperm(pages, device="cuda", dtype=torch.int32)
        if args.shuffled
        else torch.arange(pages, device="cuda", dtype=torch.int32)
    )
    table = page_ids.reshape(args.batch, pages_per_request)
    kv = torch.empty((pages, 2, 8, 16, 64), dtype=torch.float8_e4m3fn, device="cuda")
    for chunk in kv.split(512):
        chunk.copy_(torch.randn(chunk.shape, device="cuda", dtype=torch.bfloat16))
    q = torch.randn((args.batch, 64, 64), device="cuda", dtype=torch.bfloat16)
    graphs = {
        name: capture_decode(q, kv, table, lengths, page_load, stream)
        for name, page_load in [("gather", False), ("page_load", True)]
    }

    # Change a captured input in place and check that replay rewrites the output.
    q.neg_()
    expected = reference(q, kv, table, lens)
    for name, (graph, out, _) in graphs.items():
        out.fill_(float("nan"))
        graph.replay()
        error = (out.float() - expected).norm() / expected.norm()
        if not error.item() < 0.015:
            raise AssertionError(f"{name}: relative RMS {error.item():.4%}")
        print(f"{name} relative RMS vs FP32: {error.item():.4%}")
    torch.testing.assert_close(
        graphs["gather"][1], graphs["page_load"][1], rtol=0, atol=0
    )

    # Alternate graph order. Each CUDA event pair covers one split + merge replay.
    events = {name: [] for name in graphs}
    for i in range(args.repeats):
        for name in graphs if i % 2 == 0 else reversed(graphs):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record(stream)
            graphs[name][0].replay()
            end.record(stream)
            events[name].append((start, end))
    stream.synchronize()
    for name, pairs in events.items():
        times = [start.elapsed_time(end) for start, end in pairs]
        print(
            f"{name}: mean {statistics.mean(times):.3f} ms, median {statistics.median(times):.3f} ms"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument(
        "--shuffled", action="store_true", help="permute physical page IDs"
    )
    parser.add_argument(
        "--ragged", action="store_true", help="cycle through five request lengths"
    )
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument(
        "--sms", type=int, help="request a green context with this many SMs"
    )
    args = parser.parse_args()
    if min(args.batch, args.length, args.repeats) < 1 or (
        args.sms is not None and args.sms < 1
    ):
        parser.error("batch, length, repeats and SM count must be positive")
    torch.manual_seed(17)
    torch.cuda.init()
    if args.sms is None:
        stream = torch.cuda.Stream()
    else:
        from torch.cuda.green_contexts import GreenContext

        context = GreenContext.create(num_sms=args.sms)
        stream = context.Stream()
    with torch.cuda.stream(stream):
        compare(args, stream)


if __name__ == "__main__":
    main()
