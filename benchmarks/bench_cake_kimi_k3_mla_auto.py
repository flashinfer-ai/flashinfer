# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi-K3 FP8 paged MLA decode: ``backend="auto"`` (Cake programs) against the explicit backends.

For the contract that ``backend="auto"`` routes to the Cake Kimi-K3 programs on SM100 / SM103 (12 heads, one
query token per request, float8_e4m3fn query and cache, kv_lora_rank 512 + qk_rope_head_dim 64, page size 64),
time ``trtllm_batch_decode_with_kv_cache_mla`` with ``backend="trtllm-gen"`` and ``backend="cute-dsl"`` (the two
candidates the previous ``auto`` dispatch autotunes between), ``backend="cake"`` and ``backend="auto"`` over a
batch x max-sequence-length grid through ``flashinfer.testing.utils.bench_gpu_time`` (CUPTI, cold L2).  Reports
the median per backend, the speedup of Cake over each incumbent, the maximum absolute error of every backend
against an FP32 reference, whether ``auto`` reached the Cake launcher and whether its output equals
``backend="cake"`` bitwise.  Inputs are built like ``tests/mla/test_cake_kimi_k3_mla.py`` builds them.

    python benchmarks/bench_cake_kimi_k3_mla_auto.py --out bench.json
"""

import argparse
import json
import math
import os
import statistics
import time
import traceback

import torch

LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE
PAGE = 64


def _fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _make_case(batch, q_lens, kv_lens, num_heads, *, seed, device, pool_pages=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    pages_per_seq = [(k + PAGE - 1) // PAGE for k in kv_lens]
    width = max(pages_per_seq)
    total_pages = sum(pages_per_seq)
    pool_pages = pool_pages or total_pages + 8
    kv_cache = _fp8(
        torch.randn((pool_pages, PAGE, QK_DIM), generator=gen, device=device) * 0.5
    )
    perm = torch.randperm(pool_pages, generator=gen, device=device)[:total_pages]
    block_tables = torch.zeros((batch, width), dtype=torch.int32, device=device)
    off = 0
    for b, n in enumerate(pages_per_seq):
        block_tables[b, :n] = perm[off : off + n].to(torch.int32)
        off += n
    total_q = sum(q_lens)
    query = _fp8(
        torch.randn((total_q, num_heads, QK_DIM), generator=gen, device=device) * 0.5
    )
    q_indptr = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    return dict(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        q_indptr=q_indptr,
        q_lens=list(q_lens),
        kv_lens=list(kv_lens),
        num_heads=num_heads,
        bmm1_scale=1.0 / math.sqrt(LATENT + ROPE),
        bmm2_scale=1.0,
    )


def _reference(case) -> torch.Tensor:
    cache = case["kv_cache"]
    num_heads = case["num_heads"]
    q_rows = case["query"].float()
    out = torch.zeros(
        (q_rows.shape[0], num_heads, LATENT), dtype=torch.float32, device=q_rows.device
    )
    q_indptr = case["q_indptr"].tolist()
    for b, (q_len, kv_len) in enumerate(
        zip(case["q_lens"], case["kv_lens"], strict=True)
    ):
        n_pages = (kv_len + PAGE - 1) // PAGE
        pages = case["block_tables"][b, :n_pages].long()
        values = cache[pages].reshape(-1, QK_DIM)[:kv_len].float()
        q = q_rows[q_indptr[b] : q_indptr[b + 1]].reshape(q_len * num_heads, QK_DIM)
        logits = (q @ values.T) * case["bmm1_scale"]
        if q_len > 1:
            positions = torch.arange(kv_len, device=q.device)
            limit = kv_len - q_len + torch.arange(q_len, device=q.device) + 1
            mask = positions[None, :] < limit[:, None]
            logits = logits.reshape(q_len, num_heads, kv_len).masked_fill(
                ~mask[:, None, :], float("-inf")
            )
            logits = logits.reshape(q_len * num_heads, kv_len)
        probs = torch.softmax(logits, dim=-1)
        out[q_indptr[b] : q_indptr[b + 1]] = (probs @ values[:, :LATENT]).reshape(
            q_len, num_heads, LATENT
        )
    return (out * case["bmm2_scale"]).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="bench_cake_kimi_k3_mla_auto.json",
        help="JSON output path (a .md table is written next to it)",
    )
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--kv-lens", default="1024,4096,16384")
    ap.add_argument("--backends", default="trtllm-gen,cute-dsl,cake,auto")
    ap.add_argument("--repeat-ms", type=int, default=200)
    a = ap.parse_args()
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    import flashinfer.mla.cake_kimi_k3_mla as cake_kimi
    from flashinfer.mla.cake_kimi_k3_mla import workspace_bytes
    from flashinfer.testing.utils import bench_gpu_time
    from flashinfer.utils import get_compute_capability

    device = torch.device("cuda")
    cc = get_compute_capability(device)
    arch = f"sm_{cc[0]}{cc[1]}a"
    gpu = torch.cuda.get_device_name(device)
    num_heads = 12
    backends = a.backends.split(",")
    rows = []
    # Count calls into the real Cake launcher so "auto reached Cake" is a fact, not an inference.
    real_launch = cake_kimi.run_cake_kimi_k3_mla_fp8_paged_attention
    cake_calls = []

    def counting(*args, **kwargs):
        cake_calls.append(1)
        return real_launch(*args, **kwargs)

    cake_kimi.run_cake_kimi_k3_mla_fp8_paged_attention = counting
    os.environ.pop(cake_kimi.AUTO_DISABLE_ENV, None)

    for batch in (int(x) for x in a.batches.split(",")):
        for kv_len in (int(x) for x in a.kv_lens.split(",")):
            # kv_lens: a spread around kv_len so pages differ per request; the longest decides the table width.
            kv_lens = [
                max(1, kv_len - (i * 37) % min(kv_len, 300)) for i in range(batch)
            ]
            kv_lens[0] = kv_len
            seed = 7_000_000 + batch * 1000 + kv_len
            case = _make_case(
                batch, [1] * batch, kv_lens, num_heads, seed=seed, device=device
            )
            width = int(case["block_tables"].shape[1])
            if (
                width % 2
            ):  # the auto / trtllm-gen dispatch requires an even page-64 table width
                pad = torch.zeros((batch, 1), dtype=torch.int32, device=device)
                case["block_tables"] = torch.cat(
                    [case["block_tables"], pad], dim=1
                ).contiguous()
                width += 1
            max_seq_len = width * PAGE
            ref = _reference(case).float()
            query = case["query"].reshape(batch, 1, num_heads, QK_DIM)
            ws_bytes = max(workspace_bytes(batch * num_heads, 256), 256 << 20)
            row = {
                "arch": arch,
                "gpu": gpu,
                "batch": batch,
                "kv_len": kv_len,
                "table_width": width,
                "max_seq_len": max_seq_len,
                "backends": {},
            }
            outs = {}
            for backend in backends:
                workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
                out = torch.full(
                    (batch, 1, num_heads, LATENT),
                    float("nan"),
                    dtype=torch.bfloat16,
                    device=device,
                )
                kwargs = dict(
                    kv_cache=case["kv_cache"],
                    workspace_buffer=workspace,
                    qk_nope_head_dim=128,
                    kv_lora_rank=LATENT,
                    qk_rope_head_dim=ROPE,
                    block_tables=case["block_tables"],
                    seq_lens=case["seq_lens"],
                    max_seq_len=max_seq_len,
                    bmm1_scale=case["bmm1_scale"],
                    bmm2_scale=case["bmm2_scale"],
                    backend=backend,
                )
                call = lambda: trtllm_batch_decode_with_kv_cache_mla(
                    query, out=out, **kwargs
                )  # noqa: E731
                entry = {}
                try:
                    n0 = len(cake_calls)
                    call()
                    torch.cuda.synchronize()
                    entry["cake_launcher_calls"] = len(cake_calls) - n0
                    o = out.float().reshape(ref.shape)
                    err = (o - ref).abs()
                    entry["finite"] = bool(torch.isfinite(o).all())
                    entry["max_abs_err"] = float(err.max())
                    entry["max_abs_ref"] = float(ref.abs().max())
                    entry["max_err_over_max_ref"] = float(
                        err.max() / ref.abs().max().clamp_min(1e-12)
                    )
                    entry["within_1e-2"] = bool(
                        torch.allclose(o, ref, atol=1e-2, rtol=1e-2)
                    )
                    outs[backend] = out.clone()
                    times = bench_gpu_time(
                        call,
                        enable_cupti=True,
                        cold_l2_cache=True,
                        dry_run_time_ms=50,
                        repeat_time_ms=a.repeat_ms,
                    )
                    times = [float(t) for t in times]
                    entry["median_ms"] = statistics.median(times)
                    entry["min_ms"] = min(times)
                    entry["iters"] = len(times)
                except (
                    Exception
                ) as exc:  # a backend that declines the shape is evidence too
                    entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    traceback.print_exc(limit=2)
                row["backends"][backend] = entry
            if "auto" in outs and "cake" in outs:
                row["auto_equals_cake_bitwise"] = bool(
                    torch.equal(outs["auto"], outs["cake"])
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
            del case, ref, outs
            torch.cuda.empty_cache()
    result = {
        "arch": arch,
        "gpu": gpu,
        "num_heads": num_heads,
        "dtype": "float8_e4m3fn",
        "page_size": PAGE,
        "timer": "flashinfer.testing.utils.bench_gpu_time(enable_cupti=True, cold_l2_cache=True)",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": rows,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    md = [
        "| arch | batch | max_seq_len | "
        + " | ".join(f"{b} ms" for b in backends)
        + " | cake vs trtllm-gen | cake vs cute-dsl | auto->Cake | max abs err vs FP32 ref (trtllm-gen / cute-dsl / cake) | max abs ref | all within atol=rtol=1e-2 |",
        "|" + "---|" * (3 + len(backends) + 6),
    ]
    for r in rows:
        b = r["backends"]

        def ms(name):
            e = b.get(name, {})
            return f"{e['median_ms']:.4f}" if "median_ms" in e else "n/a"

        def speed(inc):
            e, c = b.get(inc, {}), b.get("cake", {})
            return (
                f"{e['median_ms'] / c['median_ms']:.2f}x"
                if "median_ms" in e and "median_ms" in c
                else "n/a"
            )

        c = b.get("cake", {})
        au = b.get("auto", {})

        def aerr(name):
            e = b.get(name, {})
            return f"{e['max_abs_err']:.3g}" if "max_abs_err" in e else "n/a"

        within = all(
            b.get(x, {}).get("within_1e-2")
            for x in backends
            if "median_ms" in b.get(x, {})
        )
        md.append(
            f"| {r['arch']} | {r['batch']} | {r['max_seq_len']} | "
            + " | ".join(ms(x) for x in backends)
            + f" | {speed('trtllm-gen')} | {speed('cute-dsl')} | {au.get('cake_launcher_calls', 'n/a')} call(s), bitwise={r.get('auto_equals_cake_bitwise', 'n/a')}"
            f" | {aerr('trtllm-gen')} / {aerr('cute-dsl')} / {aerr('cake')} | {c.get('max_abs_ref', float('nan')):.3g} | {within} |"
        )
    with open(os.path.splitext(a.out)[0] + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("BENCH_DONE", a.out)


if __name__ == "__main__":
    main()
