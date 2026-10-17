# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce B300 small-head DSv4 FP8 decode warm-cache GPU comparisons.

Run this script twice, once with --reverse, using the same FlashInfer checkout.
It measures amortized CUDA Graph GPU intervals with the repository timer.
Synthetic padded cache pools reproduce the storage-offset contract, not model
traffic or serving throughput. Native scheduling controls only override the
launcher's logical SM budget; they do not partition the GPU or change math.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
from types import FunctionType

import torch
import flashinfer
from flashinfer.mla import _core, _smallhead
from flashinfer.testing.utils import bench_gpu_time_with_cudagraph
from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes


def _fixture(rows=1, heads=16, layout="HND", width=640):
    # Nonzero origins and physical page padding reproduce the framework's
    # cache contract without allocating the complete model working set.
    # Public ordinary decode requires [batch, query_length, heads, dim] for
    # every KV layout. Rank3 below refers only to the cache, not to the query.
    dtype = torch.float8_e4m3fn
    query = torch.empty((rows + 1, 1, heads, 512), device="cuda", dtype=dtype)[1:]
    pools = []
    for page, origin in ((32, 288), (128, 219)):
        storage = torch.empty((origin + 16 * 364, 512), device="cuda", dtype=dtype)
        storage.fill_(float("nan"))
        pools.append(
            storage.as_strided(
                (16, 1, page, 512), (364 * 512, page * 512, 512, 1), origin * 512
            )
        )
    args = dict(
        query=query,
        swa_kv_cache=pools[0],
        compressed_kv_cache=pools[1],
        sparse_indices=torch.empty((rows + 1, width), dtype=torch.int32, device="cuda")[
            1:
        ],
        sparse_topk_lens=torch.empty(rows + 4, dtype=torch.int32, device="cuda")[4:],
        seq_lens=torch.empty(rows + 4, dtype=torch.int32, device="cuda")[4:],
        sinks=torch.empty(heads + 4, dtype=torch.float32, device="cuda")[4:],
        workspace_buffer=torch.empty(
            64 * 1024 * 1024 + 16, dtype=torch.uint8, device="cuda"
        )[16:],
        out=torch.empty((rows + 1, 1, heads, 512), dtype=torch.bfloat16, device="cuda")[
            1:
        ],
        bmm1_scale=512**-0.5 / (64 * 32),
        bmm2_scale=1 / 32,
        max_q_len=1,
        sparse_indices_are_storage_offsets=True,
        kv_layout="HND" if layout == "rank3" else layout,
        enable_pdl=True,
    )
    if layout == "NHD":
        args["swa_kv_cache"] = pools[0].transpose(1, 2)
        args["compressed_kv_cache"] = pools[1].transpose(1, 2)
    elif layout == "rank3":
        args["swa_kv_cache"] = pools[0][:, 0]
        args["compressed_kv_cache"] = pools[1][:, 0]
    return args, pools


def _change(args, pools, generation):
    rows = args["query"].shape[0]
    width = args["sparse_indices"].shape[1]
    torch.manual_seed(713 + generation)
    args["query"].copy_(
        (torch.randn_like(args["query"], dtype=torch.float32) * 32).to(
            args["query"].dtype
        )
    )
    for begin, end, pool in ((0, 128, pools[0]), (128, width, pools[1])):
        pool.copy_((torch.randn_like(pool, dtype=torch.float32) * 8).to(pool.dtype))
        logical = torch.randint(
            0,
            pool.shape[0] * pool.shape[2],
            (rows, end - begin),
            device="cuda",
            dtype=torch.int32,
        )
        args["sparse_indices"][:, begin:end] = (
            logical // pool.shape[2] * 364 + logical % pool.shape[2]
        )
    args["sparse_indices"][:, 3::37] = -1
    args["sparse_topk_lens"].fill_(width if generation != 1 else max(128, width - 33))
    args["seq_lens"].fill_(7 if generation == 1 else 8192)
    args["sinks"].copy_(torch.linspace(-2, 2, args["sinks"].numel(), device="cuda"))
    if generation == 2:
        args["sparse_indices"][0].fill_(-1)


def _reference(args, pools):
    query = args["query"].reshape(-1, args["sinks"].numel(), 512).double()
    ids = args["sparse_indices"].long()
    values, masks = [], []
    for begin, end, pool in ((0, 128, pools[0]), (128, ids.shape[1], pools[1])):
        span = (pool.shape[0] - 1) * 364 + pool.shape[2]
        physical = pool.as_strided((span, 512), (512, 1))
        index = ids[:, begin:end]
        position = torch.arange(begin, end, device="cuda")[None]
        valid = (
            (index >= 0)
            & (index < span)
            & (position < args["sparse_topk_lens"][:, None])
        )
        if begin == 0:
            valid &= position < args["seq_lens"][:, None]
        values.append(
            torch.where(
                valid[..., None], physical[index.clamp(0, span - 1)].double(), 0.0
            )
        )
        masks.append(valid)
    kv, valid = torch.cat(values, 1), torch.cat(masks, 1)
    scores = torch.einsum("bhd,bkd->bhk", query, kv) * args["bmm1_scale"]
    scores.masked_fill_(~valid[:, None], -float("inf"))
    sink = args["sinks"].double()[None, :, None].expand(query.shape[0], -1, 1)
    probabilities = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
    return torch.einsum("bhk,bkd->bhd", probabilities, kv) * args["bmm2_scale"]


def _check_fp64(out, expected):
    actual = out.reshape(expected.shape).double()
    error = actual - expected
    assert bool(torch.isfinite(actual).all())
    assert float(error.norm() / expected.norm().clamp_min(1e-30)) < 0.005
    assert float(error.abs().max() / expected.abs().max().clamp_min(1e-30)) < 0.01


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (10, 3), (
        "This comparison targets B300 SM103"
    )
    assert _smallhead._dsl_unavailable_reason() is None
    native = _core.trtllm_batch_decode_sparse_mla_dsv4
    physical_sms = torch.cuda.get_device_properties(0).multi_processor_count
    budgets = (16, 32, 64, 96, physical_sms)
    assert len(set(budgets)) == 5
    report = dict(
        scope=__doc__,
        gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        flashinfer_version=flashinfer.__version__,
        flashinfer_path=flashinfer.__file__,
        physical_sms=physical_sms,
        reverse=args.reverse,
        cache="warm",
        checks=[],
        rows=[],
        samples=[],
    )
    report["sources"] = {}
    for module in (_core, _smallhead):
        path = Path(inspect.getsourcefile(module))
        report["sources"][module.__name__] = dict(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def native_variant(budget):
        namespace = dict(native.__globals__)
        namespace["get_device_sm_count"] = lambda unused_device: budget
        clone = FunctionType(
            native.__code__,
            namespace,
            native.__name__,
            native.__defaults__,
            native.__closure__,
        )
        clone.__kwdefaults__ = dict(native.__kwdefaults__ or {})
        assert clone.__code__ is native.__code__
        assert {
            key for key in namespace if namespace[key] is not native.__globals__[key]
        } == {"get_device_sm_count"}
        return clone

    for heads in (16, 32):
        for rows in (1, 2, 4, 8):
            inputs, pools = _fixture(rows, heads, "rank3")
            _change(inputs, pools, 0)
            eligible = rows in ((1, 2, 4) if heads == 16 else (1, 2))
            counters, calls = [], []
            for budget in budgets:
                clone = native_variant(budget)
                for pdl in (False, True):
                    counter = torch.zeros(
                        get_trtllm_gen_multi_ctas_kv_counter_bytes(
                            rows, heads, physical_sms
                        ),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    counters.append(counter)
                    kwargs = dict(
                        inputs,
                        backend="trtllm-gen",
                        enable_pdl=pdl,
                        multi_ctas_kv_counter_buffer=counter,
                    )
                    calls.append(
                        (
                            f"native_sm{budget}_pdl{int(pdl)}_owned_counter",
                            lambda clone=clone, kwargs=kwargs: clone(**kwargs),
                        )
                    )
            calls += [
                (
                    "native_default",
                    lambda: native(**dict(inputs, backend="trtllm-gen")),
                ),
                ("public_auto", lambda: native(**inputs)),
            ]
            original_try = _smallhead.try_run
            selected = []

            def traced(**kwargs):
                value = original_try(**kwargs)
                selected.append(value)
                return value

            _smallhead.try_run = traced
            try:
                calls[-1][1]()
            finally:
                _smallhead.try_run = original_try
            assert selected == [eligible], (heads, rows, selected)
            for generation in range(3):
                _change(inputs, pools, generation)
                expected = _reference(inputs, pools)
                for arm, call in calls:
                    inputs["out"].fill_(float("nan"))
                    call()
                    if arm == "public_auto" and eligible:
                        _check_fp64(inputs["out"], expected)
                    else:
                        torch.testing.assert_close(
                            inputs["out"].reshape(expected.shape).double(),
                            expected,
                            rtol=0.1,
                            atol=0.1,
                        )
                    report["checks"].append(
                        dict(heads=heads, rows=rows, generation=generation, arm=arm)
                    )
                assert all(
                    int(torch.count_nonzero(counter)) == 0 for counter in counters
                )
            _change(inputs, pools, 0)
            expected = _reference(inputs, pools)
            order = list(range(len(calls)))
            if args.reverse:
                order.reverse()
            timings = {}
            for index in order + order[::-1]:
                arm, call = calls[index]
                values = bench_gpu_time_with_cudagraph(
                    call,
                    dry_run_iters=8,
                    repeat_iters=32,
                    num_iters_within_graph=128,
                    cold_l2_cache=False,
                )
                microseconds = [float(value) * 1000 for value in values]
                timings.setdefault(arm, []).append(statistics.median(microseconds))
                # The timer has completed captured replays; validate their output,
                # not just the eager warmup. Counters must self-reset every replay.
                if arm == "public_auto" and eligible:
                    _check_fp64(inputs["out"], expected)
                else:
                    torch.testing.assert_close(
                        inputs["out"].reshape(expected.shape).double(),
                        expected,
                        rtol=0.1,
                        atol=0.1,
                    )
                assert all(
                    int(torch.count_nonzero(counter)) == 0 for counter in counters
                )
                report["samples"].append(
                    dict(heads=heads, rows=rows, arm=arm, samples_us=microseconds)
                )
            best_native = min(
                (min(values), name)
                for name, values in timings.items()
                if name != "public_auto"
            )
            candidate = max(timings["public_auto"])
            row = dict(
                heads=heads,
                rows=rows,
                eligible=eligible,
                native_default_us=min(timings["native_default"]),
                best_native=best_native[1],
                best_native_us=best_native[0],
                public_auto_us=candidate,
                ratio_vs_best_native=best_native[0] / candidate,
                bank_medians_us=timings,
            )
            report["rows"].append(row)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "bank_medians_us"
                    }
                ),
                flush=True,
            )
            save()
    assert len(report["checks"]) == 288 and len(report["rows"]) == 8
    report["status"] = "passed_warm_public_gpu_comparison_requires_independent_repeat"
    save()


if __name__ == "__main__":
    main()
