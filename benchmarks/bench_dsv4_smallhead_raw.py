# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare raw-index decode with actual vLLM packing plus FI attention on B300.

Requires vLLM's DSv4.1 mixed sparse-index builder (tested with vLLM
0.32.0rc1.dev100+g77154ca61). No model checkpoint is loaded. All baseline arms
include the real framework index producer; the raw arm consumes the same
logical indices directly. CUDA Graph intervals include GPU work only.

The pools are small and warm, with padded physical pages and nonzero origins.
These synthetic component times do not predict model traffic or serving TPS.
Run twice, once with --reverse. Scheduling-budget controls change only the
original launcher's logical SM budget, not physical GPU allocation.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import statistics
import sys
from types import FunctionType

import torch
import flashinfer
from flashinfer.mla import _core, _smallhead, _smallhead_raw
from flashinfer.mla.dsv4_raw import (
    get_dsv4_raw_decode_workspace_size,
    try_batch_decode_sparse_mla_dsv4_raw,
)
from flashinfer.testing.utils import bench_gpu_time_with_cudagraph
from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes
from bench_dsv4_smallhead import _fixture, _reference, _check_fp64


def _raw_fixture(rows, heads, page):
    packed, pools = _fixture(rows, heads, "HND")
    # Retain nonzero origins and padded physical pages from the shared fixture.
    pools[1] = pools[1][:, :, :page]
    packed["compressed_kv_cache"] = pools[1]
    names = (
        "query",
        "swa_kv_cache",
        "compressed_kv_cache",
        "seq_lens",
        "sinks",
        "bmm1_scale",
        "bmm2_scale",
        "enable_pdl",
    )
    raw = {name: packed[name] for name in names}
    raw["query"] = packed["query"].view(rows, heads, 512)
    raw.update(
        swa_indices=torch.empty((rows + 1, 128), device="cuda", dtype=torch.int32)[1:],
        compressed_indices=torch.empty(
            (rows + 1, 512), device="cuda", dtype=torch.int32
        )[1:],
        request_ids=torch.empty(rows + 4, device="cuda", dtype=torch.int32)[4:],
        compressed_block_table=torch.empty(
            (rows + 1, 16), device="cuda", dtype=torch.int32
        )[1:],
        token_valid=torch.empty(rows + 16, device="cuda", dtype=torch.bool)[16:],
        out=torch.empty((rows + 1, heads, 512), device="cuda", dtype=torch.bfloat16)[
            1:
        ],
        workspace_buffer=torch.empty(
            get_dsv4_raw_decode_workspace_size(rows, heads),
            device="cuda",
            dtype=torch.uint8,
        ),
    )
    return raw, packed, pools


def _change(raw, packed, pools, generation):
    torch.manual_seed(831 + generation * 31)
    rows, heads, _ = raw["query"].shape
    page = pools[1].shape[2]
    raw["query"].copy_(
        (torch.randn_like(raw["query"], dtype=torch.float32) * 32).to(
            raw["query"].dtype
        )
    )
    for pool in pools:
        pool.copy_((torch.randn_like(pool, dtype=torch.float32) * 8).to(pool.dtype))
    for name, limit in (("swa_indices", 16 * 32), ("compressed_indices", 16 * page)):
        raw[name].copy_(
            torch.randint(limit, raw[name].shape, device="cuda", dtype=torch.int32)
        )
    raw["request_ids"].copy_(
        (torch.arange(rows, device="cuda", dtype=torch.int32) + generation) % rows
    )
    raw["compressed_block_table"].copy_(
        torch.randint(16, (rows, 16), device="cuda", dtype=torch.int32)
    )
    raw["token_valid"].fill_(True)
    raw["swa_indices"][:, 3::37] = -1
    raw["seq_lens"].fill_(7 if generation == 1 else 8192)
    raw["sinks"].copy_(torch.linspace(-2, 2, heads, device="cuda"))
    if generation == 1:
        raw["compressed_indices"][:, 479:] = -1
    elif generation == 2:
        raw["swa_indices"][0].fill_(-1)
        raw["compressed_indices"][0].fill_(-1)
    elif generation == 3:
        raw["token_valid"].fill_(False)
        pools[1].fill_(float("nan"))
    elif generation == 4:
        raw["compressed_indices"][:, 3::37] = -1
        raw["compressed_block_table"][:, 3] = -1
    elif generation == 5:
        raw["sinks"].fill_(10000)

    # Independent explicit mapping for the existing packed attention contract.
    # Count defines a prefix; do not compact holes or drop invalid page entries.
    swa = raw["swa_indices"].long()
    local = raw["compressed_indices"].long()
    requests = raw["request_ids"].long()
    physical = raw["compressed_block_table"][
        requests[:, None], local.clamp_min(0) // page
    ]
    packed["sparse_indices"][:, :128] = torch.where(
        swa >= 0, swa // 32 * 364 + swa % 32, -1
    )
    packed["sparse_indices"][:, 128:] = torch.where(
        (local >= 0) & (physical >= 0), physical.long() * 364 + local % page, -1
    )
    packed["sparse_topk_lens"].copy_(
        128 + ((local >= 0) & raw["token_valid"][:, None]).sum(-1)
    )


def _packed_run(packed):
    names = (
        "query",
        "swa_kv_cache",
        "compressed_kv_cache",
        "sparse_indices",
        "sparse_topk_lens",
        "seq_lens",
        "sinks",
        "workspace_buffer",
        "out",
        "bmm1_scale",
        "bmm2_scale",
        "enable_pdl",
    )
    kwargs = {name: packed[name] for name in names}
    kwargs["query"] = kwargs["query"].squeeze(1)
    kwargs["out"] = kwargs["out"].squeeze(1)
    return _smallhead.run(**kwargs)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, choices=(16, 32), default=16)
    parser.add_argument("--rows", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--page", type=int, choices=(64, 128), default=128)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    assert torch.cuda.get_device_capability() == (10, 3)
    assert options.rows in ((1, 2, 4) if options.heads == 16 else (1, 2))
    assert _smallhead._dsl_unavailable_reason() is None
    # Use the framework's actual producer rather than a Python/Torch surrogate.
    from vllm.models.deepseek_v41.common.ops import (
        build_flashinfer_mixed_sparse_indices as pack,
    )

    raw, packed, pools = _raw_fixture(options.rows, options.heads, options.page)
    raw["out"] = packed["out"].squeeze(1)
    raw["workspace_buffer"] = packed["workspace_buffer"]
    query_start = torch.arange(options.rows + 1, device="cuda", dtype=torch.int32)
    replay_start = torch.zeros(options.rows, device="cuda", dtype=torch.int32)

    def build_indices():
        return pack(
            raw["swa_indices"],
            raw["compressed_indices"],
            None,
            raw["compressed_indices"][:0, :0],
            query_start,
            raw["seq_lens"],
            raw["request_ids"],
            raw["compressed_block_table"],
            32,
            raw["compressed_block_table"],
            options.page,
            128,
            128 // options.page,
            0,
            decode_compressed_indices_are_local=True,
            decode_is_valid_token=raw["token_valid"],
            swa_block_span=364,
            compressed_block_span=364,
            replay_start=replay_start,
        )

    native = _core.trtllm_batch_decode_sparse_mla_dsv4
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    counters, calls = [], []
    for budget in (16, 32, 64, 96, sms):
        namespace = dict(native.__globals__)
        namespace["get_device_sm_count"] = lambda unused, budget=budget: budget
        clone = FunctionType(
            native.__code__,
            namespace,
            native.__name__,
            native.__defaults__,
            native.__closure__,
        )
        clone.__kwdefaults__ = dict(native.__kwdefaults__ or {})
        assert {
            key for key in namespace if namespace[key] is not native.__globals__[key]
        } == {"get_device_sm_count"}
        for pdl in (False, True):
            counter = torch.zeros(
                get_trtllm_gen_multi_ctas_kv_counter_bytes(
                    options.rows, options.heads, sms
                ),
                dtype=torch.uint8,
                device="cuda",
            )
            counters.append(counter)

            def call(clone=clone, pdl=pdl, counter=counter):
                ids, lengths = build_indices()
                return clone(
                    **dict(
                        packed,
                        sparse_indices=ids,
                        sparse_topk_lens=lengths,
                        backend="trtllm-gen",
                        enable_pdl=pdl,
                        multi_ctas_kv_counter_buffer=counter,
                    )
                )

            calls.append((f"pack_native_sm{budget}_pdl{int(pdl)}", call))

    def default():
        ids, lengths = build_indices()
        return native(
            **dict(
                packed,
                sparse_indices=ids,
                sparse_topk_lens=lengths,
                backend="trtllm-gen",
            )
        )

    def cute_packed():
        ids, lengths = build_indices()
        return _packed_run(dict(packed, sparse_indices=ids, sparse_topk_lens=lengths))

    def raw_call():
        assert try_batch_decode_sparse_mla_dsv4_raw(**raw), (
            "Raw helper unexpectedly declined"
        )

    calls += [
        ("pack_native_default", default),
        ("pack_cutedsl", cute_packed),
        ("raw_cutedsl", raw_call),
    ]
    report = dict(
        scope=__doc__,
        heads=options.heads,
        rows=options.rows,
        compressed_page=options.page,
        gpu=torch.cuda.get_device_name(),
        physical_sms=sms,
        torch_version=torch.__version__,
        flashinfer_version=flashinfer.__version__,
        vllm_version=importlib.metadata.version("vllm"),
        dsl_version=importlib.metadata.version("nvidia-cutlass-dsl"),
        reverse=options.reverse,
        cache="warm synthetic padded KV pools",
        checks=[],
        samples=[],
        sources={},
    )
    for name, obj in [
        ("framework_packer", pack),
        ("native_host", _core),
        ("packed_host", _smallhead),
        ("raw_public_helper", try_batch_decode_sparse_mla_dsv4_raw),
        ("raw_host", _smallhead_raw),
        (
            "raw_kernel",
            importlib.import_module("flashinfer.cute_dsl.attention.dsv4_smallhead_raw"),
        ),
        (
            "packed_kernel",
            importlib.import_module("flashinfer.cute_dsl.attention.dsv4_smallhead"),
        ),
        ("benchmark", sys.modules[__name__]),
    ]:
        path = Path(inspect.getsourcefile(obj))
        report["sources"][name] = dict(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def validate(arm, expected):
        if arm in ("pack_cutedsl", "raw_cutedsl"):
            _check_fp64(raw["out"], expected)
        else:
            torch.testing.assert_close(
                raw["out"].double(), expected, rtol=0.1, atol=0.1
            )
        assert all(int(torch.count_nonzero(counter)) == 0 for counter in counters)

    for generation in range(6):
        _change(raw, packed, pools, generation)
        reference = _reference(packed, pools)
        actual_ids, actual_lens = build_indices()
        assert torch.equal(actual_ids, packed["sparse_indices"])
        assert torch.equal(actual_lens, packed["sparse_topk_lens"])
        packed_output = None
        for arm, call in calls:
            raw["out"].fill_(float("nan"))
            call()
            validate(arm, reference)
            if arm == "pack_cutedsl":
                packed_output = raw["out"].view(torch.uint8).clone()
            if arm == "raw_cutedsl":
                assert torch.equal(raw["out"].view(torch.uint8), packed_output)
            report["checks"].append(dict(generation=generation, arm=arm))
    _change(raw, packed, pools, 0)
    reference = _reference(packed, pools)
    order = list(range(len(calls)))
    if options.reverse:
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
        values = [float(value) * 1000 for value in values]
        validate(arm, reference)
        timings.setdefault(arm, []).append(statistics.median(values))
        report["samples"].append(dict(arm=arm, samples_us=values))
    best_native = min(
        (min(values), arm)
        for arm, values in timings.items()
        if arm.startswith("pack_native")
    )
    candidate = max(timings["raw_cutedsl"])
    report.update(
        bank_medians_us=timings,
        best_native_arm=best_native[1],
        best_native_us=best_native[0],
        raw_us=candidate,
        packed_cutedsl_us=min(timings["pack_cutedsl"]),
        ratio_vs_best_native=best_native[0] / candidate,
        ratio_vs_packed_cutedsl=min(timings["pack_cutedsl"]) / candidate,
        status="passed",
    )
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("samples", "checks", "sources", "bank_medians_us")
            }
        )
    )


if __name__ == "__main__":
    main()
