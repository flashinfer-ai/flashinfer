# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare the HY3 fused sampler with existing FlashInfer sampling paths.

The benchmark follows the HPC-Ops workload (BF16 logits, vocabulary 120832,
temperature-only and repetition-penalty/top-k/top-p scenes), while keeping
input construction, history packing, allocation, and JIT compilation outside
the timed region. Both providers generate random numbers inside their kernels;
external Gumbel tensors are intentionally reserved for correctness tests.

Usage:
$ python -m benchmarks.bench_sampling_hy3
$ python -m benchmarks.bench_sampling_hy3 --batches 32 128 512 --hot-l2
"""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


VOCAB_SIZE = 120832
REPETITION_PENALTY = 1.1
TEMPERATURE = 1.05
TOP_K = 20
TOP_P = 0.9
SEED = 1
OFFSET = 0


def _measure(call, args):
    values = bench_gpu_time(
        call,
        dry_run_iters=args.warmup,
        repeat_iters=args.iters,
        enable_cupti=not args.cuda_events,
        cold_l2_cache=not args.hot_l2,
    )
    values = np.asarray(values, dtype=np.float64) * 1e3
    return (
        float(np.median(values)),
        float(np.percentile(values, 20)),
        float(np.percentile(values, 80)),
    )


def _make_history_masks(batch_size, device):
    # A 40% occupancy approximates the unique-token density after drawing two
    # 32K-token histories from a 120832-token vocabulary. Build both layouts
    # from the same mask so the providers see identical repetition state.
    generator = torch.Generator(device=device).manual_seed(batch_size)
    dense = (
        torch.rand((batch_size, VOCAB_SIZE), generator=generator, device=device) < 0.4
    )
    bits = dense.view(batch_size, VOCAB_SIZE // 8, 8).to(torch.uint8)
    weights = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device
    )
    packed = (bits * weights).sum(dim=-1, dtype=torch.uint8)
    return dense, packed


@torch.inference_mode()
def _benchmark_batch(batch_size, args):
    device = torch.device(f"cuda:{args.device}")
    generator = torch.Generator(device=device).manual_seed(1000 + batch_size)
    logits = torch.randn(
        (batch_size, VOCAB_SIZE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
    output = torch.empty((batch_size, 1), dtype=torch.int32, device=device)
    temperature_work = torch.empty_like(logits, dtype=torch.float32)
    full_work = torch.empty_like(temperature_work)
    positive_scaled = torch.empty_like(full_work)
    negative_scaled = torch.empty_like(full_work)
    adjusted_work = torch.empty_like(full_work)
    sample_work = torch.empty_like(full_work)
    positive_mask = torch.empty_like(logits, dtype=torch.bool)
    dense_history, packed_history = _make_history_masks(batch_size, device)
    dense_history = dense_history.clone()
    packed_history = packed_history.clone()
    slots = torch.arange(batch_size, dtype=torch.int32, device=device)
    rows = torch.arange(batch_size, dtype=torch.int64, device=device)

    def hy3_temperature():
        return flashinfer.fused_sampling_from_logits_hy3(
            logits,
            workspace_buffer=workspace,
            out=output,
            temperature=TEMPERATURE,
            seed=SEED,
            offset=OFFSET,
        )

    def flashinfer_temperature_logits():
        temperature_work.copy_(logits)
        temperature_work.mul_(1.0 / TEMPERATURE)
        return flashinfer.sampling_from_logits(
            temperature_work,
            deterministic=True,
            seed=SEED,
            offset=OFFSET,
        )

    def flashinfer_temperature_probs():
        probs = flashinfer.softmax(logits, temperature=TEMPERATURE)
        return flashinfer.sampling_from_probs(
            probs,
            deterministic=True,
            seed=SEED,
            offset=OFFSET,
        )

    def hy3_full():
        return flashinfer.fused_sampling_from_logits_hy3(
            logits,
            workspace_buffer=workspace,
            out=output,
            penalty_mask=packed_history,
            slot_id=slots,
            repetition_penalty=REPETITION_PENALTY,
            temperature=TEMPERATURE,
            softmax_policy=flashinfer.sampling.HY3_SAMPLER_SOFTMAX_AFTER_TOP_K,
            top_k=TOP_K,
            top_p=TOP_P,
            max_top_k=32,
            seed=SEED,
            offset=OFFSET,
        )

    def flashinfer_full():
        full_work.copy_(logits)
        positive_scaled.copy_(full_work).div_(REPETITION_PENALTY)
        negative_scaled.copy_(full_work).mul_(REPETITION_PENALTY)
        torch.gt(full_work, 0, out=positive_mask)
        torch.where(
            positive_mask,
            positive_scaled,
            negative_scaled,
            out=adjusted_work,
        )
        torch.where(dense_history, adjusted_work, full_work, out=sample_work)
        sample_work.mul_(1.0 / TEMPERATURE)
        token = flashinfer.top_k_top_p_sampling_from_logits(
            sample_work,
            TOP_K,
            TOP_P,
            filter_apply_order="top_k_first",
            deterministic=True,
            seed=SEED,
            offset=OFFSET,
        )
        dense_history[rows, token.long()] = True
        return token

    # Build/load all JIT modules and cached workspaces before timing.
    for call in (
        hy3_temperature,
        flashinfer_temperature_logits,
        flashinfer_temperature_probs,
        hy3_full,
        flashinfer_full,
    ):
        call()
    torch.cuda.synchronize(device)

    results = {
        "hy3_temperature": _measure(hy3_temperature, args),
        "flashinfer_temperature_logits": _measure(flashinfer_temperature_logits, args),
        "flashinfer_temperature_probs": _measure(flashinfer_temperature_probs, args),
        "hy3_full": _measure(hy3_full, args),
        "flashinfer_full": _measure(flashinfer_full, args),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--hot-l2",
        action="store_true",
        help="reuse hot inputs instead of flushing L2 between measurements",
    )
    parser.add_argument(
        "--cuda-events",
        action="store_true",
        help="use CUDA events instead of the preferred CUPTI timer",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("HY3 fused sampling benchmark skipped: CUDA is unavailable")
        return
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability(args.device) != (10, 0):
        print("HY3 fused sampling benchmark skipped: requires SM100/B200")
        return

    cache_mode = "hot" if args.hot_l2 else "cold"
    timer = "cuda_event" if args.cuda_events else "cupti_or_event_fallback"
    print(f"# cache={cache_mode}, timer={timer}, dtype=bf16, vocab={VOCAB_SIZE}")
    print("scene,batch,provider,median_us,p20_us,p80_us,speedup_vs_flashinfer_best")
    for batch_size in args.batches:
        results = _benchmark_batch(batch_size, args)
        fi_temperature_best = min(
            results["flashinfer_temperature_logits"][0],
            results["flashinfer_temperature_probs"][0],
        )
        hy3_temperature = results["hy3_temperature"][0]
        full_speedup = results["flashinfer_full"][0] / results["hy3_full"][0]
        for scene, providers, speedup in (
            (
                "temperature",
                (
                    "hy3_temperature",
                    "flashinfer_temperature_logits",
                    "flashinfer_temperature_probs",
                ),
                fi_temperature_best / hy3_temperature,
            ),
            ("full", ("hy3_full", "flashinfer_full"), full_speedup),
        ):
            for provider in providers:
                median, p20, p80 = results[provider]
                row_speedup = speedup if provider.startswith("hy3_") else 1.0
                print(
                    f"{scene},{batch_size},{provider},{median:.3f},{p20:.3f},"
                    f"{p80:.3f},{row_speedup:.3f}"
                )


if __name__ == "__main__":
    main()
