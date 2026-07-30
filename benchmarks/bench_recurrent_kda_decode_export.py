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

"""Public-API CUPTI harness for the frozen recurrent-KDA decode export.

Run this script in separate processes with ``PYTHONPATH`` pointing at the
pinned current-upstream, evolution-peer, or candidate checkout. Every mode
calls ``flashinfer.kda_decode.recurrent_kda`` with identical D128/T5 inputs.
The caller can alternate mode order across processes to form paired rounds.
"""

import argparse
import functools
import importlib
import json
import math
import statistics
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.kda_decode import recurrent_kda
from flashinfer.testing import bench_gpu_time


UPSTREAM_MAIN_SHA = "a02d94de5796650ead1c6be27b834c3a063bf45d"
EVOLUTION_PEER_SHA = "cea7f46ffc190cabf82c95a39cd0d2aa6c888c17"
CASES = tuple(
    {
        "name": f"d128_t5_b{batch_size}_h16_hv32_precomputed",
        "D": 128,
        "T": 5,
        "N": batch_size,
        "H": 16,
        "HV": 32,
        "expected_variant": "d128_t5_precomputed_gram_split1",
    }
    for batch_size in (8, 16, 32, 64, 128)
)


def _make_case(spec: dict, device: torch.device) -> dict:
    head_dim = spec["D"]
    num_tokens = spec["T"]
    num_sequences = spec["N"]
    num_heads = spec["H"]
    num_value_heads = spec["HV"]
    total_tokens = num_sequences * num_tokens
    generator = torch.Generator(device=device).manual_seed(42)

    q = torch.rand(
        (1, total_tokens, num_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.rand(
        (1, total_tokens, num_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.rand(
        (1, total_tokens, num_value_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    g = F.logsigmoid(
        torch.randn(
            (1, total_tokens, num_value_heads, head_dim),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    ).to(torch.bfloat16)
    beta = torch.sigmoid(
        torch.randn(
            (1, total_tokens, num_value_heads),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    ).to(torch.bfloat16)
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        num_tokens,
        dtype=torch.int32,
        device=device,
    )
    ssm_state_indices = torch.arange(
        1,
        num_sequences * num_tokens + 1,
        dtype=torch.int32,
        device=device,
    ).reshape(num_sequences, num_tokens)
    num_accepted_tokens = torch.ones(
        num_sequences,
        dtype=torch.int32,
        device=device,
    )
    state = (
        torch.randn(
            (
                num_sequences * num_tokens + 6,
                num_value_heads,
                head_dim,
                head_dim,
            ),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.01
    ).to(torch.bfloat16)
    output = torch.full_like(v, 123.0)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": None,
        "dt_bias": None,
        "scale": head_dim**-0.5,
        "initial_state": state,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": False,
        "lower_bound": None,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_spec_tokens": num_tokens - 1,
        "num_accepted_tokens": num_accepted_tokens,
        "output": output,
    }


def _assert_frozen_route(spec: dict, kwargs: dict) -> None:
    recurrent_module = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")
    selected = recurrent_module._select_flash_kda_decode_variant(
        q=kwargs["q"],
        k=kwargs["k"],
        v=kwargs["v"],
        g=kwargs["g"],
        beta=kwargs["beta"],
        state=kwargs["initial_state"],
        out=kwargs["output"],
        cu_seqlens=kwargs["cu_seqlens"],
        ssm_state_indices=kwargs["ssm_state_indices"].view(-1),
        num_accepted_tokens=kwargs["num_accepted_tokens"],
        scale=kwargs["scale"],
        num_tokens=spec["T"],
        num_spec_tokens=spec["T"] - 1,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        lower_bound=None,
        A_log=None,
        dt_bias=None,
        initial_state_source=None,
        beta_is_logit=False,
    )
    if selected != spec["expected_variant"]:
        raise AssertionError(
            f"{spec['name']} expected {spec['expected_variant']}, got {selected}"
        )


def _check_source(args: argparse.Namespace) -> str:
    imported_root = Path(flashinfer.__file__).resolve().parents[1]
    expected_root = args.expected_source_root.resolve()
    if imported_root != expected_root:
        raise RuntimeError(
            f"expected flashinfer from {expected_root}, imported {imported_root}"
        )
    actual_source_sha = subprocess.run(
        ["git", "-C", str(expected_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_status = subprocess.run(
        [
            "git",
            "-C",
            str(expected_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_status:
        raise RuntimeError(f"benchmark source checkout must be clean:\n{source_status}")
    if actual_source_sha != args.expected_source_sha:
        raise RuntimeError(
            f"expected source SHA {args.expected_source_sha}, got {actual_source_sha}"
        )
    pinned = {
        "upstream": UPSTREAM_MAIN_SHA,
        "evolution-peer": EVOLUTION_PEER_SHA,
    }.get(args.mode)
    if pinned is not None and actual_source_sha != pinned:
        raise RuntimeError(
            f"{args.mode} mode must use pinned {pinned}, got {actual_source_sha}"
        )
    return actual_source_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("upstream", "evolution-peer", "frozen"),
        required=True,
    )
    parser.add_argument("--expected-source-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds <= 0 or args.warmup <= 0:
        parser.error("--rounds and --warmup must be positive")

    try:
        from cupti import cupti  # noqa: F401

        cupti_python_version = version("cupti-python")
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError("reportable timings require cupti-python >= 13") from error
    if int(cupti_python_version.split(".", 1)[0]) < 13:
        raise RuntimeError(
            f"reportable timings require cupti-python >= 13, got {cupti_python_version}"
        )

    actual_source_sha = _check_source(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("this benchmark requires exact B200 / sm_100a")

    rows = []
    for spec in CASES:
        kwargs = _make_case(spec, device)
        if args.mode == "frozen":
            _assert_frozen_route(spec, kwargs)
        initial_state = kwargs["initial_state"].clone()
        measured_run = functools.partial(recurrent_kda, **kwargs)
        warmup_kwargs = dict(kwargs)
        warmup_kwargs["initial_state"] = initial_state.clone()
        warmup_kwargs["output"] = torch.empty_like(kwargs["output"])
        warmup_run = functools.partial(recurrent_kda, **warmup_kwargs)

        measured_run()
        torch.cuda.synchronize()
        samples_ms = []
        for _ in range(args.rounds):
            kwargs["initial_state"].copy_(initial_state)
            kwargs["output"].fill_(123.0)
            warmup_kwargs["initial_state"].copy_(initial_state)
            warmup_kwargs["output"].fill_(123.0)
            torch.cuda.synchronize()

            # FlashInfer's CUPTI helper performs one preflight launch and five
            # estimate launches before its explicit dry runs. Keep all of
            # those launches on a separate state/output allocation so the one
            # measured public-API call sees the same restored inputs in every
            # round.
            warmup_calls = 6 + args.warmup
            call_count = 0

            def staged_run():
                nonlocal call_count
                call_count += 1
                if call_count <= warmup_calls:
                    return warmup_run()
                if call_count == warmup_calls + 1:
                    return measured_run()
                raise RuntimeError(
                    "unexpected extra recurrent_kda call inside one-sample timing"
                )

            measured = [
                float(value)
                for value in bench_gpu_time(
                    staged_run,
                    enable_cupti=True,
                    cold_l2_cache=True,
                    use_cuda_graph=False,
                    dry_run_iters=args.warmup,
                    repeat_iters=1,
                )
            ]
            if call_count != warmup_calls + 1:
                raise RuntimeError(
                    f"expected {warmup_calls + 1} public-API calls, got {call_count}"
                )
            if len(measured) != 1:
                raise RuntimeError(
                    f"expected one timed sample for {spec['name']}, got {measured}"
                )
            samples_ms.append(measured[0])

        median_ms = float(statistics.median(samples_ms))
        if not math.isfinite(median_ms) or median_ms <= 0.0:
            raise RuntimeError(f"invalid timing for {spec['name']}: {median_ms}")
        row = {
            **spec,
            "mode": args.mode,
            "median_ms": median_ms,
            "samples_ms": samples_ms,
            "timing_backend": "CUPTI",
            "cupti_python_version": cupti_python_version,
            "cold_l2": True,
            "cuda_graph": False,
            "timing_scope": "public_recurrent_kda_gpu_activity",
            "upstream_main_sha": UPSTREAM_MAIN_SHA,
            "evolution_peer_sha": EVOLUTION_PEER_SHA,
            "source_sha": actual_source_sha,
            "sampling_protocol": {
                "rounds": args.rounds,
                "warmup_iters_per_round": args.warmup,
                "timed_iters_per_round": 1,
            },
        }
        rows.append(row)
        print(f"{args.mode:<14} {spec['name']:<43} {median_ms * 1000.0:10.3f} us")
        torch.cuda.empty_cache()

    args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
