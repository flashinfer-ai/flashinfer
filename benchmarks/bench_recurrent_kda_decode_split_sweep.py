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

"""Sweep frozen recurrent-KDA decode value splits through the public API.

This exact-SM100a/SM103a harness forces every split1/2/4/8 specialization for
the D128, H16, HV32 precomputed-gate T=1/2/4/5/6 contracts. It additionally
sweeps both direct-state T=1 schedules (split8 and split16), while retaining
the four T=1 WY schedules. T=1 uses the standard decode ABI; the other token
counts use packed speculative decode. For every coordinate, it:

1. computes a reference with the explicit public ``backend="cute-dsl"`` path;
2. checks the output and the complete mutated state from every forced frozen
   specialization against that reference; and
3. measures the CuTe-DSL and Cake public-API calls with CUPTI, cold L2, and
   one public call per timed sample.

All selector and launch-tracking monkeypatches are restored in ``finally``
blocks. State/output restoration is performed before entering
``bench_gpu_time`` and is therefore outside the timed GPU activity.
"""

import argparse
import contextlib
import functools
import importlib
import json
import math
import statistics
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.kda_decode import recurrent_kda
from flashinfer.testing import bench_gpu_time


DATA_SEED = 4242
HEAD_DIM = 128
NUM_HEADS = 16
NUM_VALUE_HEADS = 32
TOKEN_COUNTS = (1, 2, 4, 5, 6)
SEQUENCE_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128)
VALUE_SPLITS = (1, 2, 4, 8)
SUPPORTED_FLASH_KDA_DECODE_ARCHS = {(10, 0): "sm100a", (10, 3): "sm103a"}
VARIANT_PREFIXES = {
    1: "d128_t1_precomputed_split",
    2: "d128_t2_precomputed_split",
    4: "d128_t4_precomputed_split",
    5: "d128_t5_precomputed_gram_split",
    6: "d128_t6_precomputed_gram_split",
}

recurrent_module = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")


def _variant_specs_for_tokens(num_tokens: int) -> tuple[dict, ...]:
    """Return every frozen schedule that should be swept for one token count."""

    schedule_kind = "coefficient_gram" if num_tokens in (5, 6) else "wy"
    specs = [
        {
            "schedule_kind": schedule_kind,
            "value_split": value_split,
            "variant": f"{VARIANT_PREFIXES[num_tokens]}{value_split}",
        }
        for value_split in VALUE_SPLITS
    ]
    if num_tokens == 1:
        specs.extend(
            {
                "schedule_kind": "direct_state",
                "value_split": value_split,
                "variant": f"d128_t1_precomputed_direct_split{value_split}",
            }
            for value_split in (8, 16)
        )
    return tuple(specs)


def _make_case(num_tokens: int, num_sequences: int, device: torch.device) -> dict:
    total_tokens = num_sequences * num_tokens
    is_standard_decode = num_tokens == 1
    token_shape = (num_sequences, 1) if is_standard_decode else (1, total_tokens)
    generator = torch.Generator(device=device).manual_seed(
        DATA_SEED + 1000 * num_tokens + num_sequences
    )

    q = torch.rand(
        (*token_shape, NUM_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.rand(
        (*token_shape, NUM_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.rand(
        (*token_shape, NUM_VALUE_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    g = F.logsigmoid(
        torch.randn(
            (*token_shape, NUM_VALUE_HEADS, HEAD_DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    ).to(torch.bfloat16)
    beta = torch.sigmoid(
        torch.randn(
            (*token_shape, NUM_VALUE_HEADS),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
    )

    if is_standard_decode:
        cu_seqlens = None
        ssm_state_indices = None
        num_accepted_tokens = None
        num_spec_tokens = None
        state_slots = num_sequences
        api_mode = "standard_decode"
    else:
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            num_tokens,
            dtype=torch.int32,
            device=device,
        )
        ssm_state_indices = torch.arange(
            1,
            total_tokens + 1,
            dtype=torch.int32,
            device=device,
        ).reshape(num_sequences, num_tokens)
        num_accepted_tokens = torch.ones(
            num_sequences,
            dtype=torch.int32,
            device=device,
        )
        num_spec_tokens = num_tokens - 1
        state_slots = total_tokens + 6
        api_mode = "packed_spec_decode"

    initial_state = (
        torch.randn(
            (state_slots, NUM_VALUE_HEADS, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.01
    ).to(torch.bfloat16)
    output_template = torch.full_like(v, 123.0)
    return {
        "name": f"d128_t{num_tokens}_n{num_sequences}_h16_hv32_{api_mode}",
        "D": HEAD_DIM,
        "T": num_tokens,
        "N": num_sequences,
        "H": NUM_HEADS,
        "HV": NUM_VALUE_HEADS,
        "gate_mode": "precomputed",
        "api_mode": api_mode,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "num_spec_tokens": num_spec_tokens,
        "initial_state": initial_state,
        "output_template": output_template,
    }


def _call_kwargs(case: dict, *, state: torch.Tensor, output: torch.Tensor) -> dict:
    return {
        "q": case["q"],
        "k": case["k"],
        "v": case["v"],
        "g": case["g"],
        "beta": case["beta"],
        "A_log": None,
        "dt_bias": None,
        "scale": HEAD_DIM**-0.5,
        "initial_state": state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": False,
        "lower_bound": None,
        "cu_seqlens": case["cu_seqlens"],
        "ssm_state_indices": case["ssm_state_indices"],
        "num_spec_tokens": case["num_spec_tokens"],
        "num_accepted_tokens": case["num_accepted_tokens"],
        "output": output,
    }


@contextlib.contextmanager
def _forced_variant(
    variant: Optional[str],
) -> Iterator[list[str]]:
    """Force one selector result and record every frozen launch."""

    original_selector = recurrent_module._select_flash_kda_decode_variant
    original_run_frozen = recurrent_module._run_flash_kda_decode
    launched_variants = []

    def select_forced_variant(**kwargs):
        del kwargs
        return variant

    def track_frozen_launch(selected_variant, **kwargs):
        launched_variants.append(selected_variant)
        return original_run_frozen(selected_variant, **kwargs)

    recurrent_module._select_flash_kda_decode_variant = select_forced_variant
    recurrent_module._run_flash_kda_decode = track_frozen_launch
    try:
        yield launched_variants
    finally:
        recurrent_module._run_flash_kda_decode = original_run_frozen
        recurrent_module._select_flash_kda_decode_variant = original_selector


def _assert_route(
    variant: Optional[str], launched_variants: list[str], public_calls: int
) -> None:
    if variant is None:
        if launched_variants:
            raise AssertionError(
                f"explicit CuTe-DSL route unexpectedly launched {launched_variants}"
            )
        return
    expected = [variant] * public_calls
    if launched_variants != expected:
        raise AssertionError(
            f"expected {public_calls} {variant} launches, got {launched_variants}"
        )


def _run_once(
    case: dict,
    *,
    initial_state: torch.Tensor,
    variant: Optional[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.clone()
    output = case["output_template"].clone()
    backend = "cute-dsl" if variant is None else "cake"
    with _forced_variant(variant) as launched_variants:
        actual_output, final_state = recurrent_kda(
            **_call_kwargs(case, state=state, output=output),
            backend=backend,
        )
    torch.cuda.synchronize()
    _assert_route(variant, launched_variants, public_calls=1)
    if actual_output.data_ptr() != output.data_ptr():
        raise AssertionError("public recurrent_kda did not use the supplied output")
    if final_state is not state:
        raise AssertionError("public recurrent_kda did not return the mutated state")
    return actual_output, state


def _check_correctness(
    case: dict,
    *,
    initial_state: torch.Tensor,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
    variant: str,
) -> dict:
    actual_output, actual_state = _run_once(
        case,
        initial_state=initial_state,
        variant=variant,
    )
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    output_max_abs_error = float((actual_output - expected_output).abs().max().item())
    state_max_abs_error = float((actual_state - expected_state).abs().max().item())
    return {
        "passed": True,
        "reference": "public_recurrent_kda_backend_cute_dsl",
        "output_allclose": True,
        "all_state_allclose": True,
        "atol": 1e-2,
        "rtol": 1e-2,
        "output_max_abs_error": output_max_abs_error,
        "state_max_abs_error": state_max_abs_error,
    }


def _benchmark_route(
    case: dict,
    *,
    initial_state: torch.Tensor,
    variant: Optional[str],
    rounds: int,
    warmup: int,
) -> dict:
    measured_state = initial_state.clone()
    measured_output = case["output_template"].clone()
    warmup_state = initial_state.clone()
    warmup_output = case["output_template"].clone()
    backend = "cute-dsl" if variant is None else "cake"
    measured_run = functools.partial(
        recurrent_kda,
        **_call_kwargs(case, state=measured_state, output=measured_output),
        backend=backend,
    )
    warmup_run = functools.partial(
        recurrent_kda,
        **_call_kwargs(case, state=warmup_state, output=warmup_output),
        backend=backend,
    )

    # Prime compilation/module loading and the T=1 metadata cache on disjoint
    # mutable buffers. This activity is outside every timed CUPTI sample.
    with _forced_variant(variant) as launched_variants:
        warmup_run()
    torch.cuda.synchronize()
    _assert_route(variant, launched_variants, public_calls=1)

    samples_ms = []
    expected_warmup_calls = 6 + warmup
    expected_public_calls = expected_warmup_calls + 1
    for _ in range(rounds):
        measured_state.copy_(initial_state)
        measured_output.copy_(case["output_template"])
        warmup_state.copy_(initial_state)
        warmup_output.copy_(case["output_template"])
        torch.cuda.synchronize()

        call_count = 0

        def staged_run():
            nonlocal call_count
            call_count += 1
            if call_count <= expected_warmup_calls:
                return warmup_run()
            if call_count == expected_public_calls:
                return measured_run()
            raise RuntimeError(
                "unexpected extra recurrent_kda call inside one-sample timing"
            )

        with _forced_variant(variant) as launched_variants:
            measured = [
                float(value)
                for value in bench_gpu_time(
                    staged_run,
                    enable_cupti=True,
                    cold_l2_cache=True,
                    use_cuda_graph=False,
                    dry_run_iters=warmup,
                    repeat_iters=1,
                )
            ]
        if call_count != expected_public_calls:
            raise RuntimeError(
                f"expected {expected_public_calls} public-API calls, got {call_count}"
            )
        _assert_route(
            variant,
            launched_variants,
            public_calls=expected_public_calls,
        )
        if len(measured) != 1:
            raise RuntimeError(
                f"expected one timed sample for {case['name']}, got {measured}"
            )
        samples_ms.append(measured[0])

    median_ms = float(statistics.median(samples_ms))
    if not math.isfinite(median_ms) or median_ms <= 0.0:
        raise RuntimeError(f"invalid timing for {case['name']}: {median_ms}")
    return {
        "median_ms": median_ms,
        "samples_ms": samples_ms,
        "timing_backend": "CUPTI",
        "cold_l2": True,
        "cuda_graph": False,
        "timing_scope": "public_recurrent_kda_gpu_activity",
        "single_public_api_call_per_sample": True,
        "state_output_reset_outside_timed_call": True,
        "warmup_uses_disjoint_state_output": True,
        "sampling_protocol": {
            "rounds": rounds,
            "warmup_iters_per_round": warmup,
            "timed_iters_per_round": 1,
        },
    }


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
    return actual_source_sha


def main() -> None:
    parser = argparse.ArgumentParser()
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
    compute_capability = torch.cuda.get_device_capability(device)
    if compute_capability not in SUPPORTED_FLASH_KDA_DECODE_ARCHS:
        raise RuntimeError(
            "this benchmark requires exact CC 10.0 (SM100a; B200/GB200) "
            "or CC 10.3 (SM103a; B300/GB300), got "
            f"CC {compute_capability[0]}.{compute_capability[1]}"
        )
    device_properties = torch.cuda.get_device_properties(device)
    cuda_arch = SUPPORTED_FLASH_KDA_DECODE_ARCHS[compute_capability]
    print(
        f"Hardware: {device_properties.name} CC {compute_capability[0]}."
        f"{compute_capability[1]} ({cuda_arch})"
    )

    rows = []
    best_by_t_n = []
    for num_tokens in TOKEN_COUNTS:
        for num_sequences in SEQUENCE_COUNTS:
            case = _make_case(num_tokens, num_sequences, device)
            initial_state = case["initial_state"]
            expected_output, expected_state = _run_once(
                case,
                initial_state=initial_state,
                variant=None,
            )
            cute_dsl_timing = _benchmark_route(
                case,
                initial_state=initial_state,
                variant=None,
                rounds=args.rounds,
                warmup=args.warmup,
            )

            case_rows = []
            for variant_spec in _variant_specs_for_tokens(num_tokens):
                value_split = variant_spec["value_split"]
                variant = variant_spec["variant"]
                correctness = _check_correctness(
                    case,
                    initial_state=initial_state,
                    expected_output=expected_output,
                    expected_state=expected_state,
                    variant=variant,
                )
                frozen_timing = _benchmark_route(
                    case,
                    initial_state=initial_state,
                    variant=variant,
                    rounds=args.rounds,
                    warmup=args.warmup,
                )
                speedup = cute_dsl_timing["median_ms"] / frozen_timing["median_ms"]
                row = {
                    "record_type": "split_result",
                    "case": case["name"],
                    "D": case["D"],
                    "T": case["T"],
                    "N": case["N"],
                    "H": case["H"],
                    "HV": case["HV"],
                    "api_mode": case["api_mode"],
                    "gate_mode": case["gate_mode"],
                    "schedule_kind": variant_spec["schedule_kind"],
                    "value_split": value_split,
                    "variant": variant,
                    "correctness": correctness,
                    "cute_dsl": cute_dsl_timing,
                    "frozen": frozen_timing,
                    "speedup_vs_cute_dsl": speedup,
                    "data_seed": DATA_SEED + 1000 * num_tokens + num_sequences,
                    "source_sha": actual_source_sha,
                    "cupti_python_version": cupti_python_version,
                }
                rows.append(row)
                case_rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

            best_row = min(
                case_rows,
                key=lambda row: (
                    row["frozen"]["median_ms"],
                    row["value_split"],
                ),
            )
            best = {
                "record_type": "best_split",
                "T": num_tokens,
                "N": num_sequences,
                "D": HEAD_DIM,
                "H": NUM_HEADS,
                "HV": NUM_VALUE_HEADS,
                "api_mode": case["api_mode"],
                "gate_mode": case["gate_mode"],
                "best_schedule_kind": best_row["schedule_kind"],
                "best_value_split": best_row["value_split"],
                "best_variant": best_row["variant"],
                "cute_dsl_median_ms": best_row["cute_dsl"]["median_ms"],
                "best_frozen_median_ms": best_row["frozen"]["median_ms"],
                "best_speedup_vs_cute_dsl": best_row["speedup_vs_cute_dsl"],
                "source_sha": actual_source_sha,
            }
            best_by_t_n.append(best)
            print(json.dumps(best, sort_keys=True), flush=True)
            del case, expected_output, expected_state
            torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "source_sha": actual_source_sha,
        "source_root": str(args.expected_source_root.resolve()),
        "flashinfer_version": flashinfer.__version__,
        "device": device_properties.name,
        "compute_capability": list(compute_capability),
        "cuda_arch": cuda_arch,
        "sm_count": device_properties.multi_processor_count,
        "total_memory_bytes": device_properties.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "timing_backend": "CUPTI",
        "cupti_python_version": cupti_python_version,
        "matrix": {
            "D": HEAD_DIM,
            "T": list(TOKEN_COUNTS),
            "N": list(SEQUENCE_COUNTS),
            "H": NUM_HEADS,
            "HV": NUM_VALUE_HEADS,
            "value_splits": list(VALUE_SPLITS),
            "variants_by_t": {
                str(num_tokens): list(_variant_specs_for_tokens(num_tokens))
                for num_tokens in TOKEN_COUNTS
            },
            "gate_mode": "precomputed",
            "t1_api_mode": "standard_decode",
            "other_api_mode": "packed_spec_decode",
        },
        "rows": rows,
        "best_by_t_n": best_by_t_n,
    }
    args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
