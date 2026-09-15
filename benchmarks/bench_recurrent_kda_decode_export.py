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
calls ``flashinfer.kda_decode.recurrent_kda`` with an identical, deterministic
D128/T=1..6 matrix. T=1 uses the standard decode ABI, T=3 uses the measured
lower-bound contract, and the remaining token counts use packed speculative
decode with precomputed gates. The caller can alternate mode order across
processes to form paired rounds. Candidate rows compare complete output and
mutated state against the explicit CuTe-DSL backend before timing.
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
from typing import Optional

import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.kda_decode import recurrent_kda
from flashinfer.testing import bench_gpu_time


UPSTREAM_MAIN_SHA = "39f2ce47663243e25b311a7e64681d742905f974"
EVOLUTION_PEER_SHA = "cea7f46ffc190cabf82c95a39cd0d2aa6c888c17"
DATA_SEED = 42
PRECOMPUTED_SEQUENCE_COUNTS = (8, 16, 32, 64, 128)
T3_LOWER_BOUND_SEQUENCE_COUNTS = (1, 2, 4, 8, 16)
SUPPORTED_FLASH_KDA_DECODE_ARCHS = {(10, 0): "sm100a", (10, 3): "sm103a"}

# Keep the measured public-export routes in one place. T3 has one exact
# lower-bound split4 specialization. The final 30-shape matrix has
# architecture-local T1 routes and one measured SM103a T4/N8 route override.
EXPECTED_VARIANTS_BY_T = {
    2: "d128_t2_precomputed_split4",
    3: "d128_t3_lower_bound_split4",
    4: "d128_t4_precomputed_split2",
    5: "d128_t5_precomputed_gram_split1",
    6: "d128_t6_precomputed_gram_split1",
}
EXPECTED_T1_VARIANTS_BY_ARCH_AND_N = {
    "sm100a": {
        8: "d128_t1_precomputed_direct_split16",
        16: "d128_t1_precomputed_direct_split8",
        32: "d128_t1_precomputed_direct_split8",
        64: "d128_t1_precomputed_direct_split8",
        128: "d128_t1_precomputed_direct_split8",
    },
    "sm103a": {
        num_sequences: "d128_t1_precomputed_direct_split16"
        for num_sequences in PRECOMPUTED_SEQUENCE_COUNTS
    },
}
EXPECTED_SM103A_VARIANTS_BY_T_AND_N = {
    (4, 8): "d128_t4_precomputed_split1",
}


def _case_specs_for_tokens(num_tokens: int) -> tuple[dict, ...]:
    is_t3_lower_bound = num_tokens == 3
    sequence_counts = (
        T3_LOWER_BOUND_SEQUENCE_COUNTS
        if is_t3_lower_bound
        else PRECOMPUTED_SEQUENCE_COUNTS
    )
    gate_mode = "lower_bound" if is_t3_lower_bound else "precomputed"
    api_mode = "standard_decode" if num_tokens == 1 else "spec_decode"
    num_value_heads = 16 if is_t3_lower_bound else 32
    return tuple(
        {
            "name": (
                f"d128_t{num_tokens}_b{num_sequences}_h16_"
                f"hv{num_value_heads}_{api_mode}_{gate_mode}"
            ),
            "D": 128,
            "T": num_tokens,
            "N": num_sequences,
            "H": 16,
            "HV": num_value_heads,
            "api_mode": api_mode,
            "gate_mode": gate_mode,
        }
        for num_sequences in sequence_counts
    )


CASES = tuple(
    spec for num_tokens in range(1, 7) for spec in _case_specs_for_tokens(num_tokens)
)


def _hardware_metadata(device: torch.device) -> dict:
    compute_capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return {
        "device_name": properties.name,
        "device_index": device_index,
        "compute_capability": list(compute_capability),
        "cuda_arch": SUPPORTED_FLASH_KDA_DECODE_ARCHS[compute_capability],
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _make_case(spec: dict, device: torch.device) -> dict:
    head_dim = spec["D"]
    num_tokens = spec["T"]
    num_sequences = spec["N"]
    num_heads = spec["H"]
    num_value_heads = spec["HV"]
    total_tokens = num_sequences * num_tokens
    is_standard_decode = spec["api_mode"] == "standard_decode"
    if is_standard_decode != (num_tokens == 1):
        raise ValueError(
            f"{spec['name']} has inconsistent T={num_tokens} and "
            f"api_mode={spec['api_mode']}"
        )
    token_shape = (num_sequences, 1) if is_standard_decode else (1, total_tokens)
    generator = torch.Generator(device=device).manual_seed(DATA_SEED)

    q = torch.rand(
        (*token_shape, num_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.rand(
        (*token_shape, num_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.rand(
        (*token_shape, num_value_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    beta_logits = torch.randn(
        (*token_shape, num_value_heads),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    beta = torch.sigmoid(beta_logits)
    if spec["gate_mode"] == "precomputed":
        g = F.logsigmoid(
            torch.randn(
                (*token_shape, num_value_heads, head_dim),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        ).to(torch.bfloat16)
        A_log = None
        dt_bias = None
        use_gate_in_kernel = False
        lower_bound = None
    elif spec["gate_mode"] == "lower_bound":
        g = torch.randn(
            (*token_shape, num_value_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        A_log = torch.log(
            torch.rand(
                num_heads,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            + 1.0
        )
        dt_bias = torch.randn(
            num_heads * head_dim,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        use_gate_in_kernel = True
        lower_bound = -5.0
    else:
        raise ValueError(f"unknown gate mode: {spec['gate_mode']}")
    if is_standard_decode:
        cu_seqlens = None
        ssm_state_indices = None
        num_accepted_tokens = None
        num_spec_tokens = None
        state_slots = num_sequences
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
            num_sequences * num_tokens + 1,
            dtype=torch.int32,
            device=device,
        ).reshape(num_sequences, num_tokens)
        num_accepted_tokens = torch.ones(
            num_sequences,
            dtype=torch.int32,
            device=device,
        )
        num_spec_tokens = num_tokens - 1
        state_slots = num_sequences * num_tokens + 6
    state = (
        torch.randn(
            (
                state_slots,
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
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": head_dim**-0.5,
        "initial_state": state,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": use_gate_in_kernel,
        "lower_bound": lower_bound,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_spec_tokens": num_spec_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "output": output,
    }


def _parse_expected_variant_overrides(values: list[str]) -> dict[int, str]:
    expected_variants = {}
    for value in values:
        token_text, separator, variant = value.partition("=")
        if not separator or not token_text or not variant:
            raise ValueError(
                f"invalid expected variant {value!r}; expected TOKEN_COUNT=VARIANT"
            )
        try:
            num_tokens = int(token_text)
        except ValueError as error:
            raise ValueError(
                f"invalid token count {token_text!r} in expected variant {value!r}"
            ) from error
        if num_tokens not in range(1, 7):
            raise ValueError(
                f"expected variant token count must be in 1..6, got {num_tokens}"
            )
        expected_variants[num_tokens] = variant
    return expected_variants


def _expected_variant_for_spec(
    spec: dict,
    expected_variant_overrides: dict[int, str],
    cuda_arch: Optional[str] = None,
) -> str:
    num_tokens = spec["T"]
    if num_tokens in expected_variant_overrides:
        return expected_variant_overrides[num_tokens]
    if cuda_arch is None:
        compute_capability = torch.cuda.get_device_capability()
        cuda_arch = SUPPORTED_FLASH_KDA_DECODE_ARCHS[compute_capability]
    if num_tokens == 1:
        return EXPECTED_T1_VARIANTS_BY_ARCH_AND_N[cuda_arch][spec["N"]]
    if (
        cuda_arch == "sm103a"
        and (
            num_tokens,
            spec["N"],
        )
        in EXPECTED_SM103A_VARIANTS_BY_T_AND_N
    ):
        return EXPECTED_SM103A_VARIANTS_BY_T_AND_N[(num_tokens, spec["N"])]
    return EXPECTED_VARIANTS_BY_T[num_tokens]


def _assert_frozen_route(
    spec: dict,
    kwargs: dict,
    expected_variant: str,
) -> str:
    recurrent_module = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")
    selected = []
    run_frozen = recurrent_module._run_flash_kda_decode
    initial_state = kwargs["initial_state"].clone()

    def track_frozen_route(variant, **run_kwargs):
        selected.append(variant)
        return run_frozen(variant, **run_kwargs)

    recurrent_module._run_flash_kda_decode = track_frozen_route
    try:
        recurrent_kda(**kwargs, backend="cake")
    finally:
        recurrent_module._run_flash_kda_decode = run_frozen
        kwargs["initial_state"].copy_(initial_state)
    if selected != [expected_variant]:
        raise AssertionError(
            f"{spec['name']} expected one {expected_variant} launch, got {selected}"
        )
    return selected[0]


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


def _check_cake_correctness(kwargs: dict, initial_state: torch.Tensor) -> dict:
    """Compare one Cake public call with the explicit CuTe-DSL backend."""

    reference_state = initial_state.clone()
    cake_state = initial_state.clone()
    reference_output = torch.empty_like(kwargs["output"])
    cake_output = torch.empty_like(kwargs["output"])
    reference_kwargs = dict(kwargs)
    reference_kwargs["initial_state"] = reference_state
    reference_kwargs["output"] = reference_output
    cake_kwargs = dict(kwargs)
    cake_kwargs["initial_state"] = cake_state
    cake_kwargs["output"] = cake_output

    actual_reference_output, _ = recurrent_kda(
        **reference_kwargs,
        backend="cute-dsl",
    )
    actual_cake_output, _ = recurrent_kda(
        **cake_kwargs,
        backend="cake",
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual_cake_output.float(),
        actual_reference_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        cake_state.float(),
        reference_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    return {
        "checked": True,
        "reference_backend": "cute-dsl",
        "candidate_backend": "cake",
        "output_allclose": True,
        "all_state_allclose": True,
        "atol": 1e-2,
        "rtol": 1e-2,
        "output_max_abs_error": float(
            (actual_cake_output - actual_reference_output).abs().max().item()
        ),
        "state_max_abs_error": float((cake_state - reference_state).abs().max().item()),
    }


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
    parser.add_argument(
        "--expected-variant",
        action="append",
        default=[],
        metavar="TOKEN_COUNT=VARIANT",
        help=("override one frozen route assertion; repeat for multiple token counts"),
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds <= 0 or args.warmup <= 0:
        parser.error("--rounds and --warmup must be positive")
    try:
        expected_variants = _parse_expected_variant_overrides(args.expected_variant)
    except ValueError as error:
        parser.error(str(error))

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
    hardware = _hardware_metadata(device)
    print(
        "Hardware: "
        f"{hardware['device_name']} CC {hardware['compute_capability'][0]}."
        f"{hardware['compute_capability'][1]} ({hardware['cuda_arch']})"
    )

    rows = []
    cases_per_t = {
        num_tokens: sum(spec["T"] == num_tokens for spec in CASES)
        for num_tokens in range(1, 7)
    }
    case_ordinal_by_t = {num_tokens: 0 for num_tokens in range(1, 7)}
    for spec in CASES:
        case_ordinal_by_t[spec["T"]] += 1
        kwargs = _make_case(spec, device)
        selected_variant = None
        expected_variant = None
        if args.mode == "frozen":
            expected_variant = _expected_variant_for_spec(
                spec, expected_variants, hardware["cuda_arch"]
            )
            selected_variant = _assert_frozen_route(spec, kwargs, expected_variant)
        call_kwargs = dict(kwargs)
        if args.mode == "frozen":
            call_kwargs["backend"] = "cake"
        initial_state = kwargs["initial_state"].clone()
        correctness = (
            _check_cake_correctness(kwargs, initial_state)
            if args.mode == "frozen"
            else None
        )
        measured_run = functools.partial(recurrent_kda, **call_kwargs)
        warmup_kwargs = dict(call_kwargs)
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
            "hardware": hardware,
            "mode": args.mode,
            "data_seed": DATA_SEED,
            "case_ordinal_within_t": case_ordinal_by_t[spec["T"]],
            "cases_for_t": cases_per_t[spec["T"]],
            "expected_variant": expected_variant,
            "selected_variant": selected_variant,
            "selected_backend": ("cake" if args.mode == "frozen" else "source-default"),
            "correctness": correctness,
            "median_ms": median_ms,
            "samples_ms": samples_ms,
            "timing_backend": "CUPTI",
            "cupti_python_version": cupti_python_version,
            "cold_l2": True,
            "cuda_graph": False,
            "timing_scope": "public_recurrent_kda_gpu_activity",
            "single_public_api_call_per_sample": True,
            "gpu_activity_count_asserted": False,
            "state_output_reset_per_round": True,
            "warmup_uses_disjoint_state_output": True,
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

    print("\nPer-T case timing summary (absolute timings; compare matching JSON rows):")
    for num_tokens in range(1, 7):
        t_medians = [row["median_ms"] for row in rows if row["T"] == num_tokens]
        t_geomean_ms = math.exp(
            sum(math.log(value) for value in t_medians) / len(t_medians)
        )
        print(
            f"{args.mode:<14} T={num_tokens} cases={len(t_medians)} "
            f"case-geomean={t_geomean_ms * 1000.0:.3f} us"
        )

    args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
