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

"""Strict-CUPTI benchmark for the public packed Kimi K3 T=1 decode API.

This harness measures the GPU activity span of exactly one
``flashinfer.packed_kda_decode`` call. It reports absolute kernel timings; it
does not label them as serving E2E latency or infer speedup against a different
adapter. Run it only inside a Slurm GPU allocation.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

import flashinfer
from flashinfer.jit.flash_kda_packed_t1 import _variant_for_batch
from flashinfer.jit.flash_kda_packed_t1 import (
    _get_csrc_dir as _get_packed_kda_csrc_dir,
)
from flashinfer.jit.flash_kda_packed_t1 import gen_flash_kda_packed_t1_module
from flashinfer.kda_decode import packed_kda_decode
from flashinfer.testing import bench_gpu_time


HEADS = 12
HEAD_DIM = 128
MIXED_WIDTH = 3 * HEADS * HEAD_DIM
GATE_WIDTH = HEADS * HEAD_DIM
PRODUCTION_MIXED_STRIDE = 6144
PRODUCTION_GATE_STRIDE = GATE_WIDTH
PRODUCTION_BETA_STRIDE = HEADS
STATE_SLOT_ELEMENTS = HEADS * HEAD_DIM * HEAD_DIM
PRODUCTION_STATE_PADDING = 256
DEFAULT_BATCHES = (1, 8, 16, 31, 32, 64, 128, 256, 512)
SUPPORTED_ARCHS = {(10, 0): "sm100a", (10, 3): "sm103a"}
DATA_SEED = 20260805
TYPED_SOURCE_SHA256 = "24edeaf9676b12ec3301ff413080194282be0b35e604f785dffba41c0c48640e"
FROZEN_BODY_SHA256 = {
    "tile8": "d0de8869242d09bf0c1c4840a7fd73dcd32835050cdc08db58b19a2c7506d0da",
    "tile16": "d8a446e42da47e2d8cd05139c77efe9c970f2d36394b68b49649beb6bc2bbfbe",
}


def _state_view(storage, slots, slot_stride):
    return storage.as_strided(
        (slots, HEADS, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )


def _make_case(batch, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    mixed_storage = torch.randn(
        (batch, PRODUCTION_MIXED_STRIDE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.25)
    gate_storage = torch.randn(
        (batch, PRODUCTION_GATE_STRIDE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.25)
    beta_storage = torch.randn(
        (batch, PRODUCTION_BETA_STRIDE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    A_log = torch.empty(HEADS, dtype=torch.float32, device=device)
    A_log.uniform_(-2.0, -0.1, generator=generator)
    dt_bias = torch.randn(
        GATE_WIDTH,
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).mul_(0.1)

    slots = batch + 8
    state_slot_stride = STATE_SLOT_ELEMENTS + PRODUCTION_STATE_PADDING
    state_storage = torch.randn(
        slots * state_slot_stride,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    state = _state_view(state_storage, slots, state_slot_stride)
    state_indices = torch.arange(
        batch,
        0,
        -1,
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        (batch, 1, HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    return {
        "batch": batch,
        "mixed_qkv": mixed_storage[:, :MIXED_WIDTH],
        "raw_gate": gate_storage[:, :GATE_WIDTH],
        "raw_beta": beta_storage[:, :HEADS],
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_storage": state_storage,
        "state": state,
        "state_indices": state_indices,
        "output": output,
        "initial_state_storage": state_storage.clone(),
        "slots": slots,
        "state_slot_stride": state_slot_stride,
    }


def _run(case):
    return packed_kda_decode(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        case["state"],
        case["state_indices"],
        output=case["output"],
    )


def _reference(case):
    batch = case["batch"]
    packed = case["mixed_qkv"].float().reshape(batch, 3, HEADS, HEAD_DIM)
    q_raw = packed[:, 0]
    k_raw = packed[:, 1]
    q = (
        q_raw
        * torch.rsqrt((q_raw * q_raw).sum(dim=-1, keepdim=True) + 1.0e-6)
        * (HEAD_DIM**-0.5)
    )
    k = k_raw * torch.rsqrt((k_raw * k_raw).sum(dim=-1, keepdim=True) + 1.0e-6)
    value = packed[:, 2]
    gate_x = case["raw_gate"].float().reshape(batch, HEADS, HEAD_DIM)
    gate_x = gate_x + case["dt_bias"].reshape(HEADS, HEAD_DIM)
    decay = torch.exp(
        -5.0 * torch.sigmoid(torch.exp(case["A_log"])[None, :, None] * gate_x)
    )
    beta = torch.sigmoid(case["raw_beta"].float())
    indices = case["state_indices"].long()
    selected = case["state"].index_select(0, indices).float()
    decayed = selected * decay[:, :, None, :]
    prediction = torch.einsum("bhvk,bhk->bhv", decayed, k)
    delta = (value - prediction) * beta[:, :, None]
    updated = decayed + delta[:, :, :, None] * k[:, :, None, :]
    projected = torch.einsum("bhvk,bhk->bhv", updated, q)
    reference_storage = case["initial_state_storage"].clone()
    reference_state = _state_view(
        reference_storage, case["slots"], case["state_slot_stride"]
    )
    reference_state.index_copy_(0, indices, updated.to(torch.bfloat16))
    return projected.to(torch.bfloat16).unsqueeze(1), reference_state


def _restore(case):
    case["state_storage"].copy_(case["initial_state_storage"])
    case["output"].fill_(123.0)


def _check_correctness(case):
    _restore(case)
    reference_output, reference_state = _reference(case)
    result = _run(case)
    torch.cuda.synchronize()
    if result is not case["output"]:
        raise AssertionError("public packed_kda_decode did not return caller output")
    torch.testing.assert_close(
        result,
        reference_output,
        atol=1.0e-2,
        rtol=1.0e-2,
        check_dtype=False,
    )
    torch.testing.assert_close(
        case["state"],
        reference_state,
        atol=1.0e-2,
        rtol=1.0e-2,
        check_dtype=False,
    )
    output_error = (result.float() - reference_output.float()).abs()
    state_error = (case["state"].float() - reference_state.float()).abs()
    correctness = {
        "output_max_abs": float(output_error.max()),
        "output_mean_abs": float(output_error.mean()),
        "state_max_abs": float(state_error.max()),
        "state_mean_abs": float(state_error.mean()),
        "atol": 1.0e-2,
        "rtol": 1.0e-2,
    }
    _restore(case)
    torch.cuda.synchronize()
    return correctness


def _capture_graph(case):
    _restore(case)
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream(device=case["state"].device)
    capture_stream.wait_stream(torch.cuda.current_stream(case["state"].device))
    with torch.cuda.stream(capture_stream):
        _run(case)
    capture_stream.synchronize()
    _restore(case)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        _run(case)
    torch.cuda.synchronize()
    _restore(case)
    torch.cuda.synchronize()
    return graph


def _require_cupti():
    try:
        from cupti import cupti  # noqa: F401

        cupti_python_version = version("cupti-python")
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError("reportable timings require cupti-python >= 13") from error
    if int(cupti_python_version.split(".", 1)[0]) < 13:
        raise RuntimeError(
            f"reportable timings require cupti-python >= 13, got {cupti_python_version}"
        )
    return cupti_python_version


def _hardware_metadata(device):
    capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_name": properties.name,
        "device_index": (
            device.index if device.index is not None else torch.cuda.current_device()
        ),
        "compute_capability": list(capability),
        "cuda_arch": SUPPORTED_ARCHS[capability],
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _check_source(expected_source_root):
    imported_root = Path(flashinfer.__file__).resolve().parents[1]
    expected_root = expected_source_root.resolve()
    if imported_root != expected_root:
        raise RuntimeError(
            f"expected flashinfer from {expected_root}, imported {imported_root}"
        )
    source_sha = subprocess.run(
        ["git", "-C", str(expected_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(expected_root), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "reportable timing requires a clean FlashInfer checkout; dirty paths:\n"
            f"{status.rstrip()}"
        )
    return source_sha


def _verify_frozen_body_hashes():
    csrc_dir = _get_packed_kda_csrc_dir()
    actual = {}
    begin = "// BEGIN FROZEN GENERATED BODY\n"
    end = "// END FROZEN GENERATED BODY\n"
    for variant, expected_sha256 in FROZEN_BODY_SHA256.items():
        text = (csrc_dir / f"flashkda_packed_t1_{variant}.cu").read_text()
        _, begin_marker, remainder = text.partition(begin)
        body, end_marker, _ = remainder.partition(end)
        if begin_marker != begin or end_marker != end:
            raise RuntimeError(
                f"{variant} frozen body markers are missing or malformed"
            )
        actual_sha256 = hashlib.sha256(body.encode()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{variant} frozen body SHA256 mismatch: expected "
                f"{expected_sha256}, got {actual_sha256}"
            )
        actual[variant] = actual_sha256
    return actual


def _jit_binary_metadata(variant, target):
    spec = gen_flash_kda_packed_t1_module(variant, target)
    path = spec.get_library_path().resolve()
    if not path.is_file():
        raise RuntimeError(
            f"loaded packed KDA module has no auditable binary at {path}"
        )
    return {
        "spec_name": spec.name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _make_timing_runners(case, warmup_case, mode):
    if mode == "direct":
        warmup_run = lambda: _run(warmup_case)
        measured_run = lambda: _run(case)
    elif mode == "cuda_graph":
        warmup_graph = _capture_graph(warmup_case)
        measured_graph = _capture_graph(case)
        warmup_run = warmup_graph.replay
        measured_run = measured_graph.replay
    else:
        raise ValueError(f"unknown execution mode: {mode}")
    return warmup_run, measured_run


def _timed_sample(case, warmup_case, *, warmup_run, measured_run, warmup_iters):
    _restore(case)
    _restore(warmup_case)
    torch.cuda.synchronize()
    warmup_calls = 6 + warmup_iters
    call_count = 0

    def staged_run():
        nonlocal call_count
        call_count += 1
        if call_count <= warmup_calls:
            return warmup_run()
        if call_count == warmup_calls + 1:
            return measured_run()
        raise RuntimeError("unexpected extra call inside one-sample CUPTI timing")

    measured = bench_gpu_time(
        staged_run,
        enable_cupti=True,
        cold_l2_cache=True,
        use_cuda_graph=False,
        dry_run_iters=warmup_iters,
        repeat_iters=1,
    )
    if call_count != warmup_calls + 1:
        raise RuntimeError(
            f"expected {warmup_calls + 1} staged calls, got {call_count}"
        )
    if len(measured) != 1:
        raise RuntimeError(f"expected one CUPTI sample, got {measured}")
    value = float(measured[0])
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"invalid CUPTI sample: {value}")
    return value


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCHES),
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "cuda_graph", "both"),
        default="both",
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument(
        "--expected-source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    if any(batch <= 0 or batch > 65535 for batch in args.batch_sizes):
        parser.error("all batch sizes must be in [1, 65535]")
    if args.rounds <= 0 or args.warmup < 0:
        parser.error("--rounds must be positive and --warmup non-negative")
    return args


def main():
    args = _parse_args()
    cupti_python_version = _require_cupti()
    source_sha = _check_source(args.expected_source_root)
    actual_frozen_body_sha256 = _verify_frozen_body_hashes()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability not in SUPPORTED_ARCHS:
        raise RuntimeError(
            "packed KDA T=1 benchmark requires exact CC 10.0 or 10.3, got "
            f"CC {capability[0]}.{capability[1]}"
        )
    hardware = _hardware_metadata(device)
    modes = ("direct", "cuda_graph") if args.mode == "both" else (args.mode,)
    rows = []
    jit_binaries = {}

    for ordinal, batch in enumerate(args.batch_sizes):
        case = _make_case(batch, device, args.seed + ordinal)
        warmup_case = _make_case(batch, device, args.seed + 10000 + ordinal)
        correctness = _check_correctness(case)
        variant = _variant_for_batch(batch)
        binary_key = f"{variant}_{hardware['cuda_arch']}"
        if binary_key not in jit_binaries:
            jit_binaries[binary_key] = _jit_binary_metadata(
                variant, hardware["cuda_arch"]
            )
        for mode in modes:
            warmup_run, measured_run = _make_timing_runners(case, warmup_case, mode)
            samples_ms = []
            for _ in range(args.rounds):
                samples_ms.append(
                    _timed_sample(
                        case,
                        warmup_case,
                        warmup_run=warmup_run,
                        measured_run=measured_run,
                        warmup_iters=args.warmup,
                    )
                )
            median_ms = float(statistics.median(samples_ms))
            row = {
                "B": batch,
                "T": 1,
                "H": HEADS,
                "HV": HEADS,
                "K": HEAD_DIM,
                "V": HEAD_DIM,
                "dtype": "bfloat16",
                "state_dtype": "bfloat16",
                "mode": mode,
                "variant": variant,
                "target": hardware["cuda_arch"],
                "median_ms": median_ms,
                "samples_ms": samples_ms,
                "correctness": correctness,
                "timing_backend": "CUPTI",
                "timing_scope": "public_packed_kda_decode_single_kernel_gpu_span",
                "single_public_api_call_per_sample": True,
                "cold_l2": True,
                "state_output_reset_per_round": True,
                "warmup_uses_disjoint_state_output": True,
                "production_strides": {
                    "mixed_qkv": [PRODUCTION_MIXED_STRIDE, 1],
                    "raw_gate": [PRODUCTION_GATE_STRIDE, 1],
                    "raw_beta": [PRODUCTION_BETA_STRIDE, 1],
                    "state_slot": STATE_SLOT_ELEMENTS + PRODUCTION_STATE_PADDING,
                },
            }
            rows.append(row)
            print(f"{mode:<10} B={batch:<4} {variant:<6} {median_ms * 1000.0:10.3f} us")
        del case, warmup_case
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "benchmark": "flashinfer_packed_kda_decode_t1",
        "hardware": hardware,
        "source_sha": source_sha,
        "typed_source_sha256": TYPED_SOURCE_SHA256,
        "frozen_body_sha256": actual_frozen_body_sha256,
        "jit_binaries": jit_binaries,
        "cupti_python_version": cupti_python_version,
        "seed": args.seed,
        "sampling_protocol": {
            "rounds": args.rounds,
            "warmup_iters_per_round": args.warmup,
            "timed_public_api_calls_per_round": 1,
        },
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
