# Copyright (c) 2026 by FlashInfer team.
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

"""Sanitize the SM100/SM103 swap-AB CuTe MoE pipeline on one GPU.

Run manually with ``python -m pytest scripts/sanitize_cute_dsl_moe_swap_ab.py``,
not torchrun. This script is outside normal pytest discovery and is not
registered in CI. EP8 is a local 32-of-256-expert
partition, and 1/8/1024 are local input-token counts, not global counts.
Each tool/mode/token-count worker prepares GLM-5 weights once
and enumerates every shape-valid tactic without timing or autotuning. Both
routing distributions share input/output storage and one captured graph.
Compiled kernels and the graph pool are reused across tactics. Racecheck splits
the complete tactic catalog into eight contiguous shards, selectable with
pytest IDs racecheck-shard0 through racecheck-shard7 for parallel GPU runs.
Numerical checks live in the accuracy suite. Only ``swap_ab=True`` with
deterministic BF16 finalize is included; legacy kernels, atomic fused finalize,
distributed communication, NVFP4 4over6 and strict quantization are excluded.
Compute Sanitizer must support the requested 2026.3 option set;
unsupported options fail the worker instead of reducing instrumentation.
Use --basetemp to retain worker logs and compiler/coverage receipts.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from unittest import mock

import pytest


_MODULE = "scripts.sanitize_cute_dsl_moe_swap_ab"
_TOKENS = (1, 8, 1024)
_DISTRIBUTIONS = ("random", "skewed")
_MODES = ("w4a4", "per-token", "mxfp8", "w4a8", "w4a16")
_TIMEOUT = 6 * 60 * 60
# Same GLM-5 geometry as benchmarks/bench_cute_dsl_moe_distributed.py.
_HIDDEN, _INTERMEDIATE, _EXPERTS, _TOP_K = 6144, 2048, 256, 8
_NATIVE_FILTERS = ("routingIndices", "routingInitExpertCounts")
_OWNED_MODULES = {
    "Sm100BlockScaledGroupedGemmKernel": (
        "flashinfer.fused_moe.cute_dsl.blackwell.moe_swap_ab_kernel"
    ),
    "Sm100W4A16GroupedGemmKernel": (
        "flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16_kernel"
    ),
    "_MoeFinalizeKernel": "flashinfer.fused_moe.cute_dsl.blackwell.moe_finalize",
    "NVFP4QuantizePerTokenKernel": "flashinfer.quantization.kernels.nvfp4_quantize",
}


def _mode_config(mode):
    if mode == "per-token":
        return "w4a4", True
    if mode in ("w4a4", "mxfp8", "w4a8", "w4a16"):
        return mode, False
    raise ValueError(f"Unsupported sanitizer mode: {mode}")


def _num_tactic_shards(tool):
    if tool == "racecheck":
        return 8
    if tool in ("memcheck", "synccheck"):
        return 1
    raise ValueError(f"Unsupported sanitizer: {tool}")


def _owned_kernels(quant_mode, per_token):
    if quant_mode == "w4a16":
        gemm = "Sm100W4A16GroupedGemmKernel"
    elif quant_mode in ("w4a4", "mxfp8", "w4a8"):
        gemm = "Sm100BlockScaledGroupedGemmKernel"
    else:
        raise ValueError(f"Unsupported quantization mode: {quant_mode}")
    kernels = (gemm, "_MoeFinalizeKernel")
    if quant_mode == "w4a4" and per_token:
        kernels += ("NVFP4QuantizePerTokenKernel",)
    return kernels


def _sanitizer_options(tool, quant_mode, per_token):
    options = [
        "--tool",
        tool,
        "--target-processes",
        "application-only",
        "--error-exitcode",
        "86",
        "--check-exit-code",
        "yes",
        "--print-level",
        "info",
        "--print-limit",
        "0",
        "--show-backtrace",
        "yes",
        "--check-bulk-copy",
        "yes",
        "--launch-timeout",
        "600",
        "--kill",
    ]
    if tool == "memcheck":
        # The 2026.3 tensor checker misreports legal 2-CTA peer-barrier TMA. Compile
        # GEMMs with ptxas TMEM access guards instead; never suppress a report.
        options += ["--check-tensor-ops", "no", "--track-stream-ordered-races", "all"]
    elif tool == "synccheck":
        options += ["--missing-barrier-init-is-fatal", "yes"]
    elif tool == "racecheck":
        options += [
            "--racecheck-detect-level",
            "info",
            "--racecheck-report",
            "all",
            "--racecheck-memcpy-async",
            "yes",
            "--racecheck-num-workers",
            "16",
        ]
    else:
        raise ValueError(f"Unsupported sanitizer: {tool}")
    for name in (*_NATIVE_FILTERS, *_owned_kernels(quant_mode, per_token)):
        options += ["--kernel-name", f"kns={name}"]
    return options


def _terminate(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
    process.wait()


@pytest.mark.parametrize("opt_level", [3], ids=lambda n: f"O{n}")
@pytest.mark.parametrize("ep", [8], ids=lambda n: f"ep{n}")
@pytest.mark.parametrize("local_tokens", _TOKENS, ids=lambda n: f"t{n}")
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    "tool,tactic_shard",
    [pytest.param(tool, 0, id=tool) for tool in ("memcheck", "synccheck")]
    + [
        pytest.param("racecheck", shard, id=f"racecheck-shard{shard}")
        for shard in range(8)
    ],
)
def test_cute_dsl_moe_sanitizer(
    opt_level,
    ep,
    mode,
    local_tokens,
    tool,
    tactic_shard,
    tmp_path,
):
    quant_mode, use_per_token = _mode_config(mode)
    assert "LOCAL_RANK" not in os.environ, "Run with pytest, not torchrun"
    assert "PYTEST_XDIST_WORKER" not in os.environ, "Assign one pytest process per GPU"
    if sys.platform != "linux":
        pytest.skip("requires Linux and an SM100/SM103 GPU")
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None or shutil.which("cuobjdump") is None:
        pytest.skip("compute-sanitizer and cuobjdump must be on PATH")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires one CUDA GPU")
    if torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100 or SM103")
    assert not use_per_token or quant_mode == "w4a4"
    command = [
        sanitizer,
        *_sanitizer_options(tool, quant_mode, use_per_token),
        "--log-file",
        str(tmp_path / "sanitizer.log"),
        sys.executable,
        "-m",
        _MODULE,
        "--tool",
        tool,
        "--tactic-shard",
        str(tactic_shard),
        "--quant-mode",
        mode,
        "--local-tokens",
        str(local_tokens),
        "--opt-level",
        str(opt_level),
        "--ep",
        str(ep),
        "--output",
        str(tmp_path),
    ]
    # Isolate --keep-ptx artifacts from concurrent sanitizer workers.
    dump_dir = tmp_path / "cute-dumps"
    dump_dir.mkdir()
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        CUTE_DSL_DUMP_DIR=str(dump_dir),
        FLASHINFER_CUTE_DSL_DISABLE_CACHE="1",
        FLASHINFER_JIT_DEBUG="0",
        FLASHINFER_JIT_LINEINFO="1",
        FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH="0",
        TRTLLM_DISABLE_FP4_QUANT_FAST_MATH="0",
        FLASHINFER_NVFP4_4OVER6="0",
    )
    env.pop("CUTE_DSL_COMPILER_OPT", None)
    # Honor the caller's selected device; workers only ever use visible cuda:0.
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    output = tmp_path / "worker.log"
    with output.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=_TIMEOUT)
        except subprocess.TimeoutExpired:
            pytest.fail(f"Sanitizer exceeded {_TIMEOUT}s; see {output}")
        finally:
            if process.poll() != 0:
                _terminate(process)
    sanitizer_log = tmp_path / "sanitizer.log"
    details = (
        output.read_text()
        + "\n"
        + (
            sanitizer_log.read_text()
            if sanitizer_log.exists()
            else "Missing sanitizer log"
        )
    )
    assert returncode == 0, details
    receipt_path = tmp_path / "result.json"
    assert receipt_path.is_file(), details
    receipt = json.loads(receipt_path.read_text())
    assert receipt["mode"] == mode, details
    assert receipt["quant_mode"] == quant_mode and receipt["ep"] == ep, details
    assert receipt["opt_level"] == opt_level and not receipt["apply_topk_in_fc1"], (
        details
    )
    assert receipt["per_token"] is use_per_token, details
    assert receipt["swap_ab"] is True and not receipt["use_fused_finalize"], details
    assert receipt["local_tokens"] == local_tokens, details
    shards = _num_tactic_shards(tool)
    assert (
        receipt["tactic_shard"] == tactic_shard and receipt["tactic_shards"] == shards
    ), details
    start = receipt["total_tactics"] * tactic_shard // shards
    stop = receipt["total_tactics"] * (tactic_shard + 1) // shards
    assert receipt["tactic_start"] == start < stop == receipt["tactic_stop"], details
    assert receipt["tactic_indices"] == list(range(start, stop)), details
    assert len(receipt["selected_tactics"]) == stop - start, details
    expected = [(use_per_token, local_tokens, d) for d in _DISTRIBUTIONS]
    assert [
        (c["per_token"], c["local_tokens"], c["distribution"]) for c in receipt["cases"]
    ] == expected, details
    assert all(
        c["eager_tactics"]
        == c["graph_tactics"]
        == c["expected_tactics"]
        == stop - start
        for c in receipt["cases"]
    ), details
    assert set(_owned_kernels(quant_mode, use_per_token)) == set(
        receipt["cute_kernels"]
    ), details
    assert all(receipt["native_entries"][name] for name in _NATIVE_FILTERS), details
    assert receipt["native_opt_level"] == 3, details
    summary = (
        "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
        if tool == "racecheck"
        else "ERROR SUMMARY: 0 errors"
    )
    messages = [
        line.removeprefix("=========").strip()
        for line in sanitizer_log.read_text().splitlines()
        if line.strip()
    ]
    assert messages == ["COMPUTE-SANITIZER", summary], details


def _make_weights(quant_mode, local_experts):
    import torch
    from flashinfer.fused_moe import CuteDslConfig, QuantConfig, QuantFormat

    if quant_mode == "w4a4":
        pair = (QuantFormat.NVFP4, QuantFormat.NVFP4)
    elif quant_mode == "w4a16":
        pair = (QuantFormat.NVFP4, QuantFormat.BF16)
    elif quant_mode == "w4a8":
        pair = (QuantFormat.MXFP4, QuantFormat.MXFP8)
    elif quant_mode == "mxfp8":
        pair = (QuantFormat.MXFP8, QuantFormat.MXFP8)
    else:
        raise ValueError(quant_mode)
    w1 = (
        torch.randn(
            local_experts,
            2 * _INTERMEDIATE,
            _HIDDEN,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = (
        torch.randn(
            local_experts, _HIDDEN, _INTERMEDIATE, device="cuda", dtype=torch.bfloat16
        )
        / 10
    )
    weights = CuteDslConfig.prepare_weights(
        w1,
        w2,
        quant=QuantConfig(*pair),
        num_local_experts=local_experts,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        nvfp4_4over6=None,
        swap_ab=True,
    )
    weights["fc2_input_scale"] = (
        weights.get("fc2_input_scale") if quant_mode == "w4a4" else None
    )
    return weights


def _activation(x, quant_mode, per_token):
    import torch
    from flashinfer import SfLayout, mxfp8_quantize, nvfp4_quantize
    from flashinfer.quantization.nvfp4_quantization_utils import make_nvfp4_global_scale

    if quant_mode == "w4a16":
        return x, None, None, None
    if quant_mode in ("mxfp8", "w4a8"):
        q, sf = mxfp8_quantize(
            x, is_sf_swizzled_layout=False, alignment=128, backend="cuda"
        )
        return q, sf.view(torch.uint8).reshape(x.shape[0], _HIDDEN // 32), None, None
    if quant_mode == "w4a4":
        scale = (
            make_nvfp4_global_scale(
                x, per_token_activation=True, nvfp4_4over6_config=None
            )
            if per_token
            else torch.ones(1, device=x.device)
        )
        result = nvfp4_quantize(
            x,
            scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=per_token,
            backend="cute-dsl",
            nvfp4_4over6=None,
        )
        if per_token:
            q, sf, row_scale = result
        else:
            q, sf = result
            row_scale = None
        return q, sf, row_scale, scale
    raise ValueError(quant_mode)


def _routing(tokens, local_experts, offset, distribution):
    import torch

    if distribution == "random":
        ids = (
            torch.rand(tokens, _EXPERTS, device="cuda")
            .topk(_TOP_K, dim=-1)
            .indices.to(torch.int32)
        )
        # Decode must execute local GEMMs even when random EP routing misses
        # this partition. Swapping preserves distinct experts within each row.
        first = ids[0].tolist()
        if not any(offset <= expert < offset + local_experts for expert in first):
            ids[0, 0] = offset
    elif distribution == "skewed":
        ids = (
            torch.arange(offset, offset + _TOP_K, device="cuda", dtype=torch.int32)
            .expand(tokens, _TOP_K)
            .contiguous()
        )
    else:
        raise ValueError(distribution)
    scores = torch.softmax(torch.randn(tokens, _TOP_K, device="cuda"), dim=-1)
    return ids, scores


def _native_proof(output):
    from flashinfer.jit import gen_moe_utils_module
    from flashinfer.fused_moe.cute_dsl.moe_utils import _get_moe_utils_module

    _get_moe_utils_module()
    spec = gen_moe_utils_module()
    ninja = spec.ninja_path.read_text()
    flags = [
        line.strip()
        for line in ninja.replace("$\n", "").splitlines()
        if line.strip().startswith("cuda_cflags =")
    ]
    assert flags and all(
        re.findall(r"(?:^|\s)-O([0-3])(?:\s|$)", line) == ["3"]
        and "--device-debug" not in line
        for line in flags
    ), flags
    dump = subprocess.run(
        ["cuobjdump", "--dump-elf-symbols", str(spec.get_library_path())],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (output / "native-symbols.log").write_text(dump)
    entries = {
        name: sorted(
            {
                line.strip()
                for line in dump.splitlines()
                if name in line and "STT_FUNC" in line
            }
        )
        for name in _NATIVE_FILTERS
    }
    assert all(entries.values()), (
        "Sanitizer routing filter matched no compiled native entry",
        entries,
    )
    (output / "native-build.ninja").write_text(ninja)
    return entries


def _run_worker(args):
    import cutlass.cute as cute
    import torch
    from flashinfer import CuteDslMoEWrapper

    assert args.opt_level == 3 and args.ep == 8
    assert args.local_tokens in _TOKENS
    assert (args.quant_mode, args.per_token) == _mode_config(args.mode)
    assert not args.per_token or args.quant_mode == "w4a4"
    shards = _num_tactic_shards(args.tool)
    assert 0 <= args.tactic_shard < shards
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) in (
        (10, 0),
        (10, 3),
    )
    assert os.environ["FLASHINFER_CUTE_DSL_DISABLE_CACHE"] == "1"
    assert torch.cuda.device_count() == 1, "The worker must see exactly one GPU"
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    local_experts = _EXPERTS // args.ep
    offset = local_experts  # Also exercise a nonzero local expert window.
    owned = set(_owned_kernels(args.quant_mode, args.per_token))
    compiled = {}
    compile_calls = 0
    original_compile = cute.compile

    def compile_owned(program, *values, **kwargs):
        nonlocal compile_calls
        owner = getattr(program, "__self__", program)
        name = type(owner).__name__
        instrument = name in owned and type(owner).__module__ == _OWNED_MODULES[name]
        if instrument:
            compile_calls += 1
            options = kwargs.get("options", "")
            assert not re.search(r"--opt-level(?:=|\s)", options), options
            options += " --opt-level 3 --keep-ptx"
            if args.tool == "memcheck" and name.startswith("Sm100"):
                compile_options = []
                ptxas_options = []
                option_iter = iter(shlex.split(options))
                for option in option_iter:
                    if option == "--ptxas-options":
                        ptxas_options.append(next(option_iter))
                    elif option.startswith("--ptxas-options="):
                        ptxas_options.append(option.split("=", 1)[1])
                    else:
                        compile_options.append(option)
                ptxas_options.append("-g-tmem-access-check")
                compile_options.append("--ptxas-options=" + " ".join(ptxas_options))
                options = shlex.join(compile_options)
            kwargs["options"] = options
        result = original_compile(program, *values, **kwargs)
        if instrument:
            ptx = result.__ptx__
            entries = re.findall(r"\.entry\s+([^\s(]+)", ptx)
            assert any(name in entry for entry in entries), (name, entries)
            digest = hashlib.sha256(ptx.encode()).hexdigest()
            compiled.setdefault(name, {})[digest] = dict(
                entries=entries, options=kwargs["options"]
            )
        return result

    cases = []
    started = time.perf_counter()
    print(
        f"Preparing native routing: {args.quant_mode}, {args.tool}, O3, emulated EP8",
        flush=True,
    )
    native_entries = _native_proof(args.output)
    with mock.patch.object(cute, "compile", side_effect=compile_owned), torch.no_grad():
        weights = _make_weights(args.quant_mode, local_experts)
        print(
            f"Weight/routing setup complete in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        wrapper = CuteDslMoEWrapper(
            num_experts=_EXPERTS,
            top_k=_TOP_K,
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            num_local_experts=local_experts,
            local_expert_offset=offset,
            quant_mode=args.quant_mode,
            use_fused_finalize=False,
            use_cuda_graph=True,
            swap_ab=True,
        )
        per_token, tokens = args.per_token, args.local_tokens
        x = torch.randn(tokens, _HIDDEN, device="cuda", dtype=torch.bfloat16) / 10
        x, sf, row_scale, fc2_scale = _activation(x, args.quant_mode, per_token)
        ids = torch.empty(tokens, _TOP_K, device="cuda", dtype=torch.int32)
        scores = torch.empty(tokens, _TOP_K, device="cuda", dtype=torch.float32)
        output = torch.empty(tokens, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        routing = {
            d: _routing(tokens, local_experts, offset, d) for d in _DISTRIBUTIONS
        }
        if args.quant_mode == "w4a16":
            runner = wrapper._w4a16_runner
            inputs = [
                x,
                ids,
                scores,
                weights["w1_weight"],
                weights["w1_weight_sf"],
                weights["w1_alpha"],
                weights["w2_weight"],
                weights["w2_weight_sf"],
                weights["w2_alpha"],
                output,
            ]
            kwargs = {}
        elif args.quant_mode in ("w4a4", "mxfp8", "w4a8"):
            runner = wrapper._per_token_runner if per_token else wrapper._runner
            inputs = [
                x,
                sf,
                ids,
                scores,
                weights["w1_weight"],
                weights["w1_weight_sf"],
                weights["w1_alpha"],
                fc2_scale,
                weights["w2_weight"],
                weights["w2_weight_sf"],
                weights["w2_alpha"],
            ]
            if per_token:
                inputs.append(row_scale)
            inputs.append(output)
            kwargs = {"nvfp4_4over6": None}
        else:
            raise ValueError(args.quant_mode)
        assert wrapper.swap_ab and runner.swap_ab
        tactics = runner.get_valid_tactics(inputs, None)
        assert tactics and len(tactics) == len(set(tactics)), (
            "No valid tactics or duplicate tactics"
        )
        start = len(tactics) * args.tactic_shard // shards
        stop = len(tactics) * (args.tactic_shard + 1) // shards
        assert start < stop, "Empty tactic shard"
        print(
            f"{args.quant_mode} per_token={per_token} local_tokens={tokens}: shard {args.tactic_shard}/{shards}, tactics [{start}, {stop}) of {len(tactics)}; eager + graph, random + skewed",
            flush=True,
        )
        records = {
            d: dict(
                per_token=per_token,
                local_tokens=tokens,
                distribution=d,
                expected_tactics=stop - start,
                eager_tactics=0,
                graph_tactics=0,
            )
            for d in _DISTRIBUTIONS
        }
        pool = torch.cuda.graph_pool_handle()
        for index, tactic in enumerate(tactics[start:stop], start=start):
            ids.copy_(routing["random"][0])
            scores.copy_(routing["random"][1])
            runner(inputs, tactic=tactic, **kwargs)
            torch.cuda.synchronize()
            warmed_compile_calls = compile_calls
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                runner(inputs, tactic=tactic, **kwargs)
            for distribution, (route_ids, route_scores) in routing.items():
                ids.copy_(route_ids)
                scores.copy_(route_scores)
                runner(inputs, tactic=tactic, **kwargs)
                torch.cuda.synchronize()
                records[distribution]["eager_tactics"] += 1
                for _ in range(2):
                    graph.replay()
                    torch.cuda.synchronize()
                records[distribution]["graph_tactics"] += 1
            assert compile_calls == warmed_compile_calls, (
                "Graph/distribution execution recompiled an owned kernel"
            )
            graph.reset()
            print(
                f"{args.quant_mode} per_token={per_token} local_tokens={tokens}: tactic {index + 1}/{len(tactics)}",
                flush=True,
            )
        cases.extend(records.values())
    assert owned == set(compiled), (owned, set(compiled))
    (args.output / "result.json").write_text(
        json.dumps(
            dict(
                mode=args.mode,
                quant_mode=args.quant_mode,
                per_token=args.per_token,
                local_tokens=args.local_tokens,
                tactic_shard=args.tactic_shard,
                tactic_shards=shards,
                total_tactics=len(tactics),
                tactic_start=start,
                tactic_stop=stop,
                tactic_indices=list(range(start, stop)),
                selected_tactics=tactics[start:stop],
                tactic_catalog_sha256=hashlib.sha256(
                    json.dumps(tactics).encode()
                ).hexdigest(),
                ep=args.ep,
                local_experts=local_experts,
                local_expert_offset=offset,
                shape=[_HIDDEN, _INTERMEDIATE, _EXPERTS, _TOP_K],
                opt_level=args.opt_level,
                apply_topk_in_fc1=False,
                swap_ab=True,
                use_fused_finalize=False,
                output_dtype="bfloat16",
                native_opt_level=3,
                native_entries=native_entries,
                cute_kernels=compiled,
                cute_compile_calls=compile_calls,
                cases=cases,
                seconds=time.perf_counter() - started,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool", choices=("memcheck", "synccheck", "racecheck"), required=True
    )
    parser.add_argument("--quant-mode", dest="mode", choices=_MODES, required=True)
    parser.add_argument("--tactic-shard", type=int, required=True)
    parser.add_argument("--local-tokens", type=int, choices=_TOKENS, required=True)
    parser.add_argument("--opt-level", type=int, choices=(3,), required=True)
    parser.add_argument("--ep", type=int, choices=(8,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.quant_mode, args.per_token = _mode_config(args.mode)
    if not 0 <= args.tactic_shard < _num_tactic_shards(args.tool):
        parser.error(
            "--tactic-shard must be 0 for memcheck/synccheck or 0..7 for racecheck"
        )
    _run_worker(args)
