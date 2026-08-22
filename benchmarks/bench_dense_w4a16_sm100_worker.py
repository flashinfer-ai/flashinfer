#!/usr/bin/env python3
"""Run one controlled SM100/SM103 dense CuTe-DSL W4A16 measurement.

This is the worker for :mod:`bench_dense_w4a16_sm100`.  It intentionally makes
exactly one CUPTI timing call, because ``cupti.finalize()`` is process-global
teardown.  Setup, tactic selection, compilation, correctness, and repeatability
checks all happen before that timing call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


CANONICAL_TACTIC_SPEC = "256,64,256:2,1:true"
E2M1_VALUES_FP32 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_cupti() -> tuple[Any, str, int]:
    """Fail closed instead of accepting ``bench_gpu_time``'s event fallback."""
    try:
        from cupti import cupti
    except ImportError as error:
        raise RuntimeError(
            "cupti-python >= 13 is required; install it with "
            "`python -m pip install -U cupti-python`"
        ) from error
    version = _package_version("cupti-python")
    if version is None or int(version.split(".", 1)[0]) < 13:
        raise RuntimeError(f"cupti-python >= 13 is required, found {version!r}")
    try:
        timestamp = int(cupti.get_timestamp())
    except Exception as error:
        raise RuntimeError("CUPTI imported but get_timestamp() failed") from error
    if timestamp <= 0:
        raise RuntimeError(f"CUPTI returned an invalid timestamp: {timestamp}")
    return cupti, version, timestamp


def _require_autotuner_runtime_assets() -> None:
    """Fail before tactic profiling if the AutoTuner delay kernel is unusable."""
    from flashinfer.jit import env as jit_env

    delay_source = (
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal/tensorrt_llm/kernels/delayStream.cu"
    )
    install_guidance = (
        "rerun `python -m pip install -e . --no-build-isolation` from this "
        "FlashInfer checkout after syncing it"
    )
    if not delay_source.is_file():
        raise RuntimeError(
            "FlashInfer AutoTuner delay-kernel source is missing at "
            f"{delay_source}; editable-install data symlinks may have been removed; "
            f"{install_guidance}"
        )

    try:
        from flashinfer.tllm_utils import get_trtllm_utils_module

        module = get_trtllm_utils_module()
        if not hasattr(module, "delay_kernel"):
            raise RuntimeError("loaded trtllm_utils module has no delay_kernel")
    except Exception as error:
        raise RuntimeError(
            "FlashInfer AutoTuner delay-kernel JIT preflight failed; "
            f"{install_guidance}"
        ) from error


def _repo_metadata(repo: Path) -> dict[str, Any]:
    diff = _command(["git", "diff", "--binary", "HEAD"], repo)
    return {
        "revision": _command(["git", "rev-parse", "HEAD"], repo),
        "branch": _command(["git", "branch", "--show-current"], repo),
        "status_short": _command(["git", "status", "--short"], repo),
        "diff_stat": _command(["git", "diff", "--stat"], repo),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()
        if diff is not None
        else None,
    }


def _environment(
    torch: Any,
    flashinfer: Any,
    repo: Path,
    cupti_version: str,
    cupti_timestamp: int,
) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer_version": getattr(flashinfer, "__version__", None),
        "flashinfer_file": str(Path(flashinfer.__file__).resolve()),
        "cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
        "cupti_python": cupti_version,
        "nvidia_cuda_cupti": _package_version("nvidia-cuda-cupti"),
        "cuda_bindings": _package_version("cuda-bindings"),
        "cupti_preflight_timestamp": cupti_timestamp,
        "gpu_name": props.name,
        "compute_capability": [props.major, props.minor],
        "multiprocessor_count": props.multi_processor_count,
        "gpu_total_memory_bytes": props.total_memory,
        "container_image": os.environ.get("CONTAINER_IMAGE")
        or os.environ.get("NVIDIA_PYTORCH_VERSION"),
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,power.limit,clocks.max.sm",
                "--format=csv,noheader,nounits",
            ]
        ),
        "repo": _repo_metadata(repo),
    }


def _weight_cache_path(cache_dir: Path, n: int, k: int, seed: int) -> Path:
    return cache_dir / f"nvfp4_n{n}_k{k}_seed{seed}.pt"


def _make_weights(
    torch: Any,
    flashinfer: Any,
    *,
    n: int,
    k: int,
    seed: int,
    cache_dir: Path | None,
) -> tuple[Any, Any, Any, str]:
    cache_path = _weight_cache_path(cache_dir, n, k, seed) if cache_dir else None
    if cache_path is not None and cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("n") != n or cached.get("k") != k or cached.get("seed") != seed:
            raise RuntimeError(f"weight-cache metadata mismatch in {cache_path}")
        return (
            cached["b_fp4"].to(device="cuda", non_blocking=False),
            cached["b_sf"].to(device="cuda", non_blocking=False),
            cached["alpha"].to(device="cuda", non_blocking=False),
            "hit",
        )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
    weight.normal_(mean=0.0, std=0.1, generator=generator)
    global_scale = (448 * 6) / weight.float().abs().nan_to_num().max()
    b_fp4, b_sf = flashinfer.nvfp4_quantize(
        weight,
        global_scale,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    alpha = torch.tensor(
        [1.0 / global_scale.item()], device="cuda", dtype=torch.float32
    )
    del weight

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        torch.save(
            {
                "n": n,
                "k": k,
                "seed": seed,
                "b_fp4": b_fp4.cpu(),
                "b_sf": b_sf.cpu(),
                "alpha": alpha.cpu(),
            },
            temporary,
        )
        os.replace(temporary, cache_path)
    return b_fp4, b_sf, alpha, "miss"


def _dequantize_reference(
    torch: Any,
    b_fp4: Any,
    b_sf: Any,
    n: int,
    k: int,
) -> Any:
    from flashinfer.gemm.gemm_bf16_fp4 import _unswizzle_sf_128x4

    lut = torch.tensor(E2M1_VALUES_FP32, dtype=torch.float32, device="cuda")
    packed = b_fp4.to(torch.int64)
    codes = torch.stack([packed & 0xF, (packed >> 4) & 0xF], dim=-1).reshape(n, k)
    values = lut[codes]
    scales = _unswizzle_sf_128x4(b_sf, n, k // 16).view(torch.float8_e4m3fn)
    return values * scales.to(torch.float32).repeat_interleave(16, dim=1)


def _normalize_tactic(value: Any) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RuntimeError(f"unexpected SM100 W4A16 tactic: {value!r}")
    tile, cluster, raster = value
    normalized = (
        tuple(int(item) for item in tile),
        tuple(int(item) for item in cluster),
        bool(raster),
    )
    if len(normalized[0]) != 3 or len(normalized[1]) != 2:
        raise RuntimeError(f"unexpected SM100 W4A16 tactic: {value!r}")
    return normalized


def _parse_tactic(value: str) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    try:
        tile_text, cluster_text, raster_text = value.split(":")
        tile = tuple(int(item) for item in tile_text.split(","))
        cluster = tuple(int(item) for item in cluster_text.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "tactic must be TILE_M,TILE_N,TILE_K:CLUSTER_M,CLUSTER_N:true|false"
        ) from error
    raster_values = {"true": True, "false": False}
    if raster_text.lower() not in raster_values:
        raise argparse.ArgumentTypeError("tactic raster must be true or false")
    try:
        return _normalize_tactic((tile, cluster, raster_values[raster_text.lower()]))
    except RuntimeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _select_tactic(
    torch: Any,
    inputs: list[Any],
    *,
    m: int,
    enable_pdl: bool,
    tactic_mode: str,
    forced_tactic: tuple[tuple[int, ...], tuple[int, ...], bool],
    allow_experimental_tactic: bool,
    compile_opt_level: int,
    autotune_cache: Path | None,
) -> tuple[Any, tuple[tuple[int, ...], tuple[int, ...], bool], int, int]:
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.gemm.gemm_bf16_fp4_cute_dsl import (
        _SM100_BF16_FP4_CUTE_DSL_TUNING_CONFIG,
        _SM100_BF16_FP4_TACTICS,
        _cute_dsl_sm100_bf16_fp4_runner,
    )

    runner = _cute_dsl_sm100_bf16_fp4_runner(enable_pdl=enable_pdl)
    valid = [
        _normalize_tactic(value) for value in runner.get_valid_tactics(inputs, None)
    ]
    if not valid:
        raise RuntimeError("no valid SM100 dense W4A16 tactics")

    cache_preexisting = int(autotune_cache.is_file()) if autotune_cache else 0
    if tactic_mode == "canonical":
        tactic = _normalize_tactic(forced_tactic)
        # Raster direction does not affect can_implement. Normalize it only
        # when compile-gating a fixed experimental tactic outside production.
        validity_tactic = (tactic[0], tactic[1], True)
        if tactic not in valid and validity_tactic not in valid:
            if not allow_experimental_tactic:
                raise RuntimeError(
                    f"canonical tactic {tactic!r} is outside the production search "
                    "space; pass --allow-experimental-tactic to compile-gate it"
                )
            import cutlass

            from flashinfer.gemm.kernels.cute_dsl.dense_gemm_bf16_fp4_sm100 import (
                Sm100DenseGemmBf16Fp4Kernel,
            )

            a, b, _, _, _, _, _ = inputs
            public_m, k = map(int, a.shape)
            public_n = int(b.shape[0])
            if not Sm100DenseGemmBf16Fp4Kernel.can_implement(
                mnkl=(public_n, public_m, k, 1),
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.BFloat16,
                c_dtype=cutlass.BFloat16,
                a_major="k",
                b_major="k",
                c_major="m",
                mma_tiler=tactic[0],
                cluster_shape_mn=tactic[1],
                use_2cta_instrs=tactic[0][0] == 256,
            ):
                raise RuntimeError(
                    f"experimental tactic {tactic!r} cannot implement "
                    f"M={public_m}, N={public_n}, K={k}"
                )
    else:
        if autotune_cache is None:
            raise RuntimeError("auto tactic mode requires --autotune-cache")
        autotune_cache.parent.mkdir(parents=True, exist_ok=True)
        import re

        import cutlass.cute as cute

        original_compile = cute.compile

        def compile_with_requested_opt_level(*compile_args: Any, **compile_kwargs: Any):
            options = str(compile_kwargs.get("options", ""))
            if "--opt-level" in options:
                options = re.sub(
                    r"--opt-level(?:=|\s+)\d+",
                    f"--opt-level {compile_opt_level}",
                    options,
                )
            else:
                options = f"{options} --opt-level {compile_opt_level}".strip()
            compile_kwargs["options"] = options
            return original_compile(*compile_args, **compile_kwargs)

        try:
            # The production runner owns tactic validity and profiling. Override
            # only CuTe's host-side compile option so auto ranking and the timed
            # direct compile use the same requested optimization level.
            cute.compile = compile_with_requested_opt_level
            with autotune(
                True,
                cache=str(autotune_cache),
                tuning_buckets=(m,),
                round_up=False,
            ):
                chosen_runner, selected = AutoTuner.get().choose_one(
                    "bf16_fp4_cute_dsl_sm100_gemm",
                    [runner],
                    _SM100_BF16_FP4_CUTE_DSL_TUNING_CONFIG,
                    inputs,
                )
        finally:
            cute.compile = original_compile
        if chosen_runner.__class__.__name__ != runner.__class__.__name__:
            raise RuntimeError("autotuner returned an unexpected runner")
        if selected == -1:
            failed_count = len(
                AutoTuner.get().stats.failed_tactics.get(
                    "bf16_fp4_cute_dsl_sm100_gemm::CuteDslSm100Bf16Fp4Runner",
                    (),
                )
            )
            raise RuntimeError(
                "production autotuning selected fallback tactic -1 after "
                f"rejecting {failed_count} tactic(s); no timed tactic profile "
                "succeeded. Inspect the preceding [Autotuner] debug failures; "
                "this usually indicates a shared profiling/runtime failure, not "
                "an invalid tactic encoding"
            )
        tactic = _normalize_tactic(selected)
        if tactic not in valid:
            raise RuntimeError(f"autotuner returned invalid tactic {tactic!r}")

    all_tactics = [_normalize_tactic(value) for value in _SM100_BF16_FP4_TACTICS]
    tactic_index = all_tactics.index(tactic) if tactic in all_tactics else -1
    return runner, tactic, tactic_index, cache_preexisting


def _compile_tactic(
    torch: Any,
    inputs: list[Any],
    tactic: tuple[tuple[int, ...], tuple[int, ...], bool],
    *,
    enable_pdl: bool,
    compile_opt_level: int,
    transform_fragment_size: int | None,
) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Compile the selected tactic and expose its derived stage allocation."""
    import cutlass
    import cutlass.cute as cute

    from flashinfer.cute_dsl.utils import (
        current_cuda_stream,
        get_max_active_clusters,
        make_ptr,
    )
    from flashinfer.gemm.kernels.cute_dsl.dense_gemm_bf16_fp4_sm100 import (
        Sm100DenseGemmBf16Fp4Kernel,
    )

    a, b, b_descale, alpha, _, out, _ = inputs
    public_m, k = map(int, a.shape)
    public_n = int(b.shape[0])
    mma_tiler_mnk, cluster_shape_mn, raster_along_m = tactic
    cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
    for name, tensor, alignment in (
        ("b", b, 32),
        ("b_descale", b_descale, 16),
        ("a", a, 32),
        ("alpha", alpha, 16),
        ("out", out, 32),
    ):
        if tensor.data_ptr() % alignment:
            raise RuntimeError(
                f"{name} pointer is not {alignment}-byte aligned: "
                f"{tensor.data_ptr():#x}"
            )

    weight_ptr = make_ptr(
        cutlass.Float4E2M1FN, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    weight_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        b_descale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    activation_ptr = make_ptr(
        cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    alpha_ptr = make_ptr(
        cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    output_ptr = make_ptr(
        cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    max_active_clusters = int(get_max_active_clusters(cluster_size))
    if transform_fragment_size is None:
        transform_fragment_size = 128 if k == mma_tiler_mnk[2] else 32
    kernel = Sm100DenseGemmBf16Fp4Kernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=mma_tiler_mnk[0] == 256,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mn=cluster_shape_mn,
        enable_pdl=enable_pdl,
        raster_along_m=raster_along_m,
        transform_fragment_size=transform_fragment_size,
    )
    stream = current_cuda_stream()
    compiled = cute.compile(
        kernel.wrapper,
        weight_ptr,
        weight_sf_ptr,
        activation_ptr,
        alpha_ptr,
        output_ptr,
        public_n,
        public_m,
        k,
        max_active_clusters=max_active_clusters,
        stream=stream,
        options=f"--opt-level {compile_opt_level} --enable-tvm-ffi",
    )

    stage_names = (
        "num_load2trans_stage",
        "num_trans2mma_stage",
        "num_acc_stage",
        "num_c_stage",
        "num_tile_info_stage",
        "num_acc_tmem_cols",
        "num_a_tmem_cols",
        "num_tmem_alloc_cols",
    )
    missing = [name for name in stage_names if not hasattr(kernel, name)]
    if missing:
        raise RuntimeError(
            "CuTe compilation did not expose derived pipeline stages: "
            + ", ".join(missing)
        )
    stages = {name: int(getattr(kernel, name)) for name in stage_names}
    stages.update(
        {
            "configured_transform_fragment_size": transform_fragment_size,
            "num_transform_warpgroups": int(kernel.num_transform_warpgroups),
            "num_transform_warps": int(kernel.num_transform_warps),
            "threads_per_cta": int(kernel.threads_per_cta),
            "num_regs_epilogue_warps": int(kernel.num_regs_epilogue_warps),
            "num_regs_generic_warps": int(kernel.num_regs_generic_warps),
            "max_active_clusters": max_active_clusters,
        }
    )

    def run() -> Any:
        compiled(
            b.data_ptr(),
            b_descale.data_ptr(),
            a.data_ptr(),
            alpha.data_ptr(),
            out.data_ptr(),
            public_n,
            public_m,
            k,
            current_cuda_stream(),
        )
        return out

    return run, stages


def _check_correctness_and_repeatability(
    torch: Any,
    run: Callable[[], Any],
    out: Any,
    reference: Any,
    *,
    cuda_graph: bool,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    run()
    torch.cuda.synchronize()
    eager_first = out.clone()
    run()
    torch.cuda.synchronize()
    eager_repeatable = bool(torch.equal(eager_first, out))
    if not eager_repeatable:
        raise RuntimeError("selected tactic is not bitwise repeatable in eager mode")

    graph_repeatable: bool | None = None
    graph_matches_eager: bool | None = None
    if cuda_graph:
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.cuda.synchronize()
        graph_first = out.clone()
        graph_matches_eager = bool(torch.equal(eager_first, graph_first))
        if not graph_matches_eager:
            raise RuntimeError("CUDA graph output differs bitwise from eager output")
        graph.replay()
        torch.cuda.synchronize()
        graph_repeatable = bool(torch.equal(graph_first, out))
        if not graph_repeatable:
            raise RuntimeError(
                "selected tactic is not bitwise repeatable under graph replay"
            )
        del graph_first, graph

    torch.testing.assert_close(out, reference, rtol=rtol, atol=atol)
    difference = (out.float() - reference.float()).abs()
    max_abs = float(difference.max().item())
    max_rel = float((difference / reference.float().abs().clamp_min(1e-6)).max().item())
    all_finite = bool(torch.isfinite(out).all().item())
    if not all_finite:
        raise RuntimeError("selected tactic produced non-finite output")
    del difference, eager_first
    return {
        "reference_rtol": rtol,
        "reference_atol": atol,
        "reference_max_abs": max_abs,
        "reference_max_rel": max_rel,
        "output_all_finite": all_finite,
        "eager_bitwise_repeatable": eager_repeatable,
        "graph_bitwise_repeatable": graph_repeatable,
        "graph_matches_eager_bitwise": graph_matches_eager,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--enable-pdl", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--tactic-mode", choices=("auto", "canonical"), default="canonical"
    )
    parser.add_argument(
        "--forced-tactic",
        type=_parse_tactic,
        default=_parse_tactic(CANONICAL_TACTIC_SPEC),
        metavar="TILE_M,TILE_N,TILE_K:CLUSTER_M,CLUSTER_N:true|false",
        help=(
            "Tactic used by --tactic-mode canonical; defaults to the tuned "
            f"baseline {CANONICAL_TACTIC_SPEC}."
        ),
    )
    parser.add_argument(
        "--allow-experimental-tactic",
        action="store_true",
        help=(
            "Allow a canonical tactic outside the production search space after "
            "the kernel's can_implement check; never affects auto mode."
        ),
    )
    parser.add_argument("--compile-opt-level", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--transform-fragment-size",
        type=int,
        choices=(32, 64, 128),
        help="Override the production 32/128 transform-fragment heuristic.",
    )
    parser.add_argument("--autotune-cache", type=Path)
    parser.add_argument("--input-cache-dir", type=Path)
    parser.add_argument("--rtol", type=float, default=1.5e-2)
    parser.add_argument("--atol", type=float, default=1.5e-2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if min(args.m, args.n, args.k, args.repeat, args.iters) <= 0 or args.warmup < 0:
        raise ValueError("M/N/K/repeat/iters must be positive and warmup nonnegative")
    if args.n % 64 or args.k % 16:
        raise ValueError("dense W4A16 requires N % 64 == 0 and K % 16 == 0")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("correctness tolerances must be nonnegative")
    if args.tactic_mode == "auto" and args.autotune_cache is None:
        raise ValueError("--autotune-cache is required in auto tactic mode")
    if args.allow_experimental_tactic and args.tactic_mode != "canonical":
        raise ValueError("--allow-experimental-tactic requires canonical tactic mode")
    if args.transform_fragment_size is not None and args.tactic_mode != "canonical":
        raise ValueError(
            "--transform-fragment-size requires canonical tactic mode so the "
            "autotuner cannot rank a different production-fragment kernel"
        )

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) not in ((10, 0), (10, 3)):
        raise RuntimeError(
            f"SM100 or SM103 is required, got SM{props.major}{props.minor}"
        )

    _, cupti_version, cupti_timestamp = _require_cupti()
    repo = Path(__file__).resolve().parents[1]
    import flashinfer

    imported = Path(flashinfer.__file__).resolve()
    if not imported.is_relative_to(repo):
        raise RuntimeError(
            "flashinfer must resolve to this benchmark checkout: "
            f"imported={imported}, checkout={repo}"
        )
    if args.tactic_mode == "auto":
        _require_autotuner_runtime_assets()
    environment = _environment(torch, flashinfer, repo, cupti_version, cupti_timestamp)

    weight_seed = args.seed + args.n * 1000003 + args.k
    activation_seed = args.seed + args.m * 10007 + args.k
    b_fp4, b_sf, alpha, input_cache_status = _make_weights(
        torch,
        flashinfer,
        n=args.n,
        k=args.k,
        seed=weight_seed,
        cache_dir=args.input_cache_dir,
    )
    activation_generator = torch.Generator(device="cuda").manual_seed(activation_seed)
    a = torch.empty((args.m, args.k), device="cuda", dtype=torch.bfloat16)
    a.normal_(mean=0.0, std=0.5, generator=activation_generator)
    b_prepared, sf_prepared, alpha_prepared = flashinfer.prepare_bf16_fp4_weights(
        b_fp4, b_sf, alpha, backend="cute-dsl"
    )
    from flashinfer.gemm.gemm_bf16_fp4_cute_dsl import _prepare_bf16_fp4_alpha

    alpha_for_launch = _prepare_bf16_fp4_alpha(alpha_prepared, a.device)
    out = torch.empty((args.m, args.n), device="cuda", dtype=torch.bfloat16)
    inputs = [
        a,
        b_prepared,
        sf_prepared,
        alpha_for_launch,
        torch.bfloat16,
        out,
        16,
    ]
    _, tactic, tactic_index, cache_preexisting = _select_tactic(
        torch,
        inputs,
        m=args.m,
        enable_pdl=args.enable_pdl,
        tactic_mode=args.tactic_mode,
        forced_tactic=args.forced_tactic,
        allow_experimental_tactic=args.allow_experimental_tactic,
        compile_opt_level=args.compile_opt_level,
        autotune_cache=args.autotune_cache,
    )
    run, stages = _compile_tactic(
        torch,
        inputs,
        tactic,
        enable_pdl=args.enable_pdl,
        compile_opt_level=args.compile_opt_level,
        transform_fragment_size=args.transform_fragment_size,
    )

    weight_fp32 = _dequantize_reference(torch, b_fp4, b_sf, args.n, args.k)
    reference = ((a.float() @ weight_fp32.T) * alpha.to(torch.float32)).to(
        torch.bfloat16
    )
    correctness = _check_correctness_and_repeatability(
        torch,
        run,
        out,
        reference,
        cuda_graph=args.cuda_graph,
        rtol=args.rtol,
        atol=args.atol,
    )
    del reference, weight_fp32
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # There must be exactly one call to CUPTI timing in this process.  The
    # helper finalizes CUPTI before it returns, so do not issue more CUDA work.
    from flashinfer.testing.utils import bench_gpu_time

    samples_ms = [
        float(value)
        for value in bench_gpu_time(
            fn=run,
            dry_run_iters=args.warmup,
            repeat_iters=args.iters,
            enable_cupti=True,
            use_cuda_graph=args.cuda_graph,
            cold_l2_cache=True,
            sleep_after_run=True,
        )
    ]
    if len(samples_ms) != args.iters:
        raise RuntimeError(
            f"CUPTI returned {len(samples_ms)} samples, expected {args.iters}"
        )
    if any(not math.isfinite(value) or value <= 0 for value in samples_ms):
        raise RuntimeError(f"CUPTI returned invalid samples: {samples_ms!r}")

    median_ms = float(statistics.median(samples_ms))
    flops = 2 * args.m * args.n * args.k
    bytes_accessed = (
        args.m * args.k * 2
        + (args.k // 2) * args.n
        + (args.k // 16) * args.n
        + args.m * args.n * 2
    )
    result = {
        "schema_version": 1,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "invocation": [sys.executable, *sys.argv],
        "suite": args.suite,
        "case": args.case,
        "label": args.label,
        "repeat": args.repeat,
        "seed": args.seed,
        "weight_seed": weight_seed,
        "activation_seed": activation_seed,
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "backend": "cute-dsl",
        "input_dtype": "bfloat16",
        "weight_dtype": "nvfp4_e2m1_e4m3_block16",
        "accumulator_dtype": "float32",
        "output_dtype": "bfloat16",
        "alpha_dtype": "float32",
        "reference_scale_order": "fp32_accumulator_then_fp32_alpha_then_bfloat16",
        "preallocated_output": True,
        "enable_pdl": args.enable_pdl,
        "cuda_graph": args.cuda_graph,
        "cold_l2": True,
        "timing_backend": "cupti",
        "warmup": args.warmup,
        "iters": args.iters,
        "tactic_mode": args.tactic_mode,
        "forced_tactic_requested": args.forced_tactic,
        "allow_experimental_tactic": args.allow_experimental_tactic,
        "tactic_index": tactic_index,
        "tactic": tactic,
        "tactic_in_production_search_space": tactic_index >= 0,
        "compile_opt_level": args.compile_opt_level,
        "transform_fragment_size_requested": args.transform_fragment_size,
        "pipeline": stages,
        "autotune_cache": str(args.autotune_cache) if args.autotune_cache else None,
        "autotune_cache_preexisting": bool(cache_preexisting),
        "input_cache_status": input_cache_status,
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "std_ms": float(statistics.pstdev(samples_ms)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "tflops": flops / median_ms / 1e9,
        "tb_per_sec": bytes_accessed / median_ms / 1e9,
        "correctness": correctness,
        "environment": environment,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_result = args.result_json.with_name(
        f".{args.result_json.name}.{os.getpid()}.tmp"
    )
    temporary_result.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary_result, args.result_json)
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
