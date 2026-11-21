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

"""Check and time routed tile256/tile512 kernels with matched inputs."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

from ..routed_config import routed_options


@contextmanager
def routed_inputs(runner, fixture, *, seed=1111):
    """Scope benchmark-only data factories; kernel modules are never replaced."""
    import torch

    names = ("_make_token_layout", "_random_quantized_tensor", "_random_sf_bytes")
    originals = [getattr(runner, name) for name in names]

    def token_layout(**kwargs):
        layout = originals[0](**kwargs)
        assert kwargs["num_tokens"] == 512
        assert kwargs["top_k"] == kwargs["num_experts"]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        expected_routes = sum(layout.tokens_per_expert)
        if fixture:
            route_data = fixture.load("route")
            if route_data.dtype != torch.int32:
                raise ValueError("route fixture must have dtype torch.int32")
            if route_data.numel() != expected_routes:
                raise ValueError(f"route fixture must contain {expected_routes} IDs")
            if ((route_data < 0) | (route_data >= 512)).any().item():
                raise ValueError("route fixture IDs must be in [0, 512)")
            routes = route_data.reshape(-1).tolist()
        else:
            routes = torch.randint(
                0,
                512,
                (expected_routes,),
                generator=generator,
                dtype=torch.int32,
            ).tolist()
        # Wider token CGAs may pad each 512-row expert to 1024 rows. Keep
        # the same valid gathers and insert padding separately per expert.
        if len(routes) != layout.total_padded_tokens:
            padded_routes = [0] * layout.total_padded_tokens
            offset = 0
            for start, count in zip(
                layout.padded_starts[:-1], layout.tokens_per_expert, strict=True
            ):
                padded_routes[start : start + count] = routes[offset : offset + count]
                offset += count
            routes = padded_routes
        return replace(layout, expanded_to_token=routes)

    def fp4_values(shape, *, dtype_kind, fan_in, device):
        assert runner._is_packed_e2m1_dtype(dtype_kind)
        storage_shape = runner._storage_shape_for_dtype(shape, dtype_kind)
        if fixture:
            return fixture.load("B" if len(shape) == 3 else "A", device).reshape(
                storage_shape
            )
        # DKG's integer distribution is exactly representable as packed E2M1.
        codes = (0xC, 0xA, 0x0, 0x2)  # -2, -1, 0, 1
        lookup = torch.tensor(
            [codes[i % 4] | (codes[i // 4] << 4) for i in range(16)],
            dtype=torch.uint8,
            device=device,
        )
        packed = torch.empty(storage_shape, dtype=torch.uint8, device=device)
        flat = packed.view(-1)
        for start in range(0, flat.numel(), 1 << 20):
            chunk = flat[start : start + (1 << 20)]
            indices = runner._random_uint8(chunk.shape, high=16, device=device)
            chunk.copy_(lookup[indices.long()])
        return packed

    def sf_values(shape, *, cfg, device, layout_kind):
        assert not cfg.uses_mx_scale_factors
        if fixture:
            return fixture.load(
                "SFA" if layout_kind == "linear" else "SFB", device
            ).reshape(shape)
        return runner._random_uint8(shape, high=2, device=device).mul_(8).add_(0x38)

    try:
        for name, factory in zip(
            names, (token_layout, fp4_values, sf_values), strict=False
        ):
            setattr(runner, name, factory)
        yield
    finally:
        for name, factory in zip(names, originals, strict=False):
            setattr(runner, name, factory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("e256", "e384"), default="e256")
    parser.add_argument(
        "--tile-m",
        type=int,
        choices=(256, 512),
        default=256,
        help="Rows per two-CTA MMA atom",
    )
    parser.add_argument(
        "--cluster-shape", type=int, nargs=2, default=(2, 1), metavar=("X", "Y")
    )
    parser.add_argument("--mode", choices=("check", "bench"), default="check")
    parser.add_argument("--scheduler", choices=("static", "clc"))
    parser.add_argument("--rolled-mma", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--weight-pipeline", choices=("auto", "shared", "separate"), default="auto"
    )
    parser.add_argument("--work-throttle", type=int, choices=(0, 1))
    parser.add_argument("--experts", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timer", choices=("cupti", "events"), default="cupti")
    args = parser.parse_args()
    if min(args.cluster_shape) <= 0:
        parser.error("Both cluster-shape dimensions must be positive.")
    prefix = (
        args.output
        or Path("routed-results") / f"{args.case}-{args.mode}-{time.time_ns()}"
    )
    if prefix.with_suffix(".json").exists():
        parser.error(f"output already exists: {prefix.with_suffix('.json')}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_KEEP"] = ",".join(
        filter(None, (os.environ.get("CUTE_DSL_KEEP"), "ptx", "cubin"))
    )
    dump_dir = Path(
        os.environ.setdefault(
            "CUTE_DSL_DUMP_DIR", str(prefix.parent / "compiler-dumps")
        )
    )
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Set this before importing the runner: the same opt2 compiler is used for
    # correctness and timing. No compiler/toolkit paths are hard-coded here.
    os.environ["FLASHINFER_PRIMS_TS_COMPILE_OPTIONS"] = "--opt-level 2"
    import torch
    import cuda.bindings.driver as cuda_drv
    import cutlass._mlir._mlir_libs._cutlass_ir as compiler
    from .. import batched_gemm_run as runner
    from ..batched_gemm_config import make_config, compute_launch_grid
    from .routed_inputs import Fixture

    experts = args.experts or (256 if args.case == "e256" else 384)
    n = args.n or (4096 if args.case == "e256" else 6144)
    scheduler = args.scheduler or ("static" if args.case == "e256" else "clc")
    fixture = (
        Fixture(args.fixture_dir, (512, n, args.k, experts))
        if args.fixture_dir
        else None
    )
    options = routed_options(
        collective_tile_m=args.tile_m,
        cluster_shape=tuple(args.cluster_shape),
        scheduler=scheduler,
        rolled_mma=args.rolled_mma,
        weight_pipeline=args.weight_pipeline,
        work_throttle=args.work_throttle,
        max_active_clusters=torch.cuda.get_device_properties(0).multi_processor_count
        // (args.cluster_shape[0] * args.cluster_shape[1]),
    )
    cfg = make_config(**options)
    shape = dict(
        num_experts=experts,
        num_tokens=512,
        top_k=experts,
        problem_n=n,
        problem_k=args.k,
        seed=1111,
        routing_seed=1111,
        routing="balanced_round_robin",
        data_distribution="default",
        scale_c_value=1.0,
        scale_gate_value=1.0,
    )
    layout = runner._make_token_layout(
        num_tokens=512,
        num_experts=experts,
        top_k=experts,
        tile_size=cfg.token_tile_size,
        cluster_dim_in_token=cfg.token_cluster_dim,
    )
    metadata = dict(
        shape=[512, n, args.k, experts],
        scheduler=scheduler,
        config=options,
        gpu=torch.cuda.get_device_name(),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        compiler=compiler.__file__,
        compile_options="--opt-level 2",
        output="BF16, no SwiGLU",
        collective_tile=[args.tile_m, 256, 256],
        cluster_shape=list(args.cluster_shape),
        total_padded_tokens=layout.total_padded_tokens,
        token_rows_per_cga=layout.cga_tile_size,
        logical_clusters=layout.num_token_cgas
        * (
            (n + cfg.tile_n * args.cluster_shape[1] - 1)
            // (cfg.tile_n * args.cluster_shape[1])
        ),
        launch_grid=compute_launch_grid(
            cfg,
            problem_m=layout.total_padded_tokens,
            problem_n=n,
            early_exit_max_token_ctas=0,
        ),
        input_pool=512,
        routing="512 random gathers per expert, with replacement",
        ab_distribution="packed integers -2,-1,0,1",
        sf_distribution="E4M3 1 or 2",
        shared_inputs=fixture.manifest if fixture else None,
        warp_layout=asdict(cfg.warp_layout),
    )
    with routed_inputs(runner, fixture):
        if args.mode == "check":
            result = runner.reference_check(
                **shape, repeat_launches=3, output_guard_elements=256, **options
            )
            assert result and result.max_abs == 0, result
            metadata["correctness"] = vars(result)
        else:
            io = runner._build_launch_io(**shape, **options)
            tensors = {tensor.data_ptr(): tensor for tensor in io["_keepalive"]}
            audit = {}
            names = dict(
                a_dp="A", b_dp="B", sfa_dp="SFA", sfb_dp="SFB", route_map_dp="route"
            )
            for key, name in names.items():
                tensor = tensors[io[key]._pointer]
                record = dict(
                    shape=list(tensor.shape),
                    numel=tensor.numel(),
                    dtype=str(tensor.dtype),
                    nonzero=int(torch.count_nonzero(tensor).item()),
                    min=int(tensor.min().item()),
                    max=int(tensor.max().item()),
                )
                assert record["nonzero"] > 0, name
                if fixture:
                    audited_tensor = tensor
                    if name == "route":
                        layout = io["token_layout"]
                        if layout.total_padded_tokens != sum(layout.tokens_per_expert):
                            audited_tensor = torch.cat(
                                [
                                    tensor[start : start + count]
                                    for start, count in zip(
                                        layout.padded_starts[:-1],
                                        layout.tokens_per_expert,
                                        strict=True,
                                    )
                                ]
                            )
                            record["sha256_scope"] = (
                                "valid rows, excluding expert padding"
                            )
                    record["sha256"] = fixture.audit(name, audited_tensor)
                audit[key] = record
            metadata["input_audit"] = audit
            torch_stream = torch.cuda.Stream()
            stream = cuda_drv.CUstream(torch_stream.cuda_stream)
            compiled = runner._compile_for_launch(io, stream)
            # CuTe returns paths when CUTE_DSL_KEEP writes artifacts to disk,
            # and inline contents otherwise.
            ptx, cubin = compiled.__ptx__, compiled.__cubin__
            if isinstance(cubin, str):
                cubin = Path(cubin).read_bytes()
            if isinstance(ptx, str) and "\n" not in ptx:
                ptx = Path(ptx).read_text()
            prefix.with_suffix(".ptx").write_text(ptx)
            prefix.with_suffix(".cubin").write_bytes(cubin)
            us = runner.benchmark_compiled(
                compiled,
                io,
                stream,
                warmup_iters=10,
                bench_iters=50,
                num_rotated_buffers=0,
                use_cuda_graphs=False,
                use_cupti=args.timer == "cupti",
            )
            metadata.update(
                us=us,
                tflops=2 * 512 * n * args.k * experts / (us * 1e6),
                timer=args.timer,
                warmups=10,
                iterations=50,
                workspace_count=1,
            )
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
