"""Export standalone cuDNN Frost grouped kernels for FlashInfer runtime JIT."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import hashlib
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

from .activations import ACTIVATIONS, is_gated


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_graph(
    s: int,
    n: int,
    k: int,
    experts: int,
    groups: int,
    op: str = "grouped_gemm1_swiglu",
    dtype: str = "bf16",
):
    import cudnn

    if dtype not in ("bf16", "mxfp8"):
        raise ValueError(f"unsupported export dtype {dtype!r}")
    if dtype == "mxfp8" and (n % 128 or k % 128):
        raise ValueError("MXFP8 export requires N and K divisible by 128")
    data_type = (
        cudnn.data_type.FP8_E4M3 if dtype == "mxfp8" else cudnn.data_type.BFLOAT16
    )
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    token = graph.tensor(
        name="token",
        dim=[1, s, k],
        stride=[s * k, k, 1],
        data_type=data_type,
    )
    gate = graph.tensor(
        name="gate_weight",
        dim=[experts, k, n],
        stride=[k * n, 1, k],
        data_type=data_type,
    )

    def dequantize(tensor, name, token=False):
        if dtype == "bf16":
            return tensor
        sf_k = k // 32
        scales = graph.tensor(
            name=name,
            dim=[1, s, sf_k] if token else [experts, sf_k, n],
            stride=[s * sf_k, sf_k, 1] if token else [sf_k * n, 1, sf_k],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        return graph.block_scale_dequantize(
            input=tensor, descale=scales, block_size=[1, 32] if token else [32, 1]
        )

    token = dequantize(token, "token_scale", token=True)
    gate = dequantize(gate, "gate_weight_scale")
    if op == "grouped_gemm2":
        offsets = graph.tensor(
            name="first_token_offset",
            dim=[groups, 1, 1],
            stride=[1, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        output = graph.moe_grouped_matmul(
            token,
            gate,
            offsets,
            mode=cudnn.moe_grouped_matmul_mode.NONE,
            compute_data_type=cudnn.data_type.FLOAT,
            name="fc2",
        )
        output.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
        return graph
    activation = op.removeprefix("grouped_gemm1_")
    if activation not in ACTIVATIONS:
        raise ValueError(f"unsupported export op {op!r}")
    up = graph.tensor(
        name="up_weight",
        dim=[experts, k, n],
        stride=[k * n, 1, k],
        data_type=data_type,
    )
    up = dequantize(up, "up_weight_scale")
    offsets = graph.tensor(
        name="first_token_offset",
        dim=[groups, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    scale = graph.tensor(
        name="scale",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    gate_out = graph.moe_grouped_matmul(
        token,
        gate,
        offsets,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="gate_gemm",
    )
    up_out = None
    if is_gated(activation):
        up_out = graph.moe_grouped_matmul(
            token,
            up,
            offsets,
            mode=cudnn.moe_grouped_matmul_mode.NONE,
            compute_data_type=cudnn.data_type.FLOAT,
            name="up_gemm",
        )
    unary = {
        "swiglu": "swish",
        "swiglu_step": "swish",
        "silu": "swish",
        "geglu": "gelu",
        "gelu": "gelu",
        "geglu_tanh": "gelu_approx_tanh",
        "relu": "relu",
        "relu2": "relu",
    }
    activated = gate_out
    if activation in unary:
        activated = getattr(graph, unary[activation])(input=gate_out, name=activation)
    if activation == "relu2":
        activated = graph.mul(a=activated, b=activated, name="square")
    elif activation == "swiglu_step":
        activated = graph.relu(
            input=activated, negative_slope=1.0, upper_clip=7.0, name="gate_clamp"
        )
        up_out = graph.relu(
            input=up_out,
            negative_slope=1.0,
            lower_clip=-7.0,
            upper_clip=7.0,
            name="up_clamp",
        )
    elif activation == "situ":
        gate_scale = graph.tensor(
            name="gate_scale",
            dim=[1, 1, 1],
            stride=[1, 1, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        linear_scale = graph.tensor(
            name="linear_scale",
            dim=[1, 1, 1],
            stride=[1, 1, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        cap = graph.tanh(input=graph.div(a=gate_out, b=gate_scale))
        activated = graph.mul(
            a=graph.mul(a=gate_scale, b=cap), b=graph.sigmoid(input=gate_out)
        )
        up_out = graph.mul(
            a=linear_scale, b=graph.tanh(input=graph.div(a=up_out, b=linear_scale))
        )
    if up_out is not None:
        activated = graph.mul(a=activated, b=up_out, name="gated_activation")
    output = graph.mul(a=activated, b=scale, name="scale_output")
    output.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    return graph


def _standalone_source(path: Path) -> str:
    """Freeze generated code and its helpers; leave only CUDA/CuTe dependencies.

    Keep device function bodies byte-faithful. Only imports and the ordinary
    Python compile() call are rewritten. FlashInfer owns the persistent cache.
    """
    inlined: set[str] = set()

    def freeze(source: str, *, helper: bool = False) -> str:
        lines = source.splitlines(keepends=True)
        edits = []
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if helper and node.module == "__future__":
                edits.append((node.lineno - 1, node.end_lineno, ""))
            elif node.module and node.module.startswith("cudnn."):
                replacement = ""
                if node.module != "cudnn.frost.compiled_cache":
                    if node.module not in inlined:
                        inlined.add(node.module)
                        spec = importlib.util.find_spec(node.module)
                        if spec is None or spec.origin is None:
                            raise RuntimeError(
                                f"Missing cuDNN Frost helper {node.module}"
                            )
                        replacement = (
                            f"# Inlined from {node.module}\n"
                            + freeze(Path(spec.origin).read_text(), helper=True)
                            + "\n"
                        )
                    for alias in node.names:
                        if alias.asname and alias.asname != alias.name:
                            replacement += f"{alias.asname} = {alias.name}\n"
                edits.append((node.lineno - 1, node.end_lineno, replacement))
        for start, end, replacement in reversed(edits):
            lines[start:end] = [replacement]
        source = "".join(lines)
        # compile_cached adds a second disk cache and a cuDNN runtime dependency.
        lines = source.splitlines(keepends=True)
        for statement in ast.walk(ast.parse(source)):
            if isinstance(statement, ast.Return) and isinstance(
                statement.value, ast.Call
            ):
                call = statement.value
                if (
                    isinstance(call.func, ast.Name)
                    and call.func.id == "_compile_cached"
                ):
                    call.func = ast.Attribute(
                        value=ast.Name(id="cute", ctx=ast.Load()),
                        attr="compile",
                        ctx=ast.Load(),
                    )
                    call.keywords = [
                        k for k in call.keywords if k.arg not in ("cache_key", "symbol")
                    ]
                    lines[statement.lineno - 1 : statement.end_lineno] = [
                        " " * statement.col_offset
                        + "return "
                        + ast.unparse(call)
                        + "\n"
                    ]
        return "".join(lines)

    source = freeze(path.read_text())
    for node in ast.walk(ast.parse(source)):
        modules = []
        if isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        if any(m == "cudnn" or m.startswith("cudnn.") for m in modules):
            raise RuntimeError("Exported source still imports the cuDNN frontend")
    return source


def _export_source(
    generated_path: Path,
    output_dir: Path,
    artifact_id: str,
    *,
    replace: bool = False,
    template_family: str | None = None,
    swap_ab: bool = False,
) -> dict[str, Any]:
    source = _standalone_source(generated_path)
    parameters = None
    if template_family is not None:
        from .source_template import extract_template

        source, parameters = extract_template(source, swap_ab=swap_ab)
    digest = hashlib.sha256(source.encode()).hexdigest()
    # One maintained implementation per family. A producer upgrade replaces the
    # family explicitly; it must not silently add another historical template.
    name = _slug(artifact_id if template_family is None else template_family)
    if not name.startswith("cudnn_frost_"):
        name = f"cudnn_frost_{name}"
    path = output_dir / "sources" / f"{name}.py"
    record: dict[str, Any] = {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": digest,
    }
    if parameters is not None:
        record["parameters"] = parameters
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _sha256(path) == digest:
            return record
        if not replace:
            raise RuntimeError(
                f"cuDNN Frost source {path} already exists with different contents; "
                "pass --replace to update it"
            )
    temporary = path.with_suffix(f".py.tmp.{os.getpid()}")
    try:
        temporary.write_text(source)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def _compile_graph(graph: Any, config: Any, cta_group: int, scheduler: str) -> Any:
    """Call both the old and current cuDNN Frost compiler APIs.

    Older cuDNN Frost revisions selected the execution strategy with keyword
    arguments.  Current revisions encode ``cta_group`` in ``TileConfig`` and
    infer the scheduler from the selected template.
    """
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph

    parameters = inspect.signature(jit_from_cudnn_graph).parameters
    kwargs: dict[str, Any] = {"config": config}
    if "cta_group" in parameters:
        kwargs["cta_group"] = cta_group
    elif config.cta_group != cta_group:
        raise ValueError(
            f"tile config {config.name!r} has cta_group={config.cta_group}, "
            f"but --cta-group={cta_group} was requested"
        )
    if "scheduler" in parameters:
        kwargs["scheduler"] = scheduler
    elif scheduler != "clc":
        raise ValueError(
            "this cuDNN Frost revision chooses the scheduler from the template; "
            "only --scheduler=clc is supported by this exporter"
        )
    return jit_from_cudnn_graph(graph, **kwargs)


def _select_template(chain: Any, config: Any, cta_group: int, scheduler: str) -> Any:
    from cudnn.gemm.frost.kernel_registry import select_template

    parameters = inspect.signature(select_template).parameters
    args = [chain, config]
    if "cta_group" in parameters:
        args.append(cta_group)
    if "scheduler" in parameters:
        args.append(scheduler)
    return select_template(*args)


def export_one(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from cudnn.gemm.frost.compiler import force_stg_epi
    from cudnn.gemm.frost import tile_config

    # Swap-AB and extended geometries are parsed by cuDNN Frost but need not be in
    # its default heuristic catalog. Preserve compatibility with older builds.
    if hasattr(tile_config, "by_name"):
        config = tile_config.by_name(args.config)
    else:
        config = next((c for c in tile_config.CATALOG if c.name == args.config), None)
        if config is None:
            raise ValueError(f"unknown cuDNN Frost tile config {args.config!r}")
    if getattr(config, "split_k_slices", 1) != 1:
        raise ValueError(
            "the exported MoE ABI does not support split-K launch sequences"
        )
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}a"
    target = os.environ.get("CUTE_DSL_ARCH", arch).replace("_", "")
    if target != arch.replace("_", ""):
        raise ValueError("CUTE_DSL_ARCH must match the export GPU architecture")
    op = getattr(args, "op", "grouped_gemm1_swiglu")
    dtype = getattr(args, "dtype", "bf16")
    graph = _build_graph(args.s, args.n, args.k, args.experts, args.groups, op, dtype)
    with force_stg_epi(args.store_mode == "stg"):
        compiled = _compile_graph(graph, config, args.cta_group, args.scheduler)
    return _export_compiled(args, compiled, config, arch)


def _export_compiled(args, compiled, config, arch):
    """Seal a producer result, also usable by source-only export tooling."""
    if getattr(config, "cta_group", args.cta_group) != args.cta_group:
        raise ValueError("exported CTA group must match the rendered tile config")
    op = getattr(args, "op", "grouped_gemm1_swiglu")
    dtype = getattr(args, "dtype", "bf16")
    swap_ab = getattr(config, "swap_ab", False)
    activation = (
        op.removeprefix("grouped_gemm1_") if op != "grouped_gemm2" else "identity"
    )
    expected_gemms = 2 if op != "grouped_gemm2" and is_gated(activation) else 1
    if not compiled.chain.has_moe or compiled.chain.num_gemms != expected_gemms:
        raise RuntimeError(
            f"cuDNN Frost did not compile {op} as {expected_gemms} grouped GEMM(s)"
        )
    if compiled.chain.has_block_scale != (dtype == "mxfp8"):
        raise RuntimeError("cuDNN Frost compiled the wrong quantization pipeline")
    store_modes = tuple(compiled.store_modes)
    if len(store_modes) != 1 or store_modes[0] not in ("stg", "tma"):
        raise RuntimeError(
            f"unexpected cuDNN Frost output store modes: {store_modes!r}"
        )
    actual_store_mode = store_modes[0]
    if args.store_mode == "tma" and actual_store_mode != "tma":
        raise RuntimeError(
            f"cuDNN Frost did not produce the requested TMA-store kernel for {config.name}"
        )
    template = _select_template(compiled.chain, config, args.cta_group, args.scheduler)
    prefix = op if dtype == "bf16" else f"block_scale_{op}"
    artifact_id = args.id or _slug(
        f"{prefix}_{arch}_e{args.experts}_n{args.n}_k{args.k}_"
        f"g{args.groups}_{config.name}_{args.cta_group}cta_{args.scheduler}_"
        f"{actual_store_mode}"
    )
    output_dir = args.output_dir.resolve()
    source = _export_source(
        Path(compiled.generated_path),
        output_dir,
        artifact_id,
        replace=args.replace,
        template_family=f"{op}_{dtype}_{'swap_ab' if swap_ab else 'normal'}_{actual_store_mode}",
        swap_ab=swap_ab,
    )
    tma_slots: frozenset[int] = getattr(compiled, "tma_slots", frozenset())
    aux_names = [aux.name for aux in compiled.chain.aux_tensors]
    launch_tail = aux_names + ["output"] if 0 in tma_slots else ["output"] + aux_names
    expected_aux = (
        {"scale", "gate_scale", "linear_scale"} if activation == "situ" else {"scale"}
    )
    if op != "grouped_gemm2" and set(aux_names) != expected_aux:
        raise RuntimeError(f"Unexpected FC1 auxiliary tensors: {aux_names}")
    if op == "grouped_gemm2":
        launch_tail = ["output"]
    return {
        "id": artifact_id,
        "op": prefix,
        "arch": arch,
        "abi": f"cudnn_frost_{prefix}{'_swap_ab' if swap_ab else ''}_v1",
        "source": source,
        "workspace_bytes": int(compiled.workspace_bytes),
        "launch": {"tail": launch_tail},
        "contract": {
            "s": {"min": 1},
            "n": args.n,
            "k": args.k,
            "experts": args.experts,
            "groups": args.groups,
            "token_dtype": "float8_e4m3fn" if dtype == "mxfp8" else "bfloat16",
            "weight_dtype": "float8_e4m3fn" if dtype == "mxfp8" else "bfloat16",
            "output_dtype": "bfloat16",
            "activation": activation,
            **(
                {
                    "scale_dtype": "float8_e8m0fnu",
                    "block_size": 32,
                    "scale_layout": "F8_128x4",
                    "token_scale_layout": "segmented_F8_128x4",
                }
                if dtype == "mxfp8"
                else {}
            ),
        },
        "tactic": {
            "swap_ab": swap_ab,
            "template": template.file,
            "tile": config.name,
            "cta_tile": {
                "m": int(config.cta_tile_m),
                "n": int(config.cta_tile_n),
                "k_bytes": int(config.cta_tile_k_bytes),
            },
            "cta_group": args.cta_group,
            "scheduler": args.scheduler,
            "store_mode": actual_store_mode,
        },
        "producer_revision": args.cudnn_frost_revision,
    }


def _write_manifest(output_dir: Path, kernel: dict[str, Any], replace: bool) -> None:
    path = output_dir / "cudnn_frost_selected_kernels.json"
    payload: dict[str, Any] = {
        "schema_version": 2,
        "producer": {"name": "cudnn_frost"},
        "kernels": [],
    }
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != 2:
            raise RuntimeError(f"refusing to update unsupported manifest {path}")
    kernels = list(payload.get("kernels", []))
    old = next(
        (i for i, item in enumerate(kernels) if item.get("id") == kernel["id"]),
        None,
    )
    if old is not None:
        if not replace:
            raise RuntimeError(
                f"artifact {kernel['id']!r} already exists; pass --replace to update it"
            )
        kernels[old] = kernel
    else:
        kernels.append(kernel)
    payload["kernels"] = sorted(kernels, key=lambda item: item["id"])
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "mxfp8"), default="bf16")
    parser.add_argument(
        "--op",
        choices=(*(f"grouped_gemm1_{name}" for name in ACTIVATIONS), "grouped_gemm2"),
        default="grouped_gemm1_swiglu",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", required=True, help="cuDNN Frost TileConfig name")
    parser.add_argument("--cudnn-frost-revision", required=True)
    parser.add_argument("--cta-group", type=int, choices=(1, 2), default=2)
    parser.add_argument("--scheduler", choices=("clc", "static"), default="clc")
    parser.add_argument("--store-mode", choices=("tma", "stg"), default="tma")
    parser.add_argument("--id", help="stable artifact id (derived by default)")
    parser.add_argument("--s", type=int, default=1024)
    parser.add_argument(
        "--n", type=int, required=True, help="GEMM output width (FC2: hidden size)"
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="GEMM reduction width (FC2: intermediate size)",
    )
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--groups", type=int, help="defaults to --experts")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.groups is None:
        args.groups = args.experts
    kernel = export_one(args)
    _write_manifest(args.output_dir.resolve(), kernel, args.replace)
    print(f"exported {kernel['id']} to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
