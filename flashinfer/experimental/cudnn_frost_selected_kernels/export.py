"""Export standalone cuDNN Frost BF16 Python kernels for FlashInfer runtime JIT."""

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
):
    import cudnn

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    token = graph.tensor(
        name="token",
        dim=[1, s, k],
        stride=[s * k, k, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    gate = graph.tensor(
        name="gate_weight",
        dim=[experts, k, n],
        stride=[k * n, 1, k],
        data_type=cudnn.data_type.BFLOAT16,
    )
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
    if op != "grouped_gemm1_swiglu":
        raise ValueError(f"unsupported export op {op!r}")
    up = graph.tensor(
        name="up_weight",
        dim=[experts, k, n],
        stride=[k * n, 1, k],
        data_type=cudnn.data_type.BFLOAT16,
    )
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
    up_out = graph.moe_grouped_matmul(
        token,
        up,
        offsets,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="up_gemm",
    )
    activated = graph.swish(input=gate_out, name="silu")
    swiglu = graph.mul(a=activated, b=up_out, name="swiglu")
    output = graph.mul(a=swiglu, b=scale, name="scale_output")
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
) -> dict[str, str]:
    source = _standalone_source(generated_path)
    digest = hashlib.sha256(source.encode()).hexdigest()
    name = _slug(artifact_id)
    if not name.startswith("cudnn_frost_"):
        name = f"cudnn_frost_{name}"
    path = output_dir / "sources" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _sha256(path) == digest:
            return {"path": path.relative_to(output_dir).as_posix(), "sha256": digest}
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
    return {"path": path.relative_to(output_dir).as_posix(), "sha256": digest}


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
    swap_ab = getattr(config, "swap_ab", False)
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
    graph = _build_graph(args.s, args.n, args.k, args.experts, args.groups, op)
    with force_stg_epi(args.store_mode == "stg"):
        compiled = _compile_graph(graph, config, args.cta_group, args.scheduler)
    expected_gemms = 2 if op == "grouped_gemm1_swiglu" else 1
    if not compiled.chain.has_moe or compiled.chain.num_gemms != expected_gemms:
        raise RuntimeError(
            f"cuDNN Frost did not compile {op} as {expected_gemms} grouped GEMM(s)"
        )
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
    prefix = "grouped_swiglu" if op == "grouped_gemm1_swiglu" else "grouped_fc2"
    artifact_id = args.id or _slug(
        f"{prefix}_{arch}_e{args.experts}_n{args.n}_k{args.k}_"
        f"g{args.groups}_{config.name}_{args.cta_group}cta_{args.scheduler}_"
        f"{actual_store_mode}"
    )
    output_dir = args.output_dir.resolve()
    source = _export_source(
        Path(compiled.generated_path), output_dir, artifact_id, replace=args.replace
    )
    tma_slots: frozenset[int] = getattr(compiled, "tma_slots", frozenset())
    launch_tail = ["scale", "output"] if 0 in tma_slots else ["output", "scale"]
    if op == "grouped_gemm2":
        launch_tail = ["output"]
    return {
        "id": artifact_id,
        "op": op,
        "arch": arch,
        "abi": f"cudnn_frost_{op}{'_swap_ab' if swap_ab else ''}_v1",
        "source": source,
        "workspace_bytes": int(compiled.workspace_bytes),
        "launch": {"tail": launch_tail},
        "contract": {
            "s": {"min": 1},
            "n": args.n,
            "k": args.k,
            "experts": args.experts,
            "groups": args.groups,
            "token_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "activation": "silu(gate) * up" if expected_gemms == 2 else "identity",
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
    parser.add_argument(
        "--op",
        choices=("grouped_gemm1_swiglu", "grouped_gemm2"),
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
