#!/usr/bin/env python3
"""Export the pinned Kimi device closure from its base package and overlay."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

KIMI_COMMIT = "6e8cfe94e9d6c92b184fb46700c72e9f66ccb884"
KIMI_MANIFEST_SHA256 = (
    "6f4126d7c03f47f24b18c272348a3215e15df9dc257bfaa0bf564283ef69224c"
)
TRT_VERSION = "1.4.0rc3.dev202610300406"
CONTAINER_DIGEST = (
    "sha256:b0c501c715bf11dc098d886194560475936fb2e900994188fc734ba5aceff5b9"
)
KERNEL_ROOT = Path("tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe")
HOST_SOURCE = Path("tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py")
ENTRIES = (
    "api.py",
    "quant_def.py",
    "helpers/fastquant.py",
    "helpers/fastquant_mx.py",
    "kernel_src/blackwell/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py",
)
HEADER = "# Exported for FlashInfer; local changes are recorded in VENDOR.md.\n"
OPTIONS_SOURCE = '''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace-local code-generation options; no process environment overrides."""

from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType

_OPTIONS = ContextVar("kimi_k3_codegen_options", default=MappingProxyType({}))


class _Options:
    def get(self, key, default=None):
        return _OPTIONS.get().get(key, default)

    def __getitem__(self, key):
        return _OPTIONS.get()[key]


environ = _Options()


@contextmanager
def option_scope(options):
    """Freeze the caller's options for construction and compilation."""
    token = _OPTIONS.set(MappingProxyType(dict(options)))
    try:
        yield
    finally:
        _OPTIONS.reset(token)
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_closure(sources):
    pending = list(ENTRIES)
    selected = set()
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        require(name in sources, f"missing source: {name}")
        selected.add(name)
        path = Path(name)
        package = path.parent.parts
        for parent in path.parents:
            init = (parent / "__init__.py").as_posix()
            if init in sources and init not in selected:
                pending.append(init)
        for node in ast.walk(
            ast.parse(transform(name, sources[name]).decode(), filename=name)
        ):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            require(node.level <= len(package) + 1, f"external relative import: {name}")
            target = package[: len(package) + 1 - node.level]
            if node.module:
                target += tuple(node.module.split("."))
            for parts in [target] + [target + (alias.name,) for alias in node.names]:
                stem = Path(*parts)
                for candidate in (
                    stem.with_suffix(".py") if parts else None,
                    stem / "__init__.py",
                ):
                    if candidate is not None and candidate.as_posix() in sources:
                        pending.append(candidate.as_posix())
    return sorted(selected)


def replace_once(text, old, new, name):
    require(text.count(old) == 1, f"patch context changed: {name}")
    return text.replace(old, new)


def transform(name, raw):
    text = raw.decode()
    if name == "__init__.py":
        text = (
            "\n".join(text.splitlines()[:2])
            + """

from .kernel_src.blackwell.inference.mega.block_scaled_swap_ab_mega_moe_kernel import (
    BlockScaledSwapAbMegaMoeKernel as BlackwellInferenceMegaMoE,
)

__all__ = ["BlackwellInferenceMegaMoE"]
"""
        )
    if name in ("helpers/fastquant.py", "helpers/fastquant_mx.py"):
        require(
            text.count("\n_COMPILED = {}") == 1,
            f"quantizer host marker changed: {name}",
        )
        text = text[: text.index("\n_COMPILED = {}")].rstrip() + "\n"
    if name == "helpers/fastquant_mx.py":
        old = (
            'SUBNORMAL_RCP = int(os.environ.get("MEGAMOE_FASTQUANT_MX_SUBNORMAL", "1"))'
        )
        text = replace_once(text, old, "SUBNORMAL_RCP = 1", name)
    if name == "helpers/prod_config.py":
        node = next(
            node
            for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef) and node.name == "defaults_active"
        )
        lines = text.splitlines(keepends=True)
        lines[node.lineno - 1 : node.end_lineno] = [
            "def defaults_active() -> bool:\n",
            '    """The adapter admits SM100/SM103 only."""\n',
            "    return True\n",
        ]
        text = "".join(lines)
    if name == "communication/nvlink_domain/token_comm.py":
        old = """                task = Int32(pair_round * block_thread_count) + self._router_thread_idx
                if task < Int32(pair_count):
                    local_expert = task // Int32(world)
                    ring_position = task - local_expert * Int32(world)
"""
        new = """                task = Int32(pair_round * block_thread_count) + self._router_thread_idx
                local_expert = task // Int32(world)
                if task < Int32(pair_count):
                    ring_position = task - local_expert * Int32(world)
"""
        text = replace_once(text, old, new, name)
    if "os.environ" in text:
        text = text.replace("os.environ", "_kimi_options.environ")
        level = "." * (len(Path(name).parent.parts) + 1)
        line = f"from {level}helpers import codegen_options as _kimi_options\n"
        tree = ast.parse(text)
        insert = 0
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                insert = node.end_lineno
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                insert = node.end_lineno
                continue
            break
        lines = text.splitlines(keepends=True)
        lines.insert(insert, line)
        text = "".join(lines)
    lines = text.splitlines(keepends=True)
    for node in reversed(ast.parse(text).body):
        if isinstance(node, ast.ImportFrom) and node.module == "cutlass.cute.nvgpu":
            operands = [item for item in node.names if item.name == "OperandMajorMode"]
            if operands:
                require(
                    len(operands) == 1, f"unexpected OperandMajorMode import: {name}"
                )
                alias = " as " + operands[0].asname if operands[0].asname else ""
                replacement = [
                    f"from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode{alias}\n"
                ]
                rest = [
                    ast.unparse(item)
                    for item in node.names
                    if item.name != "OperandMajorMode"
                ]
                if rest:
                    replacement.append(
                        "from cutlass.cute.nvgpu import " + ", ".join(rest) + "\n"
                    )
                lines[node.lineno - 1 : node.end_lineno] = replacement
    text = "".join(lines)
    if text.encode() != raw:
        text = HEADER + text
    return text.encode()


def tactic_source(raw):
    text = raw.decode()
    lines = text.splitlines(keepends=True)
    selected = []
    for node in ast.parse(text).body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if name.startswith("_TAC_") or name in (
            "PROD1_TOKEN_BUCKETS",
            "PROD1_DEFAULT_TACTICS",
        ):
            selected.append("".join(lines[node.lineno - 1 : node.end_lineno]))
    require(len(selected) > 2, "tactic extraction failed")
    return (
        HEADER
        + "# SPDX-License-Identifier: Apache-2.0\nfrom typing import Tuple\n\n"
        + "\n".join(selected)
    ).encode()


def export(base, overlay, destination, check=False):
    manifest_bytes = (overlay / "install/base_manifest.json").read_bytes()
    require(
        digest(manifest_bytes) == KIMI_MANIFEST_SHA256,
        "Kimi install manifest does not match the pinned revision",
    )
    manifest = json.loads(manifest_bytes)
    require(manifest["trtllm_version"] == TRT_VERSION, "base version changed")
    source_map = {
        p.relative_to(base / KERNEL_ROOT).as_posix(): p.read_bytes()
        for p in (base / KERNEL_ROOT).rglob("*.py")
    }
    require(bool(source_map), "base kernel tree is empty")
    for rel, expected in manifest["replaced"].items():
        if rel.startswith(str(KERNEL_ROOT) + "/") or rel == str(HOST_SOURCE):
            require(
                digest((base / rel).read_bytes()) == expected,
                f"base hash mismatch: {rel}",
            )
    for rel, expected in manifest["overlay_sha256"].items():
        if rel.startswith(str(KERNEL_ROOT) + "/") or rel == str(HOST_SOURCE):
            payload = (overlay / "overlay" / rel).read_bytes()
            require(digest(payload) == expected, f"overlay hash mismatch: {rel}")
            if rel.startswith(str(KERNEL_ROOT) + "/"):
                source_map[Path(rel).relative_to(KERNEL_ROOT).as_posix()] = payload
    chosen = source_closure(source_map)
    outputs = {name: transform(name, source_map[name]) for name in chosen}
    outputs["helpers/codegen_options.py"] = OPTIONS_SOURCE.encode()
    outputs["tactics.py"] = tactic_source(
        (overlay / "overlay" / HOST_SOURCE).read_bytes()
    )
    record = {
        "kimi_commit": KIMI_COMMIT,
        "kimi_manifest_sha256": KIMI_MANIFEST_SHA256,
        "base_version": TRT_VERSION,
        "base_container_digest": CONTAINER_DIGEST,
        "exporter_sha256": digest(Path(__file__).read_bytes()),
        "input_sha256": {name: digest(source_map[name]) for name in chosen},
        "host_overlay_sha256": digest((overlay / "overlay" / HOST_SOURCE).read_bytes()),
        "files": {name: digest(data) for name, data in sorted(outputs.items())},
    }
    root = destination / "src"
    if check:
        require(
            set(p.relative_to(root).as_posix() for p in root.rglob("*.py"))
            == set(outputs),
            "export file set differs",
        )
        for name, data in outputs.items():
            require((root / name).read_bytes() == data, f"export differs: {name}")
        require(
            json.loads((destination / "export_manifest.json").read_text()) == record,
            "export manifest differs",
        )
    else:
        for path in root.rglob("*.py"):
            if path.relative_to(root).as_posix() not in outputs:
                path.unlink()
        for name, data in outputs.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (destination / "export_manifest.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
    print(f"{'Checked' if check else 'Exported'} {len(outputs)} Kimi source files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="directory containing the pinned tensorrt_llm package",
    )
    parser.add_argument(
        "--overlay", type=Path, required=True, help="pinned Kimi repository"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "flashinfer/moe_ep/kernel_src/sm100/kimi-k3-megamoe",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    export(args.base, args.overlay, args.output, args.check)


if __name__ == "__main__":
    main()
