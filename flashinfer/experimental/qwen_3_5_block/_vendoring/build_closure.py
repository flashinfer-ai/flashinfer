#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute the serving block's module closure from the import traces + a static (AST) module-level closure check.

    build_closure.py --fe <checkout> --traces <dir with host_declare_cpu.json / node_exec_*.json> --out closure.json [--tables closure_tables.md]

Closure = (modules the traces loaded on the serving route) + (modules whose FILE the route opened by path: kernel templates)
+ EXTRA (module-level dependencies of those that a run never exercised but an import would need) - EXCLUDE (the training
half, the cuDNN graph / backend binding, routes the serving forms never take).  Then, for every file of the closure, every
MODULE-LEVEL ``cudnn`` import must resolve inside the closure (else it is reported as a hole), and every function-level
``cudnn`` import is classified as inside / outside (outside = an excluded path, listed with the function it sits in).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--fe", required=True)
ap.add_argument("--traces", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tables", default=None)
ap.add_argument("--commit", default="977757763")
args = ap.parse_args()
FE = Path(args.fe)
PKG = FE / "python" / "cudnn"

# --------------------------------------------------------------------------- traces
traced: dict[str, dict] = {}
opened: set[str] = set()
native: dict[str, set] = {}
runs = {}
for p in sorted(Path(args.traces).glob("*.json")):
    if p.name == "nanobind_enums.json" or p.name.startswith("closure"):
        continue
    with open(p) as f:
        d = json.load(f)
    runs[p.stem] = {
        "n_cudnn_modules": len(d.get("cudnn_modules", {})),
        "facts": d.get("facts"),
        "forms": {k: v.get("stages", {}) for k, v in d.get("forms", {}).items()},
    }
    for k, v in d.get("cudnn_modules", {}).items():
        e = traced.setdefault(
            k, {"first_importers": set(), "stages": set(), "runs": set()}
        )
        e["first_importers"].add(v.get("first_importer"))
        e["stages"].add(v.get("stage"))
        e["runs"].add(p.stem)
    for f, _st in d.get("opened_files", []):
        opened.add(f.split("/python/cudnn/")[-1])
    for attr, caller, stage in d.get("native_attrs", []):
        native.setdefault(attr, set()).add(f"{caller} @ {stage}")


def mod_to_rel(mod: str) -> str | None:
    """'cudnn.a.b' -> 'a/b.py' or 'a/b/__init__.py' (relative to python/cudnn)."""
    if mod == "cudnn":
        return "__init__.py"
    rel = mod[len("cudnn.") :].replace(".", "/")
    if (PKG / (rel + ".py")).is_file():
        return rel + ".py"
    if (PKG / rel / "__init__.py").is_file():
        return rel + "/__init__.py"
    return None


def rel_to_mod(rel: str) -> str:
    rel = rel[:-3] if rel.endswith(".py") else rel
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")]
    return "cudnn" if rel == "__init__" else "cudnn." + rel.replace("/", ".")


# --------------------------------------------------------------------------- policy
EXCLUDE = {
    "cudnn": "the package root: loads the nanobind extension; the vendored root is written fresh (shim)",
    "cudnn._compiled_module": "the nanobind extension (.so): NOT vendored; its Python-visible surface is shimmed",
    "cudnn.graph": "the classic graph API facade (graph / jit / graph_cache): the block never uses it",
    "cudnn.gated_attention_block.api_bwd": "the training half (backward); dropped from the vendored package __init__ (patch P0)",
    "cudnn.gated_attention_block.kernels.qsa_kv_upcast": "the e4m3 KV-cache arm of the block-sparse decode mode (the gathered cast + the dense decode tile over a caller list): lazily imported by gated_attention_block/api.py under kv_cache_dtype=torch.float8_e4m3fn, which the Qwen 3.8 decode form declines by name until the sparse core's in-kernel dequant arm lands",
}
EXTRA = {
    # the template FILES the route executes by path are modules of the tree too -- vendored as files:
    "cudnn.gemm.frost.epilogue_codegen": "the facade twin of gemm.frost.compiler (22 lines); kept so the facade pair stays intact",
    "cudnn.gemm.frost.kernel_templates.dynamic_scheduler_counter_initialization": "17 lines; the kernel_templates package's second shared template (imported by rendered sources of other pipelines; kept so the package is whole)",
    # the separate launches the block's own declaration rules select off the fused arms (lazily imported by gated_attention_block/api.py):
    "cudnn.gated_attention_block.kernels.elementwise": "the sigmoid-gate launch (_SigmoidGate): what the DECODE form runs when the decode tile runs unsplit (decode_split_kv=1), where the gate cannot ride the split combine",
    "cudnn.gated_attention_block.kernels.cache_write": "the separate paged write-through launch (_CacheWrite): what runs when the write-through is folded neither into the fused projection's epilogue nor into the norm + RoPE launch (fuse_cache_write=False, or 128+ rows with q_len > 1 on the lowered decode chain); also a module-level dependency of the TMA norm + RoPE kernel's pool-write arm",
    "cudnn.gated_attention_block.kernels.qk_norm_rope": "the shared helpers of the two launches above (ACCESS_BYTES, lanes_per_row, vec_chunks, ...) and the ldg norm + RoPE arm of the unfused chain",
}
TEST_HELPERS = {
    "test/python/gated_attention_block/cutedsl/gated_block_reference.py": "the fp32 oracle (gated_attention_block_reference), build_rope_tables, make_inputs, RefGeometry: what a FlashInfer correctness test compares against (the quantized / backward refs inside import test-tree siblings lazily and are not used)",
}
# vendor.py's EXTRA_FILES -- files outside python/cudnn that ARE vendored (next to the closure): pinned here by the digest of their
# blob at the commit, like every closure file, so a re-vendor at a commit where the blob moved STOPS (the exporter compares before it
# copies).  The Python SDPA binder builds on this file's private frame helpers; a moved blob is re-checked against it first.
EXTRA_FILES = {
    "test/python/sdpa/frost/sdpa_binding_reference.py": "the retired Python frame implementation of the SDPA launch binder (bind_dense / bind_dense_split / bind_thd, the paged pools and tables, the split combine): the frame logic the FlashInfer-side Python binder delegates to, vendored as sdpa/fwd/_binding_reference.py",
}
GROUPS = [
    ("gated_attention_block", "gated_attention_block/"),
    ("sdpa.fwd.kernels", "sdpa/fwd/kernels/"),
    ("sdpa.fwd", "sdpa/fwd/"),
    ("sdpa", "sdpa/"),
    ("gemm.frost.sm100", "gemm/frost/sm100/"),
    ("gemm.frost", "gemm/frost/"),
    ("gemm", "gemm/"),
    ("frost.tile_dsl", "frost/tile_dsl/"),
    ("frost", "frost/"),
    ("engines", "engines/"),
    ("root", ""),
]


def group_of(rel: str) -> str:
    for g, prefix in GROUPS:
        if prefix and rel.startswith(prefix):
            return g
    return "root"


closure_mods = {m for m in traced if m not in EXCLUDE}
closure_mods |= {rel_to_mod(f) for f in opened if f.endswith(".py")}
closure_mods |= set(EXTRA)
closure_files = {}
for m in sorted(closure_mods):
    rel = mod_to_rel(m)
    if rel is None:
        print(f"WARN: module {m} has no file (a template module?)", file=sys.stderr)
        continue
    closure_files[rel] = m
# every package __init__ on the path of a closure file is needed for the import to resolve
for rel in list(closure_files):
    parts = rel.split("/")[:-1]
    for i in range(1, len(parts) + 1):
        init = "/".join(parts[:i]) + "/__init__.py"
        if (PKG / init).is_file() and init not in closure_files:
            closure_files[init] = rel_to_mod(init)
            EXTRA.setdefault(
                rel_to_mod(init),
                "package __init__ on the import path of a closure module",
            )

# --------------------------------------------------------------------------- static analysis


def imports_of(path: Path, this_mod: str):
    """(module-level, function-level) lists of absolute cudnn module names imported by ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    pkg = this_mod if path.name == "__init__.py" else this_mod.rsplit(".", 1)[0]
    top: list[tuple[str, str]] = []
    lazy: list[tuple[str, str]] = []

    def resolve(node):
        out = []
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "cudnn" or a.name.startswith("cudnn."):
                    out.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = pkg.split(".")
                parts = (
                    parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                )
                base = ".".join(parts) + ("." + node.module if node.module else "")
                for a in node.names:
                    cand = base + "." + a.name
                    out.append(cand if mod_to_rel(cand) else base)
            elif node.module and (
                node.module == "cudnn" or node.module.startswith("cudnn.")
            ):
                for a in node.names:
                    cand = node.module + "." + a.name
                    out.append(cand if mod_to_rel(cand) else node.module)
        return out

    class V(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0
            self.fn = []

        def visit_FunctionDef(self, node):
            self.fn.append(node.name)
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1
            self.fn.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_If(self, node):  # `if TYPE_CHECKING:` imports are not runtime imports
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                return
            self.generic_visit(node)

        def visit_Import(self, node):
            for m in resolve(node):
                (lazy if self.depth else top).append((m, ".".join(self.fn)))

        visit_ImportFrom = visit_Import

    V().visit(tree)
    return top, lazy


def spdx(path: Path):
    head = path.read_text().split("\n")[:6]
    lic = next(
        (
            ln.split("SPDX-License-Identifier:")[1].strip()
            for ln in head
            if "SPDX-License-Identifier:" in ln
        ),
        None,
    )
    cpr = next((ln for ln in head if "Copyright" in ln), None)
    return lic, cpr


rows = []
holes = []
lazy_out = []
for rel in sorted(closure_files):
    p = PKG / rel
    mod = closure_files[rel]
    text = p.read_bytes()
    top, lazy = imports_of(p, mod)
    top_mods = sorted({m for m, _ in top})
    lazy_mods = sorted({(m, fn) for m, fn in lazy})
    for m in top_mods:
        if m not in closure_mods and m != "cudnn":
            holes.append((rel, m))
    for m, fn in lazy_mods:
        if m not in closure_mods and m != "cudnn":
            lazy_out.append((rel, m, fn))
    lic, cpr = spdx(p)
    rows.append(
        {
            "path": "python/cudnn/" + rel,
            "module": mod,
            "group": group_of(rel),
            "bytes": len(text),
            "lines": text.count(b"\n"),
            "sha256": hashlib.sha256(text).hexdigest(),
            "license": lic,
            "copyright": cpr,
            "traced": mod in traced,
            "opened_by_path": rel in opened,
            "why": EXTRA.get(
                mod,
                "traced on the serving route"
                if mod in traced
                else "template file executed by path",
            ),
            "top_level_cudnn_imports": [m for m in top_mods if m != mod],
            "lazy_cudnn_imports": [f"{m} (in {fn})" for m, fn in lazy_mods],
            "lazy_outside_closure": [
                f"{m} (in {fn})"
                for m, fn in lazy_mods
                if m not in closure_mods and m != "cudnn"
            ],
            "root_cudnn_imports": sorted({m for m, _ in top + lazy if m == "cudnn"}),
        }
    )

# data files: non-.py files under the closure's directories, plus the template .py files executed by path
dirs = sorted({str(Path(r["path"]).parent) for r in rows})
data_files = []
for d in dirs:
    for p in sorted((FE / d).iterdir()):
        if (
            p.is_file()
            and p.suffix not in (".py", ".pyc")
            and not p.name.startswith(".")
            and p.suffix
            in (".json", ".txt", ".csv", ".yaml", ".yml", ".toml", ".cubin", ".ptx")
        ):
            data_files.append(
                {"path": str(p.relative_to(FE)), "bytes": p.stat().st_size}
            )
templates = [r for r in rows if r["opened_by_path"]]

# the test helper
helpers = []
for rel, why in TEST_HELPERS.items():
    p = FE / rel
    t = p.read_bytes()
    lic, cpr = spdx(p)
    helpers.append(
        {
            "path": rel,
            "bytes": len(t),
            "lines": t.count(b"\n"),
            "sha256": hashlib.sha256(t).hexdigest(),
            "license": lic,
            "why": why,
        }
    )

# the files vendored from outside python/cudnn (vendor.py's EXTRA_FILES): the same digest pin as a closure file
extras = []
for rel, why in EXTRA_FILES.items():
    p = FE / rel
    t = p.read_bytes()
    lic, cpr = spdx(p)
    extras.append(
        {
            "path": rel,
            "bytes": len(t),
            "lines": t.count(b"\n"),
            "sha256": hashlib.sha256(t).hexdigest(),
            "license": lic,
            "why": why,
        }
    )

# git identity: every closure file must be identical to the commit's blob
commit = args.commit
git_mismatch = []
for r in rows + helpers + extras:
    blob = subprocess.run(
        ["git", "-C", str(FE), "show", f"{commit}:{r['path']}"], capture_output=True
    ).stdout
    if hashlib.sha256(blob).hexdigest() != r["sha256"]:
        git_mismatch.append(r["path"])

total_bytes = sum(r["bytes"] for r in rows)
total_lines = sum(r["lines"] for r in rows)
by_group: dict[str, dict[str, int]] = {}
for r in rows:
    g = by_group.setdefault(r["group"], {"files": 0, "bytes": 0, "lines": 0})
    g["files"] += 1
    g["bytes"] += r["bytes"]
    g["lines"] += r["lines"]
lic_count: dict[str | None, int] = {}
for r in rows:
    lic_count[r["license"]] = lic_count.get(r["license"], 0) + 1

result = {
    "commit": commit,
    "fe": "cudnn_frontend",  # the checkout label (the path is a local fact, not a closure fact)
    "runs": runs,
    "n_files": len(rows),
    "total_bytes": total_bytes,
    "total_lines": total_lines,
    "by_group": by_group,
    "licenses": lic_count,
    "files": rows,
    "templates_executed_by_path": [r["path"] for r in templates],
    "data_files": data_files,
    "test_helpers": helpers,
    "extra_files": extras,
    "excluded": EXCLUDE,
    "holes_module_level": holes,
    "lazy_imports_outside_closure": lazy_out,
    "native_attrs": {k: sorted(v) for k, v in native.items()},
    "git_blob_mismatch": git_mismatch,
}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(result, f, indent=1)
    f.write("\n")
print(
    f"closure: {len(rows)} files, {total_bytes} bytes, {total_lines} lines; groups {by_group}; licenses {lic_count}; holes {holes}; git mismatch {git_mismatch}"
)
print(f"lazy imports outside the closure: {len(lazy_out)}")
for rel, m, fn in lazy_out:
    print(f"   {rel}: {m} (in {fn})")

if args.tables:
    L = []
    L.append("| group | files | bytes | lines |\n|---|---:|---:|---:|")
    for gname, v in sorted(by_group.items(), key=lambda kv: -kv[1]["bytes"]):
        L.append(f"| {gname} | {v['files']} | {v['bytes']:,} | {v['lines']:,} |")
    L.append(
        f"| **total** | **{len(rows)}** | **{total_bytes:,}** | **{total_lines:,}** |"
    )
    L.append("")
    cur = None
    for r in sorted(
        rows,
        key=lambda r: (
            GROUPS.index(next(x for x in GROUPS if x[0] == r["group"])),
            r["path"],
        ),
    ):
        if r["group"] != cur:
            cur = r["group"]
            L.append(f"\n#### {cur}\n")
            L.append(
                "| file | bytes | lines | license | module-level `cudnn` imports (cross-package) | lazy `cudnn` imports outside the closure |\n|---|---:|---:|---|---|---|"
            )
        own = r["group"]
        cross = (
            [
                m
                for m in r["top_level_cudnn_imports"]
                if group_of(mod_to_rel(m) or "") != own
            ]
            if r["top_level_cudnn_imports"]
            else []
        )
        L.append(
            f"| `{r['path'][len('python/cudnn/') :]}`{' (template, executed by path)' if r['opened_by_path'] else ''} | {r['bytes']:,} | {r['lines']:,} | {r['license']} | "
            + (", ".join(f"`{m[len('cudnn.') :]}`" for m in cross) if cross else "-")
            + " | "
            + (
                ", ".join(f"`{x}`" for x in r["lazy_outside_closure"])
                if r["lazy_outside_closure"]
                else "-"
            )
            + " |"
        )
    Path(args.tables).write_text("\n".join(L) + "\n")
    print("tables ->", args.tables)
