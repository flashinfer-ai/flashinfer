#!/usr/bin/env python3
"""Build minimal, workload-specific FlashInfer jit-cache wheels.

The official jit-cache ships one provider wheel per CUDA architecture holding
every ahead-of-time module. This tool builds provider wheels that hold only the
modules a deployment needs, described by a *recipe*:

    {
      "name": "llama3-h100",
      "cuda_archs": ["9.0a"],
      "usage_logs": ["usage.jsonl"],
      "modules": ["norm", "sampling"],
      "include": ["batch_prefill_with_kv_cache_dtype_q_bf16_*head_dim_qk_128_*"],
      "exclude": ["*_use_logits_cap_True*"],
      "aot_config": {"add_comm": false}
    }

Typical flow (see docs/custom_jit_cache.rst):

    # 1. Record the modules a representative workload loads.
    python scripts/custom_jit_cache.py capture --log usage.jsonl -- python serve.py
    # 2. Turn the log into an editable recipe.
    python scripts/custom_jit_cache.py init --name my-model --from-usage usage.jsonl
    # 3. Check what each architecture's provider will contain.
    python scripts/custom_jit_cache.py resolve my-model.json
    # 4. Build the provider wheel(s) and the matching shim.
    python scripts/custom_jit_cache.py build my-model.json

Commands that enumerate modules (``list``, ``resolve``, ``build``) need a
FlashInfer source checkout with an editable install and the CUDA toolkit; they
do not need a GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVIDER_DIR = PROJECT_ROOT / "flashinfer-jit-cache-provider"
SHIM_DIR = PROJECT_ROOT / "flashinfer-jit-cache"
USAGE_LOG_ENV = "FLASHINFER_JIT_USAGE_LOG"
SELECTION_ENV = "FLASHINFER_JIT_CACHE_PROVIDER_SELECTION"
RECIPE_KEYS = {
    "name",
    "cuda_archs",
    "modules",
    "include",
    "exclude",
    "usage_logs",
    "aot_config",
}


def _load_package_config():
    # Only the stdlib-only arch helpers are needed; avoid importing flashinfer.
    spec = importlib.util.spec_from_file_location(
        "_custom_jit_cache_package_config", PROVIDER_DIR / "package_config.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


normalize_cuda_architecture = _load_package_config().normalize_cuda_architecture


# --------------------------------------------------------------------------
# Usage logs and recipes
# --------------------------------------------------------------------------


def read_usage_logs(paths: list[Path]) -> dict[str, Any]:
    """Summarize ``FLASHINFER_JIT_USAGE_LOG`` files.

    Returns the nvcc modules (packageable in a jit-cache), the modules built by
    other toolchains (always compiled at runtime), the CUDA architectures and
    the FlashInfer versions seen while recording.
    """
    nvcc: set[str] = set()
    runtime_only: dict[str, str] = {}
    archs: set[str] = set()
    versions: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if record.get("toolchain") == "nvcc":
                nvcc.add(record["name"])
            else:
                runtime_only[record["name"]] = record.get("toolchain", "unknown")
            archs.update(record.get("cuda_archs") or ())
            if record.get("flashinfer_version"):
                versions.add(record["flashinfer_version"])
    return {
        "modules": sorted(nvcc),
        "runtime_only": dict(sorted(runtime_only.items())),
        "cuda_archs": sorted(archs),
        "flashinfer_versions": sorted(versions),
    }


def load_recipe(path: Path) -> dict[str, Any]:
    if path.suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:
            sys.exit("TOML recipes need Python 3.11+; use a JSON recipe instead")
        recipe = tomllib.loads(path.read_text())
    else:
        recipe = json.loads(path.read_text())
    unknown = set(recipe) - RECIPE_KEYS
    if unknown:
        sys.exit(f"{path}: unknown recipe keys {sorted(unknown)}")
    name = recipe.get("name", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        sys.exit(f"{path}: recipe needs a 'name' of letters, digits, '.', '_', '-'")
    # Usage-log paths are relative to the recipe file.
    recipe["usage_logs"] = [
        str((path.parent / log).resolve()) for log in recipe.get("usage_logs", [])
    ]
    return recipe


def recipe_archs(recipe: dict[str, Any], override: list[str] | None) -> list[str]:
    archs = override or recipe.get("cuda_archs") or []
    if not archs:
        sys.exit("No CUDA architecture: set 'cuda_archs' in the recipe or pass --arch")
    return sorted(
        {normalize_cuda_architecture(arch)[0] for arch in archs},
        key=lambda arch: (int(arch.split(".")[0]), arch),
    )


def recipe_selection(recipe: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    """Expand usage logs into a ``flashinfer.aot.ModuleSelection`` mapping."""
    usage = read_usage_logs([Path(p) for p in recipe["usage_logs"]])
    modules = sorted(set(recipe.get("modules", [])) | set(usage["modules"]))
    selection = {
        "name": recipe["name"],
        "modules": modules,
        "include": list(recipe.get("include", [])),
        "exclude": list(recipe.get("exclude", [])),
        "aot_config": dict(recipe.get("aot_config", {})),
    }
    if not (selection["modules"] or selection["include"]):
        sys.exit(
            f"Recipe {recipe['name']!r} selects nothing: add 'modules', "
            "'include' patterns or 'usage_logs'"
        )
    return selection, usage


def _source_version() -> str:
    version_file = PROJECT_ROOT / "version.txt"
    return version_file.read_text().strip() if version_file.exists() else "unknown"


def _warn_version_mismatch(usage: dict[str, Any]) -> None:
    source_version = _source_version()
    for version in usage["flashinfer_versions"]:
        base = version.split("+")[0]
        # 0.0.0 is a checkout without generated build metadata.
        if base != "0.0.0" and not base.startswith(source_version):
            print(
                f"WARNING: usage log recorded with FlashInfer {version}, but this "
                f"checkout is {source_version}. Module names can change between "
                "releases; re-capture with the version you build.",
                file=sys.stderr,
            )


# --------------------------------------------------------------------------
# Per-architecture worker (imports flashinfer with one target architecture)
# --------------------------------------------------------------------------


def _worker(arch: str, selection: dict[str, Any] | None, aot_config: dict) -> dict:
    nvcc_arch, provider_tag = normalize_cuda_architecture(arch)
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = nvcc_arch
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    from flashinfer import aot

    config = aot.get_default_config()
    config.update(aot.parse_aot_config_overrides(aot_config))
    # Share the provider build's workspace so its generated sources are reused.
    aot.configure_aot_build_env(
        PROJECT_ROOT, PROJECT_ROOT / "build" / "aot-providers" / provider_tag
    )
    specs = aot.gen_modules_from_config(config, aot.detect_sm_capabilities())
    result: dict[str, Any] = {"arch": nvcc_arch, "available": len(specs)}
    if selection is None:
        result["modules"] = [spec.name for spec in specs]
        return result
    selected = aot.select_modules(specs, aot.ModuleSelection.from_dict(selection))
    result["modules"] = [spec.name for spec in selected.selected]
    result["missing_modules"] = selected.missing_modules
    result["unmatched_patterns"] = selected.unmatched_patterns
    return result


def run_worker(arch: str, selection: dict[str, Any] | None, aot_config: dict) -> dict:
    """Run `_worker` in a fresh interpreter so each arch gets its own context."""
    payload = json.dumps(
        {"arch": arch, "selection": selection, "aot_config": aot_config}
    )
    proc = subprocess.run(
        [sys.executable, __file__, "_worker"],
        input=payload,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"Enumerating modules for {arch} failed (exit {proc.returncode})")
    # The last stdout line is the result; generators may print before it.
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_capture(args: argparse.Namespace) -> int:
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        sys.exit("capture: give the workload command after '--'")
    log = Path(args.log).resolve()
    env = dict(os.environ, **{USAGE_LOG_ENV: str(log)})
    returncode = subprocess.call(command, env=env)
    if log.exists():
        usage = read_usage_logs([log])
        print(
            f"Recorded {len(usage['modules'])} jit-cache modules and "
            f"{len(usage['runtime_only'])} runtime-compiled modules to {log}",
            file=sys.stderr,
        )
    else:
        print(f"No FlashInfer modules were loaded; {log} not written", file=sys.stderr)
    return returncode


def cmd_init(args: argparse.Namespace) -> int:
    logs = [Path(p) for p in args.from_usage]
    usage = read_usage_logs(logs)
    _warn_version_mismatch(usage)
    archs = args.arch or [
        normalize_cuda_architecture(arch)[0] for arch in usage["cuda_archs"]
    ]
    out = Path(args.output or f"{args.name}.json")
    recipe = {
        "name": args.name,
        "cuda_archs": archs,
        "usage_logs": [
            os.path.relpath(log.resolve(), out.parent.resolve()) for log in logs
        ],
        "modules": [],
        "include": [],
        "exclude": [],
        "aot_config": {},
    }
    out.write_text(json.dumps(recipe, indent=2) + "\n")
    print(
        f"Wrote {out}: {len(usage['modules'])} modules from usage logs, archs {archs}"
    )
    if usage["runtime_only"]:
        print(
            f"{len(usage['runtime_only'])} modules use other toolchains "
            "(CuTe-DSL, ...) and are compiled at runtime, not shipped in a jit-cache"
        )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    aot_config = (
        load_recipe(Path(args.recipe)).get("aot_config", {}) if args.recipe else {}
    )
    result = run_worker(args.arch, None, aot_config)
    names = [
        name
        for name in result["modules"]
        if not args.pattern or any(re.search(p, name) for p in args.pattern)
    ]
    if args.json:
        print(json.dumps(names, indent=2))
    else:
        print("\n".join(names))
        print(
            f"# {len(names)} of {result['available']} modules for {result['arch']}",
            file=sys.stderr,
        )
    return 0


def _resolve(recipe: dict[str, Any], archs: list[str]) -> tuple[dict, dict, list[dict]]:
    selection, usage = recipe_selection(recipe)
    _warn_version_mismatch(usage)
    results = [run_worker(arch, selection, selection["aot_config"]) for arch in archs]
    return selection, usage, results


def _print_resolution(usage: dict, results: list[dict], verbose: bool) -> bool:
    ok = True
    for result in results:
        print(
            f"{result['arch']}: {len(result['modules'])} of {result['available']} "
            "modules selected"
        )
        if verbose:
            for name in result["modules"]:
                print(f"    {name}")
        if result["missing_modules"]:
            print(
                f"  {len(result['missing_modules'])} requested modules are not in "
                "this arch's AOT set (built for another arch, or JIT-only):"
            )
            for name in result["missing_modules"]:
                print(f"    {name}")
        for pattern in result["unmatched_patterns"]:
            print(f"  include pattern matched nothing: {pattern}")
        ok = ok and bool(result["modules"])
    if usage["runtime_only"]:
        print(
            f"{len(usage['runtime_only'])} recorded modules use other toolchains "
            "and always compile at runtime:"
        )
        for name, toolchain in usage["runtime_only"].items():
            print(f"    [{toolchain}] {name}")
    return ok


def cmd_resolve(args: argparse.Namespace) -> int:
    recipe = load_recipe(Path(args.recipe))
    archs = recipe_archs(recipe, args.arch)
    _, usage, results = _resolve(recipe, archs)
    ok = _print_resolution(usage, results, args.verbose)
    if args.strict and any(
        r["missing_modules"] or r["unmatched_patterns"] for r in results
    ):
        ok = False
    return 0 if ok else 1


def _clean_setuptools_outputs(package_dir: Path) -> None:
    # setuptools reuses build/lib, which would leak modules of an earlier,
    # larger build of the same arch into this wheel.
    shutil.rmtree(package_dir / "build", ignore_errors=True)
    for egg_info in package_dir.glob("*.egg-info"):
        shutil.rmtree(egg_info)


def _build_wheel(package_dir: Path, out_dir: Path, env: dict[str, str]) -> Path:
    _clean_setuptools_outputs(package_dir)
    before = set(out_dir.glob("*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            # The build requires pin an isolated environment's CUDA wheels;
            # like CI, build with what the current environment provides.
            "--skip-dependency-check",
            "--wheel",
            "--outdir",
            str(out_dir),
            str(package_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    built = sorted(set(out_dir.glob("*.whl")) - before, key=lambda p: p.stat().st_mtime)
    if not built:
        # Rebuilding an identical file name replaces it in place.
        built = sorted(out_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    return built[-1]


def _wheel_modules(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/manifest.json"):
                return json.loads(archive.read(name))["modules"]
    return []


def cmd_build(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("build") is None:
        sys.exit("The 'build' package is required: pip install build")
    recipe = load_recipe(Path(args.recipe))
    archs = recipe_archs(recipe, args.arch)
    out_dir = Path(
        args.out_dir or PROJECT_ROOT / "dist" / "custom-jit-cache" / recipe["name"]
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selection, usage, results = _resolve(recipe, archs)
    if not _print_resolution(usage, results, verbose=False):
        sys.exit("At least one architecture selects no modules; see above")
    selection_path = out_dir / f"{recipe['name']}.selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")

    env = dict(os.environ)
    env[SELECTION_ENV] = str(selection_path)
    if args.local_version is not None:
        env["FLASHINFER_LOCAL_VERSION"] = args.local_version

    wheels = []
    for arch in archs:
        print(f"==> Building {recipe['name']} provider for {arch}")
        wheels.append(
            _build_wheel(
                PROVIDER_DIR,
                out_dir,
                dict(env, FLASHINFER_JIT_CACHE_PROVIDER_ARCH=arch),
            )
        )
    if not args.no_shim:
        print("==> Building flashinfer-jit-cache shim")
        wheels.append(
            _build_wheel(
                SHIM_DIR,
                out_dir,
                dict(env, FLASHINFER_JIT_CACHE_PROVIDER_ARCHS=" ".join(archs)),
            )
        )

    report = {
        "recipe": recipe["name"],
        "cuda_archs": archs,
        "wheels": [
            {
                "file": wheel.name,
                "bytes": wheel.stat().st_size,
                "modules": len(_wheel_modules(wheel)),
            }
            for wheel in wheels
        ],
        "runtime_only_modules": usage["runtime_only"],
    }
    (out_dir / f"{recipe['name']}.report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"\nWheels in {out_dir}:")
    for entry in report["wheels"]:
        print(
            f"  {entry['file']}  {entry['bytes'] / 2**20:.1f} MiB"
            + (f"  ({entry['modules']} modules)" if entry["modules"] else "")
        )
    print("\nInstall with:\n  pip install " + " ".join(str(w) for w in wheels))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("capture", help="run a workload and record the modules it loads")
    p.add_argument("--log", required=True, help="JSONL usage log to append to")
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- workload command")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("init", help="write a recipe from usage logs")
    p.add_argument("--name", required=True)
    p.add_argument("--from-usage", nargs="+", required=True, metavar="LOG")
    p.add_argument("--arch", nargs="+", help="override the archs seen in the logs")
    p.add_argument("-o", "--output", help="recipe path (default: <name>.json)")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("list", help="list the AOT modules available for one arch")
    p.add_argument("--arch", required=True)
    p.add_argument("--pattern", nargs="*", help="regex filters on module names")
    p.add_argument("--recipe", help="apply this recipe's aot_config")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("resolve", help="show what a recipe selects per arch")
    p.add_argument("recipe")
    p.add_argument("--arch", nargs="+", help="override the recipe's cuda_archs")
    p.add_argument("-v", "--verbose", action="store_true", help="list selected modules")
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail when requested modules or patterns are not available",
    )
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("build", help="build the provider wheel(s) and shim")
    p.add_argument("recipe")
    p.add_argument("--arch", nargs="+", help="override the recipe's cuda_archs")
    p.add_argument("--out-dir", help="default: dist/custom-jit-cache/<name>")
    p.add_argument(
        "--local-version",
        help="PEP 440 local label, e.g. cu130 (default: $FLASHINFER_LOCAL_VERSION)",
    )
    p.add_argument("--no-shim", action="store_true", help="build provider wheels only")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("_worker")
    p.set_defaults(func=None)

    args = parser.parse_args(argv)
    if args.cmd == "_worker":
        request = json.loads(sys.stdin.read())
        result = _worker(request["arch"], request["selection"], request["aot_config"])
        print(json.dumps(result))
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
