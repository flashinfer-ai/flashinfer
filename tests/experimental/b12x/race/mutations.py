"""Check that the race tests catch b12x races that were really shipped.

Each mutation puts a fixed race back into a b12x module (by deleting the fix's
lines from the installed source), then runs the scenarios that should expose
it in a fresh interpreter. A mutation the harness fails to catch means the
perturbation or scenario coverage is too weak for that class of bug.

    python -m tests.experimental.b12x.race.mutations              # all
    python -m tests.experimental.b12x.race.mutations dense_gemm_stage_fence

The mutated module is loaded before anything imports b12x, here and in b12x's
spawned compile workers (which re-import this module as their main module).
``B12X_RACE_MUTATION`` names the active mutation; like every non-operational
``B12X_`` variable it is part of the compile cache key, so mutated kernels
never share cache entries with real ones. Set ``B12X_RACE_FUZZ_DELAY`` too for
mutations listed with ``delay``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ENV = "B12X_RACE_MUTATION"

from .scenarios import MUTATIONS  # noqa: E402  (family modules must not import b12x at module level)


def _replace_block(source: str, old: str, new: str, where: str) -> str:
    """Replace the one run of lines matching ``old`` (compared without indentation)
    with ``new``, re-indented like the first matched line."""
    want = [line.strip() for line in old.strip("\n").splitlines()]
    lines = source.splitlines(keepends=True)
    hits = [
        i
        for i in range(len(lines) - len(want) + 1)
        if [line.strip() for line in lines[i : i + len(want)]] == want
    ]
    if len(hits) != 1:
        raise RuntimeError(f"{where}: expected one match for {want}, found {len(hits)}")
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    body = [indent + line.strip() + "\n" for line in new.strip("\n").splitlines()]
    return "".join(lines[:i] + body + lines[i + len(want) :])


def _mutate(name: str) -> None:
    spec = MUTATIONS[name]
    module_name = spec["module"]
    if module_name in sys.modules:
        raise RuntimeError(f"{module_name} imported before mutation {name}")
    found = importlib.util.find_spec(module_name)
    source = Path(found.origin).read_text()
    for old, new in spec["replace"]:
        source = _replace_block(source, old, new, f"mutation {name} in {found.origin}")
    # CuTe DSL re-reads kernel functions with inspect.getsource, so the mutated
    # module must come from a file of its own, not from text exec'd under the
    # original path.
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    path = Path(tempfile.gettempdir()) / f"b12x-race-mutation-{name}-{digest}.py"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(source)
        os.replace(tmp, path)
    parent, leaf = module_name.rsplit(".", 1)
    package = importlib.import_module(parent)
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(module_name, path)
    )
    sys.modules[module_name] = module
    module.__spec__.loader.exec_module(module)
    setattr(package, leaf, module)
    print(
        f"mutation {name} applied to {module_name} in pid {os.getpid()}",
        file=sys.stderr,
        flush=True,
    )


if os.environ.get(ENV):
    _mutate(os.environ[ENV])


def _child(name: str, replays: int) -> dict:
    import torch

    from ..conftest import require_b12x
    from . import scenarios
    from .noise import SMContention
    from .replay import ReplayCheck

    device = require_b12x()
    spec = MUTATIONS[name]
    contention = SMContention(device)
    results = []
    for case in spec["cases"]:
        scenario = scenarios.FAMILIES[spec["family"]](device, case)
        try:
            check = ReplayCheck(
                scenario.forward, scenario.outputs, reset=scenario.reset
            )
            with contention:
                report = check.run(replays=replays, contention=contention)
        finally:
            scenario.close()
        results.append(
            {
                "case": case,
                "description": scenario.description,
                "replays": report.replays,
                "bad": report.bad,
            }
        )
    return {
        "mutation": name,
        "device": torch.cuda.get_device_name(device),
        "results": results,
    }


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--child":
        print(json.dumps(_child(argv[1], int(argv[2]))))
        return 0
    names = [a for a in argv if not a.startswith("-")] or list(MUTATIONS)
    replays = int(os.environ.get("FLASHINFER_B12X_RACE_REPLAYS", "2000"))
    missed = []
    for name in names:
        env = dict(os.environ, **{ENV: name})
        if MUTATIONS[name]["delay"]:
            env.setdefault("B12X_RACE_FUZZ_DELAY", "1")
        out = subprocess.run(
            [sys.executable, "-m", __spec__.name, "--child", name, str(replays)],
            env=env,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[4],
        )
        lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
        applied = sum("applied to" in line for line in out.stderr.splitlines())
        print(f"{name}: mutation applied in {applied} processes")
        if out.returncode or not lines:
            print(f"{name}: child failed\n{out.stderr[-3000:]}")
            missed.append(name)
            continue
        result = json.loads(lines[-1])
        caught = any(r["bad"] for r in result["results"])
        for r in result["results"]:
            print(
                f"{name} on {result['device']}: {r['description']}: {r['bad']} of {r['replays']} replays differ"
            )
        print(f"{name}: {'caught' if caught else 'MISSED'}")
        if not caught:
            missed.append(name)
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
