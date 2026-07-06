# flashinfer: signature-bind smoke tests for the wan example's Ulysses
# consumers — every UlyssesContext(...) call site in the example scripts must
# bind against the adapter's real signature, so an API migration cannot leave
# an advertised entry point passing removed kwargs (as elem_bytes= once did).
# Pure CPU: the scripts are AST-scanned, never imported (no models, no GPUs).

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

_WAN_DIR = Path(__file__).resolve().parents[2] / "examples" / "pytorch" / "wan"
_SCRIPTS = sorted(p for p in _WAN_DIR.glob("*.py") if p.name != "ulysses.py")


def _ulysses_context_signature():
    spec = importlib.util.spec_from_file_location(
        "wan_example_ulysses", _WAN_DIR / "ulysses.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return inspect.signature(mod.UlyssesContext.__init__), mod


def _calls(script: Path, callee: str):
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == callee
        ):
            yield node


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_example_ulysses_context_calls_bind(script):
    sig, _mod = _ulysses_context_signature()
    calls = list(_calls(script, "UlyssesContext"))
    if not calls:
        pytest.skip(f"{script.name} has no UlyssesContext call")
    for call in calls:
        if any(kw.arg is None for kw in call.keywords):
            continue  # **kwargs splat: not statically checkable
        # bind with positional placeholders + the literal keyword names
        args = ["self"] + [object()] * len(call.args)
        kwargs = {kw.arg: object() for kw in call.keywords}
        try:
            sig.bind(*args, **kwargs)
        except TypeError as e:
            raise AssertionError(
                f"{script.name}:{call.lineno} UlyssesContext(...) does not "
                f"bind against the adapter signature: {e}"
            ) from e


def test_scan_found_the_known_consumers():
    consumers = {s.name for s in _SCRIPTS if list(_calls(s, "UlyssesContext"))}
    assert {"bench_wan_ulysses.py", "run_wan_ulysses_video.py"} <= consumers, consumers
