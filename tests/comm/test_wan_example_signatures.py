# flashinfer: signature-bind smoke tests for the wan example's Ulysses
# consumers — every UlyssesCommunicator(...) call site in the example scripts
# must bind against the real public constructor signature, so an API change
# cannot leave an advertised entry point passing removed kwargs (as the old
# adapter's elem_bytes= once did). Pure CPU: the scripts are AST-scanned,
# never imported (no models, no GPUs).

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from flashinfer.comm import UlyssesCommunicator

_WAN_DIR = Path(__file__).resolve().parents[2] / "examples" / "pytorch" / "wan"
_SCRIPTS = sorted(_WAN_DIR.glob("*.py"))


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
def test_example_communicator_calls_bind(script):
    sig = inspect.signature(UlyssesCommunicator.__init__)
    calls = list(_calls(script, "UlyssesCommunicator"))
    if not calls:
        pytest.skip(f"{script.name} has no UlyssesCommunicator call")
    for call in calls:
        if any(kw.arg is None for kw in call.keywords):
            continue  # **kwargs splat: not statically checkable
        args = ["self"] + [object()] * len(call.args)
        kwargs = {kw.arg: object() for kw in call.keywords}
        try:
            sig.bind(*args, **kwargs)
        except TypeError as e:
            raise AssertionError(
                f"{script.name}:{call.lineno} UlyssesCommunicator(...) does not "
                f"bind against the public constructor signature: {e}"
            ) from e


def test_scan_found_the_known_consumers():
    consumers = {s.name for s in _SCRIPTS if list(_calls(s, "UlyssesCommunicator"))}
    assert {"bench_wan_ulysses.py", "run_wan_ulysses_video.py"} <= consumers, consumers


def test_no_stale_adapter_references():
    # the UlyssesContext adapter was removed; nothing may quietly resurrect it
    for script in _SCRIPTS:
        text = script.read_text()
        assert "UlyssesContext" not in text, script.name
        assert "from ulysses import" not in text, script.name


def _video_module():
    spec = importlib.util.spec_from_file_location(
        "wan_example_video", _WAN_DIR / "run_wan_ulysses_video.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # CPU-only: torch/diffusers import lazily
    return mod


def test_video_preflight_geometry():
    mod = _video_module()
    seq = 32760  # default 480x832x81f token count: divisible by 6 AND 8
    # W=8 divides 40 heads: accepted
    mod.validate_ulysses_video_config(8, seq, device_count=8)
    # W=6 is a fused-kernel size and divides the token count, but 40 % 6 != 0:
    # must be rejected BEFORE communicator construction / model load
    with pytest.raises(ValueError, match="40 heads"):
        mod.validate_ulysses_video_config(6, seq, device_count=8)
    with pytest.raises(ValueError, match="positive"):
        mod.validate_ulysses_video_config(0, seq, device_count=8)
    with pytest.raises(ValueError, match="visible GPU count"):
        mod.validate_ulysses_video_config(8, seq, device_count=4)
    with pytest.raises(ValueError, match="not divisible"):
        mod.validate_ulysses_video_config(8, seq + 1, device_count=8)


def test_video_preflight_rejects_nonpositive_tokens():
    mod = _video_module()
    with pytest.raises(ValueError, match="token count must be positive"):
        mod.validate_ulysses_video_config(8, 0, device_count=8)


def test_video_cli_world_size_zero_exits_nonzero():
    # placement regression: --world-size 0 used to spawn range(0) workers and
    # exit 0 silently; the CLI preflight must fail BEFORE any worker exists
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(_WAN_DIR / "run_wan_ulysses_video.py"),
            "--world-size",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode != 0, r.stdout + r.stderr
    assert "must be positive" in (r.stdout + r.stderr)
    # the failure comes from main()'s preflight, not from a spawned worker
    assert "init_process_group" not in (r.stdout + r.stderr)
