"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import ast
import math
import pathlib

import cutlass.cute as cute
import pytest
import torch

from flashinfer.gdn_kernels import device_target as dt

GDN_KERNELS_DIR = pathlib.Path(dt.__file__).parent


class _FakeProps:
    def __init__(self, multi_processor_count: int) -> None:
        self.multi_processor_count = multi_processor_count


@pytest.fixture
def fake_devices(monkeypatch):
    """Three devices: two architectures, with cuda:0 and cuda:2 sharing one."""
    caps = {0: (9, 0), 1: (10, 0), 2: (9, 0)}
    sms = {0: 132, 1: 148, 2: 114}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: caps[i])
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: _FakeProps(sms[i])
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    # The real DSL arch comes from the host's device 0; neutralize it so these tests
    # exercise resolution alone. test_rejects_* below drive the guard directly.
    monkeypatch.setattr(dt, "_dsl_runtime_arch", lambda: None)
    dt._resolve.cache_clear()
    yield
    dt._resolve.cache_clear()


def test_device_target_reads_the_requested_device(fake_devices):
    d0 = dt.gdn_device_target("cuda:0")
    d1 = dt.gdn_device_target("cuda:1")

    assert (d0.arch, d0.num_sms, d0.use_packed_fma) == ("sm_90a", 132, False)
    assert (d1.arch, d1.num_sms, d1.use_packed_fma) == ("sm_100a", 148, True)


def test_compile_key_separates_same_arch_devices(fake_devices):
    """Cache entries hold device-resident default tensors, so same-arch devices
    must not share one -- the arch alone cannot tell them apart."""
    first, second = dt.gdn_device_target("cuda:0"), dt.gdn_device_target("cuda:2")

    assert first.arch == second.arch
    assert first.compile_key != second.compile_key


def test_index_less_device_follows_current_device(fake_devices, monkeypatch):
    assert dt.gdn_device_target("cuda").arch == "sm_90a"
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert dt.gdn_device_target("cuda").arch == "sm_100a"


def test_cute_dsl_arch_env_override_wins(fake_devices, monkeypatch):
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_90a")
    dt._resolve.cache_clear()

    target = dt.gdn_device_target("cuda:1")
    # The whole policy must follow the override, not just the arch string: packed
    # F32x2 codegen against an sm_90a target would not assemble.
    assert (target.arch, target.major, target.use_packed_fma) == ("sm_90a", 9, False)


def test_unparsable_cute_dsl_arch_is_rejected(fake_devices, monkeypatch):
    monkeypatch.setenv("CUTE_DSL_ARCH", "hopper")
    dt._resolve.cache_clear()
    with pytest.raises(ValueError, match="not a recognized arch"):
        dt.gdn_device_target("cuda:0")


def test_non_cuda_device_is_rejected(fake_devices):
    with pytest.raises(ValueError, match="require CUDA tensors"):
        dt.gdn_device_target("cpu")


def test_rejects_a_device_the_dsl_would_cross_compile_for(fake_devices, monkeypatch):
    """A mismatch here yields no JIT engine, so fail with the remedy instead."""
    monkeypatch.setattr(dt, "_dsl_runtime_arch", lambda: "sm_89")
    with pytest.raises(RuntimeError, match="CUTE_DSL_ARCH=sm_90a"):
        dt.gdn_device_target("cuda:0")


def test_accepts_a_device_matching_the_dsl_arch(fake_devices, monkeypatch):
    monkeypatch.setattr(dt, "_dsl_runtime_arch", lambda: "sm_90a")
    assert dt.gdn_device_target("cuda:0").arch == "sm_90a"


def test_compile_options_pin_arch_and_preserve_extras(fake_devices):
    extras = (cute.EnableTVMFFI(True), cute.OptLevel(3))
    options = dt.gdn_compile_options("cuda:1", *extras)

    assert isinstance(options[0], cute.GPUArch)
    assert options[0].value == "sm_100a"
    assert options[1:] == extras


def _is_cute_compile(func: ast.expr) -> bool:
    target = func.value if isinstance(func, ast.Subscript) else func
    if not isinstance(target, ast.Attribute) or target.attr != "compile":
        return False
    owner = target.value
    if isinstance(owner, ast.Name):
        return owner.id == "cute"
    return isinstance(owner, ast.Attribute) and owner.attr == "cute"


def _device_target_options(tree: ast.AST) -> set:
    """Names bound to a ``gdn_compile_options(...)`` result in this module."""
    return {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "gdn_compile_options"
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _pins_a_device_target(subscript: ast.expr, bound: set) -> bool:
    if isinstance(subscript, ast.Call):
        return (
            isinstance(subscript.func, ast.Name)
            and subscript.func.id == "gdn_compile_options"
        )
    return isinstance(subscript, ast.Name) and subscript.id in bound


# delta_rule_dsl/ and blackwell/gdn_cp_prefill.py compile through this shim, which
# takes already-pinned options from its callers. Those callers pin an arch of their
# own but key nothing by device; that is the rest of GDN-H3 (#4214).
OPTIONS_FROM_CALLER = ("custom_compile_cache.py",)


def test_every_gdn_cute_compile_pins_an_explicit_target():
    """Un-subscripted ``cute.compile`` targets whatever the DSL picks (device 0),
    and a subscript carrying no ``GPUArch`` is no better.

    A string ``options=`` kwarg is equally unsafe: the DSL replaces subscripted
    options wholesale when one is present, silently dropping the ``GPUArch``.
    """
    unpinned, string_options = [], []
    for path in sorted(GDN_KERNELS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        bound = _device_target_options(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_cute_compile(node.func):
                continue
            where = f"{path.name}:{node.lineno}"
            if path.name not in OPTIONS_FROM_CALLER and not (
                isinstance(node.func, ast.Subscript)
                and _pins_a_device_target(node.func.slice, bound)
            ):
                unpinned.append(where)
            if any(
                kw.arg == "options" and isinstance(kw.value, ast.Constant)
                for kw in node.keywords
            ):
                string_options.append(where)

    assert not unpinned and not string_options, (
        "call cute.compile[gdn_compile_options(device, ...)](...) instead; "
        f"unpinned={unpinned} string_options={string_options}"
    )


# Queries that answer for the ambient device instead of the operand's. Dispatch
# makes the operand's device current today, so a bare call agrees by luck until
# something invokes the kernel from anywhere else.
AMBIENT_DEVICE_READS = (
    "current_device",
    "current_stream",
    "gdn_compile_options",
    "gdn_device_target",
    "get_device_capability",
    "get_device_properties",
    "get_num_sm",
)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _ambient_device_reads(path: pathlib.Path) -> list:
    """Device queries naming no device, or a hardcoded one."""
    found = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        name = _call_name(node) if isinstance(node, ast.Call) else ""
        if name not in AMBIENT_DEVICE_READS:
            continue
        if (not node.args and not node.keywords) or (
            node.args and isinstance(node.args[0], ast.Constant)
        ):
            found.append(f"{path.name}:{node.lineno} {name}")
    return found


def test_device_target_adopters_read_policy_off_the_operand():
    """Launch policy and streams must follow the operand's device.

    ``num_sms`` and ``use_packed_fma`` choose tile shapes and codegen, so taking
    them from device 0 mistunes -- or miscompiles -- every other device, and a
    stream from the ambient device launches somewhere else entirely. Both are
    invisible on a single-GPU box, so guard them by AST: no GPU needed.
    """
    adopters = [
        path
        for path in sorted(GDN_KERNELS_DIR.rglob("*.py"))
        if path.name != "device_target.py" and "gdn_compile_options" in path.read_text()
    ]
    assert adopters, "nothing imports gdn_compile_options; did the helper move?"
    offenders = {
        name: reads
        for path in adopters
        for name, reads in [(path.name, _ambient_device_reads(path))]
        if reads
    }
    assert not offenders, (
        f"name the operand's device: {offenders}. Launch policy comes from "
        "gdn_device_target(q.device), streams from current_stream(q.device)."
    )


# tests/gdn/test_decode_ucache.py and benchmarks/bench_gdn_ucache_flush.py load these
# by path to re-specialize them per dtype arm, so they execute outside the package.
STANDALONE_LOADED = (
    "gdn_decode_bf16_wy_ucache.py",
    "gdn_decode_bf16_wy_ucache_flush.py",
)


@pytest.mark.parametrize("filename", STANDALONE_LOADED)
def test_by_path_modules_guard_their_relative_imports(filename):
    """A bare relative import raises ImportError when loaded outside the package."""
    tree = ast.parse((GDN_KERNELS_DIR / filename).read_text(), filename=filename)
    guarded = {
        id(node)
        for parent in ast.walk(tree)
        if isinstance(parent, ast.Try)
        for node in ast.walk(parent)
        if isinstance(node, ast.ImportFrom)
    }
    unguarded = [
        f"{filename}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level > 0
        and id(node) not in guarded
    ]
    assert not unguarded, (
        "wrap in try/ImportError with an absolute flashinfer.gdn_kernels fallback: "
        f"{unguarded}"
    )


def test_bf16_mtp_tile_v_follows_device_sm_count():
    from flashinfer.gdn_kernels import gdn_decode_bf16_state as bf16_state

    small_gpu, _ = bf16_state._get_bf16_mtp_config(4, 2, 64, 128, num_sms=16)
    large_gpu, _ = bf16_state._get_bf16_mtp_config(4, 2, 64, 128, num_sms=132)
    assert small_gpu != large_gpu


def _gdn_capable_devices() -> list[int]:
    """Indices of devices GDN supports, restricted to a single architecture.

    The DSL builds a JIT engine only when its process-global arch (``CUTE_DSL_ARCH``
    or CUDA device 0) can run the compiled target, so one process serves one arch.
    """
    by_capability: dict[tuple, list[int]] = {}
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        if capability[0] in (9, 10, 11, 12):
            by_capability.setdefault(capability, []).append(index)
    return max(by_capability.values(), key=len, default=[])


B, H, HV, K, V = 2, 4, 4, 128, 128
SCALE = 1.0 / math.sqrt(K)


def _cross_device_inputs(T: int, index: int) -> dict:
    """Identical operands on ``index``, seeded so every device gets the same bits."""
    torch.manual_seed(0)
    cpu = {
        "q": torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1,
        "k": torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1,
        "v": torch.randn(B, T, HV, V, dtype=torch.bfloat16) * 0.1,
        "a": torch.randn(B, T, HV, dtype=torch.bfloat16) * 0.1,
        "b": torch.randn(B, T, HV, dtype=torch.bfloat16) * 0.1,
        "A_log": torch.randn(HV, dtype=torch.float32) * 0.1,
        "dt_bias": torch.randn(HV, dtype=torch.float32) * 0.1,
        "state": torch.randn(B, HV, V, K, dtype=torch.bfloat16) * 0.1,
    }
    return {name: tensor.cuda(index) for name, tensor in cpu.items()}


def _pretranspose_on(index: int) -> torch.Tensor:
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    with torch.cuda.device(index):
        args = _cross_device_inputs(T=1, index=index)
        out, _ = gated_delta_rule_decode_pretranspose(
            **args, scale=SCALE, use_qk_l2norm=True
        )
        torch.cuda.synchronize(index)
        return out.float().cpu()


def _bf16_state_mtp_on(index: int) -> torch.Tensor:
    """The bf16-state MTP entry, whose cache value holds per-B default tensors.

    ``accepted_steps`` is one of them, built on ``q.device`` and dereferenced by
    the kernel, so an entry shared across devices hands the second device a
    pointer into the first device's memory.
    """
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp

    with torch.cuda.device(index):
        args = _cross_device_inputs(T=2, index=index)
        out = gated_delta_rule_mtp(
            A_log=args["A_log"],
            a=args["a"],
            dt_bias=args["dt_bias"],
            q=args["q"],
            k=args["k"],
            v=args["v"],
            b=args["b"],
            initial_state_source=args["state"],
            initial_state_indices=torch.arange(
                B, dtype=torch.int32, device=f"cuda:{index}"
            ),
            use_qk_l2norm_in_kernel=True,
            scale=SCALE,
        )
        torch.cuda.synchronize(index)
        return out.float().cpu()


CROSS_DEVICE_ENTRIES = {
    "pretranspose": _pretranspose_on,
    "bf16_state_mtp": _bf16_state_mtp_on,
}


@pytest.mark.parametrize("entry", sorted(CROSS_DEVICE_ENTRIES))
def test_decode_agrees_across_devices(entry):
    """The same decode must give the same answer on every device in one process.

    Before the device index entered the compile key, the second device reused the
    first's cache entry -- and with it the device-resident default tensors that
    entry holds. ``bf16_state_mtp`` is the arm that can actually fail: pretranspose
    keys its own per-device auxiliaries, so it is a sanity check, not a guard.
    """
    devices = _gdn_capable_devices()
    if len(devices) < 2:
        pytest.skip("needs two GDN-capable CUDA devices of the same architecture")
    first, second = devices[0], devices[1]
    run_on = CROSS_DEVICE_ENTRIES[entry]

    original_device = torch.cuda.current_device()
    try:
        torch.testing.assert_close(run_on(first), run_on(second), atol=0, rtol=0)
    finally:
        torch.cuda.set_device(original_device)
