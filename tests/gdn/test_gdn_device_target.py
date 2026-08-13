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
    """Two devices with different arch, SM count, and packed-FMA support."""
    caps = {0: (9, 0), 1: (10, 0)}
    sms = {0: 132, 1: 148}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: caps[i])
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: _FakeProps(sms[i])
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    dt._resolve.cache_clear()
    yield
    dt._resolve.cache_clear()


def test_device_target_reads_the_requested_device(fake_devices):
    d0 = dt.gdn_device_target("cuda:0")
    d1 = dt.gdn_device_target("cuda:1")

    assert (d0.arch, d0.num_sms, d0.use_packed_fma) == ("sm_90a", 132, False)
    assert (d1.arch, d1.num_sms, d1.use_packed_fma) == ("sm_100a", 148, True)


def test_compile_key_separates_devices(fake_devices):
    """A compiled artifact is pinned to the device it first ran on, so two devices
    must not share a cache entry even when they are the same architecture."""
    assert dt.gdn_device_target("cuda:0").compile_key != (
        dt.gdn_device_target("cuda:1").compile_key
    )


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


def test_every_gdn_cute_compile_pins_an_explicit_target():
    """Un-subscripted ``cute.compile`` targets whatever the DSL picks (device 0).

    A string ``options=`` kwarg is equally unsafe: the DSL replaces subscripted
    options wholesale when one is present, silently dropping the ``GPUArch``.
    """
    unpinned, string_options = [], []
    for path in sorted(GDN_KERNELS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_cute_compile(node.func):
                continue
            where = f"{path.name}:{node.lineno}"
            if not isinstance(node.func, ast.Subscript):
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


def test_bf16_mtp_tile_v_follows_device_sm_count():
    from flashinfer.gdn_kernels import gdn_decode_bf16_state as bf16_state

    small_gpu, _ = bf16_state._get_bf16_mtp_config(4, 2, 64, 128, num_sms=16)
    large_gpu, _ = bf16_state._get_bf16_mtp_config(4, 2, 64, 128, num_sms=132)
    assert small_gpu != large_gpu


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs at least two CUDA devices"
)
def test_decode_agrees_across_devices():
    """The same decode must give the same answer on every device in one process.

    A compiled CuTe-DSL artifact is pinned to the device it first ran on, so before
    the device index entered the compile key the second device silently reused the
    first device's artifact.
    """
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    B, T, H, HV, K, V = 2, 1, 4, 4, 128, 128
    torch.manual_seed(0)
    cpu_inputs = {
        "q": torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1,
        "k": torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1,
        "v": torch.randn(B, T, HV, V, dtype=torch.bfloat16) * 0.1,
        "a": torch.randn(B, T, HV, dtype=torch.bfloat16) * 0.1,
        "b": torch.randn(B, T, HV, dtype=torch.bfloat16) * 0.1,
        "A_log": torch.randn(HV, dtype=torch.float32) * 0.1,
        "dt_bias": torch.randn(HV, dtype=torch.float32) * 0.1,
        "state": torch.randn(B, HV, V, K, dtype=torch.bfloat16) * 0.1,
    }

    def run_on(index):
        # The DSL binds an artifact to the current device, so make it the operand's.
        with torch.cuda.device(index):
            args = {k: v.cuda(index) for k, v in cpu_inputs.items()}
            out, _ = gated_delta_rule_decode_pretranspose(
                **args, scale=1.0 / math.sqrt(K), use_qk_l2norm=True
            )
            torch.cuda.synchronize(index)
            return out.float().cpu()

    original_device = torch.cuda.current_device()
    try:
        torch.testing.assert_close(run_on(0), run_on(1), atol=0, rtol=0)
    finally:
        torch.cuda.set_device(original_device)
