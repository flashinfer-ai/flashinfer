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


def test_index_less_device_follows_current_device(fake_devices, monkeypatch):
    assert dt.gdn_device_target("cuda").arch == "sm_90a"
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert dt.gdn_device_target("cuda").arch == "sm_100a"


def test_cute_dsl_arch_env_override_wins(fake_devices, monkeypatch):
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_90a")
    dt._resolve.cache_clear()
    assert dt.gdn_device_target("cuda:1").arch == "sm_90a"


def test_non_cuda_device_is_rejected(fake_devices):
    with pytest.raises(ValueError, match="require CUDA tensors"):
        dt.gdn_device_target("cpu")


def test_compile_options_pin_arch_and_preserve_extras(fake_devices):
    extras = (cute.EnableTVMFFI(True), cute.OptLevel(3))
    options = dt.gdn_compile_options("cuda:1", *extras)

    assert isinstance(options[0], cute.GPUArch)
    assert options[0].value == "sm_100a"
    assert options[1:] == extras


def test_gdn_compile_sites_do_not_pass_string_options():
    """``cute.compile[opts](..., options="...")`` drops ``opts`` instead of merging.

    The DSL replaces subscripted options wholesale when a string ``options=``
    kwarg is present, so a string here would silently un-pin the target arch.
    """
    offenders = []
    for path in sorted(GDN_KERNELS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if "compile" not in ast.dump(node.func):
                continue
            for kw in node.keywords:
                if kw.arg == "options" and isinstance(kw.value, ast.Constant):
                    offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        "pass compile options via cute.compile[gdn_compile_options(device, ...)] "
        f"instead of a string options= kwarg: {offenders}"
    )


def test_bf16_mtp_tile_v_follows_device_sm_count():
    from flashinfer.gdn_kernels import gdn_decode_bf16_state as bf16_state

    small_gpu, _ = bf16_state._get_bf16_mtp_config(4, 2, 64, 128, num_sms=16)
    large_gpu, _ = bf16_state._get_bf16_mtp_config(4, 2, 64, 128, num_sms=132)
    assert small_gpu != large_gpu


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs at least two CUDA devices"
)
def test_decode_matches_across_current_device(monkeypatch):
    """Operands on cuda:1 must decode identically whatever the current device is."""
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    B, T, H, HV, K, V = 2, 1, 4, 4, 128, 128
    device = torch.device("cuda", 1)
    torch.manual_seed(0)

    def make(dtype):
        return torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1

    q, k = make(torch.bfloat16), make(torch.bfloat16)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device) * 0.1
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device) * 0.1
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    state = torch.randn(B, HV, V, K, dtype=torch.bfloat16, device=device) * 0.1

    def run():
        out, _ = gated_delta_rule_decode_pretranspose(
            q=q,
            k=k,
            v=v,
            state=state.clone(),
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            scale=1.0 / math.sqrt(K),
            use_qk_l2norm=True,
        )
        return out

    torch.cuda.set_device(0)
    from_other_device = run()
    torch.cuda.set_device(1)
    from_same_device = run()

    torch.testing.assert_close(
        from_other_device.float(), from_same_device.float(), atol=0, rtol=0
    )
