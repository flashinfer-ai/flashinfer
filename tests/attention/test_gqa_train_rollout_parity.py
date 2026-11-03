"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Training-vs-rollout forward parity of HD256 causal GQA (#7932): the training
forward ``varlen_gqa_attention_fwd`` against the rollout prefill
``BatchPrefillWithRaggedKVCacheWrapper`` on the same BF16 packed inputs.

The two kernels agree to BF16 rounding (1-2 ULP on the output, FP32 noise on
the natural-log LSE) but reduce in different orders, so the tolerance test is
the contract and the bitwise arm is a strict xfail: it starts failing the day
the forwards agree bitwise, which is the signal to promote it.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
from flashinfer.gqa_train import varlen_gqa_attention_fwd
from tests.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake_gqa_train", "cake_backend")

D = 256
SCALE = D**-0.5
WORKSPACE_BYTES = 256 << 20

# (id, segment lengths, H_q, H_kv)
LAYOUTS = (
    ("one_segment", (128,), 8, 2),
    ("ragged", (257, 1000, 64, 2048), 16, 2),
    ("two_4k", (4096, 4096), 32, 4),
)
ROLLOUT_BACKENDS = ("auto", "fa2")
# (atol, rtol) per rollout backend, from SM100 measurements: out 1-2 BF16 ULP, LSE 1.9e-6 (auto) / 1.4e-3 (fa2)
OUT_TOL = {"auto": (1e-2, 2e-2), "fa2": (2e-2, 2e-2)}
LSE_TOL = {"auto": (1e-4, 1e-5), "fa2": (5e-3, 1e-3)}


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(f"the training package does not ship this device ({capability})")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _cu(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    return cu


def _inputs(lengths, hq, hkv, device):
    gen = torch.Generator(device="cpu").manual_seed(7932)
    T = sum(lengths)
    q = torch.randn(T, hq, D, generator=gen, dtype=torch.float32)
    k = torch.randn(T, hkv, D, generator=gen, dtype=torch.float32)
    v = torch.randn(T, hkv, D, generator=gen, dtype=torch.float32)
    cu = torch.tensor(_cu(lengths), dtype=torch.int32, device=device)
    return (
        q.to(device=device, dtype=torch.bfloat16),
        k.to(device=device, dtype=torch.bfloat16),
        v.to(device=device, dtype=torch.bfloat16),
        cu,
    )


def _rollout_forward(q, k, v, cu, hq, hkv, backend, device):
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    wrapper.plan(
        cu, cu, hq, hkv, D, causal=True, sm_scale=SCALE, q_data_type=torch.bfloat16
    )
    return wrapper.run(q, k, v, return_lse=True, lse_base="ln", lse_layout="HN")


def _both_forwards(lengths, hq, hkv, backend, device):
    q, k, v, cu = _inputs(lengths, hq, hkv, device)
    out_train, lse_train = varlen_gqa_attention_fwd(
        q, k, v, cu, cu, softmax_scale=SCALE
    )
    out_roll, lse_roll = _rollout_forward(q, k, v, cu, hq, hkv, backend, device)
    torch.cuda.synchronize(device)
    return out_train, lse_train, out_roll, lse_roll


@pytest.mark.parametrize("backend", ROLLOUT_BACKENDS)
@pytest.mark.parametrize(
    ("lengths", "hq", "hkv"), [s[1:] for s in LAYOUTS], ids=[s[0] for s in LAYOUTS]
)
def test_training_forward_matches_rollout_within_tolerance(
    device, lengths, hq, hkv, backend
) -> None:
    out_train, lse_train, out_roll, lse_roll = _both_forwards(
        lengths, hq, hkv, backend, device
    )
    assert out_train.shape == out_roll.shape and out_train.dtype == out_roll.dtype
    assert lse_train.shape == lse_roll.shape
    diff = (out_train.float() - out_roll.float()).abs()
    print(
        f"\n{backend}: out max-abs {diff.max().item():.3e} mean-abs {diff.mean().item():.3e} "
        f"exact {(out_train == out_roll).float().mean().item():.4f} "
        f"lse max-abs {(lse_train - lse_roll).abs().max().item():.3e}"
    )
    atol, rtol = OUT_TOL[backend]
    torch.testing.assert_close(
        out_train.float(), out_roll.float(), atol=atol, rtol=rtol
    )
    atol, rtol = LSE_TOL[backend]
    torch.testing.assert_close(lse_train, lse_roll, atol=atol, rtol=rtol)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#7932 asks for bitwise parity; training and rollout forwards reduce in different orders",
)
@pytest.mark.parametrize("backend", ROLLOUT_BACKENDS)
@pytest.mark.parametrize(
    ("lengths", "hq", "hkv"), [s[1:] for s in LAYOUTS], ids=[s[0] for s in LAYOUTS]
)
def test_training_forward_matches_rollout_bitwise(
    device, lengths, hq, hkv, backend
) -> None:
    out_train, lse_train, out_roll, lse_roll = _both_forwards(
        lengths, hq, hkv, backend, device
    )
    assert torch.equal(out_train, out_roll)
    assert torch.equal(lse_train, lse_roll)
