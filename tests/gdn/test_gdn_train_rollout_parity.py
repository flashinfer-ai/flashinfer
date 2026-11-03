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

Training-vs-rollout forward parity of packed GDN (#7932): the single-rank
context-parallel training forward (``flashinfer.gdn_train.chunk_gdn_kcp_fwd``,
the deterministic route, and ``flashinfer.cake_gdn_train_kcp.chunk_gdn_fwd_kcp``
with ``deterministic=False``) against the rollout prefill
``chunk_gated_delta_rule`` on the same BF16 packed inputs with natural-log
gates, post-sigmoid beta and scale ``128 ** -0.5``.

The tolerance test is the contract, judged with the training route's own
reference budget (rel-RMS and max-abs relative to the largest rollout
magnitude); the bitwise arm is a strict xfail that starts failing the day the
forwards agree bitwise.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer import cake_gdn_train_kcp, gdn_train
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import is_sm100a_supported
from tests.test_helpers.cake_gdn_det_train_kcp_reference import (
    error_metrics,
    make_inputs,
)

D = 128
SCALE = D**-0.5
ROUTES = ("det", "nondet")

# (id, document lengths, (H_qk, HV))
SHAPES = (
    ("ragged-2x4", (100, 640, 77, 1000, 129), (2, 4)),
    ("one-doc-4x8", (3000,), (4, 8)),
    ("qwen3next-16x32", (2048 + 37, 1000), (16, 32)),
)
# the KCP training suites' reference budget (tests/experimental/test_cake_gdn_train_kcp_cp1.py)
REL_RMS_MAX = 1e-2
MAX_ABS_REL = 2e-2


def _supported(route: str, device: torch.device) -> bool:
    if route == "det":
        return gdn_train.is_gdn_train_supported(device)
    return cake_gdn_train_kcp.is_gdn_train_kcp_supported(device, deterministic=False)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda", torch.cuda.current_device())
    if is_sm100a_supported(device):
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13:
            pytest.skip(
                f"SM100 GDN prefill requires CUDA 13+, got {torch.version.cuda}"
            )
    return device


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _cu(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    return cu


def _training_forward(route, inputs, cu, device):
    q, k, v, g, beta = (inputs[n].to(device) for n in ("q", "k", "v", "g", "beta"))
    cu64 = torch.tensor(cu, dtype=torch.int64)
    if route == "det":
        part = gdn_train.kcp_partition(cu64, 1)
        ctx = gdn_train.build_kcp_context(part, None, 0, device)
        out, _ = gdn_train.chunk_gdn_kcp_fwd(
            q, k, v, g, beta, ctx, scale=SCALE, save_for_backward=False
        )
    else:
        part = cake_gdn_train_kcp.partition_kcp(cu64, 1, deterministic=False)
        ctx = cake_gdn_train_kcp.build_context_kcp(
            part, None, 0, device, deterministic=False
        )
        out, _ = cake_gdn_train_kcp.chunk_gdn_fwd_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            scale=SCALE,
            save_for_backward=False,
            deterministic=False,
        )
    return out[0]


def _rollout_forward(inputs, cu, device):
    q, k, v, g, beta = (
        inputs[n][0].to(device).contiguous() for n in ("q", "k", "v", "g", "beta")
    )
    return chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=SCALE,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device),
        gate_domain="log",
    )


def _both_forwards(route, lengths, heads, device):
    if not _supported(route, device):
        pytest.skip(f"the {route} training package does not support this device")
    cu = _cu(lengths)
    inputs = make_inputs(cu, *heads)
    out_train = _training_forward(route, inputs, cu, device)
    out_roll = _rollout_forward(inputs, cu, device)
    torch.cuda.synchronize(device)
    return out_train, out_roll


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    ("lengths", "heads"), [s[1:] for s in SHAPES], ids=[s[0] for s in SHAPES]
)
def test_training_forward_matches_rollout_within_tolerance(
    device, lengths, heads, route
) -> None:
    out_train, out_roll = _both_forwards(route, lengths, heads, device)
    assert out_train.shape == out_roll.shape and out_train.dtype == out_roll.dtype
    m = error_metrics(out_train, out_roll)
    print(
        f"\n{route}: out rel-RMS {m['rel_rms']:.3e} max-abs {m['max_abs']:.3e} "
        f"exact {(out_train == out_roll).float().mean().item():.4f} "
        f"ref max-abs {m['ref_max_abs']:.3e}"
    )
    assert m["nonfinite"] == 0, f"{m['nonfinite']} non-finite outputs"
    assert m["rel_rms"] <= REL_RMS_MAX, f"rel-RMS {m['rel_rms']:.3e} > {REL_RMS_MAX}"
    assert m["max_abs"] <= MAX_ABS_REL * m["ref_max_abs"], (
        f"max-abs {m['max_abs']:.3e} > {MAX_ABS_REL} x {m['ref_max_abs']:.3e} (worst {m['worst']})"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#7932 asks for bitwise parity; training and rollout forwards reduce in different orders",
)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    ("lengths", "heads"), [s[1:] for s in SHAPES], ids=[s[0] for s in SHAPES]
)
def test_training_forward_matches_rollout_bitwise(
    device, lengths, heads, route
) -> None:
    out_train, out_roll = _both_forwards(route, lengths, heads, device)
    assert torch.equal(out_train, out_roll)
