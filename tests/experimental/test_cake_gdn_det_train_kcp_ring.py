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

CPU test of the ring exchange of the KCP GDN route (``exchange="ring"``) under both launch schedules: the
``"overlap"`` schedule (test knob; the document-start tape walk issued under the first hop) must publish the same
bits as the shipped ``"wait_first"`` schedule -- forward output and every gradient, on every rank, with the norm off
and with ``use_qk_l2norm_in_kernel``.  Two partitions of a four-rank group: one whose documents cross rank boundaries
(one and two hops) and one with no crossing fragment anywhere (``N_out == 0``: no exchange at all).  The ranks are
CPU processes on a gloo group running the fake backend (the torch emulation of the programs' ABI).
"""

from __future__ import annotations

import importlib
import math
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PACKAGE = "flashinfer.experimental.cake_gdn_det_train_kcp"
W = 4
H_QK, HV = 2, 4
# contiguous layout: 256 tokens per rank
CROSSING = [
    0,
    40,
    600,
    1024,
]  # [40, 600) spans ranks 0..2 (two hops), [600, 1024) ranks 2..3 (one hop)
NO_CROSSING = [0, 256, 300, 512, 768, 1024]  # every document inside one rank
TENSORS = ("out", "dq", "dk", "dv", "dbeta", "dg")


def _mod(sub: str):
    try:
        from tests.test_helpers.cake_package import cake_package

        return cake_package("cake_gdn_det_train_kcp", sub)
    except (ImportError, KeyError):
        return importlib.import_module(f"{PACKAGE}.{sub}")


def _global_inputs(cu: list[int], seed: int = 20261029) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    T = cu[-1]
    a_log = torch.rand(HV, generator=gen) * math.log(16.0)

    def raw(H):
        scale = torch.exp2(torch.randint(-4, 5, (T, H, 1), generator=gen).float())
        return (torch.randn(T, H, 128, generator=gen) * scale).to(torch.bfloat16)

    return {
        "q": raw(H_QK),
        "k": raw(H_QK),
        "v": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
        "g": (
            -torch.exp(a_log)[None, :]
            * torch.nn.functional.softplus(torch.randn(T, HV, generator=gen))
        )
        .float()
        .contiguous(),
        "beta": torch.sigmoid(torch.randn(T, HV, generator=gen)).float().contiguous(),
        "dout": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
    }


def _local(part, cu: list[int], rank: int, glob: dict[str, torch.Tensor]):
    """This rank's operands: its fragments' token ranges in slot order (the KCP local layout)."""
    out = {}
    for name, t in glob.items():
        pieces = [
            t[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
            for f in part.fragments[rank]
            if f.length > 0
        ]
        out[name] = torch.cat(pieces, 0).contiguous()
    return out


def _pass(kcp, ctx, ops, **o) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Forward + backward (saved set) of one rank; the published tensors and the forward's launch sequence."""
    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"])
    out, saved = kcp.chunk_gdn_det_fwd_kcp(*args, ctx, backend="fake", **o)
    seq = list(kcp.LAST_CALL["fwd"]["sequence"])
    dq, dk, dv, dbeta, dg = kcp.chunk_gdn_det_bwd_kcp(
        *args, ctx, ops["dout"], saved=saved, backend="fake", **o
    )
    return {"out": out, "dq": dq, "dk": dk, "dv": dv, "dbeta": dbeta, "dg": dg}, seq


def _rank(rank: int, init_file: str, cu: list[int], crossing: bool) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=W
    )
    try:
        torch.set_flush_denormal(False)
        kcp, plan, l2n_ref = _mod("cake_kcp"), _mod("cake_kcp_plan"), _mod("cake_l2n")
        part = plan.partition_det_kcp(torch.tensor(cu, dtype=torch.int64), W, "contiguous")
        ctx = plan.build_context_det_kcp(part, dist.group.WORLD, rank, torch.device("cpu"))
        assert int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS, int(ctx.n_d_max)
        if crossing:
            assert int(ctx.N_out) > 0 and int(ctx.n_d_max) == 3, (
                int(ctx.N_out),
                int(ctx.n_d_max),
            )
        else:
            assert int(ctx.N_out) == 0, int(ctx.N_out)
        raw = _local(part, cu, rank, _global_inputs(cu))
        # the norm-off route takes L2-normalized rows (raw rows would make the delta rule diverge), the norm-on
        # route the raw rows
        hat = dict(raw)
        hat["q"], hat["k"] = (l2n_ref.qk_l2norm_ref(raw[n])[0] for n in ("q", "k"))
        for l2n in (False, True):
            ops = raw if l2n else hat
            base = {
                "exchange": "ring",
                "save_transfer": False,
                "use_qk_l2norm_in_kernel": l2n,
            }
            ref, seq_w = _pass(kcp, ctx, ops, **base)
            got, seq_o = _pass(
                kcp, ctx, ops, **base, test_knobs={"schedule": "overlap"}
            )
            hops = sorted(s for s in seq_o if s.startswith("ring:"))
            # the crossing partition exchanges through the ring hops (one and two hops); N_out == 0 exchanges nothing
            if crossing:
                assert hops == ["ring:hop1", "ring:hop2"], seq_o
            else:
                assert not hops and not any(s.startswith("a2a:") for s in seq_o), seq_o
            assert sorted(s for s in seq_w if s.startswith("ring:")) == hops, seq_w
            for name in TENSORS:
                a, b = ref[name], got[name]
                assert torch.isfinite(a.float()).all(), (rank, l2n, crossing, name)
                assert a.shape == b.shape and torch.equal(a, b), (
                    f"rank {rank} l2n={l2n} crossing={crossing}: {name} differs between the overlap and "
                    f"wait_first schedules (max |diff| {(a.float() - b.float()).abs().max().item()})"
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "cu, crossing",
    [(CROSSING, True), (NO_CROSSING, False)],
    ids=["crossing", "no_crossing"],
)
def test_ring_overlap_schedule_equals_wait_first(cu: list[int], crossing: bool) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with gloo is not available")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "rendezvous")
        mp.spawn(_rank, args=(init_file, cu, crossing), nprocs=W, join=True)
