"""Smoke: ring-contract CUDA monolith vs ring-contract Triton reference.

Both sides consume identical ring caches; compare out/state and every cache
postcondition 1:1, at ring_start=0 and at a wrapping ring_start=17 (L=22).
Run: uv run python .plans/ring_ref_smoke.py
"""

import sys
from pathlib import Path

import torch
from einops import repeat

sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "mamba"))
from triton_reference.replay_selective_state_update import (
    replay_selective_state_update,
)

from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu

torch.manual_seed(7)
device = "cuda"
batch = cache = 4
T, W = 6, 16
L = W + T  # 22
nheads, dim, dstate, ngroups = 16, 64, 128, 1
dtype = torch.bfloat16

pnat = torch.tensor(
    [0, 4, 11, 14], device=device, dtype=torch.int32
)  # 14+6>16 -> write
wrote_exp = (pnat + T) > W

x = torch.randn(batch, T, nheads, dim, device=device, dtype=dtype)
dt_base = torch.randn(batch, T, nheads, device=device, dtype=torch.float32)
dt = repeat(dt_base, "b t h -> b t h p", p=dim)
A = repeat(-torch.rand(nheads, device=device) - 0.5, "h -> h p n", p=dim, n=dstate)
B = torch.randn(batch, T, ngroups, dstate, device=device, dtype=dtype)
C = torch.randn(batch, T, ngroups, dstate, device=device, dtype=dtype)
D = repeat(torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=dim)
dt_bias = repeat(torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=dim)
state0 = torch.randn(cache, nheads, dim, dstate, device=device, dtype=dtype)

x_cache0 = torch.randn(cache, nheads, L, dim, device=device, dtype=dtype)
B_cache0 = torch.randn(cache, ngroups, L, dstate, device=device, dtype=dtype)
dt_cache0 = torch.randn(cache, nheads, L, device=device, dtype=torch.float32).abs()

work = torch.zeros(batch, 4, device=device, dtype=torch.int32)
work[:, 0] = torch.arange(batch, device=device, dtype=torch.int32)
work[:, 1] = torch.arange(batch, device=device, dtype=torch.int32)
work[:, 2] = pnat
n_writes = wrote_exp.sum().to(torch.int32).reshape(1)


def run_pair(s0: int):
    start = torch.full((cache,), s0, device=device, dtype=torch.int32)
    outs = {}
    for name in ("cuda", "triton"):
        st = state0.clone()
        out = torch.zeros_like(x)
        xc, bc, dtc = x_cache0.clone(), B_cache0.clone(), dt_cache0.clone()
        if name == "cuda":
            checkpointing_ssu(
                st,
                xc,
                bc,
                dtc,
                start.clone(),
                pnat.clone(),
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                out=out,
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
                algorithm="monolith",
            )
        else:
            replay_selective_state_update(
                st,
                xc,
                bc,
                dtc,
                start.clone(),
                pnat.clone(),
                x,
                dt,
                A,
                B,
                C,
                out,
                n_writes,
                work,
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
                mode="persistent_dynamic",
            )
        outs[name] = dict(st=st, out=out, xc=xc, bc=bc, dtc=dtc)
    return outs


def check(name, a, b, rtol=2e-2, atol=5e-1):
    torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol)
    print(f"  {name:26s} OK  (max|d|={(a.float() - b.float()).abs().max().item():.3e})")


for s0 in (0, 17):
    print(f"=== ring_start = {s0}")
    r = run_pair(s0)
    check("out", r["cuda"]["out"], r["triton"]["out"])
    check("state", r["cuda"]["st"], r["triton"]["st"])
    for slot in range(cache):
        p = int(pnat[slot])
        rows = (s0 + p + torch.arange(T)) % L  # the appended ring rows
        check(
            f"x_cache slot{slot} w={bool(wrote_exp[slot])}",
            r["cuda"]["xc"][slot][:, rows],
            r["triton"]["xc"][slot][:, rows],
        )
        check(
            f"B_cache slot{slot}",
            r["cuda"]["bc"][slot][:, rows],
            r["triton"]["bc"][slot][:, rows],
        )
        check(
            f"dt_cache slot{slot}",
            r["cuda"]["dtc"][slot][:, rows],
            r["triton"]["dtc"][slot][:, rows],
            rtol=1e-4,
            atol=1e-4,
        )
print("ALL OK")
