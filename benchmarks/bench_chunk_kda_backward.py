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

Benchmark the deterministic chunked KDA training backward against the
flash-linear-attention v0.5.2 ``chunk_kda`` Triton backward (backward only).

Both arms run the reference forward once; the timed closure is the backward
alone (``torch.autograd.grad`` on a retained graph for the reference,
``flashinfer.chunk_kda_backward`` on the forward's saved set here).  Timing
uses ``flashinfer.testing.bench_gpu_time`` with CUPTI and cold L2, and every
configuration first checks elementwise parity of all seven gradients.

Usage:
    python benchmarks/bench_chunk_kda_backward.py [--rows b1_t2048_h16 ...]
"""

import argparse

import numpy as np
import torch

from flashinfer import chunk_kda_backward
from flashinfer.testing import bench_gpu_time

ROWS = {
    "b1_t2048_h16": dict(batch=1, seq_len=2048, heads=16, value_heads=16, packed=False),
    "b1_t4096_h16": dict(batch=1, seq_len=4096, heads=16, value_heads=16, packed=False),
    "b1_t8192_h16": dict(batch=1, seq_len=8192, heads=16, value_heads=16, packed=False),
    "b4_t2048_h16": dict(batch=4, seq_len=2048, heads=16, value_heads=16, packed=False),
    "b1_t4096_h16_hv32": dict(
        batch=1, seq_len=4096, heads=16, value_heads=32, packed=False
    ),
    "packed_4x2048_h16": dict(
        batch=4, seq_len=2048, heads=16, value_heads=16, packed=True
    ),
}
LOWER_BOUND = -5.0
CHUNK = 64
D = 128


def t_ms(fn, warmup=10, iters=50):
    times_ms = bench_gpu_time(
        fn, enable_cupti=True, dry_run_iters=warmup, repeat_iters=iters
    )
    return float(np.median(times_ms))


def make_inputs(batch, seq_len, heads, value_heads, *, packed, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=gen, dtype=torch.float32, device="cuda")

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda")

    total = batch * seq_len if packed else seq_len
    b = 1 if packed else batch
    inp = dict(
        q=(rand(b, total, heads, D) - 0.5).to(torch.bfloat16),
        k=(rand(b, total, heads, D) - 0.5).to(torch.bfloat16),
        v=(rand(b, total, value_heads, D) - 0.5).to(torch.bfloat16),
        g=(randn(b, total, value_heads, D) * 0.1).to(torch.bfloat16),
        beta=randn(b, total, value_heads).to(torch.bfloat16),
        A_log=randn(value_heads),
        dt_bias=randn(value_heads * D),
        do=randn(b, total, value_heads, D).to(torch.bfloat16),
        scale=D**-0.5,
        cu_seqlens=None,
    )
    if packed:
        inp["cu_seqlens"] = torch.arange(
            0, batch * seq_len + 1, seq_len, dtype=torch.int32, device="cuda"
        )
    return inp


def reference_arms(inp):
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.gate import fused_beta_sigmoid_fwd
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.chunk_fwd import chunk_kda_fwd
    from fla.ops.utils.index import prepare_chunk_indices

    cu = inp["cu_seqlens"]
    cu_cpu = None if cu is None else cu.cpu()
    leaves = {
        n: inp[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
    }
    out = chunk_kda(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=inp["scale"],
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        chunk_size=CHUNK,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        A_log=leaves["A_log"],
        dt_bias=leaves["dt_bias"],
    )
    out = out[0] if isinstance(out, tuple) else out
    order = ("q", "k", "v", "beta", "g", "A_log", "dt_bias")
    names = {
        "q": "dq",
        "k": "dk",
        "v": "dv",
        "beta": "dbeta",
        "g": "dg",
        "A_log": "dA_log",
        "dt_bias": "dt_bias",
    }

    def reference_backward():
        grads = torch.autograd.grad(
            out, [leaves[n] for n in order], grad_outputs=inp["do"], retain_graph=True
        )
        return {names[n]: g for n, g in zip(order, grads, strict=True)}

    with torch.no_grad():
        q_norm, q_rstd = l2norm_fwd(inp["q"])
        k_norm, k_rstd = l2norm_fwd(inp["k"])
        beta = fused_beta_sigmoid_fwd(inp["beta"], 1.0)
        chunk_indices = (
            None
            if cu is None
            else prepare_chunk_indices(cu, CHUNK, cu_seqlens_cpu=cu_cpu)
        )
        fwd = chunk_kda_fwd(
            q=q_norm,
            k=k_norm,
            v=inp["v"],
            g=inp["g"],
            beta=beta,
            scale=inp["scale"],
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            chunk_indices=chunk_indices,
            chunk_size=CHUNK,
            safe_gate=True,
            lower_bound=LOWER_BOUND,
            use_gate_in_kernel=True,
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
        )

    def candidate_backward():
        return chunk_kda_backward(
            q_norm=q_norm,
            k_norm=k_norm,
            q_rstd=q_rstd,
            k_rstd=k_rstd,
            v=inp["v"],
            g=inp["g"],
            beta_logits=inp["beta"],
            beta=beta,
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            Aqk=fwd[3],
            Akk=fwd[4],
            do=inp["do"],
            scale=inp["scale"],
            lower_bound=LOWER_BOUND,
            cu_seqlens=cu,
        )

    return reference_backward, candidate_backward


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="*", default=list(ROWS), choices=list(ROWS))
    args = parser.parse_args()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"{'row':22s} {'reference ms':>13s} {'this PR ms':>11s} {'speedup':>8s}  parity"
    )
    for name in args.rows:
        cfg = ROWS[name]
        inp = make_inputs(
            cfg["batch"],
            cfg["seq_len"],
            cfg["heads"],
            cfg["value_heads"],
            packed=cfg["packed"],
            seed=460_000 + hash(name) % 10_000,
        )
        ref_bwd, cand_bwd = reference_arms(inp)
        ref, cand = ref_bwd(), cand_bwd()
        ok = all(
            torch.allclose(cand[k].float(), ref[k].float(), atol=1e-2, rtol=1e-2)
            for k in ref
        )
        second = cand_bwd()
        deterministic = all(torch.equal(cand[k], second[k]) for k in cand)
        ref_ms, cand_ms = t_ms(ref_bwd), t_ms(cand_bwd)
        print(
            f"{name:22s} {ref_ms:13.4f} {cand_ms:11.4f} {ref_ms / cand_ms:8.2f}x  "
            f"{'pass' if ok else 'FAIL'}{'' if deterministic else ' (non-deterministic!)'}"
        )


if __name__ == "__main__":
    main()
