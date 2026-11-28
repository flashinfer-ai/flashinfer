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

Tests of the experimental ``flashinfer.sampling.softcap_sample_with_stats`` (SM90 CuTe-DSL).

The reference is a direct FP32 PyTorch evaluation of the contract (exact tanh, softmax and
log-softmax) with the Gumbel noise drawn from the contract's Philox stream by
``tl.philox``. The kernel uses approximate tanh / ex2 / lg2, so:

* argmax / sample may differ only on a few FP32 near-tie rows;
* entropy must be within 5e-4 + 1e-4 * |ref| on every row;
* every probability must be within two BF16 ulps (relative 2^-6) of the reference's
  BF16-rounded value, each row's L1 distance within 1e-2, and the mean row L1 distance and
  the |mean signed row-sum difference| within 1e-4 and 5e-5 (normalization bias).

These bounds were calibrated against independent FP64 / libdevice re-implementations and
reject a wrong Philox lane, a wrong counter, unnormalized or biased probabilities and
temperature mix-ups. The graph, stream and repeated-launch tests require bit-identical
results: the kernel's reductions are fixed-order.
"""

import pytest
import torch
import triton
import triton.language as tl

import flashinfer
from flashinfer.experimental.softcap_sample_with_stats import VOCAB_SIZE

V = VOCAB_SIZE

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="softcap_sample_with_stats requires SM90",
    ),
    pytest.mark.filterwarnings("ignore::flashinfer.experimental.ExperimentalWarning"),
]


@triton.jit
def _uniform_kernel(out_ptr, seed, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    q = offs >> 2
    c0 = (q & 0xFFFFFFFF).to(tl.uint32)
    c1 = (q >> 32).to(tl.uint32)
    zero = c0 * 0
    o0, o1, o2, o3 = tl.philox(seed, c0, c1, zero, zero)
    lane = offs & 3
    o = tl.where(lane == 0, o0, tl.where(lane == 1, o1, tl.where(lane == 2, o2, o3)))
    tl.store(out_ptr + offs, tl.uint_to_uniform_float(o), mask=offs < n)


def _uniforms(rows: int, seed: int) -> torch.Tensor:
    """u[r, c] of the contract's Philox stream for rows [0, rows)."""
    n = rows * V
    out = torch.empty(n, dtype=torch.float32, device="cuda")
    _uniform_kernel[(triton.cdiv(n, 4096),)](out, seed, n, BLOCK=4096)
    return out.view(rows, V)


def reference(logits, temperature, seed, softcap):
    """FP32 evaluation of the contract, in 256-row chunks."""
    R = logits.shape[0]
    temps = temperature.repeat_interleave(R // temperature.numel())
    argmax = torch.empty(R, dtype=torch.int64, device="cuda")
    sample = torch.empty_like(argmax)
    entropy = torch.empty(R, dtype=torch.float32, device="cuda")
    probs = torch.empty(R, V, dtype=torch.bfloat16, device="cuda")
    u_all = _uniforms(R, seed)
    for r0 in range(0, R, 256):
        r1 = min(R, r0 + 256)
        t = temps[r0:r1, None]
        x = torch.tanh(logits[r0:r1].float() / softcap) * softcap
        argmax[r0:r1] = x.argmax(dim=-1)
        scaled = x * (1.0 / t.clamp_min(1e-10))
        logp = torch.log_softmax(scaled, dim=-1)
        p = logp.exp()
        entropy[r0:r1] = -(p * logp).sum(dim=-1)
        probs[r0:r1] = p.to(torch.bfloat16)
        g = -torch.log(-torch.log(u_all[r0:r1].clamp_min(1e-20)))
        noisy = torch.where(t > 0, scaled + g, scaled)
        sample[r0:r1] = noisy.argmax(dim=-1)
    return argmax, sample, entropy, probs


def make_inputs(R, n_temps, case, gen_seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(gen_seed)
    sigma = 1.5 + 2.5 * torch.rand((R, 1), generator=g, device="cuda")
    logits = torch.empty(R, V, dtype=torch.bfloat16, device="cuda")
    for r0 in range(0, R, 256):
        r1 = min(R, r0 + 256)
        logits[r0:r1] = (
            torch.randn((r1 - r0, V), generator=g, device="cuda") * sigma[r0:r1]
        ).to(torch.bfloat16)
    # confident positions: up to three distinct tokens per row lifted by U(4, 28)
    seg = V // 3
    cols = (
        torch.randint(0, seg, (R, 3), generator=g, device="cuda")
        + torch.arange(3, device="cuda") * seg
    )
    lift = 4.0 + 24.0 * torch.rand((R, 3), generator=g, device="cuda")
    on = torch.rand((R, 3), generator=g, device="cuda") < torch.tensor(
        [0.85, 0.4, 0.2], device="cuda"
    )
    vals = logits.gather(1, cols).float() + torch.where(
        on, lift, torch.zeros_like(lift)
    )
    logits.scatter_(1, cols, vals.to(torch.bfloat16))
    temperature = 0.4 + 0.4 * torch.rand((n_temps,), generator=g, device="cuda")
    if case == "edge":
        rows = torch.arange(R, device="cuda")
        wide = rows % 5 == 2
        logits[wide] = (logits[wide].float() * 3.0).clamp(-60, 60).to(torch.bfloat16)
        peaked = rows % 11 == 5
        pk_col = torch.randint(0, V, (R,), generator=g, device="cuda")
        logits[peaked, pk_col[peaked]] = 45.0
        tie = rows % 7 == 3
        t_lo = torch.randint(0, V // 2, (R,), generator=g, device="cuda")
        t_hi = t_lo + 1 + torch.randint(0, V // 2 - 1, (R,), generator=g, device="cuda")
        top = (logits.amax(dim=1).float() + 1.0).to(torch.bfloat16)
        logits[tie, t_hi[tie]] = top[tie]
        logits[tie, t_lo[tie]] = top[tie]
        per = R // n_temps
        if per > 1:
            logits[R - max(1, per // 4) :] = 0  # padded rows of the last group
        special = torch.tensor([0.0, 0.05, 1.5, 0.4], device="cuda")
        k = min(n_temps, special.numel())
        temperature[:k] = special[:k]
    return logits, temperature


def check(ref, out):
    """The calibrated bounds of the module docstring."""
    r_am, r_sm, r_en, r_pr = ref
    am, sm, en, pr = out
    R = r_am.numel()
    for name, a, b, frac in (("argmax", r_am, am, 5e-4), ("sample", r_sm, sm, 2e-3)):
        assert bool(((b >= 0) & (b < V)).all()), f"{name} index out of range"
        bad = int((a != b).sum())
        assert bad <= max(1, int(frac * R)), f"{name}: {bad} of {R} rows differ"
    assert torch.isfinite(en).all()
    err = (en.double() - r_en.double()).abs()
    assert bool((err <= 5e-4 + 1e-4 * r_en.double().abs()).all()), (
        f"entropy max error {float(err.max())}"
    )
    sum_l1 = sum_bias = 0.0
    for r0 in range(0, R, 128):
        a = r_pr[r0 : r0 + 128].float()
        b = pr[r0 : r0 + 128].float()
        assert torch.isfinite(b).all()
        diff = b - a
        assert bool((diff.abs() <= 2.0**-6 * a + 1e-37).all()), (
            "probability beyond 2 ulps"
        )
        l1 = diff.abs().sum(dim=1, dtype=torch.float64)
        assert float(l1.max()) <= 1e-2, f"row L1 {float(l1.max())}"
        sum_l1 += float(l1.sum())
        sum_bias += float(diff.sum(dtype=torch.float64))
    assert sum_l1 / R <= 1e-4, f"mean row L1 {sum_l1 / R}"
    assert abs(sum_bias) / R <= 5e-5, f"mean row-sum bias {abs(sum_bias) / R}"


@pytest.mark.parametrize(
    "rows,n_temps", [(1, 1), (24, 24), (256, 1), (300, 3), (1024, 32), (1536, 6)]
)
@pytest.mark.parametrize("case", ["serving", "edge"])
def test_matches_reference(rows, n_temps, case):
    logits, temperature = make_inputs(rows, n_temps, case)
    seed = 1234567
    out = flashinfer.sampling.softcap_sample_with_stats(logits, temperature, seed, 30.0)
    check(reference(logits, temperature, seed, 30.0), out)


@pytest.mark.parametrize("softcap", [2.0**-16, 1.0, 64.0])
def test_softcap_range(softcap):
    logits, temperature = make_inputs(64, 1, "serving", gen_seed=1)
    # Keep logits / softcap distributed as at softcap 30: a softcap far below the logits
    # saturates tanh, and the argmax among saturated columns is a rounding tie.
    logits = (logits.float() * (softcap / 30.0)).to(torch.bfloat16)
    temperature = temperature * (softcap / 30.0)
    out = flashinfer.sampling.softcap_sample_with_stats(
        logits, temperature, 99, softcap
    )
    check(reference(logits, temperature, 99, softcap), out)


def test_seed_tensor_and_determinism():
    logits, temperature = make_inputs(512, 2, "serving", gen_seed=2)
    a = flashinfer.sampling.softcap_sample_with_stats(
        logits, temperature, 2**32 - 7, 30.0
    )
    b = flashinfer.sampling.softcap_sample_with_stats(
        logits,
        temperature,
        torch.tensor([2**32 - 7], dtype=torch.int64, device="cuda"),
        30.0,
    )
    c = flashinfer.sampling.softcap_sample_with_stats(
        logits, temperature, 2**32 - 7, 30.0
    )
    for x, y, z in zip(a, b, c, strict=True):
        assert torch.equal(x, y) and torch.equal(x, z)
    d = flashinfer.sampling.softcap_sample_with_stats(logits, temperature, 5, 30.0)
    assert torch.equal(a[0], d[0]) and torch.equal(a[3], d[3])
    assert not torch.equal(a[1], d[1])  # only the sample depends on the seed


def test_out_buffers_and_side_stream():
    logits, temperature = make_inputs(256, 1, "serving", gen_seed=3)
    base = flashinfer.sampling.softcap_sample_with_stats(logits, temperature, 7, 30.0)
    out = tuple(torch.empty_like(t) for t in base)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        res = flashinfer.sampling.softcap_sample_with_stats(
            logits, temperature, 7, 30.0, out=out
        )
    torch.cuda.current_stream().wait_stream(s)
    assert all(r is o for r, o in zip(res, out, strict=True))
    for x, y in zip(base, out, strict=True):
        assert torch.equal(x, y)


def test_repeated_launches_mixed_rows():
    """Each launch must leave the exchange workspace clean for the next, whatever its size."""
    sizes = [(24, 24), (256, 1), (8, 8), (1536, 6), (1, 1), (300, 3)]
    inputs = [make_inputs(r, n, "serving", gen_seed=r) for r, n in sizes]
    base = [
        flashinfer.sampling.softcap_sample_with_stats(L, T, 11, 30.0) for L, T in inputs
    ]
    for _ in range(20):
        for (L, T), b in zip(inputs, base, strict=True):
            o = flashinfer.sampling.softcap_sample_with_stats(L, T, 11, 30.0)
            for x, y in zip(o, b, strict=True):
                assert torch.equal(x, y)


def test_cuda_graph_replay():
    logits, temperature = make_inputs(1536, 6, "serving", gen_seed=4)
    seed = torch.tensor([3], dtype=torch.int64, device="cuda")
    out = tuple(
        torch.empty(s, dtype=d, device="cuda")
        for s, d in (
            ((1536,), torch.int64),
            ((1536,), torch.int64),
            ((1536,), torch.float32),
            ((1536, V), torch.bfloat16),
        )
    )
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        flashinfer.sampling.softcap_sample_with_stats(
            logits, temperature, seed, 30.0, out=out
        )
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        flashinfer.sampling.softcap_sample_with_stats(
            logits, temperature, seed, 30.0, out=out
        )
    for value in (17, 2**31 + 5, 17):
        seed.fill_(value)
        g.replay()
        eager = flashinfer.sampling.softcap_sample_with_stats(
            logits, temperature, value, 30.0
        )
        torch.cuda.synchronize()
        for x, y in zip(out, eager, strict=True):
            assert torch.equal(x, y)


def test_invalid_arguments():
    f = flashinfer.sampling.softcap_sample_with_stats
    logits = torch.zeros(8, V, dtype=torch.bfloat16, device="cuda")
    temperature = torch.ones(2, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        f(logits[:, : V // 2].contiguous(), temperature, 0, 30.0)  # vocabulary
    with pytest.raises(ValueError):
        f(logits.float(), temperature, 0, 30.0)  # dtype
    with pytest.raises(ValueError):
        f(logits, torch.ones(3, device="cuda"), 0, 30.0)  # 3 does not divide 8
    with pytest.raises(ValueError):
        f(logits, temperature, 2**32, 30.0)  # seed range
    with pytest.raises(ValueError):
        f(logits, temperature, 0, 65.0)  # softcap range
    with pytest.raises(ValueError):
        f(logits, temperature, 0, 30.0, out=(None, None, None, None))
    with pytest.raises(ValueError):
        f(logits, temperature, 0, 30.0, backend="triton")
