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
"""

import math
import warnings

import pytest
import torch

from flashinfer.experimental.nvfp4_sparse_mla_decode import cake_backend
from flashinfer.experimental.nvfp4_sparse_mla_decode.backend import (
    HEAD_DIM,
    NUM_HEADS,
    ROW_BYTES,
    V_HEAD_DIM,
    is_valid_config,
    select_num_ctas,
)

E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
NUM_ROWS = 64 * 1024
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)
# Relative to the largest reference magnitude; the kernel measures ~2.6e-3 (P and the partial outputs are f16).
TOLERANCE = 1e-2
# One-wave cluster capacity measured on GB200 (152 SMs): clusters of c CTAs that run at once.
GB200_CAPACITY = {8: 15, 6: 23, 5: 28, 4: 36, 3: 46}
# Both implementations of the operator; every kernel test runs against each.
BACKENDS = ("cuda", "cake")
# Cluster sizes a caller may force per backend, and the ones each rejects: the hand-written kernel serves
# 3 to 8, the generated program 1 to 6 and 8 (7 is not a variant of the program: rejected, not re-planned).
FORCED_CLUSTER_SIZES = {
    "cuda": (3, 4, 5, 6, 7, 8),
    "cake": cake_backend.CLUSTER_SIZES,
}
REJECTED_CLUSTER_SIZES = {"cuda": (1, 2), "cake": (7,)}


def _is_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )


requires_sm100 = pytest.mark.skipif(
    not _is_supported_gpu(),
    reason="NVFP4 sparse MLA decode requires compute capability 10.0 or 10.3",
)


def _decode(*args, backend="cuda", **kwargs):
    import flashinfer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ExperimentalWarning on first use
        return flashinfer.mla.nvfp4_sparse_mla_decode(*args, backend=backend, **kwargs)


def _require_backend(backend: str, num_ctas=None) -> None:
    """Skip when the generated program for this device is not registered in the checkout."""
    if backend != "cake":
        return
    device = torch.device("cuda", torch.cuda.current_device())
    if not cake_backend.generated_program_available(device, num_ctas):
        which = "every cluster size" if num_ctas is None else f"cluster size {num_ctas}"
        pytest.skip(
            f"no generated NVFP4 sparse MLA decode program ({which}) registered for this device"
        )


def make_cache(num_rows: int, seed: int) -> torch.Tensor:
    """Random nvfp4_ds_mla rows: any e2m1 bytes, finite e4m3 RoPE values, positive e4m3 scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randint(
        0, 256, (num_rows, ROW_BYTES), dtype=torch.uint8, device="cuda", generator=g
    )
    rope = (torch.randn(num_rows, 64, device="cuda", generator=g) * 0.5).clamp(-8, 8)
    kv[:, 256:320] = rope.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.rand(num_rows, 32, device="cuda", generator=g) * 0.99 + 0.01
    kv[:, 320:] = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    return kv


def make_indices(num_tokens: int, topk: int, padding: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    idx = torch.stack(
        [
            torch.randperm(NUM_ROWS, device="cuda", generator=g)[:topk]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    if (
        padding == "suffix"
    ):  # short contexts: the tail of the list is empty, as a sparse indexer emits it
        for t in range(0, num_tokens, 3):
            idx[t, topk - (64 + 97 * t) % (topk // 2) :] = -1
    elif (
        padding == "scattered"
    ):  # 30 % empty slots anywhere; a clamp-to-row-0 kernel hot-spots on these
        mask = torch.rand(num_tokens, topk, device="cuda", generator=g) < 0.3
        idx[mask] = -1
    elif padding != "none":
        raise ValueError(padding)
    return idx


def dequantize_rows(kv: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """[..., 352] nvfp4_ds_mla rows at ``idx`` -> [..., 576] float32, exactly."""
    lut = torch.tensor(E2M1_VALUES, device=kv.device)
    raw = kv[idx.clamp_min(0).long()]
    lo = lut[(raw[..., :256] & 15).long()]
    hi = lut[(raw[..., :256] >> 4).long()]
    nope = torch.stack([lo, hi], -1).flatten(-2)
    b = torch.arange(32, device=kv.device)
    scale = (
        raw[..., 320 + 8 * (b % 4) + b // 4]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    nope = nope * scale.repeat_interleave(16, -1)
    rope = raw[..., 256:320].contiguous().view(torch.float8_e4m3fn).float()
    return torch.cat([nope, rope], -1)


def reference(q, kv, idx, bmm1_scale, bmm2_scale, chunk: int = 8) -> torch.Tensor:
    out = torch.empty(q.shape[0], NUM_HEADS, V_HEAD_DIM, device=q.device)
    for t0 in range(0, q.shape[0], chunk):
        sl = slice(t0, t0 + chunk)
        k = dequantize_rows(kv, idx[sl])
        s = torch.einsum("thd,tkd->thk", q[sl].float(), k) * bmm1_scale
        s = s.masked_fill((idx[sl] < 0)[:, None, :], float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)  # a token without keys gets zeros
        out[sl] = torch.einsum("thk,tkd->thd", p, k[..., :V_HEAD_DIM]) * bmm2_scale
    return out


def make_query(num_tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, device="cuda", generator=g) * 0.5
    return q.to(torch.float8_e4m3fn)


def relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


@pytest.fixture(scope="module")
def kv_cache():
    return make_cache(NUM_ROWS, seed=0)


# ---------------------------------------------------------------- plan, hand-written kernel (no GPU needed)


@pytest.mark.parametrize(
    "num_tokens, expected",
    [
        (1, 8),
        (15, 8),
        (16, 6),
        (23, 6),
        (24, 5),
        (28, 5),
        (29, 4),
        (36, 4),
        (37, 3),
        (46, 3),
        (47, 3),
        (200, 3),
    ],
)
def test_plan_picks_largest_one_wave_cluster(num_tokens, expected):
    assert select_num_ctas(num_tokens, 2048, GB200_CAPACITY) == expected


def test_plan_respects_ring_depth():
    # 512 keys = 16 stages: 8 or 6 CTAs would leave fewer than 3 stages per CTA
    assert select_num_ctas(1, 512, GB200_CAPACITY) == 5
    assert (
        not is_valid_config(512, 8)
        and not is_valid_config(512, 6)
        and is_valid_config(512, 5)
    )


@pytest.mark.parametrize(
    "topk, num_ctas", [(2048, 2), (2048, 9), (2000, 8), (0, 4), (4096, 3)]
)
def test_invalid_configs(topk, num_ctas):
    assert not is_valid_config(topk, num_ctas)


# ---------------------------------------------------------------- plan, generated program (no GPU needed)


@pytest.mark.parametrize(
    "num_tokens, expected",
    [
        (1, 8),
        (15, 8),
        (16, 6),
        (23, 6),
        (24, 5),
        (28, 5),
        (29, 4),
        (36, 4),
        (37, 3),
        (46, 3),
        (47, 2),
        (200, 2),
    ],
)
def test_cake_plan_picks_largest_one_wave_cluster(num_tokens, expected):
    # Past the one-wave capacity of 3-CTA clusters the program falls back to its smallest split, 2 (several waves).
    assert (
        cake_backend.plan_ctas(
            num_tokens=num_tokens, topk=2048, capacity=GB200_CAPACITY
        )
        == expected
    )


def test_cake_plan_needs_one_full_stage_per_cta():
    # 512 keys = 16 stages: every split up to 8 keeps a full stage per CTA, so the one-wave rule alone decides
    assert cake_backend.plan_ctas(num_tokens=1, topk=512, capacity=GB200_CAPACITY) == 8
    # 64 keys = 2 stages: only the 2-CTA split keeps a full stage per CTA; 32 keys admit no split at all
    assert cake_backend.plan_ctas(num_tokens=1, topk=64, capacity=GB200_CAPACITY) == 2
    assert cake_backend.plan_ctas(num_tokens=1, topk=32, capacity=GB200_CAPACITY) == 1


@pytest.mark.parametrize(
    "topk, num_ctas, expected",
    [
        (2048, 1, 2048),
        (2048, 2, 1024),
        (2048, 3, 704),
        (2048, 4, 512),
        (2048, 5, 416),
        (2048, 6, 352),
        (2048, 8, 256),
        (1024, 3, 352),
        (512, 8, 64),
    ],
)
def test_cake_keys_per_cta(topk, num_ctas, expected):
    assert cake_backend.keys_per_cta(topk, num_ctas) == expected


def test_cake_keys_per_cta_fit_the_index_table():
    assert cake_backend.keys_per_cta(4096, 2) == cake_backend.MAX_KEYS_PER_CTA
    with pytest.raises(ValueError, match="index table"):
        cake_backend.keys_per_cta(4096, 1)


@pytest.mark.parametrize("num_ctas", cake_backend.CLUSTER_SIZES)
def test_cake_admits_its_cluster_sizes(num_ctas):
    assert cake_backend.is_valid_split(2048, num_ctas)
    assert cake_backend.is_valid_split(1024, num_ctas)


@pytest.mark.parametrize(
    "topk, num_ctas", [(2048, 7), (2048, 0), (2048, 9), (4096, 1), (512, 5), (0, 1)]
)
def test_cake_rejects_splits(topk, num_ctas):
    # 7 is not a variant of the program; 512 keys over 5 CTAs round to 128 per CTA and leave the fifth empty
    assert not cake_backend.is_valid_split(topk, num_ctas)


# ---------------------------------------------------------------- kernels (SM100)


@requires_sm100
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_tokens", [1, 5, 15, 16, 20, 25, 28, 35, 46, 64])
@pytest.mark.parametrize("topk", [2048, 1024])
@pytest.mark.parametrize("padding", ["none", "suffix", "scattered"])
def test_matches_reference(kv_cache, backend, num_tokens, topk, padding):
    _require_backend(backend)
    q = make_query(num_tokens, seed=1)
    idx = make_indices(num_tokens, topk, padding, seed=2)
    out = _decode(q, kv_cache, idx, SM_SCALE, backend=backend)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert relative_error(out, reference(q, kv_cache, idx, SM_SCALE, 1.0)) < TOLERANCE


@requires_sm100
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_tokens", [1, 12, 40])
def test_topk_512(kv_cache, backend, num_tokens):
    _require_backend(backend)
    q = make_query(num_tokens, seed=3)
    idx = make_indices(num_tokens, 512, "suffix", seed=4)
    out = _decode(q, kv_cache, idx, SM_SCALE, backend=backend)
    assert relative_error(out, reference(q, kv_cache, idx, SM_SCALE, 1.0)) < TOLERANCE


@requires_sm100
@pytest.mark.parametrize(
    "backend, num_ctas",
    [(backend, c) for backend in BACKENDS for c in FORCED_CLUSTER_SIZES[backend]],
)
def test_explicit_cluster_sizes(kv_cache, backend, num_ctas):
    _require_backend(backend, num_ctas)
    q = make_query(12, seed=5)
    idx = make_indices(12, 2048, "scattered", seed=6)
    ref = reference(q, kv_cache, idx, SM_SCALE, 1.0)
    out = _decode(
        q, kv_cache, idx, SM_SCALE, num_ctas_per_token=num_ctas, backend=backend
    )
    assert relative_error(out, ref) < TOLERANCE


@requires_sm100
@pytest.mark.parametrize(
    "backend, num_ctas",
    [(backend, c) for backend in BACKENDS for c in REJECTED_CLUSTER_SIZES[backend]],
)
def test_rejected_cluster_sizes(kv_cache, backend, num_ctas):
    # Rejected before any kernel is built: 1 and 2 by the hand-written kernel, 7 by the generated program.
    q = make_query(2, seed=17)
    idx = make_indices(2, 2048, "none", seed=16)
    with pytest.raises(ValueError, match="num_ctas_per_token"):
        _decode(
            q, kv_cache, idx, SM_SCALE, num_ctas_per_token=num_ctas, backend=backend
        )


@requires_sm100
def test_cake_runner_binds_the_forced_split(kv_cache):
    _require_backend("cake", 4)
    q = make_query(6, seed=19)
    idx = make_indices(6, 2048, "none", seed=20)
    runner = cake_backend.prepare_nvfp4_sparse_mla_decode(
        q, kv_cache, idx, bmm1_scale=SM_SCALE, num_ctas_per_token=4
    )
    assert (runner.num_ctas, runner.keys_per_cta, runner.grid) == (4, 512, (6, 4, 1))
    out = runner()
    assert relative_error(out, reference(q, kv_cache, idx, SM_SCALE, 1.0)) < TOLERANCE


@requires_sm100
@pytest.mark.parametrize("backend", BACKENDS)
def test_tokens_without_keys_get_zeros(kv_cache, backend):
    _require_backend(backend)
    q = make_query(4, seed=7)
    idx = make_indices(4, 2048, "none", seed=8)
    idx[0] = -1
    idx[2, 1:] = -1  # a single key: the output is that key's V
    out = _decode(q, kv_cache, idx, SM_SCALE, backend=backend)
    assert (out[0] == 0).all()
    assert relative_error(out, reference(q, kv_cache, idx, SM_SCALE, 1.0)) < TOLERANCE


@requires_sm100
@pytest.mark.parametrize("backend", BACKENDS)
def test_out_buffer_and_scales(kv_cache, backend):
    _require_backend(backend)
    q = make_query(8, seed=9)
    idx = make_indices(8, 2048, "suffix", seed=10)
    out = torch.full(
        (8, NUM_HEADS, V_HEAD_DIM), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    ret = _decode(
        q, kv_cache, idx, 0.5 * SM_SCALE, bmm2_scale=2.0, out=out, backend=backend
    )
    assert ret is out
    assert (
        relative_error(out, reference(q, kv_cache, idx, 0.5 * SM_SCALE, 2.0))
        < TOLERANCE
    )


@requires_sm100
@pytest.mark.parametrize("backend", BACKENDS)
def test_block_shaped_cache(kv_cache, backend):
    _require_backend(backend)
    # vLLM passes [num_blocks, block_size, 352]; indices are flat block_id * block_size + offset
    q = make_query(6, seed=11)
    idx = make_indices(6, 2048, "none", seed=12)
    blocks = kv_cache.view(-1, 64, ROW_BYTES)
    out = _decode(q, blocks, idx, SM_SCALE, backend=backend)
    assert relative_error(out, reference(q, kv_cache, idx, SM_SCALE, 1.0)) < TOLERANCE


@requires_sm100
@pytest.mark.parametrize("backend", BACKENDS)
def test_cuda_graph_replay(kv_cache, backend):
    _require_backend(backend)
    q = make_query(20, seed=13)
    idx = make_indices(20, 2048, "suffix", seed=14)
    out = torch.empty(20, NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _decode(
        q, kv_cache, idx, SM_SCALE, out=out, backend=backend
    )  # compile + occupancy query outside capture
    eager = out.clone()
    out.zero_()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        _decode(q, kv_cache, idx, SM_SCALE, out=out, backend=backend)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@requires_sm100
@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_batch(kv_cache, backend):
    q = make_query(0, seed=15)
    idx = torch.empty(0, 2048, dtype=torch.int32, device="cuda")
    assert _decode(q, kv_cache, idx, SM_SCALE, backend=backend).shape == (
        0,
        NUM_HEADS,
        V_HEAD_DIM,
    )


@requires_sm100
@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_bad_inputs(kv_cache, backend):
    idx = make_indices(2, 2048, "none", seed=16)
    q = make_query(2, seed=17)
    with pytest.raises(ValueError, match="heads|query"):
        _decode(q[:, :8].contiguous(), kv_cache, idx, SM_SCALE, backend=backend)
    with pytest.raises(ValueError, match="query"):
        _decode(q.to(torch.bfloat16), kv_cache, idx, SM_SCALE, backend=backend)
    with pytest.raises(ValueError, match="kv_cache"):
        _decode(q, kv_cache[:, :320].contiguous(), idx, SM_SCALE, backend=backend)


@requires_sm100
def test_unknown_backend_is_rejected(kv_cache):
    idx = make_indices(1, 2048, "none", seed=18)
    q = make_query(1, seed=18)
    with pytest.raises(ValueError, match="backend"):
        _decode(q, kv_cache, idx, SM_SCALE, backend="triton")
