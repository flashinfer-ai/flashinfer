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

# Multi-rank tests of the experimental MOSCA MegaMoE backend (SM100, one NVLink
# domain). Launch with torchrun, 1 to 8 ranks:
#
#   torchrun --nproc-per-node 8 -m pytest tests/experimental/mosca/test_mosca_megamoe_multirank.py
#
# Two shapes: a small one, and the DeepSeek-V4-Pro MoE layer the catalog lists
# as validated (384 experts, hidden 7168, intermediate 3072, top-6). On the
# latter the tile-rule layer runs DSV4_PRO_DEP8_TILE_RULE, and the token counts
# cover all three of its windows (41..56, 57..80, 193..451). Checks:
#
# * torch oracle (independent of the kernel): dequantized FP4 weights of every
#   rank, FP8 activations, the SwiGLU clamp, FP8 intermediate -> relative L2
#   bound, for every switch and both layouts, at equal and uneven counts;
# * every MOSCA switch (and num_sms) against the kernel with every switch off,
#   bitwise, at equal and uneven per-rank token counts (needs no deep_gemm);
# * the switches-off kernel and every switch against the stable
#   ``sm100_fp8_fp4_bf16_deepgemm`` backend, bitwise (when ``deep_gemm`` is
#   installed), and fast_math=False on the small shape;
# * pre-quantized input (quantize_input=False) against in-backend quantization;
# * the replicated layout (in-kernel all-gather) against the partitioned
#   layout on the ``tensor_split`` slices plus an NCCL all-gather, bitwise;
# * CUDA-graph replays against eager forwards, bitwise, including a token count
#   that was never run before capture (precompile builds every configuration);
# * precompile=False: refused with more than one rank; with one rank, capturing
#   a configuration that was never built raises.

from __future__ import annotations

import datetime
import os

import pytest
import torch

CLAMP = 10.0
TOY_RULE = "t>=2&t<=16:block_m=32;t>=17:block_m=64,num_stages=4"
SHAPES = {
    "toy": dict(hidden=2048, inter=1024, top_k=4, experts=None, cap=128),
    "dsv4": dict(hidden=7168, inter=3072, top_k=6, experts=384, cap=512),
}
SWITCHES = ("small_t", "sparse", "rule", "sms")
# 20/21 straddle the small-T limit at top-6; 48, 64 and 300 sit in the three
# windows of DSV4_PRO_DEP8_TILE_RULE. Counts above a shape's capacity skip.
TOKENS = (0, 1, 7, 16, 20, 21, 48, 64, 128, 300)
UNEVEN = (0, 1, 7, 16, 21, 48, 64, 300)
NEVER_RUN_EAGERLY = 37  # no other test uses this count


def _world() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _skip_unless_ready() -> None:
    if "WORLD_SIZE" not in os.environ:
        pytest.skip("launch with torchrun (1 to 8 ranks)")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs SM100-family GPUs")


def _random_pack(local_experts, hidden, inter, seed, dev):
    from flashinfer.moe_ep import MoEWeightPack

    g = torch.Generator(device=dev).manual_seed(seed)

    def fp4(*shape):
        return torch.randint(
            -128, 128, shape, generator=g, device=dev, dtype=torch.int16
        ).to(torch.int8)

    def scale(*shape):
        return torch.exp2(torch.randint(-8, -4, shape, generator=g, device=dev).float())

    return MoEWeightPack(
        w13=fp4(local_experts, 2 * inter, hidden // 2),
        w2=fp4(local_experts, hidden, inter // 2),
        w13_scale=scale(local_experts, 2 * inter, hidden // 32),
        w2_scale=scale(local_experts, hidden, inter // 32),
    )


@pytest.fixture(scope="module", params=sorted(SHAPES))
def ctx(request):
    _skip_unless_ready()
    import torch.distributed as dist

    from flashinfer.experimental.mosca.megamoe import (
        DSV4_PRO_DEP8_TILE_RULE,
        Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig,
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MegaConfig, MoEEpLayer

    shape = dict(SHAPES[request.param])
    rank, world = _world()
    if not 1 <= world <= 8:
        pytest.skip("1 to 8 ranks")
    num_experts = shape["experts"] or 8 * world
    if num_experts % world:
        pytest.skip(f"{num_experts} experts do not split over {world} ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=20))
    group = dist.new_group(
        list(range(world)), backend="nccl", timeout=datetime.timedelta(minutes=20)
    )
    dev = torch.device("cuda", torch.cuda.current_device())
    hidden, inter = shape["hidden"], shape["inter"]
    top_k, cap = shape["top_k"], shape["cap"]
    pack = _random_pack(num_experts // world, hidden, inter, 1234 + rank, dev)
    transformed = preprocess_mega_weights(
        pack, intermediate_size=inter, hidden_size=hidden
    )
    boot = BootstrapConfig(world_size=world, rank=rank, process_group=group)
    rule = DSV4_PRO_DEP8_TILE_RULE if request.param == "dsv4" else TOY_RULE

    def fleet(per_rank_cap):
        return FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=per_rank_cap,
            token_hidden_size=hidden,
        )

    def mosca(fleet_cap=cap, quantize_input=True, **kw):
        cfg = Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig(
            intermediate_size=inter, top_k=top_k, activation_clamp=CLAMP, **kw
        )
        return MoEEpLayer(
            boot,
            fleet(fleet_cap),
            None,
            backend=MegaConfig(
                megakernel=cfg,
                quantize_input=quantize_input,
                preprocess_weights=False,
                transformed_weights=transformed,
            ),
        )

    layers = {
        "base": mosca(small_t=False),
        "small_t": mosca(small_t=True),
        "sparse": mosca(small_t=True, sparse_count=True),
        "rule": mosca(small_t=True, sparse_count=True, tile_rule=rule),
        "sms": mosca(small_t=True, num_sms=64),
        "replicated": mosca(
            fleet_cap=cap * world, small_t=True, token_layout="replicated"
        ),
    }
    if request.param == "toy":
        layers["nofast"] = mosca(small_t=True, fast_math=False)
        layers["prequant"] = mosca(small_t=True, quantize_input=False)
        if world == 1:
            # precompile=False is refused with more ranks. Its own num_sms, so no
            # precompiling layer has built its configurations (built modules are
            # shared by every layer of the process).
            layers["lazy"] = mosca(small_t=True, num_sms=32, precompile=False)
    state = dict(
        shape=request.param,
        rank=rank,
        world=world,
        group=group,
        dev=dev,
        pack=pack,
        boot=boot,
        fleet=fleet,
        mosca=mosca,
        layers=layers,
        num_experts=num_experts,
        hidden=hidden,
        inter=inter,
        top_k=top_k,
        cap=cap,
        deepgemm={},
        gathered_weights=None,
    )
    yield state
    torch.cuda.synchronize()
    for layer in [*layers.values(), *state["deepgemm"].values()]:
        layer.destroy()


def _inputs(ctx, n, seed):
    g = torch.Generator(device=ctx["dev"]).manual_seed(seed)
    x = torch.randn(n, ctx["hidden"], generator=g, device=ctx["dev"]).to(torch.bfloat16)
    scores = torch.randn(n, ctx["num_experts"], generator=g, device=ctx["dev"])
    w, ids = torch.topk(scores, ctx["top_k"], dim=-1, sorted=False)
    return x, ids.to(torch.int64), torch.softmax(w, dim=-1).float()


def _fwd(layer, x, ids, w, out=None, scales=None):
    from flashinfer.moe_ep import MoEEpTensors

    return layer.forward(
        MoEEpTensors(
            hidden_states=x, topk_ids=ids, topk_weights=w, output=out, scales=scales
        )
    )


def _all_ok(ctx, ok: bool) -> bool:
    import torch.distributed as dist

    t = torch.tensor([int(ok)], device=ctx["dev"])
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=ctx["group"])
    return bool(t.item())


def _deepgemm(ctx, **kw):
    """The stable sm100_fp8_fp4_bf16_deepgemm backend on the same weights."""
    pytest.importorskip("deep_gemm")
    from flashinfer.moe_ep import (
        MegaConfig,
        MoEEpLayer,
        Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
    )

    key = tuple(sorted(kw.items()))
    if key not in ctx["deepgemm"]:
        ctx["deepgemm"][key] = MoEEpLayer(
            ctx["boot"],
            ctx["fleet"](ctx["cap"]),
            ctx["pack"],
            backend=MegaConfig(
                megakernel=Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(
                    intermediate_size=ctx["inter"],
                    top_k=ctx["top_k"],
                    activation_clamp=CLAMP,
                    **kw,
                )
            ),
        )
    return ctx["deepgemm"][key]


def _count(ctx, tokens: int) -> int:
    """``tokens`` on every rank, or (-1) a different count per rank."""
    if tokens >= 0:
        if tokens > ctx["cap"]:
            pytest.skip(f"{tokens} tokens exceed the {ctx['shape']} capacity")
        return tokens
    return min(UNEVEN[ctx["rank"] % len(UNEVEN)], ctx["cap"])


def _gathered_weights(ctx):
    """Every rank's experts on every rank (collective; call it on every rank)."""
    import torch.distributed as dist

    if ctx["gathered_weights"] is None:
        parts = {}
        for name in ("w13", "w2", "w13_scale", "w2_scale"):
            local = getattr(ctx["pack"], name).contiguous()
            bufs = [torch.empty_like(local) for _ in range(ctx["world"])]
            dist.all_gather(bufs, local, group=ctx["group"])
            parts[name] = torch.cat(bufs)
        ctx["gathered_weights"] = parts
    return ctx["gathered_weights"]


def _oracle(ctx, parts, x, ids, w):
    """Float reference with the kernel's quantization points (FP8 in, FP8
    intermediate), independent of the kernel and local to this rank (no
    collective); experts are processed one at a time so the DeepSeek-V4-Pro
    shape stays cheap."""
    from flashinfer.experimental.mosca.megamoe.weights import (
        ceil_to_ue8m0,
        dequantize_fp4,
    )

    def q8(t):  # per-32 UE8M0 E4M3 round trip
        v = t.float().reshape(t.shape[0], t.shape[1] // 32, 32)
        sf = ceil_to_ue8m0(v.abs().amax(dim=2).clamp(1e-4) / 448.0)
        return (
            (v / sf.unsqueeze(2)).to(torch.float8_e4m3fn).float() * sf.unsqueeze(2)
        ).reshape(t.shape)

    inter = ctx["inter"]
    out = torch.zeros(x.shape[0], ctx["hidden"], device=x.device)
    if x.shape[0] == 0:
        return out.to(torch.bfloat16)
    xq = q8(x)
    for e in ids.unique().tolist():
        rows, slots = (ids == e).nonzero(as_tuple=True)
        w13 = dequantize_fp4(parts["w13"][e], parts["w13_scale"][e])
        h = (xq[rows] @ w13.t()).to(torch.bfloat16).float()
        gate, up = h[:, :inter].clamp(max=CLAMP), h[:, inter:].clamp(-CLAMP, CLAMP)
        act = torch.nn.functional.silu(gate) * up * w[rows, slots].unsqueeze(1)
        w2 = dequantize_fp4(parts["w2"][e], parts["w2_scale"][e])
        out.index_add_(0, rows, (q8(act) @ w2.t()).to(torch.bfloat16).float())
    return out.to(torch.bfloat16)


def _check_against_oracle(ctx, layer, x, ids, w) -> tuple[bool, str]:
    """Forward (collective), then the rank-local oracle comparison.

    A failure in the local part becomes this rank's verdict instead of an
    exception: every rank must still reach the same collectives (the verdict
    all-reduce, the next test's kernels), or the ranks fall out of step and
    later launches run mismatched configurations.
    """
    parts = _gathered_weights(ctx)
    y = _fwd(layer, x, ids, w)
    try:
        ref = _oracle(ctx, parts, x, ids, w)
        if ref.numel() == 0:
            return y.numel() == 0, "empty batch"
        rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
        return bool(torch.isfinite(y).all()) and rel < 0.05, f"rel L2 {rel:.4f}"
    except Exception as exc:  # noqa: BLE001 - reported through the verdict
        return False, f"oracle raised {exc!r}"


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("layer", ("base", *SWITCHES))
@pytest.mark.parametrize("tokens", (9, 48, -1))
def test_switches_match_the_torch_oracle(ctx, layer, tokens) -> None:
    n = _count(ctx, tokens)
    x, ids, w = _inputs(ctx, n, 10 + n + 7 * ctx["rank"])
    ok, detail = _check_against_oracle(ctx, ctx["layers"][layer], x, ids, w)
    assert _all_ok(ctx, ok), f"{layer}: {detail}"


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
def test_replicated_matches_the_torch_oracle(ctx) -> None:
    x, ids, w = _inputs(ctx, 64, 4321)  # the same batch on every rank
    ok, detail = _check_against_oracle(ctx, ctx["layers"]["replicated"], x, ids, w)
    assert _all_ok(ctx, ok), detail


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tokens", (*TOKENS, -1))
def test_switches_are_bitwise_equal_to_the_switches_off_kernel(ctx, tokens) -> None:
    n = _count(ctx, tokens)
    x, ids, w = _inputs(ctx, n, 100 * n + ctx["rank"])
    ref = _fwd(ctx["layers"]["base"], x, ids, w)
    bad = [
        s for s in SWITCHES if not torch.equal(_fwd(ctx["layers"][s], x, ids, w), ref)
    ]
    assert _all_ok(ctx, not bad), f"mismatch: {bad}"


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tokens", (*TOKENS, -1))
def test_switches_are_bitwise_equal_to_the_deepgemm_backend(ctx, tokens) -> None:
    reference = _deepgemm(ctx)
    n = _count(ctx, tokens)
    x, ids, w = _inputs(ctx, n, 100 * n + ctx["rank"])
    ref = _fwd(reference, x, ids, w)
    bad = [
        s
        for s in ("base", *SWITCHES)
        if not torch.equal(_fwd(ctx["layers"][s], x, ids, w), ref)
    ]
    assert _all_ok(ctx, not bad), f"mismatch: {bad}"


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
def test_fast_math_off_matches_the_deepgemm_backend(ctx) -> None:
    if ctx["shape"] != "toy":
        pytest.skip("small shape only")
    reference = _deepgemm(ctx, fast_math=False)
    x, ids, w = _inputs(ctx, 21, 77 + ctx["rank"])
    assert _all_ok(
        ctx,
        torch.equal(
            _fwd(ctx["layers"]["nofast"], x, ids, w), _fwd(reference, x, ids, w)
        ),
    )


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tokens", (1, 21, 64))
def test_prequantized_input_matches_in_backend_quantization(ctx, tokens) -> None:
    if ctx["shape"] != "toy":
        pytest.skip("small shape only")
    from flashinfer.experimental.mosca.megamoe.staging import (
        _per_token_cast_to_fp8_packed,
    )

    x, ids, w = _inputs(ctx, tokens, 300 + tokens + ctx["rank"])
    x_fp8, packed_sf = _per_token_cast_to_fp8_packed(x)
    y = _fwd(ctx["layers"]["prequant"], x_fp8, ids, w, scales=packed_sf)
    assert _all_ok(ctx, torch.equal(y, _fwd(ctx["layers"]["small_t"], x, ids, w)))


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("total", (1, 3, 8, 13, 64, 200, 512))
def test_replicated_equals_slices_plus_all_gather(ctx, total) -> None:
    import torch.distributed as dist

    counts = [len(c) for c in torch.arange(total).tensor_split(ctx["world"])]
    if max(counts) > ctx["cap"]:
        pytest.skip("beyond the per-rank capacity")
    x, ids, w = _inputs(ctx, total, 5000 + total)  # the same batch on every rank
    y = _fwd(ctx["layers"]["replicated"], x, ids, w)
    start = sum(counts[: ctx["rank"]])
    rows = slice(start, start + counts[ctx["rank"]])
    part = _fwd(ctx["layers"]["small_t"], x[rows], ids[rows], w[rows])
    gathered = [
        torch.empty(c, ctx["hidden"], device=ctx["dev"], dtype=torch.bfloat16)
        for c in counts
    ]
    dist.all_gather(gathered, part.contiguous(), group=ctx["group"])
    assert y.shape == (total, ctx["hidden"])
    assert _all_ok(ctx, torch.equal(y, torch.cat(gathered)))


def _replays_match_eager(ctx, layer, rows, seed_of, warm_rows) -> bool:
    import torch.distributed as dist

    _fwd(layer, *_inputs(ctx, warm_rows, seed_of(99)))  # prepare the workspace
    torch.cuda.synchronize()
    dist.barrier(group=ctx["group"])
    static = [t.clone() for t in _inputs(ctx, rows, seed_of(0))]
    out = torch.empty(rows, ctx["hidden"], device=ctx["dev"], dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        _fwd(layer, *static, out)
    torch.cuda.synchronize()
    ok = True
    for k in range(1, 4):
        fresh = _inputs(ctx, rows, seed_of(k))
        for dst, src in zip(static, fresh, strict=False):
            dst.copy_(src)
        dist.barrier(group=ctx["group"])
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier(group=ctx["group"])
        # Every rank runs every eager forward (a collective), whatever ``ok``.
        eager = _fwd(layer, *fresh)
        ok = torch.equal(out, eager) and ok
    del graph
    return ok


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "layout,rows", (("replicated", 8), ("replicated", 64), ("small_t", 16))
)
def test_cuda_graph_replays_match_eager(ctx, layout, rows) -> None:
    replicated = layout == "replicated"

    def seed_of(k):
        return 900 + k if replicated else 900 + k + 17 * ctx["rank"]

    ok = _replays_match_eager(ctx, ctx["layers"][layout], rows, seed_of, warm_rows=rows)
    assert _all_ok(ctx, ok)


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("layout", ("replicated", "rule"))
def test_capture_of_a_count_never_run_eagerly(ctx, layout) -> None:
    # precompile=True built every configuration at warmup, so a count that no
    # forward has staged yet captures without any build or load in capture.
    replicated = layout == "replicated"

    def seed_of(k):
        return 700 + k if replicated else 700 + k + 13 * ctx["rank"]

    ok = _replays_match_eager(
        ctx, ctx["layers"][layout], NEVER_RUN_EAGERLY, seed_of, warm_rows=1
    )
    assert _all_ok(ctx, ok)


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
def test_precompile_off_is_refused_with_more_than_one_rank(ctx) -> None:
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    if ctx["world"] == 1:
        pytest.skip("allowed with one rank")
    with pytest.raises(MoEEpConfigError, match="precompile=False"):
        ctx["mosca"](small_t=True, precompile=False)


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
def test_a_preparation_failure_on_one_rank_raises_on_every_rank(
    ctx, monkeypatch
) -> None:
    # A rank whose planner build fails must not leave its peers waiting in the
    # symmetric-memory rendezvous: every rank raises before allocating.
    from flashinfer.experimental.mosca.megamoe import jit
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    if ctx["world"] == 1:
        pytest.skip("needs more than one rank")
    # A num_sms no other layer uses, so the workspace pool has nothing to reuse.
    layer = ctx["mosca"](small_t=True, num_sms=96)
    if ctx["rank"] == 1:

        def broken():
            raise RuntimeError("injected planner build failure")

        monkeypatch.setattr(jit, "host_module", broken)
    expected, match = (
        (RuntimeError, "injected")
        if ctx["rank"] == 1
        else (MoEEpConfigError, "failed on another EP rank")
    )
    try:
        with pytest.raises(expected, match=match):
            _fwd(layer, *_inputs(ctx, 1, 51))
    finally:
        monkeypatch.undo()
        layer.destroy()


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
def test_capture_of_an_unbuilt_configuration_raises(ctx) -> None:
    if "lazy" not in ctx["layers"]:
        pytest.skip("precompile=False exists with one rank on the small shape only")
    from flashinfer.experimental.mosca.megamoe import jit
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    layer = ctx["layers"]["lazy"]
    _fwd(layer, *_inputs(ctx, 1, 41))  # builds only the 1-token configuration
    backend, workspace = layer._kernel, layer._workspace
    unbuilt = ctx["cap"]
    if jit.is_loaded(backend.kernel_params(workspace, unbuilt)):
        pytest.skip("1 and the capacity share a launch configuration")
    torch.cuda.synchronize()
    static = _inputs(ctx, unbuilt, 42)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(MoEEpConfigError, match="before CUDA graph capture"),
        torch.cuda.graph(graph, stream=torch.cuda.Stream()),
    ):
        _fwd(layer, *static)
    del graph
    torch.cuda.synchronize()
