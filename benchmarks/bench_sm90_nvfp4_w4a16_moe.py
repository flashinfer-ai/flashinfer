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

Benchmark of ``flashinfer.fused_moe.sm90_nvfp4_w4a16_moe`` (SM90 NVFP4 W4A16
fused MoE) on one GPU.

Times one full MoE call (routing given) per token count with
``flashinfer.testing.bench_gpu_time`` (cold L2 by default, CUPTI with
``--cupti``). Weight preparation, workspace allocation and JIT compilation are
outside the timed region. Reports the median time, the packed weight bytes of
the experts the routing touches divided by that time, and the useful TFLOP/s.

Optional baseline (``--baseline cutile``): FlashInfer's cuTile NVFP4-weight x
BF16-activation backend through the unified ``MoELayer`` on the same
checkpoint tensors (its own weight preparation, gate / up halves passed in the
up-first order it expects; timing only, no numerical cross-check here; see
tests/moe/test_sm90_nvfp4_w4a16_moe.py for accuracy). Not available with
``--post-norm``, which cuTile has no equivalent for.

Examples::

    # tuned shape: hidden = intermediate = 4096, 32 local experts, top-8
    python benchmarks/bench_sm90_nvfp4_w4a16_moe.py --cupti
    # DeepSeek-V3-like expert slice, with the cuTile baseline
    python benchmarks/bench_sm90_nvfp4_w4a16_moe.py --hidden 7168 --intermediate 2048 \\
        --experts 32 --top-k 8 --tokens 1 8 64 512 4096 --baseline cutile
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from flashinfer.fused_moe import (
    get_sm90_nvfp4_w4a16_moe_workspace,
    prepare_sm90_nvfp4_w4a16_weights,
    sm90_nvfp4_w4a16_moe,
    sm90_nvfp4_w4a16_moe_prefill_min_tokens,
)
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm90a_supported

DEFAULT_TOKENS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 753, 1024, 2048, 4096, 16384]


def make_checkpoint(H, I, E, seed=0):
    """ModelOpt-layout NVFP4 experts: E2M1 codes, E4M3 block scales, FP32 global scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    kw = dict(device="cuda", generator=g)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, **kw)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, **kw)
    s13 = torch.randint(0x28, 0x40, (E, 2 * I, H // 16), dtype=torch.uint8, **kw)
    s2 = torch.randint(0x28, 0x40, (E, H, I // 16), dtype=torch.uint8, **kw)
    gs13 = torch.full((E,), 1.0 / (2.0 * math.sqrt(H)), device="cuda")
    gs2 = torch.full((E,), 1.0 / (2.0 * math.sqrt(I)), device="cuda")
    return (
        w13,
        s13.view(torch.float8_e4m3fn),
        gs13,
        w2,
        s2.view(torch.float8_e4m3fn),
        gs2,
    )


def make_routing(M, E, k, mode, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if mode == "uniform":
        scores = torch.rand(M, E, device="cuda", generator=g)
    elif mode == "skewed":
        # Zipf-like expert popularity
        pop = 1.0 / torch.arange(1, E + 1, device="cuda").float()
        scores = torch.rand(M, E, device="cuda", generator=g) * pop
    else:
        raise ValueError(mode)
    w, ids = scores.topk(k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return ids.to(torch.int32), w.float()


def touched_weight_bytes(ids, E, H, I):
    used = torch.unique(ids[(ids >= 0) & (ids < E)]).numel()
    per_expert = (H // 256) * (2 * I // 256) * 35840 + (I // 256) * (H // 256) * 35840
    return used * per_expert


def to_up_first(w13, s13, gs13):
    """Gate-first gated rows (ModelOpt, as passed to the SM90 kernel) -> the up-first
    layout ``prepare_cutile_nvfp4_weights`` expects (it swaps the halves back)."""
    w13 = torch.cat(w13.chunk(2, dim=1)[::-1], dim=1)
    s13 = torch.cat(s13.chunk(2, dim=1)[::-1], dim=1)
    if gs13.dim() == 2:
        gs13 = gs13[:, [1, 0]]
    return w13, s13, gs13


def cutile_baseline(ckpt, H, I, E, k, max_tokens):
    """Callable ``f(x, ids, w)`` running the cuTile NVFP4 x BF16 backend, or raise."""
    from flashinfer.fused_moe import (
        BackendOptions,
        CuTileNvfp4Bf16Config,
        ExecutionConfig,
        ExpertConfig,
        MoEActivationPack,
        MoEConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
    )

    w13, s13, gs13, w2, s2, gs2 = ckpt
    w13, s13, gs13 = to_up_first(w13, s13, gs13)
    option = CuTileNvfp4Bf16Config()
    view = option.prepare_weights(
        w13,
        s13,
        gs13,
        w2,
        s2,
        gs2,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size=I,
    )
    pack = MoEWeightPack()
    pack.prepare_for("cutile_nvfp4", view)
    cfg = MoEConfig(
        RoutingConfig(E, k),
        QuantConfig(
            weight=QuantFormat.NVFP4,
            activation=QuantFormat.BF16,
            output=QuantFormat.BF16,
        ),
        ExpertConfig(I),
        backend=BackendOptions((option,)),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )
    layer = MoELayer(cfg)
    return lambda x, ids, w: layer(MoEActivationPack(x, None, ids, w), pack)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--intermediate", type=int, default=4096)
    p.add_argument("--experts", type=int, default=32, help="local experts (<= 32)")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    p.add_argument("--routing", choices=["uniform", "skewed"], default="uniform")
    p.add_argument(
        "--post-norm", action="store_true", help="per-route RMSNorm before the sum"
    )
    p.add_argument(
        "--prefill-min-tokens",
        type=int,
        default=None,
        help="large-batch kernel from this token count (default: automatic per shape)",
    )
    p.add_argument("--cupti", action="store_true", help="CUPTI kernel timing")
    p.add_argument(
        "--warm-l2", action="store_true", help="do not flush L2 between iterations"
    )
    p.add_argument("--baseline", choices=["none", "cutile"], default="none")
    a = p.parse_args()

    if not is_sm90a_supported(torch.device("cuda")):
        raise SystemExit("sm90_nvfp4_w4a16_moe needs an SM90 (sm_90a) GPU")
    H, I, E, k = a.hidden, a.intermediate, a.experts, a.top_k
    if not 1 <= E <= 32:
        p.error("--experts must be in [1, 32]")
    if not 1 <= k <= min(E, 8):
        p.error(f"--top-k must be in [1, min(experts, 8)] = [1, {min(E, 8)}]")
    if a.post_norm and a.baseline == "cutile":
        p.error(
            "--post-norm has no cuTile equivalent (per-route RMSNorm before the top-k "
            "sum); the comparison would time different operations"
        )
    ckpt = make_checkpoint(H, I, E)
    packed = prepare_sm90_nvfp4_w4a16_weights(*ckpt)
    ws = get_sm90_nvfp4_w4a16_moe_workspace(torch.device("cuda"), H, I, a.post_norm)
    nw = torch.rand(H, device="cuda") + 0.5 if a.post_norm else None
    base = None
    if a.baseline == "cutile":
        try:
            base = cutile_baseline(ckpt, H, I, E, k, max(a.tokens))
        except Exception as exc:  # report and continue with the kernel alone
            print(f"# cuTile baseline unavailable: {type(exc).__name__}: {exc}")

    print(
        f"# {torch.cuda.get_device_name()}  hidden={H} intermediate={I} experts={E} "
        f"top_k={k} routing={a.routing} post_norm={a.post_norm} "
        f"l2={'warm' if a.warm_l2 else 'cold'} timer={'cupti' if a.cupti else 'events'}"
    )
    print(
        f"{'tokens':>7} {'path':>8} {'us':>10} {'GB/s':>8} {'TFLOP/s':>8}"
        + (f" {'cutile_us':>10} {'speedup':>8}" if base else "")
    )
    pmin = a.prefill_min_tokens
    if pmin is None:
        pmin = sm90_nvfp4_w4a16_moe_prefill_min_tokens(H, I, E, k)
    print(f"# large-batch kernel from {pmin} tokens")
    for M in a.tokens:
        x = torch.randn(M, H, device="cuda").bfloat16()
        ids, w = make_routing(M, E, k, a.routing, seed=M)
        out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)

        def fn():
            sm90_nvfp4_w4a16_moe(
                x, ids, w, packed.w13, packed.w13_global_scale, packed.w2, packed.w2_global_scale,
                post_norm_weight=nw, out=out, workspace=ws, prefill_min_tokens=a.prefill_min_tokens,
            )  # fmt: skip

        fn()
        torch.cuda.synchronize()
        times = bench_gpu_time(fn, enable_cupti=a.cupti, cold_l2_cache=not a.warm_l2)
        us = statistics.median(times) * 1e3
        nbytes = touched_weight_bytes(ids, E, H, I)
        routes = int(((ids >= 0) & (ids < E)).sum())
        flops = 2.0 * routes * (2 * I * H + H * I)
        path = "prefill" if pmin <= M else "decode"
        line = f"{M:>7} {path:>8} {us:>10.2f} {nbytes / us / 1e3:>8.0f} {flops / us / 1e6:>8.1f}"
        if base is not None:
            base(x, ids, w)
            torch.cuda.synchronize()
            bt = (
                statistics.median(
                    bench_gpu_time(
                        lambda: base(x, ids, w),
                        enable_cupti=a.cupti,
                        cold_l2_cache=not a.warm_l2,
                    )
                )
                * 1e3
            )
            line += f" {bt:>10.2f} {bt / us:>8.2f}x"
        print(line)


if __name__ == "__main__":
    main()
