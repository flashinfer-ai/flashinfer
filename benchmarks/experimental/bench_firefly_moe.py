"""Benchmark the experimental Firefly backend against the other NVFP4 MoE backends.

Every backend runs through its own ``MoELayer`` (one candidate each) on the same canonical BF16
experts and the same NVFP4 activations, per-token (``nvfp4_quantize_per_token_cute_dsl``) or, with
``--per-tensor``, per-tensor (``TrtllmFp4Config.prepare_activations``: unit global scale), with
pre-routed sigmoid top-k routing. The FlashInfer backends are autotuned for each batch size first.
Times are the median GPU time of ``bench_gpu_time`` (CUPTI, CUDA graph, cold L2).

The pre-routed ids and weights reach each backend in its cheapest supported mode: ``UnpackedPrecomputed``
with BF16 weights where the runner accepts it (firefly, prims_ts, trtllm), else ``PackedPrecomputed`` with
FP32 weights (cute_dsl). Packing them for ``PackedPrecomputed`` would add a few PyTorch kernels to the
trtllm and prims_ts calls (about 7-11 us on B200), timed with them.

    python benchmarks/experimental/bench_firefly_moe.py --num-tokens 1,16,256,4096
    python benchmarks/experimental/bench_firefly_moe.py --hidden-size 7168 \\
        --intermediate-size 2048 --num-experts 256 --top-k 8 --backends firefly,prims_ts
    python benchmarks/experimental/bench_firefly_moe.py --per-tensor
"""

import argparse
import inspect
import warnings

import numpy as np
import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CuteDslConfig,
    ExecutionConfig,
    ExpertConfig,
    FireflyNvfp4Config,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    PrimsTsConfig,
    QuantConfig,
    QuantFormat,
    ReLU2,
    RoutingConfig,
    RoutingInputMode,
    SwiGLU,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.quantization import nvfp4_quantize_per_token_cute_dsl
from flashinfer.testing import bench_gpu_time

BACKENDS = {
    "firefly": (FireflyNvfp4Config(), "firefly_nvfp4"),
    "prims_ts": (PrimsTsConfig(), "prims_ts"),
    "cute_dsl": (CuteDslConfig(), "cute_dsl"),
    "trtllm": (TrtllmFp4Config(), "trtllm_fp4_routed"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--intermediate-size", type=int, default=768)
    p.add_argument("--num-tokens", default="1,4,16,64,256,1024,4096")
    p.add_argument("--activation", choices=("swiglu", "relu2"), default="swiglu")
    p.add_argument("--fused-finalize", action="store_true")
    p.add_argument(
        "--per-tensor",
        action="store_true",
        help="per-tensor activations (QuantConfig(per_token_scale=False))",
    )
    p.add_argument("--backends", default="firefly,prims_ts,cute_dsl,trtllm")
    args = p.parse_args()

    num_experts, top_k = args.num_experts, args.top_k
    hidden, inter = args.hidden_size, args.intermediate_size
    ms = [int(m) for m in args.num_tokens.split(",")]
    act = SwiGLU() if args.activation == "swiglu" else ReLU2()
    device = torch.device("cuda")
    torch.manual_seed(0)
    rows = 2 * inter if act.is_gated else inter
    w1 = (torch.randn(num_experts, rows, hidden, device=device) * 0.02).bfloat16()
    w2 = (torch.randn(num_experts, hidden, inter, device=device) * 0.02).bfloat16()

    quant = QuantConfig(
        weight=QuantFormat.NVFP4,
        activation=QuantFormat.NVFP4,
        per_token_scale=not args.per_tensor,
    )
    layers = {}
    unpacked = set()  # backends taking UnpackedPrecomputed routing
    for name in args.backends.split(","):
        cfg, key = BACKENDS[name]
        kw = dict(
            num_local_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=inter,
            activation=act,
        )
        if "quant" in inspect.signature(cfg.prepare_weights).parameters:
            kw["quant"] = QuantConfig(
                weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4
            )
        modes = getattr(_BACKEND_RUNNERS[type(cfg)], "supported_routing_modes", ())
        if RoutingInputMode.UnpackedPrecomputed in modes:
            unpacked.add(name)
        pack = MoEWeightPack()
        pack.prepare_for(key, cfg.prepare_weights(w1, w2, **kw))
        config = MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=quant,
            experts=ExpertConfig(intermediate_size=inter),
            activation=act,
            finalize=MoEFinalizeConfig(use_fused_finalize=args.fused_finalize),
            execution=ExecutionConfig(tune_max_num_tokens=max(ms)),
            backend=BackendOptions(candidates=(cfg,)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layers[name] = (MoELayer(config, device=device), pack)

    gsi = torch.tensor([1.0 / 2688.0], device=device)
    print(
        f"E={num_experts} top_k={top_k} H={hidden} I={inter} {args.activation}"
        f"{' fused finalize' if args.fused_finalize else ''}"
        f"{' per-tensor' if args.per_tensor else ' per-token'}; median us (CUPTI, cold L2)"
    )
    print(f"{'M':>6} " + " ".join(f"{n:>12}" for n in layers))
    for m in ms:
        x = torch.randn(m, hidden, device=device, dtype=torch.bfloat16)
        if args.per_tensor:
            q, sf = TrtllmFp4Config.prepare_activations(x, quant=quant)
            gs = None
        else:
            q, sf, gs = nvfp4_quantize_per_token_cute_dsl(x, gsi, sf_layout=2)
            sf, gs = sf.view(torch.float8_e4m3fn).reshape(m, -1), gs.reshape(-1)
        logits = torch.randn(m, num_experts, device=device)
        wts, ids = torch.sigmoid(logits).topk(top_k, dim=-1)
        ids = ids.int()
        packed = MoEActivationPack(
            hidden_states_q=q,
            hidden_states_scale=sf,
            topk_ids=ids,
            topk_weights=wts,
            per_token_scale=gs,
        )
        unpacked_pack = MoEActivationPack(
            hidden_states_q=q,
            hidden_states_scale=sf,
            topk_ids=ids,
            topk_weights=wts.bfloat16(),
            per_token_scale=gs,
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        )
        times = []
        for name, (layer, pack) in layers.items():
            act_pack = unpacked_pack if name in unpacked else packed
            with autotune(True):
                layer(act_pack, pack)
            torch.cuda.synchronize()
            t = bench_gpu_time(
                lambda layer=layer, pack=pack, act_pack=act_pack: layer(act_pack, pack),
                enable_cupti=True,
                use_cuda_graph=True,
                cold_l2_cache=True,
            )
            times.append(float(np.median(t)) * 1e3)
        print(f"{m:>6} " + " ".join(f"{t:>12.1f}" for t in times), flush=True)


if __name__ == "__main__":
    main()
