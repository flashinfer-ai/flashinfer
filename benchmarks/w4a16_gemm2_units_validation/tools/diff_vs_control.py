"""Cross-tree differential check of the whole W4A16 MoE at model dimensions.

`--save OUT.pt` runs launch_w4a16_moe for every case in the tree on PYTHONPATH and stores the
final [tokens, hidden] outputs (order-invariant: moe_sort's within-expert order does not
reach the unpermuted output). `--compare A.pt B.pt` requires bit equality (int16 views).
`--oracle` additionally checks each saved output against the independent W4A16 oracle.
"""

import argparse
import sys

import torch

PRESETS = {
    # name: hidden, per-rank intermediate, local experts, top_k, activation kwargs
    "v3-tp8": (7168, 256, 256, 8, dict(act="swiglu")),
    "glm5-tp8": (6144, 256, 256, 8, dict(act="swiglu")),
    "k3-tp8": (3584, 384, 896, 16, dict(act="situ", situ_beta=4.0, situ_linear_beta=25.0)),
    "v41f-tp4": (5120, 576, 384, 6, dict(act="swiglu", swiglu_limit=10.0)),
    "n3u-tp8": (2048, 640, 512, 22, dict(act="relu2")),
    "v3-tp4": (7168, 512, 256, 8, dict(act="swiglu")),
}
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
TACTICS = {
    "c11-r8": ((128, 8, 256), (1, 1), True),
    "c11-r16": ((128, 16, 256), (1, 1), True),
    "c21-r16": ((128, 16, 256), (2, 1), True),
    "c11-r32": ((128, 32, 256), (1, 1), True),
    "c11-r64": ((128, 64, 256), (1, 1), True),
    "2cta-r32": ((256, 32, 256), (2, 1), True),
}


def run(save_path, oracle):
    from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import launch_w4a16_moe
    from flashinfer.tllm_enums import ActivationType
    from tests.moe.utils import create_moe_tensors

    if oracle:
        from tests.moe_ep.test_helpers.w4a16_reference import (
            assert_w4a16_bits,
            split_w4a16_reference_from_tensors,
        )
    results = {}
    for preset, (hidden, inter, experts, top_k, act) in PRESETS.items():
        gated = act["act"] != "relu2"
        for num_tokens in TOKENS:
            tensors = create_moe_tensors(
                num_tokens=num_tokens,
                hidden_size=hidden,
                intermediate_size=inter,
                num_experts=experts,
                num_local_experts=experts,
                top_k=top_k,
                gated=gated,
                seed=1000 + num_tokens,
            )
            activation = {
                "swiglu": ActivationType.Swiglu,
                "situ": ActivationType.Swiglu,
                "relu2": ActivationType.Relu2,
            }[act["act"]]
            kwargs = {}
            if "swiglu_limit" in act:
                kwargs["swiglu_limit"] = act["swiglu_limit"]
            if act["act"] == "situ":
                kwargs.update(situ_beta=act["situ_beta"], situ_linear_beta=act["situ_linear_beta"])
            for tname, gemm in TACTICS.items():
                out = launch_w4a16_moe(
                    x=tensors["x_bf16"],
                    token_selected_experts=tensors["token_selected_experts"],
                    token_final_scales=tensors["token_final_scales"],
                    w1_weight=tensors["w1_weight"],
                    w1_weight_sf=tensors["w1_weight_sf"],
                    w1_alpha=tensors["w1_alpha"],
                    w2_weight=tensors["w2_weight"],
                    w2_weight_sf=tensors["w2_weight_sf"],
                    w2_alpha=tensors["w2_alpha"],
                    num_experts=experts,
                    num_local_experts=experts,
                    local_expert_offset=0,
                    moe_output=torch.empty(
                        (num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
                    ),
                    use_fused_finalize=False,
                    enable_pdl=True,
                    activation_type=activation,
                    tactic=(gemm, gemm),
                    **kwargs,
                )
                torch.cuda.synchronize()
                key = f"{preset}/t{num_tokens}/{tname}"
                results[key] = out.cpu()
                if oracle and num_tokens in (8, 64) and tname in ("c11-r16", "c21-r16"):
                    ref_kwargs = {k: v for k, v in kwargs.items()}
                    reference = split_w4a16_reference_from_tensors(
                        tensors, activation_type=activation, **ref_kwargs
                    )
                    assert_w4a16_bits(out, reference)
                    print("oracle ok", key, flush=True)
            print("done", preset, num_tokens, flush=True)
            del tensors
            torch.cuda.empty_cache()
    torch.save(results, save_path)


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    assert set(a) == set(b), (sorted(set(a) ^ set(b)))[:10]
    bad = [k for k in a if not torch.equal(a[k].view(torch.int16), b[k].view(torch.int16))]
    finite = all(torch.isfinite(v.float()).all() for v in a.values())
    print(f"cases={len(a)} mismatches={len(bad)} finite={finite}")
    for k in bad[:20]:
        print("MISMATCH", k)
    return not bad and finite


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--compare", nargs=2)
    args = p.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    run(args.save, args.oracle)
