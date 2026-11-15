"""Compile representative W4A16 grouped-GEMM specializations and record PTX/SASS hashes.

Run once per source tree (control and candidate) with PYTHONPATH pointing at that tree and
CUTE_DSL_KEEP=ptx,cubin. Compare the two receipts with --compare: specializations whose
work unit is one M tile must have identical normalized PTX and SASS.

PTX `.loc`/`.file` debug directives are dropped before hashing: they carry source paths and
line numbers, which differ between trees without changing code.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def normalize_ptx(ptx: str) -> str:
    lines = []
    for line in ptx.splitlines():
        s = line.strip()
        if s.startswith(".loc") or s.startswith(".file") or s.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def normalize_sass(sass: str) -> str:
    keep = []
    for line in sass.splitlines():
        if "code for sm_" in line or "Fatbin" in line or line.strip().startswith("....."):
            continue
        keep.append(re.sub(r"/\*[0-9a-f]{4,}\*/", "", line))
    return "\n".join(keep)


CASES = [
    # (name, kind, hidden, intermediate, experts, top_k, activation, tactic, fused_finalize)
    ("fc1-swiglu-tp8-c11", "fc1", 7168, 256, 256, 8, "swiglu", ((128, 16, 256), (1, 1), True), False),
    ("fc1-swiglu-tp8-2cta", "fc1", 7168, 256, 256, 8, "swiglu", ((256, 128, 256), (2, 1), True), False),
    ("fc1-relu2-n3u-c11", "fc1", 2048, 640, 512, 22, "relu2", ((128, 16, 256), (1, 1), True), False),
    ("fc2-ep-k2048-c11", "fc2", 7168, 2048, 32, 8, None, ((128, 16, 256), (1, 1), True), False),
    ("fc2-v41f-k576-c11", "fc2", 5120, 576, 384, 6, None, ((128, 16, 256), (1, 1), True), False),
    ("fc2-n3u-k640-c21", "fc2", 2048, 640, 512, 22, None, ((128, 16, 256), (2, 1), True), False),
    ("fc2-tp8-k256-2cta", "fc2", 7168, 256, 256, 8, None, ((256, 16, 256), (2, 1), True), False),
    ("fc2-tp8-k256-route128", "fc2", 7168, 256, 256, 8, None, ((128, 128, 256), (1, 1), True), False),
    ("fc2-relu2-finalize-k512", "fc2", 2048, 512, 64, 8, None, ((128, 16, 256), (1, 1), True), True),
    # Grouped in the candidate (recorded, expected to differ):
    ("fc2-tp8-k256-c11-grouped", "fc2", 7168, 256, 256, 8, None, ((128, 16, 256), (1, 1), True), False),
    ("fc2-k3-k384-c11-grouped", "fc2", 3584, 384, 896, 16, None, ((128, 8, 256), (1, 1), True), False),
    ("fc2-tp8-k256-c21-grouped", "fc2", 7168, 256, 256, 8, None, ((128, 16, 256), (2, 1), True), False),
]


def run_cases(out: Path, only=None):
    import cutlass.cute as cute
    import torch

    from flashinfer.fused_moe.cute_dsl.blackwell import moe_w4a16 as launcher
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        allocate_moe_sort_buffers,
        get_max_num_permuted_tokens,
        moe_sort,
    )
    from flashinfer.tllm_enums import ActivationType
    from tests.moe.utils import create_moe_tensors

    out.mkdir(parents=True, exist_ok=True)
    receipt = {"tree": os.environ.get("PYTHONPATH"), "kernels": []}
    original = cute.compile
    current = {}

    def compile_capture(*args, **kwargs):
        kernel = getattr(args[0], "__self__", args[0])
        compiled = original(*args, **kwargs)
        if type(kernel).__name__ != "Sm100W4A16GroupedGemmKernel":
            return compiled
        stem = f"{current['name']}-{'fc1' if kernel.gather_input else 'fc2'}"
        ptx = compiled.__ptx__
        cubin = compiled.__cubin__
        if isinstance(cubin, str):
            cubin = Path(cubin).read_bytes()
        (out / f"{stem}.ptx").write_text(ptx)
        (out / f"{stem}.cubin").write_bytes(cubin)
        sass = subprocess.check_output(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(out / f"{stem}.cubin")],
            text=True,
        )
        (out / f"{stem}.sass").write_text(sass)
        res = subprocess.check_output(
            ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(out / f"{stem}.cubin")],
            text=True,
        )
        receipt["kernels"].append(
            dict(
                stem=stem,
                case=current["name"],
                m_tiles_per_unit=getattr(kernel, "m_tiles_per_unit", 1),
                stages=dict(
                    load=kernel.num_load2trans_stage,
                    trans=kernel.num_trans2mma_stage,
                    acc=kernel.num_acc_stage,
                    c=kernel.num_c_stage,
                    tile_info=kernel.num_tile_info_stage,
                    acc_cols=kernel.num_acc_tmem_cols,
                    a_cols=kernel.num_a_tmem_cols,
                ),
                ptx_sha256=hashlib.sha256(normalize_ptx(ptx).encode()).hexdigest(),
                sass_sha256=hashlib.sha256(normalize_sass(sass).encode()).hexdigest(),
                resources=" ".join(res.split()),
            )
        )
        return compiled

    cute.compile = compile_capture
    try:
        for name, kind, hidden, inter, experts, top_k, act, tactic, finalize in CASES:
            if only and name not in only:
                continue
            current["name"] = name
            launcher._kernel_cache.clear()
            gated = act == "swiglu"
            num_tokens = 64
            route_tile = tactic[0][1]
            t = create_moe_tensors(
                num_tokens=num_tokens,
                hidden_size=hidden,
                intermediate_size=inter,
                num_experts=experts,
                num_local_experts=experts,
                top_k=top_k,
                gated=gated or kind == "fc2",
                swap_ab=False,
            )
            bufs = allocate_moe_sort_buffers(
                num_tokens=num_tokens,
                num_experts=experts,
                top_k=top_k,
                num_local_experts=experts,
                tile_tokens_dim=route_tile,
                device="cuda",
            )
            (t2e, t2l, e2p, p2e, _, nne) = moe_sort(
                token_selected_experts=t["token_selected_experts"],
                token_final_scales=t["token_final_scales"],
                num_experts=experts,
                top_k=top_k,
                local_expert_offset=0,
                num_local_experts=experts,
                tile_tokens_dim=route_tile,
                enable_pdl=True,
                **bufs,
            )
            slots = get_max_num_permuted_tokens(num_tokens, top_k, experts, route_tile)
            common = dict(
                tile_idx_to_expert_idx=t2e,
                tile_idx_to_mn_limit=t2l,
                num_non_exiting_tiles=nne,
                num_local_experts=experts,
                swiglu_alpha=1.0,
                swiglu_beta=0.0,
                swiglu_limit=float("inf"),
                situ_beta=None,
                situ_linear_beta=None,
                enable_pdl=True,
                tactic=tactic,
                swap_ab=False,
            )
            if kind == "fc1":
                w = t["w1_weight"]
                out_rows = int(w.size(1)) // (2 if gated else 1)
                inter_buf = torch.empty((slots, out_rows), dtype=torch.bfloat16, device="cuda")
                launcher._run_grouped_gemm(
                    weight=w, weight_sf=t["w1_weight_sf"], activations=t["x_bf16"],
                    alpha=t["w1_alpha"], output=inter_buf,
                    activation_type=ActivationType.Swiglu if gated else ActivationType.Relu2,
                    use_fused_finalize=False, permuted_idx_to_expanded_idx=p2e,
                    token_final_scales=None, gather_input=True, gather_top_k=top_k, **common,
                )
            else:
                act_in = torch.randn((slots, inter), dtype=torch.bfloat16, device="cuda") / 8
                if finalize:
                    out_buf = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
                else:
                    out_buf = torch.empty((slots, hidden), dtype=torch.bfloat16, device="cuda")
                launcher._run_grouped_gemm(
                    weight=t["w2_weight"], weight_sf=t["w2_weight_sf"], activations=act_in,
                    alpha=t["w2_alpha"], output=out_buf, activation_type=None,
                    use_fused_finalize=finalize,
                    permuted_idx_to_expanded_idx=p2e if finalize else None,
                    token_final_scales=t["token_final_scales"] if finalize else None,
                    gather_input=False, gather_top_k=1, **common,
                )
            torch.cuda.synchronize()
            print("compiled", name, flush=True)
    finally:
        cute.compile = original
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def compare(a_path: Path, b_path: Path):
    a = {k["stem"]: k for k in json.loads(a_path.read_text())["kernels"]}
    b = {k["stem"]: k for k in json.loads(b_path.read_text())["kernels"]}
    ok = True
    for stem in sorted(set(a) | set(b)):
        ka, kb = a.get(stem), b.get(stem)
        if ka is None or kb is None:
            print(f"MISSING {stem}: control={ka is not None} candidate={kb is not None}")
            ok = False
            continue
        same_ptx = ka["ptx_sha256"] == kb["ptx_sha256"]
        same_sass = ka["sass_sha256"] == kb["sass_sha256"]
        grouped = kb.get("m_tiles_per_unit", 1) > 1
        status = "identical" if same_ptx and same_sass else "DIFFERENT"
        print(
            f"{stem:40s} units={kb.get('m_tiles_per_unit', 1)} ptx={'=' if same_ptx else 'x'} "
            f"sass={'=' if same_sass else 'x'} {status}{' (grouped: expected)' if grouped else ''}"
        )
        print(f"    control:   {ka['resources']}")
        print(f"    candidate: {kb['resources']}")
        if not grouped and not (same_ptx and same_sass):
            ok = False
    print("RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--only", nargs="*")
    p.add_argument("--compare", nargs=2, type=Path)
    args = p.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    run_cases(args.out, args.only)
