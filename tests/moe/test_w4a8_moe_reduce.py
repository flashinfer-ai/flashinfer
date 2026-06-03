# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness test for the Triton fused MoE finalize (moe_reduce + build_reduce_index)."""

import pytest
import torch

try:
    from flashinfer.fused_moe.cute_dsl.moe_reduce_triton import (
        build_reduce_index,
        moe_reduce,
    )

    _OK = True
except Exception:
    _OK = False


def _ref(route, wts, c_all, num_tokens, H):
    all_tok = torch.cat([r.long() for r in route])
    all_w = torch.cat([w.float() for w in wts]).unsqueeze(1)
    ref = torch.zeros(num_tokens, H, dtype=torch.float32, device=c_all.device)
    ref.index_add_(0, all_tok, all_w * c_all.float())
    return ref


@pytest.mark.skipif(
    not _OK or not torch.cuda.is_available(), reason="needs triton+CUDA"
)
@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("H", [512, 4096])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_moe_reduce(top_k, H, out_dtype):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    E, M_TOK = 8, 1024
    tok = torch.arange(M_TOK, device=dev)
    # balanced: token t -> experts (t+s)%E for s in range(top_k); each token exactly top_k.
    route = [
        torch.cat([tok[(tok + s) % E == e] for s in range(top_k)]).to(torch.int32)
        for e in range(E)
    ]
    wts = [torch.rand(route[e].numel(), device=dev) + 0.5 for e in range(E)]
    num_permuted = sum(r.numel() for r in route)
    c_all = (torch.randn(num_permuted, H, device=dev) / 4).to(out_dtype)

    out = torch.empty(M_TOK, H, device=dev, dtype=out_dtype)
    idx, scales = build_reduce_index(route, wts, M_TOK, top_k)
    moe_reduce(c_all, out, idx, scales, top_k)

    ref = _ref(route, wts, c_all, M_TOK, H)
    rel = (out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-3)
    assert rel < 0.03, (
        f"max rel diff {rel:.4f} too high (top_k={top_k} H={H} {out_dtype})"
    )


if __name__ == "__main__":
    for tk in (1, 2, 3):
        for H in (512, 4096):
            for dt in (torch.float16, torch.bfloat16):
                test_moe_reduce(tk, H, dt)
                print(f"PASS top_k={tk} H={H} {dt}")
    print("ALL PASS")
