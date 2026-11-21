"""Frozen-state KDA decode on the experimental tcgen05 / UMMA backend (SM10x).

python examples/experimental/kda_output_only_umma.py
"""

import torch
import torch.nn.functional as F

from flashinfer.kda_kernels.kda_decode_wy_output_only import kda_wy_output_only


def main():
    B, T, H = 8, 4, 12
    dev = "cuda"
    q = torch.randn(B, T, H, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = F.logsigmoid(torch.randn(B, T, H, 128, device=dev)).bfloat16()  # log gate
    beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16)
    pool = (torch.randn(B, H, 128, 128, device=dev) * 0.1).bfloat16()  # [slot, H, V, K]
    idx = torch.arange(B, device=dev, dtype=torch.int32)
    out = kda_wy_output_only(q, k, v, g, beta, pool, idx, backend="umma")
    ref = kda_wy_output_only(q, k, v, g, beta, pool, idx, backend="wy")
    print("max |umma - wy| =", (out.float() - ref.float()).abs().max().item())


if __name__ == "__main__":
    main()
