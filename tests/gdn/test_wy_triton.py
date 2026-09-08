"""Validation of the Triton WY kernel against the PyTorch WY reference.

Mirrors the structure of test_wy_kernel_strict.py but exercises the new
`gated_delta_rule_mtp_wy_triton` implementation.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from reference_delta_rule import verify_delta_rule_wy  # noqa: E402
from flashinfer.gdn_kernels.gdn_decode_wy_triton import (  # noqa: E402
    gated_delta_rule_mtp_wy_triton,
)


def run(B=1, T=8, H=16, HV=32, K=128, V=128, seed=42, state_update=False):
    torch.manual_seed(seed)
    device = "cuda"

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    v_ = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=device) * 0.1
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device) * 0.1
    b_t = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    # WY kernel expects [pool, HV, V, K] bf16 (V-major)
    state_vk = torch.randn(B, HV, V, K, dtype=torch.bfloat16, device=device) * 0.01

    scale = 1.0 / math.sqrt(K)

    # Reference state layout is [B, HV, K, V]
    state_kv = state_vk.float().permute(0, 1, 3, 2).contiguous()

    ref_out, ref_state, _ = verify_delta_rule_wy(
        q.float(),
        k.float(),
        v_.float(),
        state_kv.clone(),
        A_log,
        a.float(),
        dt_bias,
        b_t.float(),
        scale_factor=scale,
        use_l2_norm=True,
        state_dtype=torch.float32,
    )

    state_kernel = state_vk.clone()
    tri_out = gated_delta_rule_mtp_wy_triton(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v_,
        b=b_t,
        initial_state_source=state_kernel,
        initial_state_indices=torch.arange(B, dtype=torch.int32, device=device),
        disable_state_update=not state_update,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )

    diff = (tri_out.float() - ref_out.float()).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    ref_max = ref_out.float().abs().max().item()
    rel = max_d / max(ref_max, 1e-10)
    ok_out = max_d < 5e-2 or rel < 5e-2
    print(
        f"  out:   max={max_d:.2e}  mean={mean_d:.2e}  ref_max={ref_max:.2e}  rel={rel:.2e}  {'OK' if ok_out else 'BAD'}"
    )

    ok_state = True
    if state_update:
        # Kernel writes h_new as [B, HV, V, K]; reference returns [B, HV, K, V]
        got_kv = state_kernel.float().permute(0, 1, 3, 2).contiguous()
        sd = (got_kv - ref_state.float()).abs()
        s_max = sd.max().item()
        s_ref = ref_state.float().abs().max().item()
        s_rel = s_max / max(s_ref, 1e-10)
        ok_state = s_max < 5e-2 or s_rel < 5e-2
        print(
            f"  state: max={s_max:.2e}  ref_max={s_ref:.2e}  rel={s_rel:.2e}  {'OK' if ok_state else 'BAD'}"
        )

    return ok_out and ok_state


if __name__ == "__main__":
    configs = [
        {"B": 1, "T": 8, "H": 16, "HV": 32},
        {"B": 4, "T": 8, "H": 16, "HV": 32},
        {"B": 1, "T": 4, "H": 16, "HV": 32},
        {"B": 2, "T": 16, "H": 16, "HV": 32},
        {"B": 1, "T": 8, "H": 16, "HV": 64},
        {"B": 1, "T": 8, "H": 16, "HV": 32, "state_update": True},
    ]
    all_ok = True
    for cfg in configs:
        print(f"\nconfig: {cfg}")
        if not run(**cfg):
            all_ok = False
    print("\n" + ("ALL PASSED" if all_ok else "SOME FAILED"))
    sys.exit(0 if all_ok else 1)
