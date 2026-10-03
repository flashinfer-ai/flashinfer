"""Verify-only vcache sibling: outputs == flush kernel w/ never-flush sentinel;
state bit-unchanged; hist untouched; drafts appended at [P, P+T).

Run: pytest tests/gdn/test_decode_vcache_verify.py -v
     (or standalone: python tests/gdn/test_decode_vcache_verify.py)
"""
import importlib.util, math, os, sys, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_vcache import (  # noqa: E402
    gated_delta_rule_mtp_vcache as vver,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_vcache_flush import (  # noqa: E402
    ST_TORCH,
    gated_delta_rule_mtp_vcache_flush as vflush,
)

H = HK = 16
HV = 64
K = V = 128
W = 16
RING = 32
T = 4
SCALE = 1 / math.sqrt(K)
B = 8


def _run_check():
    torch.set_grad_enabled(False)
    dev = "cuda"
    torch.manual_seed(3)
    kc = torch.randn(B, HK, RING, K, dtype=torch.bfloat16, device=dev)
    vc = torch.randn(B, HV, RING, V, dtype=torch.bfloat16, device=dev)
    ac = torch.randn(B, HV, RING, dtype=torch.float32, device=dev) * 0.1
    bc = torch.randn(B, HV, RING, dtype=torch.float32, device=dev)
    hist = torch.tensor([0, 1, 3, 5, 8, 10, 11, 12], dtype=torch.int32, device=dev)
    # near-wrap bases: several windows/appends cross the 31 -> 0 boundary
    base = torch.tensor([0, 5, 28, 30, 12, 25, 31, 22], dtype=torch.int32, device=dev)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, HK, K, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev) * 0.1
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=dev) * 0.1
    S0 = torch.randn(B, HV, V, K, dtype=ST_TORCH, device=dev) * 0.1
    idx = torch.arange(B, dtype=torch.int32, device=dev)
    S0_in = S0.clone()
    hist_in = hist.clone()
    base_in = base.clone()
    kc2, vc2, ac2, bc2 = kc.clone(), vc.clone(), ac.clone(), bc.clone()
    o1 = vver(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
        initial_state_source=S0, initial_state_indices=idx,
        k_cache=kc, v_cache=vc, a_cache=ac, b_cache=bc,
        hist_len=hist, cache_base=base, scale=SCALE,
    )
    torch.cuda.synchronize()
    # reference: flush wrapper with the same never-flush sentinel, cloned buffers
    S0b = S0_in.clone()
    histb = hist_in.clone()
    o2 = vflush(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
        initial_state_source=S0b, initial_state_indices=idx,
        k_cache=kc2, v_cache=vc2, a_cache=ac2, b_cache=bc2,
        hist_len=histb, cache_base=base_in.clone(), flush_min=W - T + 1,
        restart_hist_on_flush=False, scale=SCALE,
    )
    torch.cuda.synchronize()
    Pl = hist_in.long()
    app_ok = True
    for bi in range(B):
        P = int(Pl[bi])
        wr = [(int(base_in[bi]) + P + j) % RING for j in range(T)]
        app_ok &= torch.equal(kc[bi][:, wr, :], k[bi].transpose(0, 1))
        app_ok &= torch.equal(vc[bi][:, wr, :], v[bi].transpose(0, 1))
    print(
        "out==flush-sentinel:", torch.equal(o1, o2),
        "| state unchanged:", torch.equal(S0, S0_in),
        "| hist untouched:", torch.equal(hist, hist_in),
        "| base untouched:", torch.equal(base, base_in),
        "| ring append at [P,P+T):", bool(app_ok),
    )
    assert torch.equal(o1, o2), "verify-only outputs != flush kernel sentinel path"
    assert torch.equal(S0, S0_in), "verify-only must not write the state pool"
    assert torch.equal(hist, hist_in), "verify-only must not reset hist_len"
    assert torch.equal(base, base_in), "verify-only must not slide cache_base"
    assert app_ok, "drafts must append at (base+P+s) & RING_MASK"
    print("VERIFY SIBLING PASS")


def test_vcache_verify_sibling():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SM90+ required")
    _run_check()


if __name__ == "__main__":
    _run_check()
