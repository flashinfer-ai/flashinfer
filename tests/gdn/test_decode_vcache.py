"""Correctness: gated_delta_rule_mtp_vcache_flush vs fp32 per-request recurrence.
Draft outputs (at window rows [P:P+T]) and folded state (flushers -> fold of the
P committed ring rows; verifiers -> unchanged) vs decode_delta_rule, per request."""
import importlib.util, math, os, sys, torch
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO,"tests","gdn"))
from reference_delta_rule import decode_delta_rule
def _load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
_vcf=_load("vcf",os.path.join(REPO,"flashinfer/gdn_kernels/gdn_decode_bf16_wy_vcache_flush.py"))
vflush=_vcf.gated_delta_rule_mtp_vcache_flush
ST_TORCH=_vcf.ST_TORCH  # state dtype (GDN_VCACHE_STATE_DTYPE: bf16 default / fp16)
torch.set_grad_enabled(False);dev="cuda"
H=HK=16;HV=64;K=V=128;W=16;T=4;RING=32;SCALE=1/math.sqrt(K)
REGIMES={"mild":(0.1,0.1,0.1,1.0),"strong":(1.0,1.0,1.0,1.0),"big":(0.1,0.1,0.1,30.0)}

def replay_req(bi,P,base0,S0b,kc,vc,ac,bc,q,k,v,a,b,A_log,dt_bias,sd):
    """fp32/bf16 recurrence for one request over [committed 0:P | draft 0:T]; returns
    (draft_outs[T,HV,V], state_after_committed[HV,V,K])."""
    st=S0b.transpose(-2,-1).contiguous().to(sd).unsqueeze(0)  # [1,HV,K,V]
    outs=[];state_at_P=None
    for t in range(P+T):
        if t<P:
            rt=(base0+t)%RING
            kt=kc[:,rt,:].float()[None];vt=vc[:,rt,:].float()[None]
            at=ac[:,rt].float()[None];bt=bc[:,rt].float()[None]  # rings are fp32; kernel reads them fp32
            qt=torch.zeros(1,H,K,device=dev)
        else:
            j=t-P
            kt=k[j].float()[None];vt=v[j].float()[None]
            at=a[j].to(torch.bfloat16).float()[None];bt=b[j].to(torch.bfloat16).float()[None]
            qt=q[j].float()[None]
        if t==P: state_at_P=st.clone()
        o,st=decode_delta_rule(qt,kt,vt,st,A_log=A_log,a=at,dt_bias=dt_bias,b=bt,scale_factor=SCALE,
            softplus_beta=1.0,softplus_threshold=20.0,use_l2_norm=True,state_dtype=sd)
        if t>=P: outs.append(o[0].float())
    if state_at_P is None: state_at_P=st.clone()  # P+T==... (P>=... ) safety
    return torch.stack(outs,0), state_at_P[0].transpose(-2,-1).contiguous().float()

def run(B,P_list,flush_min,regime,seed=0,base_off=0):
    sA,sdt,sa,sS=REGIMES[regime];torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    pool=B
    kc=torch.randn(pool,HK,RING,K,dtype=torch.bfloat16,device=dev)
    vc=torch.randn(pool,HV,RING,V,dtype=torch.bfloat16,device=dev)
    ac=torch.randn(pool,HV,RING,dtype=torch.float32,device=dev)*sa
    bc=torch.randn(pool,HV,RING,dtype=torch.float32,device=dev)
    hist=torch.tensor(P_list,dtype=torch.int32,device=dev)
    # per-request ring window origins, incl. NEAR-WRAP bases so the live
    # window and the appends exercise the & RING_MASK addressing
    base=torch.tensor([(7*i+base_off)%RING for i in range(B)],dtype=torch.int32,device=dev)
    q=torch.randn(B,T,H,K,dtype=torch.bfloat16,device=dev);k=torch.randn(B,T,HK,K,dtype=torch.bfloat16,device=dev)
    v=torch.randn(B,T,HV,V,dtype=torch.bfloat16,device=dev);a=torch.randn(B,T,HV,dtype=torch.bfloat16,device=dev)*sa
    b=torch.randn(B,T,HV,dtype=torch.bfloat16,device=dev)
    A_log=torch.randn(HV,dtype=torch.float32,device=dev)*sA;dt_bias=torch.randn(HV,dtype=torch.float32,device=dev)*sdt
    S0=torch.randn(B,HV,V,K,dtype=ST_TORCH,device=dev)*sS;idx=torch.arange(B,dtype=torch.int32,device=dev)
    S0_in=S0.clone()
    kc2,vc2,ac2,bc2=kc.clone(),vc.clone(),ac.clone(),bc.clone()
    hist2,base2=hist.clone(),base.clone()
    out=vflush(A_log=A_log,a=a,dt_bias=dt_bias,q=q,k=k,v=v,b=b,initial_state_source=S0,initial_state_indices=idx,
        k_cache=kc2,v_cache=vc2,a_cache=ac2,b_cache=bc2,
        hist_len=hist2,cache_base=base2,flush_min=flush_min,restart_hist_on_flush=True,
        use_qk_l2norm_in_kernel=True,scale=SCALE)
    torch.cuda.synchronize()
    # ring property checks (per request): appends PAST the window at
    # (base+P+j)&31; committed rows UNCHANGED (a flush never overwrites);
    # cursor commit: flushers base'=(base+P)&31,len'=0; verifiers unchanged.
    ring_ok=True
    for bi in range(B):
        P=int(hist[bi]);b0=int(base[bi])
        wr=[(b0+P+j)%RING for j in range(T)]
        ring_ok &= torch.equal(kc2[bi][:,wr,:], k[bi].transpose(0,1))
        ring_ok &= torch.equal(vc2[bi][:,wr,:], v[bi].transpose(0,1))
        ring_ok &= torch.equal(ac2[bi][:,wr], a[bi].float().t())
        ring_ok &= torch.equal(bc2[bi][:,wr], b[bi].float().t())
        cm=[(b0+i)%RING for i in range(P)]
        if cm:
            ring_ok &= torch.equal(kc2[bi][:,cm,:], kc[bi][:,cm,:])
            ring_ok &= torch.equal(vc2[bi][:,cm,:], vc[bi][:,cm,:])
        fl=P>=flush_min
        exp_b=(b0+P)%RING if fl else b0
        exp_h=0 if fl else P
        ring_ok &= int(base2[bi])==exp_b and int(hist2[bi])==exp_h
    is_flush=hist>=flush_min
    e_out=e_out_seq=e_state=e_state_seq=out_mag=0.0;ver_ok=True
    for bi in range(B):
        P=int(hist[bi])
        o32,s32=replay_req(bi,P,int(base[bi]),S0_in[bi],kc[bi],vc[bi],ac[bi],bc[bi],q[bi],k[bi],v[bi],a[bi],b[bi],A_log,dt_bias,torch.float32)
        o16,s16=replay_req(bi,P,int(base[bi]),S0_in[bi],kc[bi],vc[bi],ac[bi],bc[bi],q[bi],k[bi],v[bi],a[bi],b[bi],A_log,dt_bias,torch.bfloat16)
        e_out=max(e_out,(out[bi].float()-o32).abs().max().item());out_mag=max(out_mag,o32.abs().max().item())
        e_out_seq=max(e_out_seq,(o16-o32).abs().max().item())
        exp_s = s32 if is_flush[bi] else S0_in[bi].float()
        e_state=max(e_state,(S0[bi].float()-exp_s).abs().max().item())
        if is_flush[bi]: e_state_seq=max(e_state_seq,(s16-s32).abs().max().item())
        if not is_flush[bi]: ver_ok=ver_ok and torch.equal(S0[bi],S0_in[bi])
    obar=max(8e-3,2.0*e_out_seq);sbar=max(2e-2*max(sS,1.0),2.0*e_state_seq)
    rel_out=e_out/max(out_mag,1e-6)
    ok=(e_out<=obar or rel_out<=1e-2) and e_state<=sbar and ver_ok and ring_ok
    print(f"[{regime:>6} B={B} fmin={flush_min} P={P_list}] out {e_out:.2e}/{obar:.1e} rel{rel_out:.1e} state {e_state:.2e}/{sbar:.1e} ver={ver_ok} ring={ring_ok}{'' if ok else '  <-- FAIL'}",flush=True)
    return 0 if ok else 1

def test_vcache_fp16_state_commits_more_precisely_than_bf16():
    """The point of GDN_VCACHE_STATE_DTYPE=fp16 (PR #4081 parity): the fp16
    checkpoint (10 mantissa bits) folds closer to the fp32 truth than bf16
    (7 bits). Loads one module instance per state dtype (env is read at
    import) and compares the same fold on the same inputs."""
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SM90+ required")
    kp = os.path.join(REPO, "flashinfer/gdn_kernels/gdn_decode_bf16_wy_vcache_flush.py")
    arms = {}
    for name, env in (("bf16", ""), ("fp16", "fp16")):
        os.environ["GDN_VCACHE_STATE_DTYPE"] = env
        arms[name] = _load(f"vcf_{name}", kp)
    os.environ.pop("GDN_VCACHE_STATE_DTYPE", None)
    errs = {"bf16": 0.0, "fp16": 0.0}
    for seed in (7, 8, 9, 10):
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        B = 4; P = 12
        kc = torch.randn(B, HK, RING, K, dtype=torch.bfloat16, device="cuda")
        vc = torch.randn(B, HV, RING, V, dtype=torch.bfloat16, device="cuda")
        ac = torch.randn(B, HV, RING, dtype=torch.float32, device="cuda") * 0.1
        bc = torch.randn(B, HV, RING, dtype=torch.float32, device="cuda")
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, T, HK, K, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda") * 0.1
        b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda")
        A_log = torch.randn(HV, dtype=torch.float32, device="cuda") * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda") * 0.1
        S0f = torch.randn(B, HV, V, K, dtype=torch.float32, device="cuda") * 0.1
        idx = torch.arange(B, dtype=torch.int32, device="cuda")
        hist = torch.full((B,), P, dtype=torch.int32, device="cuda")
        base = torch.zeros(B, dtype=torch.int32, device="cuda")
        # fp32 truth of the fold (committed rows only; both arms fold the same rows)
        _, s32 = replay_req(0, P, 0, S0f[0].to(torch.bfloat16), kc[0], vc[0], ac[0],
                            bc[0], q[0], k[0], v[0], a[0], b[0], A_log, dt_bias,
                            torch.float32)
        for name, m2 in arms.items():
            Sarm = S0f.to(m2.ST_TORCH).clone()
            s32_in = S0f.to(m2.ST_TORCH)  # same start point per arm dtype
            m2.gated_delta_rule_mtp_vcache_flush(
                A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
                initial_state_source=Sarm, initial_state_indices=idx,
                k_cache=kc.clone(), v_cache=vc.clone(), a_cache=ac.clone(),
                b_cache=bc.clone(), hist_len=hist.clone(),
                cache_base=base.clone(), flush_min=P,
                restart_hist_on_flush=False, scale=SCALE)
            torch.cuda.synchronize()
            _, sref = replay_req(0, P, 0, s32_in[0], kc[0], vc[0], ac[0], bc[0],
                                 q[0], k[0], v[0], a[0], b[0], A_log, dt_bias,
                                 torch.float32)
            errs[name] += (Sarm[0].float() - sref).abs().mean().item()
    print(f"mean committed-state err: bf16 {errs['bf16']:.3e}  fp16 {errs['fp16']:.3e}")
    assert errs["fp16"] < errs["bf16"], (
        f"fp16 state should commit more precisely: {errs}"
    )


def test_vcache_flush_matches_fp32_reference():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cc=torch.cuda.get_device_capability()
    if cc[0]<9: pytest.skip("SM90+ required")
    fails=0
    for rn in REGIMES:
        fails+=run(4,[0,10,5,12],9,rn,0)
        fails+=run(4,[12,12,12,12],12,rn,1)
        fails+=run(4,[0,3,7,11],12,rn,2)
        fails+=run(8,[0,1,2,3,9,10,11,12],9,rn,3)
        fails+=run(8,[12,5,12,8,12,3,12,11],9,rn,4,base_off=26)  # wrapped windows
    assert fails==0, f"{fails} vcache correctness cases failed"

if __name__ == "__main__":
    fails=0;print("GPU:",torch.cuda.get_device_name())
    for rn in REGIMES:
        fails+=run(4,[0,10,5,12],9,rn,0)
        fails+=run(4,[12,12,12,12],12,rn,1)
        fails+=run(4,[0,3,7,11],12,rn,2)
        fails+=run(8,[0,1,2,3,9,10,11,12],9,rn,3)
        fails+=run(8,[12,5,12,8,12,3,12,11],9,rn,4,base_off=26)  # wrapped windows
    print("ALL PASS" if fails==0 else f"{fails} FAIL")
    sys.exit(1 if fails else 0)
