"""Correctness: gated_delta_rule_mtp_vcache_flush vs fp32 per-request recurrence.
Draft outputs (at window rows [P:P+T]) and folded state (flushers -> fold of the
P committed ring rows; verifiers -> unchanged) vs decode_delta_rule, per request."""
import importlib.util, math, os, sys, torch
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO,"tests","gdn"))
from reference_delta_rule import decode_delta_rule
def _load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
vflush=_load("vcf",os.path.join(REPO,"flashinfer/gdn_kernels/gdn_decode_bf16_wy_vcache_flush.py")).gated_delta_rule_mtp_vcache_flush
torch.set_grad_enabled(False);dev="cuda"
H=HK=16;HV=64;K=V=128;W=16;T=4;SCALE=1/math.sqrt(K)
REGIMES={"mild":(0.1,0.1,0.1,1.0),"strong":(1.0,1.0,1.0,1.0),"big":(0.1,0.1,0.1,30.0)}

def replay_req(bi,P,S0b,kc,vc,ac,bc,q,k,v,a,b,A_log,dt_bias,sd):
    """fp32/bf16 recurrence for one request over [committed 0:P | draft 0:T]; returns
    (draft_outs[T,HV,V], state_after_committed[HV,V,K])."""
    st=S0b.transpose(-2,-1).contiguous().to(sd).unsqueeze(0)  # [1,HV,K,V]
    outs=[];state_at_P=None
    for t in range(P+T):
        if t<P:
            kt=kc[:,t,:].float()[None];vt=vc[:,t,:].float()[None]
            at=ac[:,t].float()[None];bt=bc[:,t].float()[None]  # rings are fp32; kernel reads them fp32
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

def run(B,P_list,flush_min,regime,seed=0):
    sA,sdt,sa,sS=REGIMES[regime];torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    pool=B
    kc=torch.randn(pool,HK,W,K,dtype=torch.bfloat16,device=dev)
    vc=torch.randn(pool,HV,W,V,dtype=torch.bfloat16,device=dev)
    ac=torch.randn(pool,HV,W,dtype=torch.float32,device=dev)*sa
    bc=torch.randn(pool,HV,W,dtype=torch.float32,device=dev)
    hist=torch.tensor(P_list,dtype=torch.int32,device=dev)
    q=torch.randn(B,T,H,K,dtype=torch.bfloat16,device=dev);k=torch.randn(B,T,HK,K,dtype=torch.bfloat16,device=dev)
    v=torch.randn(B,T,HV,V,dtype=torch.bfloat16,device=dev);a=torch.randn(B,T,HV,dtype=torch.bfloat16,device=dev)*sa
    b=torch.randn(B,T,HV,dtype=torch.bfloat16,device=dev)
    A_log=torch.randn(HV,dtype=torch.float32,device=dev)*sA;dt_bias=torch.randn(HV,dtype=torch.float32,device=dev)*sdt
    S0=torch.randn(B,HV,V,K,dtype=torch.bfloat16,device=dev)*sS;idx=torch.arange(B,dtype=torch.int32,device=dev)
    S0_in=S0.clone()
    out=vflush(A_log=A_log,a=a,dt_bias=dt_bias,q=q,k=k,v=v,b=b,initial_state_source=S0,initial_state_indices=idx,
        k_cache=kc.clone(),v_cache=vc.clone(),a_cache=ac.clone(),b_cache=bc.clone(),
        hist_len=hist.clone(),flush_min=flush_min,restart_hist_on_flush=True,use_qk_l2norm_in_kernel=True,scale=SCALE)
    torch.cuda.synchronize()
    is_flush=hist>=flush_min
    e_out=e_out_seq=e_state=e_state_seq=out_mag=0.0;ver_ok=True
    for bi in range(B):
        P=int(hist[bi])
        o32,s32=replay_req(bi,P,S0_in[bi],kc[bi],vc[bi],ac[bi],bc[bi],q[bi],k[bi],v[bi],a[bi],b[bi],A_log,dt_bias,torch.float32)
        o16,s16=replay_req(bi,P,S0_in[bi],kc[bi],vc[bi],ac[bi],bc[bi],q[bi],k[bi],v[bi],a[bi],b[bi],A_log,dt_bias,torch.bfloat16)
        e_out=max(e_out,(out[bi].float()-o32).abs().max().item());out_mag=max(out_mag,o32.abs().max().item())
        e_out_seq=max(e_out_seq,(o16-o32).abs().max().item())
        exp_s = s32 if is_flush[bi] else S0_in[bi].float()
        e_state=max(e_state,(S0[bi].float()-exp_s).abs().max().item())
        if is_flush[bi]: e_state_seq=max(e_state_seq,(s16-s32).abs().max().item())
        if not is_flush[bi]: ver_ok=ver_ok and torch.equal(S0[bi],S0_in[bi])
    obar=max(8e-3,2.0*e_out_seq);sbar=max(2e-2*max(sS,1.0),2.0*e_state_seq)
    rel_out=e_out/max(out_mag,1e-6)
    ok=(e_out<=obar or rel_out<=1e-2) and e_state<=sbar and ver_ok
    print(f"[{regime:>6} B={B} fmin={flush_min} P={P_list}] out {e_out:.2e}/{obar:.1e} rel{rel_out:.1e} state {e_state:.2e}/{sbar:.1e} ver={ver_ok}{'' if ok else '  <-- FAIL'}",flush=True)
    return 0 if ok else 1

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
    assert fails==0, f"{fails} vcache correctness cases failed"

if __name__ == "__main__":
    fails=0;print("GPU:",torch.cuda.get_device_name())
    for rn in REGIMES:
        fails+=run(4,[0,10,5,12],9,rn,0)
        fails+=run(4,[12,12,12,12],12,rn,1)
        fails+=run(4,[0,3,7,11],12,rn,2)
        fails+=run(8,[0,1,2,3,9,10,11,12],9,rn,3)
    print("ALL PASS" if fails==0 else f"{fails} FAIL")
    sys.exit(1 if fails else 0)
