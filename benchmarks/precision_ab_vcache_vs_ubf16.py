"""Quantify the raw-v precision thesis: does materializing u in bf16 (#4081's
u-ring dtype) introduce fold error vs solving u at fold time from raw inputs?

Three folds of the SAME P=12 committed history into the same S0:
  TRUTH : full fp32 recurrence (fp32 state, fp32 everything)
  RAW-V : u solved in fp32 from bf16 v/k-hat + fp32 a/b decay chain, fold
          accumulated fp32, ONE bf16 rounding at the state store (our kernel's
          rounding points) — plus the ACTUAL vcache kernel for cross-check.
  U-BF16: the SAME fp32-solved u ROUNDED TO BF16 (u-ring dtype), then the same
          fp32 fold + one bf16 store round (#4081's payload rounding point).
The only difference between RAW-V and U-BF16 is the u bf16 rounding.

Run (from the repo root): python benchmarks/precision_ab_vcache_vs_ubf16.py"""
import importlib.util, math, os, sys, torch
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO,"tests","gdn"))
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
vcf=load("vcf",os.path.join(REPO,"flashinfer/gdn_kernels/gdn_decode_bf16_wy_vcache_flush.py")).gated_delta_rule_mtp_vcache_flush
torch.set_grad_enabled(False);dev="cuda"
H=HK=16;HV=64;K=V=128;W=16;T=4;SCALE=1/math.sqrt(K);B=8;P=12
REG={"mild":(0.1,0.1,0.1,1.0),"strong":(1.0,1.0,1.0,1.0)}
print("GPU:",torch.cuda.get_device_name(),f"| B={B} P={P} fold of 12 committed rows, max|err| vs fp32 truth")
print(f"{'regime':>8} {'seed':>4} {'RAW-V(model)':>13} {'RAW-V(kernel)':>14} {'U-BF16(model)':>14} {'u-rounding penalty':>18}")
for rn,(sA,sdt,sa,sS) in REG.items():
    agg=[0.0,0.0,0.0]
    for seed in range(5):
        torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        kc=torch.randn(B,HK,W,K,dtype=torch.bfloat16,device=dev)
        vc=torch.randn(B,HV,W,V,dtype=torch.bfloat16,device=dev)
        ac=torch.randn(B,HV,W,dtype=torch.float32,device=dev)*sa
        bc=torch.randn(B,HV,W,dtype=torch.float32,device=dev)
        q=torch.randn(B,T,H,K,dtype=torch.bfloat16,device=dev);k=torch.randn(B,T,HK,K,dtype=torch.bfloat16,device=dev)
        v=torch.randn(B,T,HV,V,dtype=torch.bfloat16,device=dev);a=torch.randn(B,T,HV,dtype=torch.bfloat16,device=dev)*sa
        b=torch.randn(B,T,HV,dtype=torch.bfloat16,device=dev)
        A_log=torch.randn(HV,dtype=torch.float32,device=dev)*sA;dt_bias=torch.randn(HV,dtype=torch.float32,device=dev)*sdt
        S0=torch.randn(B,HV,V,K,dtype=torch.bfloat16,device=dev)*sS;idx=torch.arange(B,dtype=torch.int32,device=dev)
        # per-head decay/gate chains, fp32, from the fp32 a/b rings (GQA expand for k)
        g=(-torch.exp(A_log)[None,:,None]*torch.nn.functional.softplus(ac+dt_bias[None,:,None]))  # [B,HV,W]
        beta=torch.sigmoid(bc)                                                                     # [B,HV,W]
        khat=torch.nn.functional.normalize(kc.float(),p=2,dim=-1)                                  # [B,HK,W,K]
        khat_e=khat.repeat_interleave(HV//HK,dim=1)                                                # [B,HV,W,K]
        # ---- TRUTH: fp32 recurrence (fp32 v as-stored = bf16 ring values widened) ----
        S=S0.float().transpose(-2,-1).contiguous()  # [B,HV,K,V]
        us_f32=[]
        G=torch.zeros(B,HV,device=dev)
        Gs=[]
        for t in range(P):
            S=S*torch.exp(g[:,:,t])[...,None,None]
            pred=torch.einsum("bhk,bhkv->bhv",khat_e[:,:,t],S)
            u=beta[:,:,t][...,None]*(vc[:,:,t,:].float()-pred)   # fp32-solved u from bf16 v
            us_f32.append(u)
            S=S+torch.einsum("bhk,bhv->bhkv",khat_e[:,:,t],u)
            G=G+g[:,:,t];Gs.append(G.clone())
        S_true=S.transpose(-2,-1).contiguous()  # [B,HV,V,K] fp32 (truth for these inputs)
        # ---- RAW-V model: identical math, ONE bf16 round at the end ----
        S_rawv=S_true.to(torch.bfloat16).float()
        # ---- U-BF16 model: same us but rounded to bf16 (the u-ring), same fold ----
        Su=S0.float().transpose(-2,-1).contiguous()
        GP=Gs[-1]
        Su=Su*torch.exp(GP)[...,None,None]
        for t in range(P):
            w=torch.exp(GP-Gs[t])                                 # exp(G_P - G_i)
            u16=us_f32[t].to(torch.bfloat16).float()              # <-- THE u ROUNDING
            Su=Su+torch.einsum("bhk,bhv->bhkv",khat_e[:,:,t],w[...,None]*u16)
        S_ub=Su.transpose(-2,-1).contiguous().to(torch.bfloat16).float()
        # ---- RAW-V actual kernel ----
        Sk=S0.clone()
        vcf(A_log=A_log,a=a,dt_bias=dt_bias,q=q,k=k,v=v,b=b,initial_state_source=Sk,initial_state_indices=idx,
            k_cache=kc.clone(),v_cache=vc.clone(),a_cache=ac.clone(),b_cache=bc.clone(),
            hist_len=torch.full((B,),P,dtype=torch.int32,device=dev),flush_min=P,
            restart_hist_on_flush=False,scale=SCALE)
        torch.cuda.synchronize()
        e_rawv=(S_rawv-S_true).abs().max().item()
        e_kern=(Sk.float()-S_true).abs().max().item()
        e_ub=(S_ub-S_true).abs().max().item()
        agg[0]=max(agg[0],e_rawv);agg[1]=max(agg[1],e_kern);agg[2]=max(agg[2],e_ub)
        print(f"{rn:>8} {seed:>4} {e_rawv:13.4e} {e_kern:14.4e} {e_ub:14.4e} {e_ub/max(e_rawv,1e-9):17.1f}x",flush=True)
    print(f"{rn:>8}  MAX {agg[0]:13.4e} {agg[1]:14.4e} {agg[2]:14.4e} {agg[2]/max(agg[0],1e-9):17.1f}x")
