"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""
"""Public TF32 prefill behavior, independent of the producer repository."""
import pytest
import torch
import torch.nn.functional as F
from flashinfer import prepare_tf32_kda_prefill

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10,3), reason='TF32 KDA export requires SM103a')


@pytest.mark.parametrize('lengths',[(17,), (64,), (17,65)])
@pytest.mark.parametrize('heads',[1,6,12])
def test_tf32_prefill_prepared(lengths,heads):
    torch.manual_seed(42)
    tokens=sum(lengths);shape=(1,tokens,heads,128)
    q,k,v,g=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
    v_original=v.clone()
    beta=torch.randn(shape[:-1],device='cuda',dtype=torch.bfloat16)
    a=torch.zeros(heads,device='cuda');bias=torch.full((heads,128),-2.,device='cuda')
    pool=torch.randn((len(lengths)+2,heads,128,128),device='cuda')*.1
    original=pool.clone();indices=torch.arange(len(lengths),device='cuda',dtype=torch.int32)+1
    offsets=[0];cp_offsets=[0]
    for n in lengths:offsets.append(offsets[-1]+n);cp_offsets.append(cp_offsets[-1]+(n+63)//64)
    cu=torch.tensor(offsets,device='cuda',dtype=torch.int64)
    starts=torch.tensor(cp_offsets,device='cuda',dtype=torch.int64)
    cp=torch.empty((cp_offsets[-1],heads,128,128),device='cuda',dtype=torch.bfloat16)
    out=torch.empty_like(q)
    call=prepare_tf32_kda_prefill(q,k,v,g,beta,A_log=a,dt_bias=bias,out=out,
        initial_state=pool,final_state=pool,cu_seqlens=cu,sequence_lengths=lengths,
        state_indices=indices,state_checkpoints=cp,checkpoint_cu_starts=starts,checkpoint_every_n_tokens=64)
    call.launch();torch.cuda.synchronize()
    qn=F.normalize(q.float(),dim=-1).bfloat16().double()
    kn=F.normalize(k.float(),dim=-1).bfloat16().double()
    decay=(-5.*(g.double()+bias.double()).sigmoid()).exp()
    active=beta.float().sigmoid().double()
    expected=torch.empty_like(out,dtype=torch.float64);checkpoints=[];final=[]
    for seq,(start,end) in enumerate(zip(offsets,offsets[1:])):
        state=original[seq+1].double()
        for token in range(start,end):
            if (token-start)%64==0:checkpoints.append(state.clone())
            state=state*decay[0,token,:,None,:]
            residual=(v_original[0,token].double()-(state*kn[0,token,:,None,:]).sum(-1))*active[0,token,:,None]
            state=state+residual[:,:,None]*kn[0,token,:,None,:]
            expected[0,token]=(state*qn[0,token,:,None,:]).sum(-1)*128**-.5
        final.append(state)
    for actual,reference in [(out,expected),(pool[indices],torch.stack(final)),(cp,torch.stack(checkpoints))]:
        assert torch.isfinite(actual).all()
        rrmse=(actual.double()-reference).norm()/reference.norm()
        assert rrmse<.01
    assert torch.equal(pool[0],original[0]) and torch.equal(pool[-1],original[-1])
    # A caller owns graph capture; the prepared object reuses its storage.
    pool.copy_(original);v.copy_(v_original)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call.launch()
    pool.copy_(original);v.copy_(v_original);graph.replay();torch.cuda.synchronize()
    assert (out.double()-expected).norm()/expected.norm()<.01


def test_tf32_rejects_bf16_external_state():
    q=torch.zeros((1,16,1,128),device='cuda',dtype=torch.bfloat16)
    state=torch.zeros((1,1,128,128),device='cuda',dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='FP32'):
        prepare_tf32_kda_prefill(q,q,q,q,q[...,0],A_log=torch.zeros(1,device='cuda'),
            dt_bias=torch.zeros((1,128),device='cuda'),out=torch.empty_like(q),initial_state=state)
