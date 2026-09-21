"""Analytical E2M1/UE8M0 inputs, runtime alpha, current-stream graph replay."""
import pytest
import torch
from flashinfer.fp4_gemm import prepare_fp4_gemm

_MODEL = [(m,n,k,None) for m in (16,128,512,4096) for n,k in ((4608,5120),(5120,2304))]
_CASES = [(m,n,k,stages,alpha) for m,n,k,stages in _MODEL for alpha in (1.,-.75)]
_CASES += [(256,128,2048,7,.5),(256,128,256,None,1.),(256,128,256,None,-.75)]


@pytest.mark.parametrize('m,n,k,num_stages,alpha',_CASES)
def test_fp4_values_alpha_stream_replay(m,n,k,num_stages,alpha):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10,3):
        pytest.skip('SM103a required')
    if torch.cuda.get_device_properties(0).multi_processor_count != 152:
        pytest.skip('The exported schedules require 152 SMs')
    a=torch.full((m,k//2),0x22,dtype=torch.uint8,device='cuda')
    b=torch.full((n,k//2),0x22,dtype=torch.uint8,device='cuda')
    sfa=torch.full((k//128,m),0x7f7f7f7f,dtype=torch.int32,device='cuda')
    sfb=torch.full((k//128,n),0x7f7f7f7f,dtype=torch.int32,device='cuda')
    plan=prepare_fp4_gemm(a,b,sfa,sfb,m=m,alpha=alpha,num_stages=num_stages)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            plan.run()
    stream.synchronize()
    for negative in (False,True,False):
        with torch.cuda.stream(stream):
            a.fill_(0xaa if negative else 0x22)
            plan.storage.fill_(float('nan'))
            graph.replay()
        stream.synchronize()
        expected=torch.full_like(plan.output,(-1 if negative else 1)*k*alpha)
        torch.testing.assert_close(plan.output,expected,atol=0,rtol=0)
