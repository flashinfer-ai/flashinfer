"""Analytical mixed E4M3/E2M1 values and changed-input current-stream replay."""
import pytest
import torch
from flashinfer.fp8_fp4_gemm import prepare_fp8_fp4_gemm

_CASES=[(m,n,k,None,128,32) for m in (1,16,128,512,4096)
        for n,k in ((4608,5120),(5120,2304))]
_CASES += [(256,256,256,None,128,32),(4096,7168,4096,None,224,128),
           (256,224,128,'bk128_s6',128,128),(4096,7168,4096,'bk128_s6',128,128),
           (256,128,256,'bk256_s4',128,128)]


@pytest.mark.parametrize('m,n,k,variant,block_n,gran_k_a',_CASES)
def test_mixed_values_packing_stream_replay(m,n,k,variant,block_n,gran_k_a):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(10,3):
        pytest.skip('SM103a required')
    if torch.cuda.get_device_properties(0).multi_processor_count!=152:
        pytest.skip('The exported schedules require 152 SMs')
    a=torch.full((m,k),0x38,dtype=torch.uint8,device='cuda').view(torch.float8_e4m3fn)
    b=torch.full((n,k//2),0x22,dtype=torch.uint8,device='cuda')
    # Constant scale1 has exponent127. BK256 repeats gran128 scale bytes;
    # native packing keeps one byte per declared granularity.
    scale_gran=32 if variant=='bk256_s4' else gran_k_a
    sfa=torch.full(((k+4*scale_gran-1)//(4*scale_gran),(m+3)//4*4),
                   0x7f7f7f7f,dtype=torch.int32,device='cuda')
    sfb=torch.full(((k+127)//128,(n+3)//4*4),0x7f7f7f7f,dtype=torch.int32,device='cuda')
    plan=prepare_fp8_fp4_gemm(a,b,sfa,sfb,m=m,variant=variant,block_n=block_n,gran_k_a=gran_k_a)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):plan.run()
    stream.synchronize()
    for sign in (1,-1,1):
        with torch.cuda.stream(stream):
            a.view(torch.uint8).fill_(0x38 if sign>0 else 0xb8)
            plan.storage.fill_(float('nan'))
            graph.replay()
        stream.synchronize()
        torch.testing.assert_close(plan.output,torch.full_like(plan.output,sign*k),atol=0,rtol=0)
