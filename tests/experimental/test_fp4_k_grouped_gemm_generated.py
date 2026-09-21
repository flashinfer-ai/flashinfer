"""Independent packed FP4 values, exact empty groups and current-stream graph replay."""
import pytest
import torch
from flashinfer.fp4_k_grouped_gemm import prepare_fp4_k_grouped_gemm

_CASES=[{'m': 4608, 'N': 5120, 'group_ks': [8192, 0, 4096], 'k_alignment': 256, 'use_psum_layout': True, 'output_dtype': 'bf16', 'accumulate': False}, {'m': 4608, 'N': 5120, 'group_ks': [12288, 0, 6144], 'k_alignment': 256, 'use_psum_layout': True, 'output_dtype': 'fp32', 'accumulate': True}, {'m': 4608, 'N': 5120, 'group_ks': [16384, 0, 8192], 'k_alignment': 256, 'use_psum_layout': True, 'output_dtype': 'fp32', 'accumulate': False}, {'m': 5120, 'N': 2304, 'group_ks': [8192, 0, 4096], 'k_alignment': 256, 'use_psum_layout': True, 'output_dtype': 'bf16', 'accumulate': False}, {'m': 5120, 'N': 2304, 'group_ks': [12288, 0, 6144], 'k_alignment': 256, 'use_psum_layout': True, 'output_dtype': 'fp32', 'accumulate': True}, {'m': 5120, 'N': 2304, 'group_ks': [16384, 0, 8192], 'k_alignment': 256, 'use_psum_layout': True, 'output_dtype': 'fp32', 'accumulate': False}, {'m': 256, 'N': 128, 'group_ks': [257, 0, 511], 'k_alignment': 256, 'use_psum_layout': True, 'output_dtype': 'bf16'}, {'m': 256, 'N': 128, 'group_ks': [257, 0, 511], 'k_alignment': 768, 'use_psum_layout': False, 'output_dtype': 'fp32', 'accumulate': True}, {'m': 256, 'N': 128, 'group_ks': [257, 511, 0], 'k_alignment': 256, 'use_psum_layout': True, 'output_dtype': 'fp32'}, {'m': 256, 'N': 128, 'group_ks': [2049, 0, 511], 'use_psum_layout': True, 'output_dtype': 'bf16'}, {'m': 256, 'N': 128, 'group_ks': [2049, 0, 511], 'use_psum_layout': False, 'output_dtype': 'fp32'}, {'m': 256, 'N': 128, 'group_ks': [2049, 0, 511], 'use_psum_layout': True, 'output_dtype': 'fp32', 'accumulate': True}]


def require_target():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(10,3):
        pytest.skip('SM103a required')
    if torch.cuda.get_device_properties(0).multi_processor_count!=152:
        pytest.skip('The exported nonempty schedules require152 SMs')


def fixture(case):
    m,n,ks=case['m'],case['N'],case['group_ks']
    physical_m=(m+255)//256*256
    alignment=case.get('k_alignment',256)
    padded=[(k+alignment-1)//alignment*alignment for k in ks]
    total=sum(padded)
    a=torch.zeros((physical_m,total//2),dtype=torch.uint8,device='cuda')
    b=torch.zeros((n,total//2),dtype=torch.uint8,device='cuda')
    cursor=0
    for k,pk in zip(ks,padded):
        for value in (a,b):
            value[:,cursor//2:(cursor+k)//2]=0x22
            if k%2:value[:,(cursor+k)//2]=0x02
        cursor+=pk
    sfa=torch.full((total//128,physical_m),0x7f7f7f7f,dtype=torch.int32,device='cuda')
    sfb=torch.full((total//128,n),0x7f7f7f7f,dtype=torch.int32,device='cuda')
    dtype=torch.float32 if case.get('output_dtype','bf16')=='fp32' else torch.bfloat16
    initial=torch.full((len(ks),physical_m,n),.25 if case.get('accumulate',False) else 0,
                       dtype=dtype,device='cuda')
    out=initial.clone()
    options={k:v for k,v in case.items() if k!='N'}
    plan=prepare_fp4_k_grouped_gemm(a,b,sfa,sfb,**options,out=out)
    assert plan.storage is out
    return a,a.clone(),initial,plan


@pytest.mark.parametrize('case',_CASES)
def test_grouped_values_stream_changed_input_replay(case):
    require_target()
    a,positive,initial,plan=fixture(case)
    negative=positive|0x88  # Sign of both FP4 nibbles; padded zero becomes signed zero.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):plan.run()
    stream.synchronize()
    accumulate=case.get('accumulate',False)
    for sign,bytes_ in ((1,positive),(-1,negative),(1,positive)):
        with torch.cuda.stream(stream):
            a.copy_(bytes_)
            plan.storage.copy_(initial)
        for replay_count in (1,2,3):
            with torch.cuda.stream(stream):graph.replay()
            stream.synchronize()
            multiplier=replay_count if accumulate else 1
            for group,k in enumerate(case['group_ks']):
                expected=initial[group]+sign*multiplier*k
                torch.testing.assert_close(plan.storage[group],expected,atol=0,rtol=0)


@pytest.mark.parametrize('dtype,accumulate',[('bf16',False),('fp32',False),('fp32',True)])
@pytest.mark.parametrize('psum',[False,True])
def test_all_empty_zero_or_preserve_graph(dtype,accumulate,psum):
    require_target()
    case=dict(m=256,N=128,group_ks=[0,0,0],k_alignment=256,
              output_dtype=dtype,accumulate=accumulate,use_psum_layout=psum)
    _,_,initial,plan=fixture(case)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):plan.run()
    stream.synchronize()
    for value in (.25,-2.,8.):
        with torch.cuda.stream(stream):
            plan.storage.fill_(value)
            graph.replay()
        stream.synchronize()
        expected=torch.full_like(plan.storage,value if accumulate else 0)
        torch.testing.assert_close(plan.storage,expected,atol=0,rtol=0)
