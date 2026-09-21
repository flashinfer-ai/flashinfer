"""Exercise the prepared source export without the source compiler installed."""
import pytest
import torch


@pytest.mark.parametrize('accumulate', [False, True])
def test_fp8_prepared_output_and_accumulator(tmp_path, accumulate):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    prop=torch.cuda.get_device_properties(0)
    if (prop.major,prop.minor,prop.multi_processor_count)!=(10,3,152):
        pytest.skip('This specialization requires sm_103a with 152 SMs')
    from flashinfer.experimental.deepgemm_fp8_gemm import prepare_fp8_gemm_1d1d
    a=torch.ones((4096,4096),device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(torch.uint8)
    b=torch.ones((7168,4096),device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(torch.uint8)
    sfa=torch.full((8,4096),0x7f7f7f7f,device='cuda',dtype=torch.uint32)
    sfb=torch.full((8,7168),0x7f7f7f7f,device='cuda',dtype=torch.uint32)
    out=torch.full((4096,7168),.25 if accumulate else 0,device='cuda',
                   dtype=torch.float32 if accumulate else torch.bfloat16)
    address=out.data_ptr()
    plan=prepare_fp8_gemm_1d1d(a,b,sfa,sfb,out,accumulate=accumulate,cache_dir=tmp_path)
    plan.run()
    torch.testing.assert_close(out,torch.full_like(out,4096.25 if accumulate else 4096),atol=0,rtol=0)
    assert out.data_ptr()==address
    if accumulate:
        plan.run()
        torch.testing.assert_close(out,torch.full_like(out,8192.25),atol=0,rtol=0)
