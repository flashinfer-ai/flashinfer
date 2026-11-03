import functools,sys,torch,cutlass,cutlass.cute as cute
from cutlass import testing
from ..api import DTYPES
from ..mqa_logits_util import KernelLaunch
@functools.cache
def _compile_gemm(qkv_dtype,mode,k,cap,packed,dtype,ctas,splits,sample_size,signature,safe_relu,tuning,heads=64,glm_short_recovery=False,glm_mid_idle=False, pdl=False):
    if mode != 'recover':raise ValueError('only recovery GEMM is supported')
    if qkv_dtype=='fp8':
        if glm_mid_idle:from .fp8_glmflash_queue_recovery import FP8PrefillMQALogitsKernel as GEMM
        else:from .fp8_queue_recovery import FP8PrefillMQALogitsKernel as GEMM
    else:
        if glm_short_recovery:from .fp4_glmflash_queue_recovery import FP4PrefillMQALogitsKernel as GEMM
        else:from .fp4_queue_recovery import FP4PrefillMQALogitsKernel as GEMM
    attrs=dict(pdl=pdl,num_heads=heads,head_dim=128,num_sms=ctas,epi_dtype=cutlass.Float32,output_dtype=dtype,
               mode=mode,top_k=k,capacity=cap,splits=splits,packed=packed,sample_size=sample_size,use_relu_trick=safe_relu)
    if qkv_dtype=='mxfp4':
        attrs.update(use_batched_store=False)
        if mode=='filter':attrs.update(max_w_in_reg=32,num_epi_subtiles=4,math_registers=192,store_registers=104)
    elif mode=='filter':attrs.update(umma_n=256,max_w_in_reg=4,num_epi_subtiles=4,math_registers=192,store_registers=104,weights_prepared=True)
    if mode in ('filter','recover'):attrs.update(dict(tuning))
    kernel=GEMM(**attrs)
    # Fix contiguous head/feature dimensions for TMA and MMA, while rows remain dynamic.
    tensors=[]
    for index,(d,shape) in enumerate(signature):
        dims=tuple(cute.sym_int() if x<0 else x for x in shape)
        order=tuple(reversed(range(len(shape))))
        if qkv_dtype=='mxfp4' and index in (2,3,4):
            order=(1,0,2) if index==2 else (0,1)
        if index in (len(signature)-10, len(signature)-5):
            # Dense recovery can exceed 2^31 elements (4K Q x 1M K).
            # Keep the inherited GEMM's INT64 row-stride ABI.
            tensors.append(cute.runtime.make_fake_tensor(DTYPES[d],dims,
                           stride=(cute.sym_int64(),1),assumed_align=16))
        else:
            tensors.append(cute.runtime.make_fake_compact_tensor(DTYPES[d],dims,
                           stride_order=order,assumed_align=16))
    return cute.compile(kernel,*tensors,cutlass.Int32(1),cutlass.Int32(1),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi')
