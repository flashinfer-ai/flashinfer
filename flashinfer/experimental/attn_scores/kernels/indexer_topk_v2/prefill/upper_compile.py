import functools,sys,torch,cutlass,cutlass.cute as cute
from cutlass import testing
from ..api import DTYPES
from ..mqa_logits_util import KernelLaunch

@functools.cache
def compile_sampler(qkv,k,cap,dtype,sms,sigma,signature,safe,qstages,heads=64,qsa_weights=0,qsa_unit_scale=False,short_domain=False,dsv4=False,pro_common_domain=False,fixed_samples=0,glm_short_quantile=False,glm_prepare_sf=False,glm_short_margin=False,rank_sigma=0.0,count_override=0, pdl=False):
 if count_override>0:fixed_samples=count_override
 if qkv=='fp8':
  from .sampler_upper_fp8 import FP8PrefillMQALogitsKernel as GEMM
 elif glm_short_margin:from .sampler_upper_glmflash_fp4 import FP4PrefillMQALogitsKernel as GEMM
 else:from .sampler_upper_fp4 import FP4PrefillMQALogitsKernel as GEMM
 attrs=dict(pdl=pdl,fixed_samples=fixed_samples,pro_common_domain=pro_common_domain,dsv4=dsv4,qsa_weights=qsa_weights,qsa_unit_scale=qsa_unit_scale,num_heads=heads,head_dim=128,num_sms=sms,epi_dtype=cutlass.Float32,output_dtype=dtype,mode='sample',top_k=k,capacity=cap,splits=1,packed=False,sample_size=512,use_relu_trick=safe,sample_sigma=sigma)
 if qkv=='fp8':attrs.update(glm_short_quantile=glm_short_quantile)
 if qkv=='fp8':attrs.update(umma_n=256,max_umma_pipeline=False,max_w_in_reg=0,num_epi_subtiles=16,math_registers=48,store_registers=48,num_sms=sms*2)
 else:attrs.update(use_batched_store=False,num_umma_stages=1,num_kv_stages=4,num_epi_subtiles=8,max_w_in_reg=64,math_registers=112,store_registers=32)
 if heads in (4,32):
  attrs.update(num_epi_subtiles=1,max_w_in_reg=heads,math_registers=48)
  if qkv=='fp8':attrs.update(umma_n=64 if heads==4 else 256,math_registers=48)
 if heads==32:attrs.update(math_registers=96 if qkv=='fp8' else 112,store_registers=32)
 if heads==32 and qkv=='fp8':attrs.update(math_registers=48,store_registers=32,num_epi_subtiles=4,num_kv_stages=2)
 if heads==32 and qkv=='fp8' and short_domain:attrs.update(num_kv_stages=2,sample_shared=True)
 if heads==32 and qkv=='mxfp4':attrs.update(num_q_stages=3,sample_shared_capacity=1024)
 if glm_prepare_sf:attrs.update(dsv4_prepare_sf=True)
 if dsv4 and qkv=='mxfp4':attrs.update(dsv4_prepare_sf=True,dsv4_shared_scores=True,dsv4_short_top16=short_domain,num_q_stages=3)
 if dsv4:
  attrs.update(dsv4_positions=True,dsv4_bulk_positions=True)
  if qkv=='fp8':attrs.update(num_kv_stages=2)
  else:attrs.update(num_epi_subtiles=4)
 if heads==32 and k==512 and qkv=='fp8' and fixed_samples>0:attrs.update(num_kv_stages=1)
 if heads==4 and qkv=='fp8' and fixed_samples==256:attrs.update(sample_shared=True)
 if heads==4 and qkv=='mxfp4' and fixed_samples==256:attrs.update(sample_shared_capacity=256)
 if rank_sigma>0:attrs.update(sample_rank_sigma=rank_sigma)
 if count_override>0:attrs.update(sample_count_override=count_override)
 if count_override>0 and qkv=='fp8':attrs.update(num_kv_stages=2,sample_shared=False)
 tensors=[]
 for i,(d,shape) in enumerate(signature):
  dims=tuple(cute.sym_int() if x<0 else x for x in shape);order=tuple(reversed(range(len(shape))))
  if qkv=='mxfp4' and i in (2,3,4):order=(1,0,2) if i==2 else (0,1)
  if i == len(signature)-8:t=cute.runtime.make_fake_tensor(DTYPES[d],dims,stride=(cute.sym_int64(),1),assumed_align=16)
  else:t=cute.runtime.make_fake_compact_tensor(DTYPES[d],dims,stride_order=order,assumed_align=16)
  tensors.append(t)
 return cute.compile(GEMM(**attrs),*tensors,cutlass.Int32(1),cutlass.Int32(1),cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi')
