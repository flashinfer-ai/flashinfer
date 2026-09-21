"""Prepared grouped packed FP4 E2M1 GEMM with BF16/FP32 output on SM103a."""
from __future__ import annotations
import functools
import json
from pathlib import Path


@functools.cache
def _catalog():
    return json.loads(Path(__file__).with_name('kgroup_catalog.json').read_text())


@functools.cache
def load_program(name):
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec, sm103a_nvcc_flags
    record=_catalog()['programs'][name]
    spec=gen_jit_spec(name=name,
        sources=[env.FLASHINFER_CSRC_DIR/p.removeprefix('csrc/') for p in record['sources']],
        extra_cuda_cflags=[*sm103a_nvcc_flags,*record['compile_flags'],
                          '--device-entity-has-hidden-visibility=false'],
        extra_ldflags=['-lcuda'],extra_include_paths=[env.FLASHINFER_CSRC_DIR,env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False)
    module=spec.build_and_load()
    return module,{**record,'library_path':str(spec.get_library_path())}


def route_key(options):
    return json.dumps([options['M'],options['N'],list(options['group_ks']),options['num_sms'],
        options.get('k_alignment',256),bool(options.get('use_psum_layout',True)),
        options.get('output_dtype','bf16'),bool(options.get('accumulate',False)),
        options.get('num_stages',7)],separators=(',',':'))


class GroupedFP4Plan:
    """Prepared independent K-group products, with optional in-place FP32 addition.

    Each group owns out[group]. Padding contains encoded zero and valid scales.
    run() uses the current stream; one output/descriptor workspace has one owner.
    For all-empty groups, overwrite clears the output and accumulation preserves it.
    """
    def __init__(self,a,b,a_scales,b_scales,*,m,group_ks,k_alignment=256,
                 use_psum_layout=True,output_dtype='bf16',accumulate=False,num_stages=7,
                 out=None,grouped_layout=None,descriptor_workspace=None):
        import torch
        if a.device.type!='cuda' or torch.cuda.get_device_capability(a.device)!=(10,3):
            raise RuntimeError('Grouped FP4 GEMM requires the validated SM103a target')
        group_ks=tuple(int(k) for k in group_ks)
        if m<1 or not group_ks or any(k<0 for k in group_ks):
            raise ValueError('m must be positive and group_ks a nonempty list of nonnegative K sizes')
        if k_alignment<256 or k_alignment%256:
            raise ValueError('k_alignment must be a positive multiple of256')
        if output_dtype not in ('bf16','fp32') or (accumulate and output_dtype!='fp32'):
            raise ValueError('Output must be bf16/fp32; accumulation requires fp32')
        if a.ndim!=2 or b.ndim!=2 or a.dtype not in (torch.int8,torch.uint8) or b.dtype not in (torch.int8,torch.uint8):
            raise ValueError('A and B must contain packed E2M1 int8/uint8 bytes')
        n=b.shape[0]
        if n<1 or n%128:
            raise ValueError('N must be positive and divisible by128')
        physical_m=(m+255)//256*256
        padded_ks=[(k+k_alignment-1)//k_alignment*k_alignment for k in group_ks]
        total_k=sum(padded_ks)
        if tuple(a.shape)!=(physical_m,total_k//2) or tuple(b.shape)!=(n,total_k//2):
            raise ValueError('A/B must use independently padded concatenated groups and physical M aligned to256')
        for tensor,shape in ((a_scales,((total_k+127)//128,physical_m)),
                             (b_scales,((total_k+127)//128,n))):
            if tensor.dtype not in (torch.int32,torch.uint32) or tuple(tensor.shape)!=shape:
                raise ValueError(f'Packed UE8M0 scales must be int32/uint32 with shape {shape}')
        dtype=torch.bfloat16 if output_dtype=='bf16' else torch.float32
        if out is None:
            if accumulate:
                raise ValueError('In-place accumulation requires caller-provided initialized out')
            out=torch.empty((len(group_ks),physical_m,n),dtype=dtype,device=a.device)
        if out.dtype!=dtype or tuple(out.shape)!=(len(group_ks),physical_m,n):
            raise ValueError('out must match output_dtype and [groups, physical_M, N]')
        expected_layout=[]
        cursor=0
        for logical_k,padded_k in zip(group_ks,padded_ks):
            expected_layout.append(cursor+logical_k if use_psum_layout else padded_k)
            cursor+=padded_k
        if grouped_layout is None:
            grouped_layout=torch.tensor(expected_layout,dtype=torch.int32,device=a.device)
        elif grouped_layout.dtype!=torch.int32 or tuple(grouped_layout.shape)!=(len(group_ks),):
            raise ValueError('grouped_layout must contain one int32 per group')
        tensors=(a,b,a_scales,b_scales,out,grouped_layout)
        if any(t.device!=a.device or not t.is_contiguous() for t in tensors):
            raise ValueError('Operands, scales, layout and output must be contiguous on one CUDA device')
        # Preparation validates optional caller metadata once, outside replay/timing.
        if grouped_layout.cpu().tolist()!=expected_layout:
            raise ValueError('grouped_layout does not encode the declared logical/padded groups')
        sms=torch.cuda.get_device_properties(a.device).multi_processor_count
        self.options=dict(M=m,N=n,group_ks=group_ks,num_sms=sms,k_alignment=k_alignment,
            use_psum_layout=use_psum_layout,output_dtype=output_dtype,accumulate=accumulate,num_stages=num_stages)
        self.storage,self.output=out,out[:,:m]
        self.empty,self.accumulate=total_k==0,bool(accumulate)
        self.descriptor_workspace=descriptor_workspace
        self.bindings=dict(A=a.view(torch.uint8),B=b.view(torch.uint8),SFA=a_scales.view(torch.uint32),
            SFB=b_scales.view(torch.uint32),C_tma=out,grouped_layout=grouped_layout,
            M=physical_m,N=n,K=total_k,num_groups=len(group_ks))
        if self.empty:
            self._retained=tensors
            return
        try:
            route=_catalog()['routes'][route_key(self.options)]
        except KeyError as error:
            raise NotImplementedError(f'No exported grouped FP4 schedule for {self.options}') from error
        cfg=route['config']
        for key in ('grid_m','grid_n'):
            self.bindings[key]=cfg[key]
        self.bindings.update(zip(('grid_x','grid_y','grid_z'),cfg['grid']))
        module,record=load_program(route['program'])
        workspace_bytes=record['tma_workspace_bytes']
        if workspace_bytes:
            if descriptor_workspace is None:
                descriptor_workspace=torch.empty(workspace_bytes,dtype=torch.uint8,device=a.device)
            if descriptor_workspace.dtype!=torch.uint8 or descriptor_workspace.device!=a.device or not descriptor_workspace.is_contiguous() or descriptor_workspace.numel()<workspace_bytes or descriptor_workspace.data_ptr()%128:
                raise ValueError('Descriptor workspace must be aligned contiguous CUDA uint8 storage')
        args=tuple(descriptor_workspace if kind=='workspace' else self.bindings[name]
                   for kind,name in record['arg_plan'])
        self._submission=module[record['ffi_entry']],args
        self._retained=(module,record,tensors,descriptor_workspace,self.bindings)
        self.descriptor_workspace=descriptor_workspace

    def run(self):
        import tvm_ffi
        if self.empty:
            if not self.accumulate:
                self.storage.zero_()
            return self.output
        with tvm_ffi.use_torch_stream():
            entry,args=self._submission
            entry(*args)
        return self.output
