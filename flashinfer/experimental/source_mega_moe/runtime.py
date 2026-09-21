"""Prepared single-rank fused MegaMoE using packed FP4/FP8 weights on SM103a."""
from __future__ import annotations
import functools,json
from pathlib import Path
from types import SimpleNamespace
from .bindings import make_bindings

@functools.cache
def catalog():
    return json.loads(Path(__file__).with_name('catalog.json').read_text())

def route_key(args):
    keys=('num_tokens','num_experts','top_k','hidden','intermediate','num_shared_experts','num_sms',
          'routed_weight_dtype','activation_clamp','fast_math')
    return json.dumps({k:args[k] for k in keys},sort_keys=True,separators=(',',':'))

@functools.cache
def load_program(name):
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec,sm103a_nvcc_flags
    record=catalog()['programs'][name]
    spec=gen_jit_spec(name=name,sources=[env.FLASHINFER_CSRC_DIR/p.removeprefix('csrc/') for p in record['sources']],
        extra_cuda_cflags=[*sm103a_nvcc_flags,*record['compile_flags'],'--device-entity-has-hidden-visibility=false'],
        extra_ldflags=['-lcuda'],extra_include_paths=[env.FLASHINFER_CSRC_DIR,env.FLASHINFER_INCLUDE_DIR],use_fast_math=False)
    return spec.build_and_load(),{**record,'library_path':str(spec.get_library_path())}

class MegaMoEPlan:
    """Prepared complete world-one pipeline with kernel-owned reusable counters.

    Inputs are packed E4M3 activations [T,H], packed scale words int32[T,H/128],
    int64 expert indices [T,topk], and FP32 routing weights [T,topk].
    weights contains B1/B2 uint8 tensors with the native gate/up interleave and
    FP4 nibble packing (or E4M3 bytes), and SFB1/SFB2 uint32 scale tensors in
    group-folded [E*K/128,N] order. SB1/SB2 and SSFB1/SSFB2 hold one FP8 shared
    expert in the same order without the expert dimension.

    Caller-provided workspace is a freshly zero-initialized byte buffer of the
    catalog's exact layout. bind and run never zero counters. The kernel cleans
    its reusable state, so repeated run and CUDA graph replay have no host reset.
    Initialization/copies happen in preparation. Default plans exclusively own
    private descriptors and run contains one submission. A borrowed descriptor
    workspace remains mutable and is refreshed before every consumer launch.
    Changing tensor addresses/layout requires a new plan; content updates remain
    supported. Prepared callables own the tensors and this plan owns the module.
    A plan retains all pointer owners and cannot execute concurrently on streams.
    """
    def __init__(self,x,x_sf,topk_idx,topk_weights,*,weights,num_experts,intermediate,
                 routed_weight_dtype='fp4',num_shared_experts=1,activation_clamp=10.0,
                 fast_math=True,num_sms=None,workspace=None,out=None,descriptor_workspace=None):
        import torch
        if x.device.type!='cuda' or torch.cuda.get_device_capability(x.device)!=(10,3):
            raise RuntimeError('This exported catalog requires SM103a')
        actual_sms=torch.cuda.get_device_properties(x.device).multi_processor_count
        if actual_sms!=152:raise RuntimeError('This exported catalog requires 152 physical SMs')
        t,h=x.shape;tk=topk_idx.shape[1];sms=actual_sms if num_sms is None else int(num_sms)
        args=dict(num_tokens=t,hidden=h,top_k=tk,num_experts=num_experts,intermediate=intermediate,
                  num_shared_experts=num_shared_experts,num_sms=sms,routed_weight_dtype=routed_weight_dtype,
                  activation_clamp=activation_clamp,fast_math=fast_math)
        route=catalog()['routes'][route_key(args)];record=route['layout'];config=SimpleNamespace(**record['config'])
        if workspace is None:workspace=torch.zeros(record['nbytes'],dtype=torch.uint8,device=x.device)
        if workspace.dtype!=torch.uint8 or workspace.device!=x.device or not workspace.is_contiguous() or workspace.numel()<record['nbytes'] or workspace.data_ptr()%128:
            raise ValueError('Workspace must be aligned contiguous CUDA uint8 with the catalog extent')
        raw=workspace.reshape(-1)
        views={name:raw.narrow(0,f['offset'],f['nbytes']).view(getattr(torch,f['dtype'])).reshape(f['shape']) for name,f in record['fields'].items()}
        views['shared_l1_acts']=views['x']
        if out is None:out=torch.empty((t,h),dtype=torch.bfloat16,device=x.device)
        if out.dtype!=torch.bfloat16 or tuple(out.shape)!=(t,h) or out.device!=x.device or not out.is_contiguous():
            raise ValueError('Output must be contiguous BF16[T,H] on the input device')
        self.route,self.workspace,self.views,self.output,self.config=route,workspace,views,out,config
        self.update_inputs(x,x_sf,topk_idx,topk_weights)
        pack=2 if routed_weight_dtype=='fp4' else 1;e=num_experts;i=intermediate
        specs={'B1':(torch.uint8,(e*2*i,h//pack)),'B2':(torch.uint8,(e*h,i//pack)),
               'SFB1':(torch.uint32,(e*h//128,2*i)),'SFB2':(torch.uint32,(e*i//128,h)),
               'SB1':(torch.uint8,(2*i,h)),'SB2':(torch.uint8,(h,i)),
               'SSFB1':(torch.uint32,(h//128,2*i)),'SSFB2':(torch.uint32,(i//128,h))}
        for name,(dtype,shape) in specs.items():
            tensor=weights[name]
            if tensor.dtype!=dtype or tuple(tensor.shape)!=shape or tensor.device!=x.device or not tensor.is_contiguous():
                raise ValueError(f'{name} must be contiguous {dtype}{shape} on the input device')
        layout=SimpleNamespace(config=config,num_sms=sms)
        self.bindings=make_bindings(layout,views,weights,out)
        self.bindings.update(grid_x=sms,grid_y=1,grid_z=1)
        module,program=load_program(route['program'])
        size=program['tma_workspace_bytes']
        private_descriptors = descriptor_workspace is None
        if size:
            if descriptor_workspace is None:descriptor_workspace=torch.empty(size,dtype=torch.uint8,device=x.device)
            if descriptor_workspace.dtype!=torch.uint8 or descriptor_workspace.device!=x.device or not descriptor_workspace.is_contiguous() or descriptor_workspace.numel()<size or descriptor_workspace.data_ptr()%128:
                raise ValueError('Descriptor workspace must be aligned CUDA uint8 with the recorded extent')
        call_args=tuple(descriptor_workspace if kind=='workspace' else self.bindings[name] for kind,name in program['arg_plan'])
        if size and private_descriptors:
            import tvm_ffi
            with tvm_ffi.use_torch_stream():
                prepared=module[program['ffi_prepare_entry']](*call_args)
            # Initialization is outside capture. Completing it here makes the
            # owned immutable maps ready for later serialized execution streams.
            torch.cuda.current_stream(x.device).synchronize()
            self._submission=(prepared,())
        else:
            # Explicit borrowed descriptor scratch remains mutable: refresh it
            # before every consumer, including every CUDA graph replay.
            self._submission=(module[program['ffi_entry']],call_args)
        self._retained=(module,program,weights,workspace,out,descriptor_workspace,x,x_sf,topk_idx,topk_weights)

    def update_inputs(self,x,x_sf,topk_idx,topk_weights):
        """Refresh packed inputs between serialized launches without resetting counters."""
        import torch
        c=self.config;t,h=c.num_tokens,c.hidden
        specs=((x,torch.float8_e4m3fn,(t,h)),(x_sf,torch.int32,(t,h//128)),
               (topk_idx,torch.int64,(t,c.top_k)),(topk_weights,torch.float32,(t,c.top_k)))
        for tensor,dtype,shape in specs:
            if tensor.dtype!=dtype or tuple(tensor.shape)!=shape or tensor.device!=self.workspace.device or not tensor.is_contiguous():
                raise ValueError(f'Packed input must be contiguous {dtype}{shape} on the workspace device')
        for name,value in zip(('x','x_sf','topk_idx','topk_weights'),(x,x_sf,topk_idx,topk_weights)):
            self.views[name][:t].copy_(value)
        if c.num_shared_experts:
            token=torch.arange(t,device=x.device,dtype=torch.int64);local=token%c.block_m
            rows=token//c.block_m*((c.block_m+127)//128*128)+(local&~127)+(local&31)*4+((local>>5)&3)
            self.views['shared_l1_sf'][:,rows]=x_sf.T

    def run(self):
        import tvm_ffi
        with tvm_ffi.use_torch_stream():
            entry,args=self._submission
            entry(*args)
        return self.output

def prepare_mega_moe(*args,**kwargs):
    return MegaMoEPlan(*args,**kwargs)
