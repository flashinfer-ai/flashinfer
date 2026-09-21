"""Prepared native packed FP4 GEMM with FP32 alpha and BF16 output on SM103a."""
from __future__ import annotations

import functools
import json
from pathlib import Path


@functools.cache
def _catalog():
    return json.loads(Path(__file__).with_name("fp4_gemm_catalog.json").read_text())


@functools.cache
def load_program(name):
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec, sm103a_nvcc_flags
    record = _catalog()["programs"][name]
    spec = gen_jit_spec(name=name,
        sources=[env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]],
        extra_cuda_cflags=[*sm103a_nvcc_flags, *record["compile_flags"],
                           "--device-entity-has-hidden-visibility=false"],
        extra_ldflags=["-lcuda"], extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False)
    module = spec.build_and_load()
    return module, {**record, "library_path": str(spec.get_library_path())}


def route_key(options):
    return ':'.join(str(options.get(key, default)) for key, default in (
        ('M',None),('N',None),('K',None),('num_sms',None),('num_stages',None),
        ('block_n',128),('epilogue_store_n',32)))


class Fp4GemmPlan:
    """Submit packed E2M1 A @ B.T with packed UE8M0 scales and BF16 output.

    A/B are contiguous int8 or uint8 [storage_M,K/2]/[N,K/2]. Even logical K
    elements occupy the low nibble. Each scale covers 32 K elements; four
    UE8M0 bytes pack into one uint32 at scales[K/128,MN]. Scales have unit
    MN stride. storage_M follows the catalog's exact production route.
    Alpha multiplies FP32 accumulators before BF16 conversion. The plan owns
    no packing/quantization kernel. Mutable tensor contents can be replayed
    on the current stream; concurrent use of one output/workspace is invalid.
    """
    def __init__(self, a, b, a_scales, b_scales, *, m, alpha=1.0, out=None,
                 num_stages=None, block_n=128, epilogue_store_n=32, descriptor_workspace=None):
        import torch
        if a.device.type != 'cuda' or torch.cuda.get_device_capability(a.device) != (10,3):
            raise RuntimeError('Native FP4 GEMM requires the validated SM103a target')
        if a.ndim != 2 or b.ndim != 2 or a.dtype not in (torch.int8,torch.uint8) or b.dtype not in (torch.int8,torch.uint8):
            raise ValueError('A/B must be packed int8/uint8 matrices')
        n,k = b.shape[0], b.shape[1]*2
        if a.shape[1]*2 != k:
            raise ValueError('A/B packed K dimensions must agree')
        sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        self.options = dict(M=m,N=n,K=k,num_sms=sms,num_stages=num_stages,block_n=block_n,
                            epilogue_store_n=epilogue_store_n)
        try:
            route = _catalog()['routes'][route_key(self.options)]
        except KeyError as error:
            raise NotImplementedError(f'No exported native FP4 schedule for {self.options}') from error
        cfg = route['config']
        # A source-selected specialization may use alpha in dispatch; its
        # exact predicate is carried in the catalog, not inferred by target.
        if cfg['required_alpha'] is not None and float(alpha) != cfg['required_alpha']:
            raise NotImplementedError('This exported production route requires its declared alpha')
        if a.shape[0] != cfg['input_m']:
            raise ValueError(f"A must have {cfg['input_m']} physical rows for this production route")
        for tensor,mn in ((a_scales,cfg['input_m']),(b_scales,n)):
            if tensor.dtype not in (torch.int32,torch.uint32) or tuple(tensor.shape) != (k//128,mn):
                raise ValueError('Packed scales must be int32/uint32[K/128,storage_MN]')
        if out is None:
            out = torch.empty((cfg['output_m'],n),dtype=torch.bfloat16,device=a.device)
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (cfg['output_m'],n):
            raise ValueError('Output must be BF16 with the production physical M and logical N')
        tensors=(a,b,a_scales,b_scales,out)
        if any(t.device != a.device or not t.is_contiguous() for t in tensors):
            raise ValueError('Packed operands, scales and output must be contiguous on one CUDA device')
        self.bindings=dict(A=a.view(torch.uint8),B=b.view(torch.uint8),SFA=a_scales.view(torch.uint32),
            SFB=b_scales.view(torch.uint32),C_tma=out,M=cfg['output_m'],N=n,K=k,
            grid_m=cfg['grid_m'],grid_n=cfg['grid_n'],K_tiles=k//256,alpha=float(alpha),
            grid_x=cfg['grid'][0],grid_y=cfg['grid'][1],grid_z=cfg['grid'][2])
        module,record=load_program(route['program'])
        workspace_bytes=record['tma_workspace_bytes']
        if workspace_bytes:
            if descriptor_workspace is None:
                descriptor_workspace=torch.empty(workspace_bytes,dtype=torch.uint8,device=a.device)
            if descriptor_workspace.dtype != torch.uint8 or descriptor_workspace.device != a.device or not descriptor_workspace.is_contiguous() or descriptor_workspace.numel()<workspace_bytes or descriptor_workspace.data_ptr()%128:
                raise ValueError('Descriptor workspace must be aligned contiguous CUDA uint8 storage')
        args=tuple(descriptor_workspace if kind=='workspace' else self.bindings[name]
                   for kind,name in record['arg_plan'])
        self._submission=(module[record['ffi_entry']],args)
        self._retained=(module,record,tensors,descriptor_workspace,self.bindings)
        self.storage,self.output=out,out[:m]
        self.descriptor_workspace=descriptor_workspace

    def run(self):
        import tvm_ffi
        with tvm_ffi.use_torch_stream():
            entry,args=self._submission
            entry(*args)
        return self.output
