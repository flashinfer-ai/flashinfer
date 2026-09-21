"""Prepared FP8 GEMM launch for the exported 152-SM specializations."""
from functools import lru_cache
import json
from pathlib import Path

from .ptx_builder import build_from_ptx

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


@lru_cache(maxsize=None)
def load_program(key,cache_dir):
    catalog=json.loads((HERE/'catalog.json').read_text())
    record=dict(catalog['programs'][key])
    out=Path(cache_dir)/key;out.mkdir(mode=0o700,parents=True,exist_ok=True)
    module=build_from_ptx(ptx_path=ROOT/record['sources'][0],binding_path=ROOT/record['sources'][1],
        module_ident=record['module_ident'],arch='sm_103a',ptxas_options=record['compile_flags'],
        workdir=out/'build',receipt_path=out/'build.json',include_paths=[ROOT/'csrc',ROOT/'include'])
    receipt=json.loads((out/'build.json').read_text())
    record.update(library_path=receipt['artifact_path'],build_receipt=str(out/'build.json'),
                  assembled_cubin_sha256=receipt['assembled_cubin_sha256'])
    return module,record


class Fp8GemmPlan:
    def __init__(self,entry,args,bindings,record):
        self.entry,self.args,self.bindings,self.record=entry,args,bindings,record

    def run(self):
        import tvm_ffi
        with tvm_ffi.use_torch_stream():
            self.entry(*self.args)


def prepare_fp8_gemm_1d1d(a,b,sfa,sfb,out,*,accumulate=False,cache_dir):
    """Prepare M4096/N7168/K4096 with prepacked MN-major UE8M0 scale words.

    ``a``/``b`` contain FP8 E4M3 bytes. Forward writes BF16 ``out``; accumulation
    reads and updates FP32 ``out`` in place. Restore the initializer before each
    independent accumulated evaluation. Packing and allocations precede run().
    """
    import torch
    case='wgrad' if accumulate else 'forward'
    catalog=json.loads((HERE/'catalog.json').read_text())
    key=catalog['routes'][case]
    if tuple(a.shape)!=(4096,4096) or tuple(b.shape)!=(7168,4096) or tuple(out.shape)!=(4096,7168):
        raise ValueError('This prepared specialization requires M4096/N7168/K4096')
    if out.dtype!=(torch.float32 if accumulate else torch.bfloat16):
        raise TypeError('Accumulator/output dtype does not match the selected specialization')
    device=torch.cuda.get_device_properties(out.device)
    if (device.major,device.minor,device.multi_processor_count)!=(10,3,152):
        raise ValueError('This prepared specialization requires sm_103a with152SMs')
    module,record=load_program(key,str(Path(cache_dir).resolve()))
    bindings=dict(A=a,B=b,SFA=sfa,SFB=sfb,C_tma=out,M=4096,N=7168,K=4096,
                  grid_m=32,grid_n=32,K_tiles=32,grid_x=152,grid_y=1,grid_z=1)
    args=tuple(bindings[key] for _,key in record['arg_plan'])
    return Fp8GemmPlan(module[record['ffi_entry']],args,bindings,record)
