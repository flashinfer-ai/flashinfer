"""Independent native SM120 QMMA GEMM reference, ascending K64 accumulation.

No target kernel, pipeline, scheduler, layout helper, or target output is used.
Inputs are logical row-major packed E2M1 and raw E4M3 scales, block size 16.
This is an implementation-semantic oracle; it is not exact real arithmetic.
"""
from pathlib import Path
import ctypes,fcntl,hashlib,os,subprocess
import torch
_lib=None

def _library():
    global _lib
    if _lib is None:
        source=Path(__file__).with_suffix('.cu')
        key=hashlib.sha256(source.read_bytes()).hexdigest()[:20]
        root=Path(os.environ.get('NATIVE_QMMA_CACHE',str(Path.home()/'.cache/native_qmma_reference')))
        root.mkdir(parents=True,exist_ok=True)
        target=root/(key+'.so')
        with (root/(key+'.lock')).open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not target.exists():
                subprocess.run(['nvcc','-std=c++17','-O2','-gencode=arch=compute_120a,code=sm_120a','--shared','-Xcompiler','-fPIC',str(source),'-o',str(target)],check=True)
        _lib=ctypes.CDLL(str(target))
        _lib.native_qmma.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p]
        _lib.native_qmma.restype=ctypes.c_int
    return _lib

class NativeQmmaReferenceLauncher:
    def __init__(self,sf_vec_size=16,mma_tiler_mn=(16,8),cluster_shape_mn=(1,1)):
        if sf_vec_size!=16:raise ValueError('NVFP4 reference requires scale block 16')
    def __call__(self,*,a,sfa,b,sfb,n,k):
        if k%64 or k<=0:raise ValueError('K must be a positive multiple of 64')
        m=a.shape[0]
        tensors=(a,sfa,b,sfb)
        if any(t.device!=a.device or not t.is_cuda for t in tensors):raise ValueError('one CUDA device required')
        if torch.cuda.get_device_capability(a.device)!=(12,0):raise ValueError('SM120 required')
        if tuple(a.shape)!=(m,k//2) or tuple(b.shape)!=(n,k//2):raise ValueError('packed A/B shape mismatch')
        if tuple(sfa.shape)!=(m,k//16) or tuple(sfb.shape)!=(n,k//16):raise ValueError('raw scale shape mismatch')
        if any(t.dtype not in (torch.uint8, torch.float4_e2m1fn_x2) for t in (a,b)):
            raise ValueError('A/B must store packed E2M1 bytes')
        if any(t.dtype not in (torch.uint8, torch.float8_e4m3fn) for t in (sfa,sfb)):
            raise ValueError('SFA/SFB must store raw E4M3 bytes')
        with torch.cuda.device(a.device):
            bufs=[t.contiguous() for t in tensors]
            out=torch.empty((m,n),device=a.device,dtype=torch.float32)
            if m==0 or n==0:return out
            err=_library().native_qmma(*(t.data_ptr() for t in bufs),out.data_ptr(),m,n,k,torch.cuda.current_stream(a.device).cuda_stream)
            if err:raise RuntimeError(f'native reference CUDA launch error {err}')
            return out
