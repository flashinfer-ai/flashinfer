"""DLPack adapter for PyTorch versions that cannot export packed FP4.

Export the owning uint8 view, describing the same bytes using DLPack's
Float4E2M1FN (17), four bits, two lanes. Shape/strides remain in packed bytes.
The capsule retains the original PyTorch storage owner and deleter.
"""
import ctypes as C
import torch

class Device(C.Structure):
    _fields_=[('type',C.c_int),('id',C.c_int)]
class Dtype(C.Structure):
    _fields_=[('code',C.c_uint8),('bits',C.c_uint8),('lanes',C.c_uint16)]
class Tensor(C.Structure):
    _fields_=[('data',C.c_void_p),('device',Device),('ndim',C.c_int),('dtype',Dtype),('shape',C.POINTER(C.c_int64)),('strides',C.POINTER(C.c_int64)),('offset',C.c_uint64)]
get_pointer=C.pythonapi.PyCapsule_GetPointer
get_pointer.argtypes=[C.py_object,C.c_char_p]
get_pointer.restype=C.c_void_p

class PackedFloat4:
    def __init__(self,tensor):
        self.tensor=tensor
    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()
    def __dlpack__(self,stream=None):
        cap=self.tensor.view(torch.uint8).__dlpack__(stream=stream)
        dl=C.cast(get_pointer(cap,b'dltensor'),C.POINTER(Tensor)).contents
        if (dl.dtype.code,dl.dtype.bits,dl.dtype.lanes)!=(1,8,1):
            raise RuntimeError('Unexpected uint8 DLPack encoding')
        dl.dtype=Dtype(*{torch.float4_e2m1fn_x2:(17,4,2),torch.float8_e4m3fn:(10,8,1),torch.float8_e5m2:(12,8,1)}[self.tensor.dtype])
        return cap

def wrap(tensor):
    if isinstance(tensor,torch.Tensor) and tensor.dtype in (torch.float4_e2m1fn_x2,torch.float8_e4m3fn,torch.float8_e5m2):
        return PackedFloat4(tensor)
    return tensor
