#
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Logical host ABI for kernel_cake_mxfp4_situ_moe_9f7ee7d869f496fe083e.
import torch

def normalize(args):
    if len(args) != 31:
        raise TypeError('expected 31 kernel arguments')
    values = []
    t = args[0]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('A' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    if t.ndim < 3 or t.stride(-1) != 1:
        raise ValueError('invalid TMA rank or inner stride for A')
    dims = (128, 128, t.size(-3), (t.numel() // (t.size(-1) * t.size(-2) * t.size(-3))),)
    if any(d <= 0 for d in dims):
        raise ValueError('TMA dimensions must be positive')
    if dims[0] < 128:
        raise ValueError('TMA box exceeds source extent')
    if dims[1] < 128:
        raise ValueError('TMA box exceeds source extent')
    if dims[2] < 2:
        raise ValueError('TMA box exceeds source extent')
    if dims[3] < 1:
        raise ValueError('TMA box exceeds source extent')
    values.extend(dims)
    stride_bits = 128 * 4
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[1] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    stride_bits = 16384 * 4
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[2] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    stride_bits = (16384 * t.size(-3)) * 4
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[3] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    t = args[1]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('SFA' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    if t.ndim < 3 or t.stride(-1) != 1:
        raise ValueError('invalid TMA rank or inner stride for SFA')
    dims = (128, 4, t.size(-3), (t.numel() // (t.size(-1) * t.size(-2) * t.size(-3))),)
    if any(d <= 0 for d in dims):
        raise ValueError('TMA dimensions must be positive')
    if dims[0] < 128:
        raise ValueError('TMA box exceeds source extent')
    if dims[1] < 4:
        raise ValueError('TMA box exceeds source extent')
    if dims[2] < 2:
        raise ValueError('TMA box exceeds source extent')
    if dims[3] < 1:
        raise ValueError('TMA box exceeds source extent')
    values.extend(dims)
    stride_bits = 128 * 8
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[1] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    stride_bits = 512 * 8
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[2] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    stride_bits = (512 * t.size(-3)) * 8
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[3] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    t = args[2]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('B' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[3]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('SFB' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[4]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('out' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[5]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('tile_idx_to_expert_idx' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[6]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('tile_idx_to_mn_limit' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[7]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('num_non_exiting_tiles' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[8]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('tile_idx_to_row_group' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[9]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('alpha' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[10]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('permuted_idx_to_expanded_idx' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[11]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('token_final_scales' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    values.append(args[12])
    values.append(args[13])
    values.append(args[14])
    values.append(args[15])
    values.append(args[16])
    values.append(args[17])
    values.append(args[18])
    t = args[19]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('situ_beta' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[20]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('situ_linear_beta' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[21]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('act_sf' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[22]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('zero_buf' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    values.append(args[23])
    values.append(args[24])
    values.append(args[25])
    values.append(args[26])
    t = args[27]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('dbg' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    if type(args[28]) is not int or args[28] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[28])
    if type(args[29]) is not int or args[29] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[29])
    if type(args[30]) is not int or args[30] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[30])
    return tuple(values)

class Kernel:
    def __init__(self, executor):
        self.executor = executor
        self.cached = None

    def run(self, *args):
        signature = tuple((a.data_ptr(), a.dtype, a.device, tuple(a.shape), tuple(a.stride()))
                          if isinstance(a, torch.Tensor) else (type(a), a) for a in args)
        if self.cached is None or self.cached[0] != signature:
            self.cached = (signature, normalize(args))
        self.executor(*self.cached[1])
