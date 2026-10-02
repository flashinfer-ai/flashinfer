#
# Copyright (c) 2023 by FlashInfer team.
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

# Logical host ABI for kernel_cake_w4a8_received_tokens_19aff305db55d3968abc.
import torch

def normalize(args):
    if len(args) != 11:
        raise TypeError('expected 11 kernel arguments')
    values = []
    t = args[0]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('A' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    if t.ndim < 2 or t.stride(-1) != 1:
        raise ValueError('invalid TMA rank or inner stride for A')
    if not (((2 * t.size(-1)) % 256) == 0):
        raise ValueError('TMA descriptor check failed for A')
    if not ((((t.numel() // (t.size(-1) * t.size(-2))) * t.size(-2)) % 128) == 0):
        raise ValueError('TMA descriptor check failed for A')
    dims = (128, 128, 2, ((2 * t.size(-1)) // 256), (((t.numel() // (t.size(-1) * t.size(-2))) * t.size(-2)) // 128),)
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
    if dims[4] < 1:
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
    stride_bits = 32768 * 4
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[3] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    stride_bits = (256 * t.size(-1)) * 4
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[4] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    t = args[1]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('B' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    if t.ndim < 2 or t.stride(-1) != 1:
        raise ValueError('invalid TMA rank or inner stride for B')
    if not ((t.size(-1) % 128) == 0):
        raise ValueError('TMA descriptor check failed for B')
    dims = (128, t.size(-2), (t.size(-1) // 128),)
    if any(d <= 0 for d in dims):
        raise ValueError('TMA dimensions must be positive')
    if dims[0] < 128:
        raise ValueError('TMA box exceeds source extent')
    if dims[1] < 8:
        raise ValueError('TMA box exceeds source extent')
    if dims[2] < 2:
        raise ValueError('TMA box exceeds source extent')
    values.extend(dims)
    stride_bits = t.size(-1) * 8
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[1] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    stride_bits = 128 * 8
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[2] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    t = args[2]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('SFA' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    if t.ndim < 3 or t.stride(-1) != 1:
        raise ValueError('invalid TMA rank or inner stride for SFA')
    dims = (t.size(-1), t.size(-2), t.size(-3), (t.numel() // (t.size(-1) * t.size(-2) * t.size(-3))),)
    if any(d <= 0 for d in dims):
        raise ValueError('TMA dimensions must be positive')
    if dims[0] < 32:
        raise ValueError('TMA box exceeds source extent')
    if dims[1] < 32:
        raise ValueError('TMA box exceeds source extent')
    if dims[2] < 1:
        raise ValueError('TMA box exceeds source extent')
    if dims[3] < 1:
        raise ValueError('TMA box exceeds source extent')
    values.extend(dims)
    stride_bits = t.size(-1) * 8
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[1] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    stride_bits = (t.size(-1) * t.size(-2)) * 8
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[2] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    stride_bits = ((t.size(-1) * t.size(-2)) * t.size(-3)) * 8
    if stride_bits < 0 or stride_bits % 128 or (stride_bits == 0 and dims[3] != 1):
        raise ValueError('TMA stride must be a nonnegative 16-byte multiple')
    values.append(stride_bits // 128)
    t = args[3]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('SFB' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[4]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('C' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[5]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('cta_to_expert' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[6]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('cta_to_limit' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[7]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('nonexiting_ctas' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    if type(args[8]) is not int or args[8] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[8])
    if type(args[9]) is not int or args[9] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[9])
    if type(args[10]) is not int or args[10] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[10])
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
