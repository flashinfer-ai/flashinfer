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

# Logical host ABI for kernel_cake_mxfp4_situ_moe_872da1f1589f27428db1.
import torch

def normalize(args):
    if len(args) != 13:
        raise TypeError('expected 13 kernel arguments')
    values = []
    t = args[0]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('rows' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[1]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('perm' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[2]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('weights' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[3]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('out' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[4]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('narrow_count' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[5]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('wide_count' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    values.append(args[6])
    values.append(args[7])
    values.append(args[8])
    values.append(args[9])
    if type(args[10]) is not int or args[10] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[10])
    if type(args[11]) is not int or args[11] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[11])
    if type(args[12]) is not int or args[12] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[12])
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
