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

# Logical host ABI for kernel_cake_w4a8_received_tokens_9c4fa8da35e8dbc19fde.
import torch

def normalize(args):
    if len(args) != 14:
        raise TypeError('expected 14 kernel arguments')
    values = []
    t = args[0]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('ids' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[1]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('expanded_to_permuted' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[2]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('permuted_to_token' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[3]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('permuted_to_expanded' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[4]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('cta_to_expert' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[5]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('cta_to_limit' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[6]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('permuted_size' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[7]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('nonexiting_ctas' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    values.append(args[8])
    values.append(args[9])
    values.append(args[10])
    if type(args[11]) is not int or args[11] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[11])
    if type(args[12]) is not int or args[12] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[12])
    if type(args[13]) is not int or args[13] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[13])
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
