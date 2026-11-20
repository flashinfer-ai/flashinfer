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

# Logical host ABI for kernel_cake_mxfp4_situ_moe_7f291b75fc80b9a071f0.
import torch

def normalize(args):
    if len(args) != 38:
        raise TypeError('expected 38 kernel arguments')
    values = []
    t = args[0]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('ids_src' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[1]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('weights_src' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[2]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('ids_dst' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[3]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('weights_dst' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[4]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('output' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[5]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('tile_expert' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[6]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('tile_limit' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[7]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('expanded' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[8]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('permuted' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[9]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('padded_total' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[10]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('active_total' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[11]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('wide_list' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[12]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('wide_count' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[13]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('narrow_list' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[14]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('narrow_count' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[15]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('all_list' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[16]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('all_count' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[17]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('wide_expert' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[18]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('wide_limit' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[19]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('chunk_counts' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    values.append(args[20])
    values.append(args[21])
    values.append(args[22])
    values.append(args[23])
    values.append(args[24])
    values.append(args[25])
    values.append(args[26])
    values.append(args[27])
    values.append(args[28])
    values.append(args[29])
    values.append(args[30])
    values.append(args[31])
    values.append(args[32])
    values.append(args[33])
    values.append(args[34])
    if type(args[35]) is not int or args[35] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[35])
    if type(args[36]) is not int or args[36] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[36])
    if type(args[37]) is not int or args[37] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[37])
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
