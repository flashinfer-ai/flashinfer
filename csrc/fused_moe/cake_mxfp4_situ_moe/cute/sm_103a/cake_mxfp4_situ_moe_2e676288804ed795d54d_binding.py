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

# Logical host ABI for kernel_cake_mxfp4_situ_moe_2e676288804ed795d54d.
import torch

def normalize(args):
    if len(args) != 24:
        raise TypeError('expected 24 kernel arguments')
    values = []
    t = args[0]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('expert_idx' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[1]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('mn_limit' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[2]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('num_groups_ptr' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[3]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('alt_expert_idx' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[4]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('alt_mn_limit' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[5]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('alt_num_groups_ptr' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[6]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('base_active_ptr' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[7]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('wide_list' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[8]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('wide_count' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[9]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('alt_wide_list' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[10]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('alt_wide_count' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[11]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('narrow_list' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[12]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('narrow_count' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[13]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('narrow_count_base' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    t = args[14]
    if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        raise ValueError('trace' + ' must be a contiguous CUDA tensor')
    values.append(t.view(-1))
    values.append(args[15])
    values.append(args[16])
    values.append(args[17])
    values.append(args[18])
    values.append(args[19])
    values.append(args[20])
    if type(args[21]) is not int or args[21] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[21])
    if type(args[22]) is not int or args[22] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[22])
    if type(args[23]) is not int or args[23] <= 0:
        raise ValueError('grid dimensions must be positive integers')
    values.append(args[23])
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
