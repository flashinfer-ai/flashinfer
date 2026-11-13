# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cutlass.cute.runtime import from_dlpack


def to_cute_tensor(
    t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True
):
    """Convert a torch tensor to a CuTe tensor with dynamic layout marking."""
    tensor = from_dlpack(
        t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi
    )
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)
