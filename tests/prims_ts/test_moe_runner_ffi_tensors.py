# Copyright (c) 2026 by FlashInfer team.
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

import torch
import tvm_ffi

from flashinfer.prims_ts.moe.runner import _torch_views_of_ffi_tensors


def test_torch_tensor_is_preserved():
    tensor = torch.arange(4)

    (view,) = _torch_views_of_ffi_tensors([tensor])

    assert view is tensor


def test_ffi_tensor_can_be_converted_repeatedly():
    tensor = torch.arange(4)
    ffi_tensor = tvm_ffi.from_dlpack(tensor)

    (first_view,) = _torch_views_of_ffi_tensors([ffi_tensor])
    (second_view,) = _torch_views_of_ffi_tensors([ffi_tensor])

    assert first_view.data_ptr() == tensor.data_ptr()
    assert second_view.data_ptr() == tensor.data_ptr()
    torch.testing.assert_close(first_view, tensor)
    torch.testing.assert_close(second_view, tensor)
