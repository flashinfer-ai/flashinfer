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

"""Disk fixtures of the actual DKG kernel inputs, including packed storage.

Both kernels use K-major packed A/B, linear SFA and R128c4 SFB. Hash the
physical bytes, not a contiguous copy of DKG's logical sub-byte tensor view.
All host transfers and hashing happen outside the measured kernel region.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def physical_bytes(tensor, size=None):
    assert tensor.storage_offset() == 0
    raw = tensor.as_strided((tensor.numel(),), (1,)).view(torch.uint8)
    return raw if size is None else raw[:size]


def sha256(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


class Fixture:
    def __init__(self, path, shape):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.source_tensors = self.manifest["tensors"].copy()
        assert self.manifest["shape"][:3] == list(shape)[:3]
        assert self.manifest["shape"][3] >= shape[3]
        self.experts = shape[3]
        self.source_experts = self.manifest["shape"][3]
        self.prefix = self.experts < self.source_experts
        if self.prefix:
            self.manifest["source_shape"] = self.manifest["shape"]
            self.manifest["shape"] = list(shape)
            for name in ("B", "SFB", "route"):
                value = self._array(name)
                self.manifest["tensors"][name] = {
                    "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
                    "bytes": value.nbytes,
                    "dtype": self.manifest["tensors"][name]["dtype"],
                }

    def _array(self, name):
        array = np.load(self.path / (name + ".npy"), mmap_mode="c", allow_pickle=False)
        assert (
            hashlib.sha256(memoryview(array).cast("B")).hexdigest()
            == self.source_tensors[name]["sha256"]
        ), f"source checksum mismatch: {name}"
        if self.prefix and name in ("B", "SFB", "route"):
            array = array[: array.size // self.source_experts * self.experts]
        return array

    def load(self, name, device="cpu"):
        # Copy-on-write memory map avoids a second full host copy of B.
        array = self._array(name)
        assert (
            hashlib.sha256(memoryview(array).cast("B")).hexdigest()
            == self.manifest["tensors"][name]["sha256"]
        )
        tensor = torch.from_numpy(array)
        if name in ("A", "B", "SFA", "SFB") and tensor.dtype != torch.uint8:
            raise ValueError(
                f"{name} fixture must use packed uint8 storage, got {tensor.dtype}"
            )
        return tensor.to(device=device)

    def audit(self, name, tensor):
        # A matching prefix must not hide corruption elsewhere in its source.
        self._array(name)
        digest = sha256(physical_bytes(tensor))
        assert digest == self.manifest["tensors"][name]["sha256"], (name, digest)
        return digest
