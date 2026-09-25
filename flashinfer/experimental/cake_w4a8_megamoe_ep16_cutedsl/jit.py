# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collectively distribute one CuTe image and prepare ordinary eager launches."""

import functools
import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist

from ...jit import env
from ...jit.cute_dsl_core import JitSpecCuteDsl
from .driver import DeviceImage, PreparedLaunch, native_module


class ImageSpec(JitSpecCuteDsl):
    """Retain the exact embedded device image beside the usual CuTe object."""

    @property
    def cubin_path(self):
        return self.object_path.with_suffix(".cubin")

    def try_load(self):
        if not self.cubin_path.is_file() or not self.object_path.is_file():
            return None
        image = self.cubin_path.read_bytes()
        if (
            not image.startswith(b"\x7fELF")
            or self.object_path.read_bytes().count(image) != 1
        ):
            return None
        return super().try_load()

    def _export(self):
        super()._export()
        image = self._compiled_kernel.__cubin__
        if (
            not isinstance(image, bytes)
            or self.object_path.read_bytes().count(image) != 1
        ):
            raise RuntimeError("CuTe object must contain its exact device image once")
        temporary = self.cubin_path.with_suffix(f".cubin.tmp.{os.getpid()}")
        temporary.write_bytes(image)
        os.replace(temporary, self.cubin_path)


class _Module:
    def __init__(self, device, identity, path):
        self.device = device
        self.identity = identity
        self.object_path = path

    def prepare(self, arguments, stream, owner):
        return PreparedLaunch(self.device, arguments, stream, owner)


@functools.cache
def _load_image(path, image_hash, device_index):
    if torch.cuda.current_device() != device_index:
        raise RuntimeError("the image must load on its owning device")
    image = Path(path).read_bytes()
    if hashlib.sha256(image).hexdigest() != image_hash:
        raise RuntimeError("CuTe image digest mismatch")
    return DeviceImage(image)


def make_spec():
    from .kernel import compile_program

    source_dir = Path(__file__).resolve().parent
    names = ("kernel.py", "abi.py", "driver.py", "submission.cpp", "jit.py")
    source_hash = hashlib.sha256(
        b"".join((source_dir / name).read_bytes() for name in names)
    ).hexdigest()
    return ImageSpec(
        "cake_w4a8_megamoe_ep16_cutedsl", "unified", compile_program, source_hash
    )


def load_module(group):
    spec = make_spec()
    metadata = [None] * dist.get_world_size(group)
    dist.all_gather_object(metadata, spec.expected_meta, group=group)
    if any(item != metadata[0] for item in metadata):
        raise RuntimeError(
            "all ranks must use identical CuTe sources and compiler stacks"
        )
    if spec.expected_meta["arch"] != "sm103a":
        raise ValueError("the compiled target must be SM103a")
    payload = [None]
    if dist.get_rank(group) == 0:
        try:
            spec.build_and_load()
            blob, image = spec.object_path.read_bytes(), spec.cubin_path.read_bytes()
            if blob.count(image) != 1:
                raise RuntimeError(
                    "device image is not the image embedded in the CuTe object"
                )
            payload[0] = dict(
                blob=blob,
                image=image,
                object_sha256=hashlib.sha256(blob).hexdigest(),
                cubin_sha256=hashlib.sha256(image).hexdigest(),
            )
        except Exception as error:
            payload[0] = dict(error=f"{type(error).__name__}: {error}")
    dist.broadcast_object_list(payload, src=dist.get_global_rank(group, 0), group=group)
    result = payload[0]
    if "error" in result:
        raise RuntimeError("CuTe collective compilation failed: " + result["error"])
    blob, image = result["blob"], result["image"]
    if (
        hashlib.sha256(blob).hexdigest() != result["object_sha256"]
        or hashlib.sha256(image).hexdigest() != result["cubin_sha256"]
        or blob.count(image) != 1
    ):
        raise RuntimeError("CuTe object/image broadcast digest mismatch")
    directory = env.FLASHINFER_JIT_DIR / "cake_w4a8_megamoe_ep16_cutedsl_objects"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (result["object_sha256"] + ".o")
    for destination, content in ((path, blob), (path.with_suffix(".cubin"), image)):
        if not destination.exists() or destination.read_bytes() != content:
            temporary = destination.with_suffix(f".tmp.{os.getpid()}")
            temporary.write_bytes(content)
            os.replace(temporary, destination)
    device = _load_image(
        str(path.with_suffix(".cubin")),
        result["cubin_sha256"],
        torch.cuda.current_device(),
    )
    native_module()  # Host-only compilation belongs to collective setup.
    identity = dict(
        spec.expected_meta,
        object_sha256=result["object_sha256"],
        cubin_sha256=result["cubin_sha256"],
        entry=spec.symbol,
        device_entry=device.symbol,
        submission="prepared-eager-cuLaunchKernelEx",
    )
    return _Module(device, identity, path)
