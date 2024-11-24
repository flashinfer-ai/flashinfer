"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pathlib
import threading
from typing import Any, Callable, List, Tuple

import torch

from .core import logger


def write_if_different(path: pathlib.Path, content: str) -> None:
    if path.exists():
        with open(path, "r") as f:
            if f.read() == content:
                return
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def parallel_load_modules(
    load_module_func_args: List[Tuple[Callable, List[Any]]],
):
    threads = []
    exceptions = []

    def wrapper(func, args):
        try:
            func(*args)
        except Exception as e:
            exceptions.append((func, e))

    for func, args in load_module_func_args:
        thread = threading.Thread(target=wrapper, args=(func, args))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if exceptions:
        for func, e in exceptions:
            print(f"Exception occurred in {func.__name__}: {e}")
        raise RuntimeError("One or more exceptions occurred during module loading")

    logger.info("Finished loading modules")


dtype_map = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
    torch.float8_e5m2: "__nv_fp8_e5m2",
    torch.int8: "int8_t",
    torch.uint8: "uint8_t",
    torch.int32: "int32_t",
    torch.uint32: "uint32_t",
    torch.int64: "int64_t",
    torch.uint64: "uint64_t",
}

filename_safe_dtype_map = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "e4m3",
    torch.float8_e5m2: "e5m2",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.int32: "i32",
    torch.uint32: "u32",
    torch.int64: "i64",
    torch.uint64: "u64",
}

pos_encoding_mode_literal = {
    0: "PosEncodingMode::kNone",
    1: "PosEncodingMode::kRoPELlama",
    2: "PosEncodingMode::kALiBi",
}

mask_mode_literal = {
    0: "MaskMode::kNone",
    1: "MaskMode::kCausal",
    2: "MaskMode::kCustom",
}
