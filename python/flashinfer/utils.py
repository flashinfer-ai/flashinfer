"""
Copyright (c) 2023 by FlashInfer team.

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
import torch


class RotaryMode:
    NONE = 0
    LLAMA = 1

    FORMAT2STR = {0: "NONE", 1: "LLAMA"}


class TensorLayout:
    NHD = 0
    HND = 1

    FORMAT2STR = {0: "NHD", 1: "HND"}


def expand_5d(x: torch.Tensor, kv_layout: str):
    if not x.ndim in [4, 5]:
        raise ValueError("x must be 4D or 5D")
    if x.ndim == 4:
        # page_size == 1
        if kv_layout == "NHD":
            # expand to 5D on the 3nd last dimension
            return x.unsqueeze(-3)
        elif kv_layout == "HND":
            # expand to 5D on the 2nd last dimension
            return x.unsqueeze(-2)
        else:
            raise KeyError("Invalid kv_layout {}".format(kv_layout))
    return x


def check_rotary_mode(rotary_mode: str):
    if not hasattr(RotaryMode, rotary_mode):
        raise KeyError("Invalid rotary_mode {}".format(rotary_mode))


def check_kv_layout(kv_layout: str):
    if not hasattr(TensorLayout, kv_layout):
        raise KeyError("Invalide kv_layout {}".format(kv_layout))
