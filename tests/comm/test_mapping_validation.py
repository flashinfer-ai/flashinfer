# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import pytest

from flashinfer.comm.mapping import Mapping


@pytest.mark.parametrize("dim", ["tp_size", "pp_size", "cp_size"])
def test_auto_parallel_rejects_non_unit_parallelism(dim):
    """auto_parallel forces tp/pp/cp_rank to 0, so every parallelism degree
    must be 1. Each of the three must be rejected on its own."""
    with pytest.raises(ValueError, match="auto parallel"):
        Mapping(world_size=1, rank=0, auto_parallel=True, **{dim: 4})


def test_auto_parallel_accepts_unit_parallelism():
    mapping = Mapping(
        world_size=1, rank=0, tp_size=1, pp_size=1, cp_size=1, auto_parallel=True
    )
    assert (mapping.tp_rank, mapping.pp_rank, mapping.cp_rank) == (0, 0, 0)


def test_non_auto_parallel_allows_context_parallelism():
    mapping = Mapping(
        world_size=4, rank=1, tp_size=1, pp_size=1, cp_size=4, auto_parallel=False
    )
    assert mapping.cp_rank == 1
