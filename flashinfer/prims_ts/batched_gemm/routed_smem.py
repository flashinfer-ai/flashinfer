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

"""Opt-in shared-memory placement for native routed SFA."""

from dataclasses import replace

import cutlass
import cutlass.cute as cute
from cutlass.experimental.task_scheduling.memory import SmemAllocator


def make_smem_allocator(cfg):
    """Use the same placement policy for host validation and kernel tracing."""
    if cfg.routed_low_smem:
        return RoutedTile256SmemAllocator(cfg)
    if cfg.routed_sfa_first:
        return RoutedSfaFirstSmemAllocator()
    return SmemAllocator()


class RoutedSfaFirstSmemAllocator(SmemAllocator):
    """Keep native SFA at low addresses without fixing other payload offsets."""

    def compute_layout(self):
        # The base allocator stably packs by alignment. Native SFA and the
        # operand buffers have the same alignment, so this stable ordering
        # puts SFA first while preserving all other payload/barrier ordering.
        # Do not reorder pipeline registration: it defines barrier ownership.
        self._allocations.sort(
            key=lambda allocation: not allocation.name.endswith("_sf_ldg")
        )
        super().compute_layout()


class RoutedTile256SmemAllocator(SmemAllocator):
    """Keep control storage below the operand buffers for the validated recipe.

    Validation restricts buffer depths and tile geometry before this allocator
    is used. Assert sizes, alignment and disjoint lifetimes here as a second
    guard against silently invalidating the fixed placement.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def control_regions(self):
        if self.cfg.is_static_persistent:
            return {
                "clc_response": (0, 32),
                "workid_mbarriers": (456, 32),
                "tmem_dealloc_mbar": (488, 8),
                "tmem_ptr_i32": (496, 8),
            }
        return {
            "clc_response": (528, 48),
            "workid_mbarriers": (456, 48),
            "work_throttle_mbarriers": (576, 48),
            "tmem_dealloc_mbar": (504, 8),
            "tmem_ptr_i32": (512, 8),
        }

    def barrier_regions(self):
        regions = {
            "LdgstsActivations": (40, 8),
            "ProxyCluster": (296, 8),
            "TmemC": (424, 2),
        }
        if self.cfg.has_tma_route:
            assert self.cfg.uses_routed_weight_pipeline
            # TMA A and LDGSTS SFA have distinct completion mechanisms. Keep
            # both rings below the payloads without moving existing barriers.
            del regions["LdgstsActivations"]
            regions.update(SmemTmaGather=(40, 8), SmemSfA=(624, 8))
        if self.cfg.uses_routed_weight_pipeline:
            regions["TmaWeights"] = (168, 8)
        else:
            regions.update(SmemB=(168, 8), SmemSfB=(624, 8))
        return regions

    def compute_layout(self):
        super().compute_layout()
        assert 0 < self.cfg.num_bytes_c_smem_scratch <= 16384
        controls = self.control_regions()
        payloads = {
            "_sc": (1024, self.cfg.num_bytes_c_smem_scratch),
            "_sf_ldg": (17408, 16384),
            "_gather": (33792, 131072),
            "_g4": (33792, 131072),
            "_b": (164864, 131072),
            "_sfb": (295936, 32768),
        }
        regions = {}
        for allocation in self._allocations:
            name = allocation.name
            if name in controls:
                offset, size = controls[name]
            else:
                matches = [v for suffix, v in payloads.items() if name.endswith(suffix)]
                assert len(matches) == 1, name
                offset, size = matches[0]
            assert allocation.size_bytes == size, (name, allocation.size_bytes, size)
            assert offset % allocation.alignment == 0, (name, offset)
            allocation.offset = offset
            regions[name] = (offset, size)
        for name, (offset, stages) in self.barrier_regions().items():
            regions["barrier:" + name] = (offset, stages * 16)
        ordered = sorted(
            (offset, offset + size, name) for name, (offset, size) in regions.items()
        )
        for left, right in zip(ordered, ordered[1:], strict=False):
            assert left[1] <= right[0], (left, right)
        assert max(end for _, end, _ in ordered) <= 328704
        # Keep the validated dynamic-SMEM launch size through the SFB end.
        self._total = 328704

    def assign_barrier_ptrs(self):
        if self._smem_base is None:
            return
        offsets = self.barrier_regions()
        offsets["WorkQueue"] = (456, self.cfg.num_stages_workid)
        for resource in [*self._barrier_resources, *self._barrier_groups]:
            config = resource.pipeline_config
            if config.barrier_ptr is not None:
                continue
            offset, stages = offsets[resource.name]
            assert config.num_stages == stages, (resource.name, config.num_stages)
            assert getattr(resource, "num_barriers_per_stage", 2) == 2
            pointer = cute.make_ptr(
                cutlass.Int64,
                self._smem_base.data_ptr() + offset,
                mem_space=cutlass.AddressSpace.smem,
            )
            resource.pipeline_config = replace(config, barrier_ptr=pointer)
