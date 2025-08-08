"""
Copyright (c) 2025 by FlashInfer team.

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

import argparse
import csv
import json
from collections import namedtuple
from enum import Enum
from typing import Any, Dict, List, Tuple

import torch
from tg4perfetto import TraceGenerator


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    """
    Decode a profiler tag into (block_idx, group_idx, event_idx, event_type, sm_id).
    Tag layout:
      bits 0-1: event_type
      bits 2-11: event_idx
      bits 12-23: block_group_idx
      bits 24-31: sm_id
    """
    sm_id = (tag >> 24) & 0xFF
    block_group_idx = (tag >> 12) & 0xFFF
    event_idx = (tag >> 2) & 0x3FF
    event_type = tag & 0x3
    block_idx = block_group_idx // num_groups
    group_idx = block_group_idx % num_groups
    return block_idx, group_idx, event_idx, event_type, sm_id


def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    event_names: List[str],
    file_name: str,
) -> None:
    assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    tgen = TraceGenerator(file_name)

    pid_map = {}
    tid_map = {}
    track_map: Dict[Tuple[int, int, int], Any] = {}

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, event_idx, event_type, sm_id = decode_tag(
            tag, num_blocks, num_groups
        )

        # create trackers
        if block_idx not in pid_map:
            pid_map[block_idx] = tgen.create_group(f"sm_{sm_id}_block_{block_idx}")
        pid = pid_map[block_idx]
        if (block_idx, group_idx) not in tid_map:
            tid_map[(block_idx, group_idx)] = pid.create_group(f"group_{group_idx}")
        tid = tid_map[(block_idx, group_idx)]
        event = event_names[event_idx]

        if (block_idx, group_idx, event_idx) in track_map:
            track = track_map[(block_idx, group_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(block_idx, group_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()
