# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Cross-stream ordering for the per-device state the impl modules share.

Every impl of the fused GDN decode step keeps some state per device that
outlives a call -- the persistent kernel's grid-barrier buffer, the
CuTe-DSL impl's ``part``/``qkv_act`` workspace.  Launches issued on one
stream are ordered by that stream, but a caller that switches streams would
otherwise have two calls in flight over the same buffer.

Both shipped impls order such a call instead of racing it; the mechanism is
identical, so it lives here rather than being copied per impl (the README's
impl-module interface makes it a requirement for new impls too).
"""

from typing import Dict

import torch


def order_after_previous_stream(
    last_stream: Dict[str, torch.cuda.Stream], device: torch.device
) -> None:
    """Serialize shared per-device state across streams instead of racing it.

    ``last_stream`` is the caller module's ``device -> stream`` record of
    which stream last used the state; it is updated in place.  A call
    arriving on a different stream than the previous one first makes its own
    stream wait on an event recorded on the previous stream, so the earlier
    call is done with the shared buffers before the new one touches them.
    Cross-stream use is then correct, at the price of serializing those
    calls -- which the in-place conv/ssm state pools already require of any
    caller sharing a pool.  Steady state (one stream, the serving case)
    costs one stream compare and nothing on the device.

    Skipped during capture: ``torch.cuda.graph`` already forks the capture
    stream from the caller's stream, and touching another stream mid-capture
    is illegal.  NOT covered, and a caller-side requirement: two *replays* of
    captured graphs running concurrently on different streams share this
    state (as they share the caller's state pools).
    """
    if torch.cuda.is_current_stream_capturing():
        return
    key = str(device)
    current = torch.cuda.current_stream(device)
    previous = last_stream.get(key)
    if previous is not None and previous != current:
        # The event is created on the current *device*, which must be the one
        # the streams belong to.
        with torch.cuda.device(device):
            event = torch.cuda.Event()
            event.record(previous)
        current.wait_event(event)
    last_stream[key] = current
