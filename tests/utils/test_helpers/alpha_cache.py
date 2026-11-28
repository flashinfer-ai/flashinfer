"""
Copyright (c) 2026 by FlashInfer team.

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

import sys
import pytest
import torch


def fill_alpha_one_late(monkeypatch) -> list:
    """Delay the cached 1.0's fill; its memory holds 0 until the fill runs.

    Returns a list that gets one entry per delayed fill, so a test can check
    that the fill it delays really ran. Its ``synced`` attribute counts the
    synchronize calls on the private stream after a fill was queued. A test
    checks it as well as the output: a host that pauses longer than the delay
    would hide a missing synchronize.

    It patches torch.cuda.Stream and torch.ones for the whole process until the
    test ends, so use it only in tests that run no other threads.
    """
    sleep = getattr(torch.cuda, "_sleep", None)  # a private PyTorch helper
    if sleep is None:
        pytest.skip("needs torch.cuda._sleep to delay the fill")

    class Fills(list):
        synced = 0

    fills = Fills()
    real_stream = torch.cuda.Stream

    class Private(real_stream):
        def synchronize(self):
            if fills:
                fills.synced += 1
            return super().synchronize()

    private = Private()
    real_ones = torch.ones

    class Stream(real_stream):
        # Only the torch.cuda.Stream(device=...) call made directly in
        # _prepare_alpha_for_launch gets `private`; every other Stream, such as
        # the one torch.cuda.current_stream() builds, is a real one. `private`
        # is not an instance of this subclass, so Python does not call
        # __init__ on it again when __new__ returns it.
        def __new__(cls, *args, **kwargs):
            caller = sys._getframe(1).f_code.co_name
            if caller == "_prepare_alpha_for_launch":
                return private
            return super().__new__(cls, *args, **kwargs)

    # On the private stream, return a tensor that holds 0 until a busy kernel
    # finishes, then 1.0.
    def ones(*args, **kwargs):
        if torch.cuda.current_stream() != private:
            return real_ones(*args, **kwargs)
        one = torch.zeros(*args, **kwargs)
        private.synchronize()
        # Keep the private stream busy before 1.0 is written.
        sleep(200_000_000)
        fills.append(one)
        return one.fill_(1.0)

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch, "ones", ones)
    return fills
