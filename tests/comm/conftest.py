# Copyright (c) 2024 by FlashInfer team.
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
import torch.distributed as dist


def pytest_sessionfinish(session, exitstatus):
    """Cleanup torch.distributed at the end of pytest session.

    This runs after all tests complete but before Python shutdown,
    avoiding the "destroy_process_group() was not called" warning.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


"""
Shared test utilities for comm tests.
"""

import ctypes
import os

from flashinfer.comm.mnnvl import MnnvlMemory


def _check_pidfd_permissions() -> bool:
    """Check if pidfd_getfd syscall is available and permitted.

    This is required for MNNVL in containers - the SYS_PTRACE capability
    must be available for cross-process file descriptor sharing.
    """
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        SYS_pidfd_open = 434
        SYS_pidfd_getfd = 438

        # Try to open our own process and get our own fd
        my_pid = os.getpid()
        pidfd = syscall(SYS_pidfd_open, my_pid, 0)
        if pidfd < 0:
            return False

        # Try pidfd_getfd on stdin (fd=0) - this tests the permission
        # We don't actually need the result, just checking if it's permitted
        test_fd = syscall(SYS_pidfd_getfd, pidfd, 0, 0)
        os.close(pidfd)

        if test_fd < 0:
            err = ctypes.get_errno()
            if err == 1:  # EPERM - permission denied (container issue)
                return False
            # Other errors (like EBADF) are OK - permission check passed
        else:
            os.close(test_fd)

        return True
    except Exception:
        return False


def mnnvl_available() -> bool:
    """Check if MNNVL is fully available (hardware + container permissions)."""
    return MnnvlMemory.supports_mnnvl() and _check_pidfd_permissions()
