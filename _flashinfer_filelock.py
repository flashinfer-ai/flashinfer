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

"""Native descriptor locks without importing FlashInfer's GPU dependencies."""

import errno
import os
from typing import NoReturn

if os.name == "nt":
    from portalocker import AlreadyLocked, LockException, LockFlags

    # The default CRT backend limits blocking waits to about ten seconds.
    from portalocker.portalocker import Win32Locker as _Locker

    _locker = _Locker()

    def _raise_lock_error(error: LockException) -> NoReturn:
        """Keep existing contention and native-error handlers working."""
        if isinstance(error, AlreadyLocked):
            raise BlockingIOError(errno.EAGAIN, error.strerror) from error
        cause = error.__cause__
        if isinstance(cause, (OSError, EOFError)):
            raise cause from None
        winerror = getattr(cause, "winerror", None)
        if winerror is not None:
            # Windows OSError derives the POSIX errno from the supplied winerror.
            raise OSError(0, error.strerror, None, winerror) from error
        raise OSError(errno.EIO, str(error)) from error

    def lock(fd: int, *, shared: bool = False, blocking: bool = True) -> None:
        """Lock a caller-owned descriptor; nonblocking contention raises BlockingIOError."""
        flags = LockFlags.SHARED if shared else LockFlags.EXCLUSIVE
        if not blocking:
            flags |= LockFlags.NON_BLOCKING
        try:
            _locker.lock(fd, flags)
        except LockException as error:
            _raise_lock_error(error)

    def unlock(fd: int) -> None:
        """Release a lock without closing or modifying the caller's file."""
        try:
            _locker.unlock(fd)
        except LockException as error:
            _raise_lock_error(error)

else:
    import fcntl

    def lock(fd: int, *, shared: bool = False, blocking: bool = True) -> None:
        """Lock a caller-owned descriptor; nonblocking contention raises BlockingIOError."""
        flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(fd, flags)

    def unlock(fd: int) -> None:
        """Release a lock without closing or modifying the caller's file."""
        fcntl.flock(fd, fcntl.LOCK_UN)
