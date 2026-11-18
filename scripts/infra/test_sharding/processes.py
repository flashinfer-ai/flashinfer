"""Start pytest subprocesses in their own process group and end the whole group.

POSIX runs each child as a session leader and signals the group with
``os.killpg``: SIGTERM, a grace period, then SIGKILL.

Windows has no process-group signals that reach every descendant, so each child
is started with ``CREATE_NEW_PROCESS_GROUP`` and assigned to its own Job Object.
The job tracks descendants even after the leader exits. Termination sends
CTRL_BREAK_EVENT to the console process group, waits for the grace period, then
calls ``TerminateJobObject``. If the job cannot be set up, the launch fails:
without a job, descendants of an exited leader could not be found again.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import weakref
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

if sys.platform == "win32":
    # Reported in termination_signal fields in place of the POSIX signal names.
    GRACEFUL_TERMINATION = "CTRL_BREAK_EVENT"
    FORCED_TERMINATION = "TerminateJobObject"
else:
    GRACEFUL_TERMINATION = "SIGTERM"
    FORCED_TERMINATION = "SIGKILL"


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.QueryInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    _kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL

    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    # Exit code for processes ended by TerminateJobObject: the shell's status
    # for SIGKILL, outside pytest's 0/1 so it never reads as a test result.
    _FORCED_EXIT_CODE = 137

    class _JobAccounting(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    def _win_error(function: str) -> OSError:
        error = ctypes.get_last_error()
        return OSError(error, f"{function} failed: {ctypes.FormatError(error)}")

    class _Job:
        def __init__(self, handle: int) -> None:
            self.handle = handle
            self._finalizer = weakref.finalize(self, _kernel32.CloseHandle, handle)

        def active_processes(self) -> int:
            info = _JobAccounting()
            if not _kernel32.QueryInformationJobObject(
                self.handle,
                _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
                None,
            ):
                raise _win_error("QueryInformationJobObject")
            return int(info.ActiveProcesses)

        def terminate(self) -> None:
            if not _kernel32.TerminateJobObject(self.handle, _FORCED_EXIT_CODE):
                raise _win_error("TerminateJobObject")

    _jobs: weakref.WeakKeyDictionary[subprocess.Popen[Any], _Job] = (
        weakref.WeakKeyDictionary()
    )

    def _assign_job(process: subprocess.Popen[Any]) -> None:
        # The child runs before it joins the job. Python and pytest start up
        # long before they spawn descendants, so none escape in practice.
        job_handle = _kernel32.CreateJobObjectW(None, None)
        if not job_handle:
            raise _win_error("CreateJobObjectW")
        job = _Job(job_handle)
        process_handle = _kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, process.pid
        )
        if not process_handle:
            raise _win_error("OpenProcess")
        try:
            if not _kernel32.AssignProcessToJobObject(job_handle, process_handle):
                raise _win_error("AssignProcessToJobObject")
        finally:
            _kernel32.CloseHandle(process_handle)
        _jobs[process] = job

    def _job(process: subprocess.Popen[Any]) -> _Job:
        try:
            return _jobs[process]
        except KeyError:
            raise ValueError(
                f"process {process.pid} was not started by start_process_group"
            ) from None

    def _group_alive(process: subprocess.Popen[Any]) -> bool:
        return _job(process).active_processes() > 0

    def _request_graceful_exit(process: subprocess.Popen[Any]) -> bool:
        if not _group_alive(process):
            return False
        # Without a shared console the group cannot receive console events;
        # the grace period still applies before the forced kill.
        with suppress(OSError):
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
        return True

    def _force_exit(process: subprocess.Popen[Any]) -> bool:
        job = _job(process)
        if job.active_processes() == 0:
            return False
        job.terminate()
        return True

else:

    def _group_alive(process: subprocess.Popen[Any]) -> bool:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _request_graceful_exit(process: subprocess.Popen[Any]) -> bool:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    def _force_exit(process: subprocess.Popen[Any]) -> bool:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return False
        return True


def start_process_group(
    command: Sequence[str], **popen_kwargs: Any
) -> subprocess.Popen[Any]:
    """Start ``command`` as the leader of a group ``terminate_process_group`` ends."""
    if sys.platform == "win32":
        flags = popen_kwargs.pop("creationflags", 0)
        process = subprocess.Popen(
            command,
            creationflags=flags | subprocess.CREATE_NEW_PROCESS_GROUP,
            **popen_kwargs,
        )
        try:
            _assign_job(process)
        except OSError:
            # Refuse a group that could not be cleaned up once its leader exits.
            process.kill()
            process.wait()
            raise
        return process
    return subprocess.Popen(command, start_new_session=True, **popen_kwargs)


def terminate_process_group(
    process: subprocess.Popen[Any], grace_seconds: float
) -> str:
    """End ``process`` and its descendants; return how the group was ended.

    The result is ``GRACEFUL_TERMINATION``, ``FORCED_TERMINATION``,
    ``exit-<code>`` when the group was already gone but the leader's exit code
    was known, or ``already-exited``.
    """
    prior_returncode = process.poll()
    if not _request_graceful_exit(process):
        process.poll()
        return (
            f"exit-{prior_returncode}"
            if prior_returncode is not None
            else "already-exited"
        )

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while True:
        process.poll()
        if not _group_alive(process):
            process.wait()
            return GRACEFUL_TERMINATION
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))

    if not _force_exit(process):
        process.wait()
        return GRACEFUL_TERMINATION
    process.wait()
    return FORCED_TERMINATION
