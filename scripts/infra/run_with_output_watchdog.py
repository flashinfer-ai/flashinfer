#!/usr/bin/env python3
"""Run a command and diagnose it if it stops producing output."""

from __future__ import annotations

import argparse
import contextlib
import heapq
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


COMPILER_PROCESS_NAMES = {
    "ninja",
    "sccache",
    "nvcc",
    "cicc",
    "ptxas",
    "fatbinary",
    "gcc",
    "g++",
    "cc1plus",
    "ld",
    "c++",
}


def _run_diagnostic(command: list[str], timeout: float = 10) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"diagnostic command failed: {error}\n"
    return result.stdout


def _read_proc_file(pid: int, name: str, limit: int = 64 * 1024) -> str:
    path = Path("/proc") / str(pid) / name
    try:
        return path.read_text(errors="replace")[:limit]
    except OSError as error:
        return f"unavailable: {error}\n"


def _read_system_file(path: Path, limit: int = 64 * 1024) -> str:
    try:
        return path.read_text(errors="replace")[:limit]
    except OSError as error:
        return f"unavailable: {error}\n"


def _compiler_pids() -> list[int]:
    if not Path("/proc").is_dir():
        return []
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
            command_names = {
                Path(argument.decode(errors="replace")).name
                for argument in arguments
                if argument
            }
            process_name = (entry / "comm").read_text().strip()
        except OSError:
            continue
        if process_name in COMPILER_PROCESS_NAMES or any(
            name in COMPILER_PROCESS_NAMES or name.endswith(("-gcc", "-g++", "-ld"))
            for name in command_names
        ):
            matches.append(int(entry.name))
    return sorted(matches)


def _fd_snapshot(pid: int, limit: int = 64) -> str:
    fd_dir = Path("/proc") / str(pid) / "fd"
    try:
        entries = sorted(fd_dir.iterdir(), key=lambda path: int(path.name))[:limit]
    except OSError as error:
        return f"unavailable: {error}\n"
    lines = []
    for entry in entries:
        try:
            target = os.readlink(entry)
        except OSError as error:
            target = f"unavailable: {error}"
        lines.append(f"{entry.name}: {target}")
    return "\n".join(lines) + "\n"


def _thread_snapshot(pid: int, limit: int = 256) -> str:
    task_dir = Path("/proc") / str(pid) / "task"
    try:
        tasks = sorted(task_dir.iterdir(), key=lambda path: int(path.name))[:limit]
    except OSError as error:
        return f"unavailable: {error}\n"
    rendered = []
    for task in tasks:
        tid = int(task.name)
        rendered.append(f"thread {tid}\n")
        for name in ("comm", "wchan", "syscall", "stack"):
            rendered.extend(
                (
                    f"--- /proc/{pid}/task/{tid}/{name} ---\n",
                    _read_system_file(task / name),
                )
            )
    return "".join(rendered)


def _newest_files(root: Path, limit: int = 100, timeout: float = 20) -> str:
    newest: list[tuple[float, int, str]] = []
    deadline = time.monotonic() + timeout
    for directory, _, filenames in os.walk(root):
        if time.monotonic() >= deadline:
            break
        for filename in filenames:
            path = Path(directory) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            item = (stat.st_mtime, stat.st_size, str(path))
            if len(newest) < limit:
                heapq.heappush(newest, item)
            else:
                heapq.heappushpop(newest, item)
    return "".join(
        f"{modified:.6f} {size} {path}\n" for modified, size, path in sorted(newest)
    )


def _collect_diagnostics(root_pid: int, idle_seconds: float) -> str:
    sections = [
        (
            "watchdog",
            f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
            f"root_pid={root_pid}\n"
            f"no_output_seconds={idle_seconds:.1f}\n",
        ),
        ("uptime", _run_diagnostic(["uptime"])),
        ("memory", _run_diagnostic(["free", "-h"])),
        ("virtual memory", _run_diagnostic(["vmstat", "1", "5"])),
        ("filesystem", _run_diagnostic(["df", "-h", ".", "/tmp"])),
        (
            "cgroup memory events",
            _read_system_file(Path("/sys/fs/cgroup/memory.events")),
        ),
        (
            "cpu pressure",
            _read_system_file(Path("/proc/pressure/cpu")),
        ),
        (
            "memory pressure",
            _read_system_file(Path("/proc/pressure/memory")),
        ),
        (
            "io pressure",
            _read_system_file(Path("/proc/pressure/io")),
        ),
        (
            "process tree",
            _run_diagnostic(
                [
                    "ps",
                    "-eo",
                    "pid,ppid,pgid,sid,stat,etime,time,pcpu,pmem,rss,wchan:24,args",
                    "--forest",
                ]
            ),
        ),
        ("sccache stats", _run_diagnostic(["sccache", "--show-stats"])),
        (
            "sccache advanced stats",
            _run_diagnostic(["sccache", "--show-adv-stats"]),
        ),
        (
            "newest build files",
            _newest_files(Path.cwd()),
        ),
    ]

    rendered = [
        "JIT-cache build output watchdog fired while the command was still alive.\n"
    ]
    for title, body in sections:
        rendered.extend((f"\n===== {title} =====\n", body))

    for pid in _compiler_pids():
        rendered.append(f"\n===== compiler process {pid} =====\n")
        for name in ("cmdline", "status", "wchan", "syscall", "stack", "io"):
            body = _read_proc_file(pid, name).replace("\0", " ")
            rendered.extend((f"--- /proc/{pid}/{name} ---\n", body))
        rendered.extend((f"--- /proc/{pid}/fd ---\n", _fd_snapshot(pid)))
        rendered.extend((f"--- /proc/{pid}/task ---\n", _thread_snapshot(pid)))

    return "".join(rendered)


def _terminate_process_group(
    process: subprocess.Popen[bytes], grace_seconds: float
) -> None:
    def process_group_has_live_members() -> bool:
        proc_root = Path("/proc")
        if proc_root.is_dir():
            try:
                entries = list(proc_root.iterdir())
            except OSError:
                entries = []
            else:
                for entry in entries:
                    if not entry.name.isdigit():
                        continue
                    try:
                        stat_fields = (
                            (entry / "stat").read_text().rpartition(")")[2].split()
                        )
                        state = stat_fields[0]
                        process_group = int(stat_fields[2])
                    except (IndexError, OSError, ValueError):
                        continue
                    if process_group == process.pid and state != "Z":
                        return True
                return False

        try:
            result = subprocess.run(
                ["ps", "-axo", "pgid=,stat="],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        else:
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    fields = line.split(None, 1)
                    if len(fields) == 2 and fields[0] == str(process.pid):
                        if not fields[1].startswith("Z"):
                            return True
                return False

        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def wait_for_process_group(timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while process_group_has_live_members():
            process.poll()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.1, remaining))
        process.poll()
        return True

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return

    if not wait_for_process_group(grace_seconds):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        wait_for_process_group(5)

    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


def run(args: argparse.Namespace) -> int:
    child_environment = os.environ.copy()
    child_environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        args.command,
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None

    def forward_signal(signum: int, _frame: object) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    last_output = time.monotonic()

    while selector.get_map():
        idle_seconds = time.monotonic() - last_output
        remaining = args.timeout_seconds - idle_seconds
        if remaining <= 0:
            diagnostics = _collect_diagnostics(process.pid, idle_seconds)
            sys.stderr.write(f"\n{diagnostics}")
            sys.stderr.flush()
            try:
                if args.diagnostics_file:
                    path = Path(args.diagnostics_file)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(diagnostics)
            except OSError as error:
                sys.stderr.write(
                    f"watchdog could not write {args.diagnostics_file}: {error}\n"
                )
                sys.stderr.flush()
            finally:
                _terminate_process_group(process, args.term_grace_seconds)
            return 124

        events = selector.select(timeout=min(remaining, 1.0))
        for key, _ in events:
            chunk = os.read(key.fd, 64 * 1024)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                last_output = time.monotonic()
            else:
                selector.unregister(key.fileobj)

    return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--term-grace-seconds", type=float, default=120)
    parser.add_argument("--diagnostics-file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.timeout_seconds <= 0 or args.term_grace_seconds < 0:
        parser.error("timeouts must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
