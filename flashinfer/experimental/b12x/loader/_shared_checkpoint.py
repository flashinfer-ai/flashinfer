"""Collective completion of routed checkpoint writes within a host-local group."""

from __future__ import annotations

from array import array
from bisect import bisect_right
from concurrent.futures import Future
import os
from pathlib import Path
import socket
from threading import Thread
import time

from ._shared_read_plan import Copy, plan_reads


# Failed collectives cannot prove that peer mappings have retired.
_retained_sessions = []


class SharedReadGroup:
    """Read each planned source span once and scatter through CUDA IPC.

    Every member must call ``finish`` at the same explicit routing boundary.
    This object never makes ordinary rank-local weight flushes collective.
    The supplied CPU process group owns communication timeout and rank failure.
    """

    def __init__(self, process_group, device):
        import torch
        import torch.distributed as dist

        self.group, self.device = process_group, device
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.native = self.executor = None
        self._initialization = self._initializer = None
        self.failed = False
        self.unsafe = False
        self.files = {}
        self.totals = {}
        self.summary = {}
        self.progress = None
        self.epoch = 0
        identity = dict(host=socket.gethostname(), boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        uuid=str(torch.cuda.get_device_properties(device).uuid))
        members = self._gather(identity)
        if (len({(m["host"], m["boot"]) for m in members}) != 1
                or len({m["uuid"] for m in members}) != self.world_size):
            raise ValueError("shared checkpoint reads require distinct CUDA devices on one host")

    def _gather(self, value):
        import torch.distributed as dist

        result = [None] * self.world_size
        dist.all_gather_object(result, value, group=self.group)
        return result

    def _phase(self, name, action):
        started = time.perf_counter()
        value = error = None
        try:
            if self.progress is not None:
                self.progress(name)
            value = action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.totals[name + "_seconds"] = self.totals.get(name + "_seconds", 0) + time.perf_counter() - started
        states = self._gather(dict(epoch=self.epoch, phase=name, error=error))
        errors = [f"rank {rank}: {s['error']}" for rank, s in enumerate(states) if s["error"]]
        if any(s["epoch"] != self.epoch or s["phase"] != name for s in states):
            errors.append("shared checkpoint completion boundaries differ between ranks")
        return value, errors

    @staticmethod
    def _identity(fd):
        stat = os.fstat(fd)
        path = str(Path(os.readlink(f"/proc/self/fd/{fd}")).resolve(strict=True))
        reopened = os.stat(path)
        def fields(stat):
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

        if fields(stat) != fields(reopened):
            raise RuntimeError("checkpoint pathname no longer identifies its open descriptor")
        return (path, *fields(stat))

    def start(self, session):
        """Initialize the rank-local reader while model weights are being routed."""
        if self._initialization is not None or self.executor is not None:
            return
        if session._gds is None:
            raise ValueError("shared checkpoint reads require cuFile")
        self.native = session._gds
        self._initialization = Future()
        arguments = (self.device, session.io_threads,
                     *(p.function for p in session._copy_programs))

        def initialize():
            try:
                self._initialization.set_result(self.native.owner_create(*arguments))
            except BaseException as exc:
                self._initialization.set_exception(exc)

        self._initializer = Thread(target=initialize, name="b12x-checkpoint-init")
        try:
            self._initializer.start()
        except Exception as exc:
            self._initializer = None
            self._initialization.set_exception(exc)

    def _prepare(self, session):
        import torch

        session._validate_source_identities()
        if self.executor is None:
            self.start(session)
            self.executor = self._initialization.result()
        records, host_records = array("Q"), array("Q")
        for index in range(0, len(session.records), 8):
            row = session.records[index:index + 8]
            (host_records if row[4] == 2 else records).extend(row)
        if host_records:
            session._execute(host_records)
        allocations = {}
        seen_storage = set()
        for destination in session.destinations:
            if destination.device.type != "cuda" or not destination.nbytes:
                continue
            storage = destination.untyped_storage()
            if storage._cdata in seen_storage:
                continue
            seen_storage.add(storage._cdata)
            base, size, handle = self.native.owner_export(self.executor, storage.data_ptr(), storage.nbytes())
            allocations[base] = (size, handle)
        ordered = sorted(allocations)
        identities = {fd: self._identity(fd) for fd in {records[i] for i in range(0, len(records), 8)}}
        wire = array("Q")
        destination_bytes = 0
        ranges = []
        sources = sorted(set(identities.values()))
        source_index = {identity: index for index, identity in enumerate(sources)}
        for index in range(0, len(records), 8):
            fd, offset, width, pointer, expand, rows, source_stride, destination_stride = records[index:index + 8]
            allocation = bisect_right(ordered, pointer) - 1
            if allocation < 0:
                raise ValueError("routed destination has no exported allocation")
            base = ordered[allocation]
            extent = (rows - 1) * destination_stride + width * (1 + expand)
            if pointer - base + extent > allocations[base][0]:
                raise ValueError("routed destination exceeds its exported allocation")
            ranges.append((pointer, pointer + extent))
            destination_bytes += rows * width * (1 + expand)
            wire.extend((source_index[identities[fd]], offset, width, pointer - base,
                         expand, rows, source_stride, destination_stride, allocation))
        ranges.sort()
        if any(previous[1] > following[0] for previous, following in zip(ranges, ranges[1:], strict=False)):
            raise ValueError("overlapping shared destination envelopes require a rank-local dependency")
        torch.cuda.current_stream(self.device).synchronize()
        local_io = dict(payload_bytes=session.payload_bytes,
                        physical_bytes=session.stats(flush=False)["physical_bytes"])
        return dict(sources=sources, allocations=[(base, *allocations[base]) for base in ordered],
                    local_io=local_io,
                    records=wire.tobytes(), destination_bytes=destination_bytes)

    def _plan(self, members):
        import torch

        sources = sorted({tuple(source) for m in members for source in m["sources"]})
        source_index = {identity: index for index, identity in enumerate(sources)}
        for identity in sources:
            if identity not in self.files:
                fd = os.open(identity[0], os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
                try:
                    if self._identity(fd) != identity:
                        raise RuntimeError("shared checkpoint source identity changed")
                except BaseException:
                    os.close(fd)
                    raise
                self.files[identity] = fd
        copies = []
        for rank, member in enumerate(members):
            pointers = [
                base if rank == self.rank else self.native.owner_import(self.executor, handle, size)
                for base, size, handle in member["allocations"]
            ]
            records = array("Q")
            records.frombytes(member["records"])
            if len(records) % 9:
                raise ValueError("invalid collective checkpoint descriptor length")
            for index in range(0, len(records), 9):
                file, offset, width, delta, expand, rows, source_stride, destination_stride, allocation = records[index:index + 9]
                if file >= len(member["sources"]) or allocation >= len(pointers):
                    raise ValueError("invalid collective source or allocation index")
                extent = (rows - 1) * destination_stride + width * (1 + expand)
                if delta + extent > member["allocations"][allocation][1]:
                    raise ValueError("collective checkpoint write exceeds exported allocation")
                source = source_index[tuple(member["sources"][file])]
                copy = Copy(source, offset, width, pointers[allocation] + delta,
                            expand, rows, source_stride, destination_stride)
                if copy.end > sources[source][3]:
                    raise ValueError("shared checkpoint source exceeds its file extent")
                copies.append(copy)
        chunks, fragments = plan_reads(copies, rank=self.rank, world_size=self.world_size)
        for index in range(0, len(chunks), 5):
            chunks[index] = self.files[sources[chunks[index]]]
        self.native.owner_execute(self.executor, chunks, fragments,
                                  torch.cuda.current_stream(self.device).cuda_stream, True)
        return chunks, fragments

    def _execute(self, plan):
        import torch

        result = self.native.owner_execute(
            self.executor, *plan, torch.cuda.current_stream(self.device).cuda_stream,
        )
        for identity, fd in self.files.items():
            if self._identity(fd) != identity:
                raise RuntimeError("shared checkpoint source changed during execution")
        return result

    def finish(self, session):
        if self.failed:
            raise RuntimeError("shared checkpoint group failed; the load cannot continue")
        started = time.perf_counter()
        safe = False
        errors = []
        try:
            prepared, errors = self._phase("prepare", lambda: self._prepare(session))
            if not errors:
                members = self._gather(prepared)
                plan, errors = self._phase("plan", lambda: self._plan(members))
            if not errors:
                result, errors = self._phase("execute", lambda: self._execute(plan))
                if result is not None:
                    for key, value in result.items():
                        self.totals[key] = value if key == "gds_version" else self.totals.get(key, 0) + value
                if not errors:
                    reports = self._gather(result)
                    if sum(report["destination_bytes"] for report in reports) != sum(member["destination_bytes"] for member in members):
                        errors.append("shared scatter byte coverage differs from routed destinations")
                    self.summary = dict(
                        ranks=self.world_size,
                        payload_bytes=sum(member["local_io"]["payload_bytes"] for member in members),
                        physical_bytes=sum(member["local_io"]["physical_bytes"] for member in members)
                        + sum(report["physical_bytes"] for report in reports),
                        shared_physical_bytes=self.summary.get("shared_physical_bytes", 0)
                        + sum(report["physical_bytes"] for report in reports),
                        shared_transfer_seconds=self.summary.get("shared_transfer_seconds", 0)
                        + max(report["execution_seconds"] for report in reports),
                    )
            _, retire_errors = self._phase("unmap", lambda: (
                self.native.owner_unmap(self.executor) if self.executor is not None else None
            ))
            errors.extend(retire_errors)
            safe = not retire_errors
            if errors:
                raise RuntimeError("shared checkpoint completion failed: " + "; ".join(errors))
            self.totals["completion_seconds"] = self.totals.get("completion_seconds", 0) + time.perf_counter() - started
            self.totals["epochs"] = self.totals.get("epochs", 0) + 1
            self.totals["staging_bytes"] = 18677760
            self.epoch += 1
        except BaseException:
            self.failed = True
            if not safe:
                self.unsafe = True
                _retained_sessions.append((self, session, list(session.destinations)))
            raise
        finally:
            if safe:
                session.records = array("Q")
                session.destinations.clear()

    def close(self):
        if self.unsafe:
            return
        if self._initializer is not None:
            self._initializer.join()
            self._initializer = None
        if self._initialization is not None:
            # Initialization errors are reported collectively by prepare.
            if self._initialization.exception() is None:
                self.executor = self._initialization.result()
            self._initialization = None
        if self.executor is not None:
            self.native.owner_close(self.executor)
            self.executor = None
        for fd in self.files.values():
            os.close(fd)
        self.files.clear()
