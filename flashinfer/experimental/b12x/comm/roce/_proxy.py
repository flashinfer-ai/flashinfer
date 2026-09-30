"""Build and bind the RDMA proxy shared library (``_roce_proxy.c``).

The proxy is plain C over libibverbs.  It is compiled once per source hash
with the host C compiler into the b12x cache directory, so the package stays
pure Python for packaging purposes while the RDMA posting loop runs without
the interpreter on its critical path.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path

_SOURCE = Path(__file__).with_name("_roce_proxy.c")
_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None

BLOB_STRUCT_ERR = "roce proxy blob size mismatch"


def _traffic_class() -> int:
    """IP DSCP/ECN byte for every RoCE queue pair, resolved once at setup."""
    for name in ("B12X_ROCE_TRAFFIC_CLASS", "NCCL_IB_TC"):
        raw = os.getenv(name)
        if raw is not None:
            try:
                value = int(raw, 0)
            except ValueError:
                raise ValueError(f"{name} must be an integer from 0 to 255") from None
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be an integer from 0 to 255")
            return value
    return 0


def _cache_dir() -> Path:
    """Directory for the compiled proxy library: ``B12X_ROCE_CACHE_DIR``, else ``<XDG cache>/b12x/roce``."""
    override = os.getenv("B12X_ROCE_CACHE_DIR")
    if override:
        return Path(override)
    root = os.getenv("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return Path(root) / "b12x" / "roce"


def _compiler() -> str:
    """Host C compiler used to build the proxy (``CC``, gcc, cc, or clang)."""
    for candidate in (os.getenv("CC"), "gcc", "cc", "clang"):
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "b12x.comm.roce needs a C compiler (gcc/cc/clang) and libibverbs headers "
        "to build its RDMA proxy"
    )


def _build() -> Path:
    """Compile the proxy source into the cache directory keyed by source hash; returns the .so path."""
    source = _SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"roce_proxy-{digest}.so"
    if target.exists():
        return target
    tmp = cache / f".roce_proxy-{digest}-{os.getpid()}.so"
    cmd = [
        _compiler(),
        "-O2",
        "-std=gnu11",
        "-shared",
        "-fPIC",
        "-o",
        str(tmp),
        str(_SOURCE),
        "-libverbs",
        "-lpthread",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "failed to build the b12x RoCE proxy: " + " ".join(cmd) + "\n" + proc.stderr
        )
    os.replace(tmp, target)
    return target


def load() -> ctypes.CDLL:
    """Return the bound proxy library, building it on first use."""

    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        lib = ctypes.CDLL(str(_build()), use_errno=True)
        u64 = ctypes.c_uint64
        p = ctypes.c_void_p
        lib.roce_abi_version.restype = ctypes.c_int
        lib.roce_abi_version.argtypes = []
        lib.roce_layout.restype = ctypes.c_int
        lib.roce_layout.argtypes = [ctypes.c_int, u64, ctypes.POINTER(u64)]
        lib.roce_blob_bytes.restype = u64
        lib.roce_blob_bytes.argtypes = []
        lib.roce_create.restype = p
        lib.roce_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            p,
            u64,
            u64,
            ctypes.c_char_p,
            u64,
        ]
        lib.roce_local_blob.restype = ctypes.c_int
        lib.roce_local_blob.argtypes = [p, p, u64]
        lib.roce_connect.restype = ctypes.c_int
        lib.roce_connect.argtypes = [p, p, u64]
        lib.roce_start.restype = ctypes.c_int
        lib.roce_start.argtypes = [p]
        lib.roce_stop.restype = None
        lib.roce_stop.argtypes = [p]
        lib.roce_failed.restype = ctypes.c_int
        lib.roce_failed.argtypes = [p]
        lib.roce_error.restype = ctypes.c_char_p
        lib.roce_error.argtypes = [p]
        lib.roce_stat.restype = u64
        lib.roce_stat.argtypes = [p, ctypes.c_int]
        lib.roce_hca_stat.restype = u64
        lib.roce_hca_stat.argtypes = [p, ctypes.c_int, ctypes.c_int]
        lib.roce_destroy.restype = None
        lib.roce_destroy.argtypes = [p]
        if lib.roce_abi_version() != 4:
            raise RuntimeError("unexpected b12x RoCE proxy ABI version")
        _LIB = lib
        return lib


class Layout:
    """Byte offsets of the pinned region shared by the kernel and the proxy."""

    __slots__ = (
        "recv_off",
        "flag_off",
        "send_off",
        "ctrl_off",
        "total_bytes",
        "flag_stride",
        "slots",
    )

    def __init__(self, world_size: int, slot_bytes: int) -> None:
        """Create the proxy context: open the HCAs, register the pinned region, create the queue pairs."""
        out = (ctypes.c_uint64 * 7)()
        if load().roce_layout(int(world_size), int(slot_bytes), out) != 0:
            raise ValueError(
                f"unsupported RoCE geometry: world_size={world_size} slot_bytes={slot_bytes} "
                "(2..16 ranks, slot_bytes a positive multiple of 4096)"
            )
        (
            self.recv_off,
            self.flag_off,
            self.send_off,
            self.ctrl_off,
            self.total_bytes,
            self.flag_stride,
            self.slots,
        ) = (int(v) for v in out)


class Proxy:
    """Owns one RDMA proxy context for one rank."""

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        hca_names: tuple[str, ...],
        gid_index: int,
        region_ptr: int,
        region_bytes: int,
        slot_bytes: int,
    ) -> None:
        """Create the proxy context: open the HCAs, register the pinned region, create the queue pairs."""
        self.traffic_class = _traffic_class()
        self._lib = load()
        names = (ctypes.c_char_p * len(hca_names))(*[n.encode() for n in hca_names])
        err = ctypes.create_string_buffer(512)
        self._ctx = self._lib.roce_create(
            int(world_size),
            int(rank),
            names,
            len(hca_names),
            int(gid_index),
            self.traffic_class,
            ctypes.c_void_p(int(region_ptr)),
            int(region_bytes),
            int(slot_bytes),
            err,
            len(err),
        )
        if not self._ctx:
            raise RuntimeError(
                f"RoCE proxy setup failed: {err.value.decode(errors='replace')}"
            )
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.hca_names = tuple(hca_names)

    def local_blob(self) -> bytes:
        """Serialized connection record (region address, keys, queue-pair numbers) to send to every peer."""
        n = int(self._lib.roce_blob_bytes())
        buf = ctypes.create_string_buffer(n)
        if self._lib.roce_local_blob(self._ctx, buf, n) != 0:
            raise RuntimeError(BLOB_STRUCT_ERR)
        return buf.raw

    def connect(self, blobs: list[bytes]) -> None:
        """Connect the queue pairs from every rank's ``local_blob``, in rank order."""
        n = int(self._lib.roce_blob_bytes())
        if len(blobs) != self.world_size or any(len(b) != n for b in blobs):
            raise RuntimeError(BLOB_STRUCT_ERR)
        joined = b"".join(blobs)
        buf = ctypes.create_string_buffer(joined, len(joined))
        if self._lib.roce_connect(self._ctx, buf, len(joined)) != 0:
            raise RuntimeError(f"RoCE queue-pair connect failed: {self.error()}")

    def start(self) -> None:
        """Start the proxy thread; a restart resumes from the last posted sequence."""
        if self._lib.roce_start(self._ctx) != 0:
            raise RuntimeError(f"RoCE proxy thread failed to start: {self.error()}")

    def stop(self) -> None:
        """Stop the proxy thread; ``start`` resumes from the last posted op."""
        self._lib.roce_stop(self._ctx)

    def failed(self) -> bool:
        """True once the proxy thread has stopped on an error."""
        return bool(self._lib.roce_failed(self._ctx))

    def error(self) -> str:
        """The proxy's last error message, empty when none."""
        raw = self._lib.roce_error(self._ctx)
        return raw.decode(errors="replace") if raw else ""

    def stats(self) -> dict[str, int | list[int]]:
        """Counters: ops posted, RDMA writes completed, last posted sequence."""
        return {
            "ops_posted": int(self._lib.roce_stat(self._ctx, 0)),
            "writes_completed": int(self._lib.roce_stat(self._ctx, 1)),
            "last_seq": int(self._lib.roce_stat(self._ctx, 2)),
            "writes_completed_per_hca": [
                int(self._lib.roce_hca_stat(self._ctx, hca, 0))
                for hca in range(len(self.hca_names))
            ],
            "bytes_posted_per_hca": [
                int(self._lib.roce_hca_stat(self._ctx, hca, 1))
                for hca in range(len(self.hca_names))
            ],
        }

    def close(self) -> None:
        """Stop the thread and release the RDMA resources."""
        ctx, self._ctx = self._ctx, None
        if ctx:
            self._lib.roce_destroy(ctx)

    def __del__(self) -> None:  # pragma: no cover - defensive teardown
        """Release the RDMA resources if ``close`` was never called."""
        with contextlib.suppress(Exception):
            self.close()


__all__ = ["Layout", "Proxy", "load"]
