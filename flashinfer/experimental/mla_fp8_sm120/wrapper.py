"""Isolated FlashInfer-derived SM120 FP8 MLA with plan/run and CUDA graph support.

The loaded FlashInfer package and serving process are never patched. Sources
remain in this directory; build products live in the user JIT cache. Q/KV are
contiguous absorbed-MLA tensors with 576 dims.
The BF16-Q entry includes row quantization; run_prequantized accepts FP8 Q.
"""

import ctypes as C
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import fcntl
import json
import os
import shutil
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def build(bm=32, bn=32, stages=1, groups=2, share_p=False, shard_qk=False):
    """Compile the research ABI into a locked user cache, never into the package.

    Source checkouts use their own FlashInfer headers. The explicit include
    override exists only to reproduce the historical 0.6.15.post1 experiment.
    """
    if bm not in (16, 32, 64) or bn not in (32, 64):
        raise ValueError("Unsupported Q/KV tile size.")
    if stages not in (1, 2) or groups not in (2, 4):
        raise ValueError("Unsupported stage count or output dimension groups.")
    if shard_qk and (not share_p or bn % (16 * groups)):
        raise ValueError("Sharded QK requires share_p and at least 16 KV columns per group.")
    package = Path(importlib.util.find_spec("flashinfer").origin).parent
    checkout = ROOT.parents[2]
    default_include = checkout / "include"
    if not (default_include / "flashinfer").is_dir():
        default_include = package / "data/include"
    include = Path(os.environ.get("FLASHINFER_MLA_FP8_INCLUDE_DIR", default_include))
    if not (include / "flashinfer/attention/mla.cuh").is_file():
        raise FileNotFoundError(f"Missing FlashInfer headers under {include}")
    # Source-only harnesses may borrow dependency headers from an installed wheel.
    installed = Path(
        importlib.metadata.distribution("flashinfer-python").locate_file("flashinfer")
    )
    includes = [include]
    for dependency in ("cutlass/include", "cccl/libcudacxx/include"):
        candidates = [
            checkout / "3rdparty" / dependency,
            package / "data" / dependency,
            installed / "data" / dependency,
        ]
        selected = next(
            (x for x in candidates if x.is_dir() and any(x.iterdir())), None
        )
        if selected is None:
            raise FileNotFoundError(f"Missing dependency headers: {dependency}")
        includes.append(selected)
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc (CUDA 12.8 or later) is required for SM120a.")
    flags = [
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_120a,code=sm_120a",
        "-Xptxas=-v",
        f"-DTILE_Q={bm}",
        f"-DTILE_KV={bn}",
        f"-DSTAGES={stages}",
        f"-DD_GROUPS={groups}",
        f"-DSHARE_P={int(share_p)}",
        f"-DSHARD_QK={int(shard_qk)}",
    ]
    compiler = subprocess.check_output([nvcc, "--version"], text=True)
    fingerprint = hashlib.sha256()
    for source in (ROOT / "mla_fp8.cu", ROOT / "scheduler_fp8.cuh"):
        fingerprint.update(source.read_bytes())
    # Include helper changes as well as the experimental CUDA specialization.
    for header in sorted((include / "flashinfer").rglob("*.cuh")):
        fingerprint.update(str(header.relative_to(include)).encode())
        fingerprint.update(header.read_bytes())
    fingerprint.update(
        json.dumps([nvcc, compiler, flags, list(map(str, includes))]).encode()
    )
    digest = fingerprint.hexdigest()[:16]
    name = f"mla_q{bm}_k{bn}_s{stages}_d{groups}_p{int(share_p)}_qk{int(shard_qk)}_{digest}"
    base = Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", Path.home()))
    cache = base / ".cache/flashinfer/experimental/mla_fp8_sm120"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (name + ".so")
    with (cache / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            if os.environ.get("FLASHINFER_DISABLE_JIT") == "1":
                raise RuntimeError(
                    "FP8 MLA module is not cached and FLASHINFER_DISABLE_JIT=1."
                )
            temporary = cache / (name + f".{os.getpid()}.tmp.so")
            cmd = [
                nvcc,
                *flags,
                *["-I" + str(x) for x in includes],
                str(ROOT / "mla_fp8.cu"),
                "-o",
                str(temporary),
            ]
            log_path = cache / (name + ".log")
            with log_path.open("w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(log_path.read_text())
            temporary.replace(target)
            (cache / (name + ".json")).write_text(
                json.dumps(
                    dict(
                        command=cmd,
                        compiler=compiler,
                        include=str(include),
                        digest=digest,
                    ),
                    indent=2,
                )
                + "\n"
            )
    return target


@lru_cache(None)
def module(bm, bn, stages, groups, share_p, shard_qk):
    lib = C.CDLL(str(build(bm, bn, stages, groups, share_p, shard_qk)))
    P, I, Z = C.c_void_p, C.c_int, C.c_size_t
    lib.fp8_plan.argtypes = [P, Z, P, P, Z, P, P, P, I, I, I, I, P, P]
    lib.fp8_run.argtypes = [P, P, P, P, P, P, P, P, P, P, I, I, I, C.c_float, I, P]
    lib.fp8_attributes.argtypes = [I, I, P]
    for name in ("fp8_plan", "fp8_run", "fp8_attributes"):
        getattr(lib, name).restype = I
    return lib


class NativeMLA:
    def __init__(
        self,
        workspace,
        qo_indptr,
        kv_indptr,
        indices,
        lengths,
        *,
        heads=20,
        page_size=16,
        causal=False,
        bm=32,
        bn=32,
        stages=1,
        groups=2,
        workers=None,
        fused=True,
        sm_scale=1 / 16,
        share_p=False,
        shard_qk=False,
    ):
        import torch

        with torch.cuda.device(workspace.device):
            self._initialize(
                workspace,
                qo_indptr,
                kv_indptr,
                indices,
                lengths,
                heads=heads,
                page_size=page_size,
                causal=causal,
                bm=bm,
                bn=bn,
                stages=stages,
                groups=groups,
                workers=workers,
                fused=fused,
                sm_scale=sm_scale,
                share_p=share_p,
                shard_qk=shard_qk,
            )

    def _initialize(
        self,
        workspace,
        qo_indptr,
        kv_indptr,
        indices,
        lengths,
        *,
        heads,
        page_size,
        causal,
        bm,
        bn,
        stages,
        groups,
        workers,
        fused,
        sm_scale,
        share_p,
        shard_qk,
    ):
        import torch

        self.torch = torch
        if (
            not workspace.is_cuda
            or workspace.dtype != torch.uint8
            or not workspace.is_contiguous()
            or workspace.numel() < 128 * 1024**2
        ):
            raise ValueError(
                "Provide a contiguous CUDA uint8 workspace of at least 128 MiB."
            )
        if heads <= 0 or page_size not in (1, 16, 32, 64, 128):
            raise ValueError("Expected positive heads and page size 1/16/32/64/128.")
        if torch.cuda.get_device_capability(workspace.device) != (12, 0):
            raise ValueError("This experimental binary targets SM120.")
        self.lib = module(bm, bn, stages, groups, share_p, shard_qk)
        self.config = dict(
            bm=bm, bn=bn, stages=stages, groups=groups, fused=fused, share_p=share_p, shard_qk=shard_qk
        )
        self.heads, self.page_size, self.causal, self.scale = (
            heads,
            page_size,
            causal,
            sm_scale,
        )
        attrs = (C.c_int * 5)()
        self._check(self.lib.fp8_attributes(causal, fused, attrs))
        self.attributes = dict(
            zip(
                (
                    "registers",
                    "shared_bytes",
                    "active_ctas_per_sm",
                    "threads",
                    "local_bytes",
                ),
                attrs,
                strict=False,
            )
        )
        sms = torch.cuda.get_device_properties(workspace.device).multi_processor_count
        if workers is None:
            workers = sms
        if fused and workers > attrs[2] * sms:
            raise ValueError(
                f"Cooperative grid needs {workers} CTAs; residency permits {attrs[2] * sms}."
            )
        if workers < 2:
            raise ValueError("At least two persistent workers are required.")
        if workers > 440:
            raise ValueError(
                "This prototype bounds the persistent grid at 440 workers."
            )
        self.config["workers"] = workers
        metadata = (qo_indptr, kv_indptr, lengths, indices)
        if any(x.ndim != 1 or x.dtype != torch.int32 for x in metadata):
            raise ValueError("Metadata must be one-dimensional int32 tensors.")
        qh, kh, lh = [
            x.to(device="cpu").contiguous() for x in (qo_indptr, kv_indptr, lengths)
        ]
        if len(qh) != len(lh) + 1 or len(kh) != len(qh):
            raise ValueError("Invalid ragged metadata shapes.")
        if len(lh) == 0 or int(qh[0]) != 0 or int(kh[0]) != 0:
            raise ValueError("Use a nonempty batch and zero-based indptr arrays.")
        if bool((qh[1:] < qh[:-1]).any()) or bool((kh[1:] < kh[:-1]).any()):
            raise ValueError("Indptr must be nondecreasing.")
        if bool((lh < 0).any()) or (causal and bool((lh < qh[1:] - qh[:-1]).any())):
            raise ValueError("Invalid KV/query lengths for causal attention.")
        if bool(((kh[1:].to(torch.int64) - kh[:-1]) * page_size < lh).any()):
            raise ValueError("Page table is shorter than KV length.")
        packed = (qh[1:].to(torch.int64) - qh[:-1]) * heads
        cluster = 2 if int(packed.sum()) // len(lh) > bm else 1
        tiles = int(((packed + cluster * bm - 1) // (cluster * bm)).sum())
        if tiles + 2 * workers >= 16384:
            raise ValueError("Query metadata exceeds this prototype planner capacity.")
        self.tokens = int(qh[-1])
        self.indices = indices.to(
            device=workspace.device, dtype=torch.int32
        ).contiguous()
        if self.indices.numel() < int(kh[-1]):
            raise ValueError("Not enough physical page indices.")
        self.workspace = workspace
        self.iw = torch.empty(4 * 1024**2, device=workspace.device, dtype=torch.uint8)
        self.hostiw = torch.empty_like(self.iw, device="cpu", pin_memory=True)
        self.info = (C.c_int64 * 18)()
        self._check(
            self.lib.fp8_plan(
                workspace.data_ptr(),
                workspace.numel() * workspace.element_size(),
                self.iw.data_ptr(),
                self.hostiw.data_ptr(),
                self.iw.numel(),
                qh.data_ptr(),
                kh.data_ptr(),
                lh.data_ptr(),
                len(lh),
                heads,
                causal,
                workers,
                self.info,
                torch.cuda.current_stream(workspace.device).cuda_stream,
            )
        )
        offset = self.info[15]
        work_ptr = self.hostiw[offset : offset + 4 * (self.info[1] + 1)].view(
            torch.int32
        )
        self.scheduling = dict(
            grid=[self.info[0], self.info[1]],
            works=int(work_ptr[-1]),
            queues_with_work=int((work_ptr[1:] > work_ptr[:-1]).sum()),
        )
        self.q8 = torch.empty(
            (self.tokens, heads, 576),
            device=workspace.device,
            dtype=torch.float8_e4m3fn,
        )
        self.qs = torch.empty(
            (self.tokens, heads), device=workspace.device, dtype=torch.float32
        )
        self.out = torch.empty(
            (self.tokens, heads, 512), device=workspace.device, dtype=torch.bfloat16
        )
        self.lse = torch.empty(
            (self.tokens, heads), device=workspace.device, dtype=torch.float32
        )

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"Native FlashInfer FP8 CUDA status {status}")

    def run_prequantized(self, q8, kv8, qs, ks):
        with self.torch.cuda.device(self.workspace.device):
            return self._run_prequantized(q8, kv8, qs, ks)

    def _run_prequantized(self, q8, kv8, qs, ks):
        t = self.torch
        for x in (q8, kv8, qs, ks):
            if (
                not x.is_cuda
                or x.device != self.workspace.device
                or not x.is_contiguous()
            ):
                raise ValueError(
                    "Inputs must be contiguous on the workspace CUDA device."
                )
        if (
            q8.shape != self.q8.shape
            or kv8.ndim != 3
            or kv8.shape[1:] != (self.page_size, 576)
        ):
            raise ValueError("Expected Q [T,H,576], KV [pages,page_size,576].")
        if q8.dtype != t.float8_e4m3fn or kv8.dtype != t.float8_e4m3fn:
            raise ValueError("Q and KV must use FP8 E4M3.")
        if (
            qs.dtype != t.float32
            or ks.dtype != t.float32
            or qs.numel() != q8.numel() // 576
            or ks.numel() != kv8.numel() // 576
        ):
            raise ValueError("Expected one FP32 scale per Q head and per KV token.")
        self._check(
            self.lib.fp8_run(
                self.workspace.data_ptr(),
                self.iw.data_ptr(),
                self.info,
                q8.data_ptr(),
                kv8.data_ptr(),
                qs.data_ptr(),
                ks.data_ptr(),
                self.indices.data_ptr(),
                self.out.data_ptr(),
                self.lse.data_ptr(),
                self.heads,
                self.page_size,
                self.causal,
                self.scale,
                self.config["fused"],
                t.cuda.current_stream(self.workspace.device).cuda_stream,
            )
        )
        return self.out

    def run(self, q, kv8, ks):
        from .quantization import quantize_rows

        if (
            q.shape != self.q8.shape
            or q.dtype != self.torch.bfloat16
            or not q.is_contiguous()
            or q.device != self.workspace.device
        ):
            raise ValueError(
                "Expected contiguous BF16 Q [T,H,576] on the workspace device."
            )
        with self.torch.cuda.device(self.workspace.device):
            quantize_rows(q, self.q8, self.qs)
        return self.run_prequantized(self.q8, kv8, self.qs, ks)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--bm", type=int, default=32)
    p.add_argument("--bn", type=int, default=32)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--groups", type=int, default=2)
    a = p.parse_args()
    print(build(a.bm, a.bn, a.stages, a.groups))
