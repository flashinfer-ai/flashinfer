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

---------------------------------------------------------------------------
Perf bench for the GDN ucache verify+flush kernels: the HMMA kernel's dtype
arms (``--arm bf16 | fp16_state | fp16_io | ring_fp16 | fp16_state_cache``,
backend pinned to HMMA) and the UMMA (tcgen05 / TMEM) backend of the
fp16_state arm (``--arm umma``, plus ``umma_ph10`` / ``umma_ph5`` / ``umma_lcg``
with stochastic rounding of the flushed state), and the Rubin (sm_107a) build of the
UMMA kernel (``--arm rubin`` / ``rubin_u16`` / ``rubin_ph5`` / ``rubin_ph10`` /
``rubin_lcg``; flashinfer/gdn_kernels/gdn_replay_mtp_umma_rubin.py).

Prints one row per batch size, one column per flush rate. Steady-state
operating points: at ``--T 4`` flush_min = 13, verify rows at P = 12, flush
rows at P = 13 scattered at exact counts; ``--T 8`` uses the lazy
flush_min = 17 - T = 9 (verify rows at P = 8, flush rows at P = 9). The
closure is captured as a CUDA graph and benched on the replay (CUPTI, cold
L2): eager calls carry ~25 us of host launch overhead that serving (always
graph-captured) never pays.

Cold-L2 method (``--l2``):
  zero (default): ``bench_gpu_time(cold_l2_cache=True)``, which flushes with
      ``buffer.zero_()``. That leaves L2 full of DIRTY lines, and the timed
      kernel pays their write-back to DRAM -- several us on short launches.
  read: L2 is flushed by a read-only pass over a 2x-L2 buffer (clean lines),
      then the kernel's own CUPTI record is taken (exactly one kernel per
      call is asserted). Cold L2 without the flush's write-back in the
      measurement.
  warm: no flush between replays (``bench_gpu_time(cold_l2_cache=False)``);
      at large B the working set is many times L2 anyway.

UMMA / Rubin arms only:
  --localized: die-localized batch split for GPUs with two locality domains
      (Rubin): the rows are split in halves, each half's state pool and rings
      are allocated in one die's memory (cuMemCreate on a locality domain) and
      launched on a green-context stream of that die's SMs (cuDevSmResourceSplit
      by locality domain, with backfill so every SM is used); fork / join is
      captured in the graph and the time is the CUPTI span of both launches.
      Needs a driver and cuda-python with the CUDA 13.4 locality-domain APIs.
  Rubin arms pass ``num_sms`` = the launch's SMs and ``flush_hint`` from the case
      ("none" at 0 % flush, "flush_first" with --row-order flush_first, else "some")
      on localized launches, so the kernel's automatic choice applies there
      (persistent at T = 4-8 on sm_107a).
  --persistent / --no-persistent: force ``persistent=True`` / ``False`` (Rubin arms).
  --ugpu (Rubin arms): the kernel's own uGPU localization -- the state pool and rings from
      ``ugpu_ucache_buffers`` (key heads [t H / 2, (t + 1) H / 2) of every slot in die t's
      memory) and one API call, which forks one launch per die onto green-context streams;
      the time is the CUPTI span of both launches. ``flush_hint`` as for --localized.

DRAM %: ``--dram-bytes`` (JSON list of {"T", "B", "rate", "bytes"}, e.g. ncu
``dram__bytes.sum`` per step, collected with ``--single-call`` under ncu) and
``--peak-tbps`` (the GPU's sustained DRAM peak) print bytes / time / peak next
to each time.

Anchors (B200, median of 1000, 2026-07-19, ``--l2 zero``): B=32/20% ~= 32 us,
B=256/20% ~= 163 us, B=256/0% ~= 134 us. Regressions >5% are real.

Run:
  python benchmarks/bench_gdn_ucache_flush.py --arm umma --T 4 --l2 read \
      --no-commit --batches 512 --rates 0 30 --iters 200
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch

from flashinfer.testing import bench_gpu_time

DEV = "cuda"
H, HV, K, V = 16, 64, 128, 128  # Qwen3.5-122B GDN @ TP1
T, W = 4, 16  # W = max history window (kernel W_RING); T is set by --T
RING = 32  # physical ring depth (kernel RING_SLOTS)
FLUSH_MIN = 13
SCALE = 1.0 / math.sqrt(K)

# --arm choices: dtype is fixed at module import (env-gated), so each arm
# loads its own copy of the flush module.
#   bf16       : bf16 inputs + bf16 state pool (default serving config)
#   fp16_state : bf16 inputs + fp16 state pool (GDN_UCACHE_STATE_DTYPE=fp16)
#   fp16_io    : fp16 inputs + fp16 state pool (GDN_UCACHE_IO_DTYPE=fp16)
#   ring_fp16  : bf16 inputs + bf16 state + fp16 u/k rings
#                (GDN_UCACHE_RING_DTYPE=fp16 — the vLLM/Triton ring rule)
# tuple: (io_env, state_env, ring_env, io_dtype, state_dtype, ring_dtype)
ARMS = {
    "bf16": (None, None, None, torch.bfloat16, torch.bfloat16, torch.bfloat16),
    "fp16_state": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "fp16_io": ("fp16", None, None, torch.float16, torch.float16, torch.float16),
    "ring_fp16": (None, None, "fp16", torch.bfloat16, torch.bfloat16, torch.float16),
    "fp16_state_cache": (
        None,
        "fp16",
        "fp16",
        torch.bfloat16,
        torch.float16,
        torch.float16,
    ),
    # UMMA (tcgen05 / TMEM) drop-in for the fp16_state arm: same tensors and contract
    # (flashinfer/gdn_kernels/gdn_replay_mtp_umma.py)
    "umma": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    # ... with an fp16 u ring (vLLM's u-cache dtype; the k ring stays bf16)
    "umma_u16": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    # ... with stochastic rounding of the flushed state (fixed seed; the bits do not
    # change the work): Philox 5 / 10 rounds, LCG
    "umma_ph5": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "umma_ph10": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "umma_lcg": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    # Rubin (sm_107a) build of the UMMA kernel, same arms
    # (flashinfer/gdn_kernels/gdn_replay_mtp_umma_rubin.py)
    "rubin": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "rubin_u16": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "rubin_ph5": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "rubin_ph10": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "rubin_lcg": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
}
_UMMA_SR = {
    "umma": {},
    "umma_u16": {},
    "umma_ph5": dict(stochastic_rounding="philox", philox_rounds=5),
    "umma_ph10": dict(stochastic_rounding="philox", philox_rounds=10),
    "umma_lcg": dict(stochastic_rounding="lcg"),
}
_UMMA_SR.update({"rubin" + k[4:]: v for k, v in list(_UMMA_SR.items())})
# history length of the non-flushing (verify) rows: None = flush_min - 1 (--hist full)
P_VERIFY = None
# --row-order: which request each CTA runs (umma arms only; the kernel reads the permutation)
ROW_ORDER = "none"


def make_row_order(hl_src, flush_min, mode):
    """Row permutation for ``mode``: identity (same work, exercises the kernel's row_order
    load), flush_first (longest CTAs first), verify_tail (the last two waves of CTAs hold no
    flushing row: their flush rows are swapped with verify rows from the front)."""
    B = hl_src.numel()
    order = torch.arange(B, dtype=torch.int64, device=hl_src.device)
    if mode == "identity":
        return order.to(torch.int32)
    is_flush = hl_src >= flush_min
    if mode == "flush_first":
        return torch.argsort((~is_flush).to(torch.int32), stable=True).to(torch.int32)
    assert mode == "verify_tail", mode
    slots = torch.cuda.get_device_properties(hl_src.device).multi_processor_count * 4
    tail = min(B, 2 * -(-slots // H))  # two waves of CTAs, in rows (H CTAs per row)
    tail_pos = order[B - tail :]
    tail_flush = tail_pos[is_flush[tail_pos]]
    head_pos = order[: B - tail]
    head_verify = head_pos[~is_flush[head_pos]][: tail_flush.numel()]
    n = head_verify.numel()
    if n:
        a, b = tail_flush[:n].clone(), head_verify.clone()
        order[a], order[b] = b, a
    return order.to(torch.int32)


_FLUSH_PATH = str(
    Path(__file__).resolve().parents[1]
    / "flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py"
)


def load_flush(arm):
    io_env, state_env, ring_env, io_dtype, state_dtype, ring_dtype = ARMS[arm]
    if arm in _UMMA_SR:
        if arm.startswith("rubin"):
            from flashinfer.gdn_kernels.gdn_replay_mtp_umma_rubin import (
                gated_delta_rule_mtp_ucache_flush_umma_rubin as fn,
            )
        else:
            from flashinfer.gdn_kernels.gdn_replay_mtp_umma import (
                gated_delta_rule_mtp_ucache_flush_umma as fn,
            )

        if _UMMA_SR[arm]:
            seed = torch.tensor([0x5EED], dtype=torch.int64, device=DEV)
            fn = functools.partial(fn, rand_seed=seed, **_UMMA_SR[arm])
        return fn, io_dtype, state_dtype, ring_dtype
    old = {
        k: os.environ.pop(k, None)
        for k in (
            "GDN_UCACHE_IO_DTYPE",
            "GDN_UCACHE_STATE_DTYPE",
            "GDN_UCACHE_RING_DTYPE",
        )
    }
    if io_env:
        os.environ["GDN_UCACHE_IO_DTYPE"] = io_env
    if state_env:
        os.environ["GDN_UCACHE_STATE_DTYPE"] = state_env
    if ring_env:
        os.environ["GDN_UCACHE_RING_DTYPE"] = ring_env
    try:
        spec = importlib.util.spec_from_file_location(f"uc_flush_{arm}", _FLUSH_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # the dtype arms time the HMMA kernel (on SM100 the fp16_state arm would
    # otherwise auto-route to the UMMA backend, which has its own arms above)
    fn = functools.partial(mod.gated_delta_rule_mtp_ucache_flush, backend="hmma")
    return fn, io_dtype, state_dtype, ring_dtype


torch.manual_seed(0)


def graphed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return lambda: g.replay()


class DieSplit:
    """Die-localized batch split (see --localized in the module docstring): one green-context
    stream per locality domain and allocations backed by one domain's memory."""

    def __init__(self):
        from cuda.bindings import driver as drv

        self.d = drv
        torch.zeros(1, device=DEV)  # a current context
        self.dev = self._ck(drv.cuDeviceGet(torch.cuda.current_device()))
        attr = drv.CUdevice_attribute
        self.ndom = self._ck(
            drv.cuDeviceGetAttribute(
                attr.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, self.dev
            )
        )
        if self.ndom != 2:
            raise SystemExit(
                f"--localized: the device reports {self.ndom} locality domain(s), needs 2"
            )
        nsm = torch.cuda.get_device_properties(DEV).multi_processor_count
        sm = self._ck(
            drv.cuDeviceGetDevResource(
                self.dev, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
        )
        flags = drv.CUdevSmResourceGroup_flags
        params = []
        for i in range(self.ndom):
            p = drv.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
            p.smCount = nsm // self.ndom
            p.coscheduledSmCount = 2
            p.flags = int(flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID) | int(
                flags.CU_DEV_SM_RESOURCE_GROUP_BACKFILL
            )
            p.localityDomainId = i
            params.append(p)
        groups = self._ck(drv.cuDevSmResourceSplit(self.ndom, sm, 0, params))[0]
        self.streams, self.sms = [], []
        for g in groups:
            desc = self._ck(drv.cuDevResourceGenerateDesc([g], 1))
            ctx = self._ck(
                drv.cuGreenCtxCreate(
                    desc,
                    self.dev,
                    drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
                )
            )
            st = self._ck(
                drv.cuGreenCtxStreamCreate(
                    ctx, drv.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
                )
            )
            self.streams.append(torch.cuda.ExternalStream(int(st)))
            self.sms.append(int(g.sm.smCount))
        self._allocs = []

    def _ck(self, r):
        err = r[0] if isinstance(r, tuple) else r
        if err != self.d.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error {err}")
        if isinstance(r, tuple):
            return r[1] if len(r) == 2 else r[1:]
        return None

    def copy_to_domain(self, t, dom):
        """A copy of ``t`` in memory backed by locality domain ``dom``."""
        drv = self.d
        t = t.contiguous()
        nbytes = t.numel() * t.element_size()
        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = (
            drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        )
        prop.location.localized.deviceId = int(self.dev)
        prop.location.localized.localityDomainId = dom
        gran = self._ck(
            drv.cuMemGetAllocationGranularity(
                prop,
                drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
        size = (nbytes + gran - 1) // gran * gran
        h = self._ck(drv.cuMemCreate(size, prop, 0))
        ptr = int(self._ck(drv.cuMemAddressReserve(size, 0, 0, 0)))
        self._ck(drv.cuMemMap(ptr, size, 0, h, 0))
        acc = drv.CUmemAccessDesc()
        acc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        acc.location.id = int(self.dev)
        acc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._ck(drv.cuMemSetAccess(ptr, size, [acc], 1))
        self._allocs.append((ptr, size, h))

        class _Raw:
            __cuda_array_interface__ = {
                "shape": (nbytes,),
                "typestr": "|u1",
                "data": (ptr, False),
                "version": 3,
            }

        raw = torch.as_tensor(_Raw(), device=DEV)
        src = t.view(-1).view(torch.uint8)
        # a fresh mapping can serve SMs stale lines right after the copy-engine fill:
        # verify on the SMs and copy again until it reads back
        for _ in range(4):
            raw.copy_(src)
            torch.cuda.synchronize()
            if torch.equal(raw, src):
                break
        else:
            raise RuntimeError("localized copy does not read back")
        return raw.view(t.dtype).view(t.shape)

    def release(self):
        torch.cuda.synchronize()
        for ptr, size, h in self._allocs:
            self._ck(self.d.cuMemUnmap(ptr, size))
            self._ck(self.d.cuMemAddressFree(ptr, size))
            self._ck(self.d.cuMemRelease(h))
        self._allocs = []


def make_case(
    B,
    seed,
    io_dtype=torch.bfloat16,
    state_dtype=torch.bfloat16,
    ring_dtype=torch.bfloat16,
    u_dtype=None,
):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=g, device=DEV) * sc).to(io_dtype)

    q, k = rn(B, T, H, K), rn(B, T, H, K)
    v, a, b = rn(B, T, HV, V, sc=0.5), rn(B, T, HV, sc=0.5), rn(B, T, HV)
    A_log = (
        torch.full((HV,), -3.0, device=DEV)
        + torch.rand(HV, generator=g, device=DEV) * 0.3
    ).to(io_dtype)
    dt_bias = rn(HV, sc=0.5)
    pool = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(state_dtype)
    # 32-deep physical rings, fully populated (rows outside the live window
    # are masked by the kernel; values just need to be finite).
    kh = torch.randn(B, H, RING, K, generator=g, device=DEV)
    kc = (kh / kh.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(ring_dtype)
    uc = (torch.randn(B, HV, RING, V, generator=g, device=DEV) * 0.3).to(
        u_dtype or ring_dtype
    )
    la = -(torch.rand(B, HV, RING, generator=g, device=DEV) * 0.3 + 0.003)
    gc = torch.cumsum(la, dim=-1).float().contiguous()
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    return q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, idx


def bench_point(
    uc_flush,
    B,
    rate_pct,
    iters,
    seed,
    io_dtype,
    state_dtype,
    ring_dtype=torch.bfloat16,
    base=0,
    no_commit=False,
    l2="zero",
    u_dtype=None,
    split=None,
    persistent=None,
    single_call=False,
    rubin=False,
    ugpu=False,
):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, idx = make_case(
        B, seed, io_dtype, state_dtype, ring_dtype, u_dtype
    )
    nf = 0 if rate_pct == 0 else max(1, round(B * rate_pct / 100))
    mask = torch.zeros(B, dtype=torch.bool, device=DEV)
    if nf:
        g_cpu = torch.Generator().manual_seed(seed + 3)
        mask[torch.randperm(B, generator=g_cpu)[:nf].to(DEV)] = True
    hl_src = torch.where(
        mask,
        torch.tensor(FLUSH_MIN, dtype=torch.int32, device=DEV),
        torch.tensor(
            FLUSH_MIN - 1 if P_VERIFY is None else P_VERIFY,
            dtype=torch.int32,
            device=DEV,
        ),
    )
    hl = hl_src.clone()
    cb_src = torch.full((B,), base, dtype=torch.int32, device=DEV)
    cb = cb_src.clone()
    # one launch per part: (rows, stream, (pool, k ring, u ring, g ring), state indices, SMs)
    nsm = torch.cuda.get_device_properties(DEV).multi_processor_count
    if ugpu:  # the kernel's uGPU buffers: head halves of every slot, one per die
        from flashinfer.gdn_kernels.gdn_replay_mtp_umma_rubin import (
            ugpu_ucache_buffers,
        )

        bufs = ugpu_ucache_buffers(B, H, HV, u_dtype=uc.dtype, device=DEV)
        for dst, src in zip(bufs, (pool, kc, uc, gc), strict=True):
            src5 = src.view(src.shape[0], 2, src.shape[1] // 2, *src.shape[2:])
            dst.copy_(src5)
            torch.cuda.synchronize()
            assert torch.equal(dst, src5), "uGPU buffer copy does not read back"
        parts = [(slice(0, B), None, tuple(bufs), idx, nsm)]
        del pool, kc, uc, gc
    elif split is None:
        parts = [(slice(0, B), None, (pool, kc, uc, gc), idx, nsm)]
    else:
        half = B // 2
        parts = []
        for dom, rows in enumerate((slice(0, half), slice(half, B))):
            bufs = tuple(split.copy_to_domain(x[rows], dom) for x in (pool, kc, uc, gc))
            n = rows.stop - rows.start
            ar = torch.arange(n, dtype=torch.int32, device=DEV)
            parts.append((rows, split.streams[dom], bufs, ar, split.sms[dom]))
        del pool, kc, uc, gc

    def part_kwargs(rows, n_sms):
        kw = {}
        if ROW_ORDER != "none":
            kw["row_order"] = make_row_order(hl_src[rows], FLUSH_MIN, ROW_ORDER)
        if split is not None and not rubin:  # heads per CTA for the die's SMs
            from flashinfer.gdn_kernels.gdn_replay_mtp_umma import auto_heads_per_cta

            n = rows.stop - rows.start
            kw["heads_per_cta"] = auto_heads_per_cta(n, H, n_sms, HV // H)
        if persistent is not None:
            kw["persistent"] = persistent
        if rubin and (split is not None or persistent or ugpu):
            # the launch's SMs: heads per CTA and the persistent choice follow them (uGPU: the
            # kernel takes each die's SMs itself)
            if not ugpu:
                kw["num_sms"] = n_sms
            kw["flush_hint"] = (
                "none"
                if rate_pct == 0
                else ("flush_first" if ROW_ORDER == "flush_first" else "some")
            )
        return kw

    extras = [part_kwargs(p[0], p[4]) for p in parts]

    def call(part, extra):
        rows, _, (pool_, kc_, uc_, gc_), idx_, _ = part
        uc_flush(
            A_log,
            a[rows],
            dt_bias,
            q=q[rows],
            k=k[rows],
            v=v[rows],
            b=b[rows],
            initial_state_source=pool_,
            initial_state_indices=idx_,
            k_cache=kc_,
            u_cache=uc_,
            g_cache=gc_,
            hist_len=hl[rows],
            cache_base=cb[rows],
            scale=SCALE,
            flush_min=FLUSH_MIN,
            restart_hist_on_flush=not no_commit,
            **extra,
        )

    def fn():
        if split is None:
            call(parts[0], extras[0])
        else:  # fork / join over the per-die streams
            src = torch.cuda.current_stream()
            ev = torch.cuda.Event()
            ev.record(src)
            dones = []
            for part, extra in zip(parts, extras, strict=True):
                part[1].wait_event(ev)
                with torch.cuda.stream(part[1]):
                    call(part, extra)
                done = torch.cuda.Event()
                done.record(part[1])
                dones.append(done)
            for done in dones:
                src.wait_event(done)
        if not no_commit:
            # wrapper committed cursors for flushed rows; restore them
            hl.copy_(hl_src)
            cb.copy_(cb_src)

    try:
        if single_call:  # one eager call, no timing (e.g. under a profiler)
            fn()
            torch.cuda.synchronize()
            return float("nan")
        if l2 == "read":
            assert split is None and not ugpu, (
                "--l2 read times exactly one kernel per call"
            )
            return kernel_us_read_flush(graphed(fn), iters)
        times = bench_gpu_time(
            graphed(fn),
            enable_cupti=True,
            cold_l2_cache=l2 == "zero",
            dry_run_iters=10,
            repeat_iters=iters,
        )
        return float(np.median(times)) * 1000.0  # us
    finally:
        if split is not None:
            parts.clear()
            torch.cuda.synchronize()
            split.release()


_READ_FLUSH = {}


def kernel_us_read_flush(runner, iters):
    """Median kernel time (us) with a cold, CLEAN L2: before each call a read-only pass
    over a 2x-L2 buffer evicts everything (no dirty lines left behind), then the
    graph replay runs alone. The kernel's own CUPTI activity record is the time
    (torch.profiler), and exactly one kernel per call is asserted."""
    from torch.profiler import ProfilerActivity, profile

    from flashinfer.testing.utils import get_l2_cache_size

    if not _READ_FLUSH:
        _READ_FLUSH["buf"] = torch.ones(
            2 * get_l2_cache_size() // 4, dtype=torch.int32, device=DEV
        )
        _READ_FLUSH["out"] = torch.empty((), dtype=torch.int64, device=DEV)
    buf, out = _READ_FLUSH["buf"], _READ_FLUSH["out"]

    def gpu_events(prof):
        return [
            e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]

    for _ in range(10):
        runner()
    torch.cuda.synchronize()
    # The call alone must be exactly one GPU kernel (no memsets / copies / extra launches).
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            runner()
        torch.cuda.synchronize()
    names = {e.name for e in gpu_events(prof)}
    assert len(gpu_events(prof)) == 5 and len(names) == 1, (
        f"expected one kernel per call, got {len(gpu_events(prof))} for 5 calls: {names}"
    )
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            torch.sum(
                buf, dim=0, out=out
            )  # read-only L2 flush (its kernels are dropped below)
            torch.cuda.synchronize()
            runner()
            torch.cuda.synchronize()
    kern = [e for e in gpu_events(prof) if e.name in names]
    assert len(kern) == iters, (len(kern), iters)
    return float(np.median([e.device_time for e in kern]))


def main():
    global T, FLUSH_MIN, P_VERIFY, HV
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--T", type=int, default=4, help="draft tokens (flush_min = 17 - T)"
    )
    ap.add_argument(
        "--hv", type=int, default=64, help="value heads (16 key heads; HV / H = 1..4)"
    )
    ap.add_argument(
        "--l2",
        choices=["zero", "read", "warm"],
        default="zero",
        help="cold-L2 method: zero = bench_gpu_time's zero_() flush; "
        "read = read-only flush + kernel-only CUPTI record; warm = no flush "
        "(see module docstring)",
    )
    ap.add_argument(
        "--json", type=str, default=None, help="append rows to this JSON file"
    )
    ap.add_argument(
        "--hist",
        choices=["full", "mid", "zero"],
        default="full",
        help="history length of the non-flushing rows: full = flush_min - 1 (default), "
        "mid = (flush_min - 1) // 2, zero = 0",
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    ap.add_argument("--rates", type=int, nargs="+", default=[0, 20, 40, 80])
    ap.add_argument(
        "--arm",
        choices=list(ARMS),
        default="bf16",
        help="dtype config: bf16 | fp16_state | fp16_io | ring_fp16 | fp16_state_cache | umma (UMMA kernel, "
        "fp16 state) | umma_u16 (UMMA, fp16 u ring) | umma_ph5 / umma_ph10 / umma_lcg (UMMA with "
        "stochastic rounding)",
    )
    ap.add_argument(
        "--no-commit",
        action="store_true",
        help="pure-kernel timing: disable the wrapper's standalone "
        "cursor commit AND the per-iter cursor restores (the "
        "kernel never mutates cursors, so iterations are "
        "identical). Without this flag the timed graph also "
        "contains ~4-6us of commit/restore elementwise ops — "
        "fine for wrapper-level A/Bs, misleading for "
        "kernel-level ones.",
    )
    ap.add_argument(
        "--base",
        type=int,
        default=0,
        help="ring window origin for all rows (28 exercises the "
        "wrapped-window path: base+P crosses RING_SLOTS)",
    )
    ap.add_argument(
        "--row-order",
        choices=["none", "identity", "flush_first", "verify_tail"],
        default="none",
        help="umma arms: which request each CTA runs (see make_row_order); identity "
        "isolates the cost of the kernel's row_order load from the ordering itself",
    )
    ap.add_argument(
        "--localized",
        action="store_true",
        help="umma / rubin arms: die-localized batch split over the two locality "
        "domains (Rubin; see module docstring)",
    )
    ap.add_argument(
        "--ugpu",
        action="store_true",
        help="rubin arms: the kernel's uGPU localization (ugpu_ucache_buffers, one launch "
        "per die; see module docstring)",
    )
    ap.add_argument(
        "--persistent",
        action="store_true",
        help="rubin arms: force persistent=True (default: the kernel's automatic choice)",
    )
    ap.add_argument(
        "--no-persistent",
        action="store_true",
        help="rubin arms: force persistent=False",
    )
    ap.add_argument(
        "--single-call",
        action="store_true",
        help="run each point once without timing (for profilers, e.g. ncu dram__bytes.sum)",
    )
    ap.add_argument(
        "--dram-bytes",
        type=str,
        default=None,
        help='JSON list of {"T", "B", "rate", "bytes"}: DRAM bytes per step, for DRAM %%',
    )
    ap.add_argument(
        "--peak-tbps",
        type=float,
        default=None,
        help="sustained DRAM peak (TB/s) for DRAM %% (with --dram-bytes)",
    )
    args = ap.parse_args()
    if (args.localized or args.persistent) and args.arm not in _UMMA_SR:
        ap.error("--localized / --persistent: umma or rubin arms only")
    if (args.persistent or args.no_persistent) and not args.arm.startswith("rubin"):
        ap.error("--persistent / --no-persistent: rubin arms only")
    if args.persistent and args.no_persistent:
        ap.error("--persistent and --no-persistent")
    if args.ugpu and (args.localized or not args.arm.startswith("rubin")):
        ap.error("--ugpu: rubin arms only, without --localized")
    pz = True if args.persistent else (False if args.no_persistent else None)
    split = DieSplit() if args.localized else None
    dram_bytes = {}
    if args.dram_bytes:
        import json

        for d in json.loads(Path(args.dram_bytes).read_text()):
            dram_bytes[(d["T"], d["B"], d["rate"])] = d["bytes"]
    global ROW_ORDER
    ROW_ORDER = args.row_order
    T = args.T
    HV = args.hv
    FLUSH_MIN = W - T + 1
    P_VERIFY = {"full": FLUSH_MIN - 1, "mid": (FLUSH_MIN - 1) // 2, "zero": 0}[
        args.hist
    ]

    uc_flush, io_dtype, state_dtype, ring_dtype = load_flush(args.arm)
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | fused verify+flush, "
        f"arm={args.arm} (io={io_dtype}, state={state_dtype}, "
        f"ring={ring_dtype}), "
        f"T={T} W={W} ring={RING} base={args.base} fm={FLUSH_MIN} P_verify={P_VERIFY} "
        + (f"localized split {split.sms} SMs " if split else "")
        + ("uGPU buffers " if args.ugpu else "")
        + {True: "persistent ", False: "non-persistent ", None: ""}[pz]
        + f"H={H} HV={HV} K=V={K} | "
        f"CUDA-graph replay, CUPTI ({args.l2} L2), median of {args.iters}",
        flush=True,
    )
    recs = []
    hdr = "   B | " + " | ".join(f"{r:3d}% (us)" for r in args.rates)
    print(hdr)
    print("-" * len(hdr))
    for B in args.batches:
        row = [
            bench_point(
                uc_flush,
                B,
                r,
                args.iters,
                1000 + B + r,
                io_dtype,
                state_dtype,
                ring_dtype,
                base=args.base,
                no_commit=args.no_commit,
                l2=args.l2,
                u_dtype=torch.float16 if args.arm.endswith("_u16") else None,
                split=split,
                persistent=pz,
                single_call=args.single_call,
                rubin=args.arm.startswith("rubin"),
                ugpu=args.ugpu,
            )
            for r in args.rates
        ]

        def dram_pct(r, t):
            nb = dram_bytes.get((T, B, r))
            if nb is None or not args.peak_tbps or math.isnan(t):
                return None
            return 100.0 * nb / (t * 1e-6) / (args.peak_tbps * 1e12)

        pcts = [dram_pct(r, t) for r, t in zip(args.rates, row, strict=True)]
        print(
            f"{B:4d} | "
            + " | ".join(
                f"{t:9.2f}" + (f" ({p:4.1f}% DRAM)" if p is not None else "")
                for t, p in zip(row, pcts, strict=True)
            ),
            flush=True,
        )
        recs += [
            dict(
                arm=args.arm,
                T=T,
                B=B,
                rate=r,
                l2=args.l2,
                no_commit=args.no_commit,
                folded_rows=0 if r == 0 else max(1, round(B * r / 100)),
                hist=args.hist,
                p_verify=P_VERIFY,
                row_order=args.row_order,
                localized=args.localized,
                ugpu=args.ugpu,
                persistent={True: True, False: False, None: "auto"}[pz],
                us=t,
                dram_pct=p,
            )
            for r, t, p in zip(args.rates, row, pcts, strict=True)
        ]
    if args.json and not args.single_call:  # single-call rows carry no timing (NaN)
        import json

        path = Path(args.json)
        prev = json.loads(path.read_text()) if path.exists() else []
        key = lambda d: (  # noqa: E731
            d["arm"],
            d["T"],
            d["B"],
            d["rate"],
            d["l2"],
            d["no_commit"],
            d.get("hist", "full"),
            d.get("localized", False),
            d.get("ugpu", False),
            d.get("persistent", "auto"),
        )
        new = {key(d) for d in recs}
        path.write_text(
            json.dumps([d for d in prev if key(d) not in new] + recs, indent=1)
        )


if __name__ == "__main__":
    main()
