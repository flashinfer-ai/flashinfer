"""Seeded reject-or-correct checks for the paged decode wrapper (#3605).

Phase 1 covers ``BatchDecodeWithPagedKVCacheWrapper`` on the fa2 kernels: the
CUDA-core decode kernel (``use_tensor_cores=False``) and the tensor-core path
that reuses the fa2 prefill kernel (``use_tensor_cores=True``), plus ``auto``.

FLASHINFER_PAGED_FUZZ_NUM_TESTS controls the random cases (default 150).
FLASHINFER_PAGED_FUZZ_SEED sets the first seed (default 0).
FLASHINFER_PAGED_FUZZ_ONLY_SEED selects comma-separated seeds for reproduction.
Valid cases must run; mutations must reject cleanly or preserve the oracle.
"""

import math
import os
import random
import re
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Optional

import pytest
import torch

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.utils import get_compute_capability
from tests.infra.test_helpers.fuzz_ledger import Finding, FuzzLedger

BACKENDS = ("fa2", "auto")
NUM_TESTS = int(os.environ.get("FLASHINFER_PAGED_FUZZ_NUM_TESTS", "150"))
START_SEED = int(os.environ.get("FLASHINFER_PAGED_FUZZ_SEED", "0"))
ONLY_SEED = os.environ.get("FLASHINFER_PAGED_FUZZ_ONLY_SEED", "")
SEEDS = (
    [int(seed) for seed in ONLY_SEED.split(",") if seed.strip()]
    if ONLY_SEED
    else list(range(START_SEED, START_SEED + NUM_TESTS))
)
# Rejections raised by Python validation before any native call.
CLEAN_ERRORS = (ValueError, TypeError, NotImplementedError)
# Native host-side checks (TVM-FFI ICHECK, FLASHINFER_ERROR) raise RuntimeError
# before a launch; accepted where the table below says the native layer owns it.
HOST_ERRORS = CLEAN_ERRORS + (RuntimeError,)
# Group sizes compiled into the CUDA-core decode kernel (DISPATCH_GQA_GROUP_SIZE).
CUDA_CORE_GROUPS = (1, 2, 3, 4, 6, 8, 16)


def _known(*mutations, kernels=("cuda-core", "tensor-core")):
    cases = {(mutation, kernel) for mutation in mutations for kernel in kernels}
    return cases.__contains__


def _any(*predicates):
    return lambda case: any(predicate(case) for predicate in predicates)


# Cases are keyed (case, decode kernel). The gaps below were found by this
# fuzzer on main 5ced14a7a. Besides #1654, there are five gaps, each with its
# own issue and fix PR; a fix deletes its entry:
#   1. output strides: run() writes out/lse without a stride check (OOB writes)
#   2. run-time checks: run() does not check q and the KV cache against plan()
#   3. plan validation: plan() accepts invalid indptr/indices/last_page_len
#   4. planner: page_size=0 raises SIGFPE in plan()
#   5. replan atomicity: a late-rejected replan corrupts the live plan
# Quarantined entries crash the process or access memory out of bounds, so
# they are skipped and never launched; the rest run and xfail until fixed.
LEDGER = FuzzLedger(
    "batch-paged-decode",
    [
        Finding(
            match=_any(
                _known("metadata_dtype_indptr"),
                _known(
                    "metadata_dtype_indices",
                    "metadata_dtype_last_page_len",
                    kernels=("tensor-core",),
                ),
            ),
            reason="#1654: plan() does not validate index dtypes (open PR #5105): "
            "int64 indptr JIT-compiles an int64 module (CUDA-core: compile error; "
            "tensor-core: the int32 qo_indptr is read as int64), int64 indices and "
            "last_page_len are read as int32 on the tensor-core path",
            quarantine=True,
        ),
        Finding(
            match=_known("page_size_zero"),
            reason="#3605 gap 4 (to file): page_size=0 raises SIGFPE in the planner",
            quarantine=True,
        ),
        Finding(
            match=_known(
                "indptr_length",
                "last_page_len_length",
                "metadata_rank_indptr",
                "metadata_rank_indices",
                "metadata_rank_last_page_len",
            ),
            reason="#1654: plan() does not validate index rank and length (open PR "
            "#5105); mismatches fail late with messages that name no argument",
        ),
        Finding(
            match=_known(
                "out_noncontiguous", "lse_noncontiguous", "lse_unrequested_dtype"
            ),
            reason="#9674 (#3605 gap 1): run() writes out/lse as dense buffers "
            "without a stride check, and an lse passed with return_lse=False is never "
            "validated (fp32 LSE written into an fp16 buffer overruns it)",
            quarantine=True,
        ),
        Finding(
            match=_any(
                _known(
                    "q_rows_extra",
                    "q_heads_drift",
                    "head_dim_drift",
                    "kv_heads_drift",
                    "kv_page_size_drift",
                    "v_dtype",
                    "q_last_dim_strided",
                    "kv_last_dim_strided",
                    "window_left_drift",
                ),
                _known("q_rows_short", kernels=("cuda-core",)),
            ),
            reason="#3605 gap 2 (to file): run() does not check q and the paged KV "
            "cache against plan() (rows, heads, head_dim, page size, V dtype, last-dim "
            "stride); a window_left drift is a bare assert",
        ),
        Finding(
            match=_any(
                _known(
                    "indices_short",
                    "last_page_len_zero",
                    "last_page_len_over",
                    "last_page_len_negative",
                ),
                _known(
                    "indptr_nonmonotonic", "heads_not_divisible", kernels=("cuda-core",)
                ),
            ),
            reason="#3605 gap 3 (to file): plan() does not validate metadata values "
            "(monotonic indptr, len(indices) >= indptr[-1], 1 <= last_page_len <= "
            "page_size, num_qo_heads % num_kv_heads on the CUDA-core path)",
        ),
        Finding(
            match=_known(
                "kv_pool_short",
                "v_pool_short",
                "page_id_out_of_pool",
                "page_id_negative",
            ),
            reason="#3605 gap 2 (to file): run() does not check page indices against "
            "the K/V pool size (cf. #6741 for CUTLASS MLA)",
        ),
        Finding(
            match=_known("graph_late_rejection"),  # keyed by the graph test
            reason="#3605 gap 5 (to file): plan() copies indptr/indices/last_page_len "
            "into the wrapper (and the captured CUDA-graph buffers) before checks that "
            "can still reject it, so a rejected replan corrupts the live plan",
        ),
    ],
)
# Backends this phase validates; auto resolving elsewhere (cuDNN on SM100) is skipped.
PHASE_BACKENDS = ("fa2",)
UNIT_ROUNDOFF = {torch.bfloat16: 2**-8, torch.float16: 2**-11}
BOUND_FACTOR = 8
# Soft-capped logits go through PTX tanh.approx.f32 (maximum relative error
# 2**-11); the bound allows a relative 2**-10 on each capped logit.
TANH_APPROX_ERROR = 2**-10
# Anchor heads put a large score on one chosen token, so a lost page, token or
# KV chunk changes their output even when LSE is not requested. A weaker
# contrast token next to it keeps the soft cap visible in the output: the cap
# compresses the gap between the two logits.
ANCHOR_Q, ANCHOR_K, CONTRAST_K = 8.0, 48.0, 40.0


@dataclass(frozen=True)
class Config:
    seed: int
    kv_lengths: tuple[int, ...]
    num_kv_heads: int = 2
    group: int = 4
    head_dim: int = 128
    page_size: int = 16
    kv_layout: str = "NHD"
    dtype: torch.dtype = torch.float16
    fp8_kv: bool = False
    tensor_cores: bool = False
    window_left: int = -1
    soft_cap: float = 0.0
    sm_scale: Optional[float] = None
    kv_scales: Optional[tuple[float, float]] = None
    return_lse: bool = True
    kv_form: str = "tuple"
    q_strided: bool = False
    metadata_device: str = "cpu"

    @property
    def num_qo_heads(self):
        return self.num_kv_heads * self.group

    @property
    def kv_dtype(self):
        return torch.float8_e4m3fn if self.fp8_kv else self.dtype


@dataclass
class Problem:
    config: Config
    q: torch.Tensor
    q_dense: torch.Tensor
    kv_cache: object
    k_pages: torch.Tensor
    v_pages: torch.Tensor
    pages: tuple[tuple[int, ...], ...]
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    nan_page: int


@dataclass
class Reference:
    out: torch.Tensor
    lse: torch.Tensor
    magnitude: torch.Tensor
    # sum of |V| over the visible tokens: bounds subnormal rounding of P.
    mass: torch.Tensor
    # Output and LSE error allowed for the approximate tanh of the soft cap.
    soft: torch.Tensor
    lse_soft: torch.Tensor


def sample_config(seed, backend):
    rng = random.Random(seed)
    page = rng.choice((1, 16, 32, 64))
    lengths = (1, max(1, page - 1), page, page + 1)
    if rng.random() < 0.15:
        # Many short requests fill the grid, so the planner skips split-KV.
        batch, lengths = rng.randint(32, 96), lengths + (33,)
    else:
        batch, lengths = rng.randint(1, 8), lengths + (127, 128, 129, 1000, 4096)
    num_kv_heads = rng.choice((1, 2, 4, 8))
    group = rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 12, 16))
    while num_kv_heads > 1 and num_kv_heads * group > 64:
        num_kv_heads //= 2
    fp8_kv = rng.random() < 0.25
    return Config(
        seed,
        tuple(rng.choice(lengths) for _ in range(batch)),
        num_kv_heads=num_kv_heads,
        group=group,
        head_dim=rng.choice((64, 128, 256)),
        page_size=page,
        kv_layout=rng.choice(("NHD", "HND")),
        dtype=rng.choice((torch.float16, torch.bfloat16)),
        fp8_kv=fp8_kv,
        tensor_cores=rng.random() < 0.5,
        window_left=rng.choice((-1, -1, -1, -1, 0, 1, max(2, page - 1), 31, 200)),
        soft_cap=rng.choice((0.0, 0.0, 0.0, 30.0, 50.0)),
        sm_scale=rng.choice((None, None, None, 0.05, 0.2)),
        kv_scales=rng.choice((None, (0.5, 2.0), (1.5, 0.75))) if fp8_kv else None,
        return_lse=rng.random() < 0.6,
        kv_form=rng.choice(("tuple", "packed", "strided")),
        q_strided=rng.random() < 0.25,
        metadata_device=rng.choice(("cpu", "cuda")),
    )


def _layout(pages_nhd, layout):
    """[pool, page, kv_heads, dim] -> the wrapper's NHD or HND page layout."""
    return pages_nhd if layout == "NHD" else pages_nhd.transpose(1, 2)


def _kv_leaves(kv_cache):
    if isinstance(kv_cache, tuple):
        return list(kv_cache)
    return [kv_cache]


def _build_cache(k_pages, v_pages, cfg):
    k, v = (_layout(t, cfg.kv_layout).contiguous() for t in (k_pages, v_pages))
    if cfg.kv_form == "packed":
        return torch.stack((k, v), dim=1)
    if cfg.kv_form == "strided":
        # Padded head rows: a kernel that reads past head_dim meets NaN.
        def padded(t):
            storage = torch.full(
                (*t.shape[:-1], t.shape[-1] + 16), float("nan"), device=t.device
            ).to(t.dtype)
            view = storage[..., : t.shape[-1]]
            view.copy_(t)
            return view

        return padded(k), padded(v)
    return k, v


def _fill_page(p, page, value):
    for leaf in _kv_leaves(p.kv_cache):
        leaf[page] = value


def make_problem(cfg, device="cuda", *, poison_padding=True):
    generator = torch.Generator().manual_seed(cfg.seed)
    page_size, heads, kv_heads, dim = (
        cfg.page_size,
        cfg.num_qo_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
    )
    counts = [(length + page_size - 1) // page_size for length in cfg.kv_lengths]
    pool_size = sum(counts) + 2
    order = torch.randperm(pool_size, generator=generator).tolist()
    pages, cursor = [], 0
    for count in counts:
        pages.append(tuple(order[cursor : cursor + count]))
        cursor += count
    nan_page = order[cursor]
    q = torch.randn(len(counts), heads, dim, generator=generator) * 0.5
    k = torch.randn(pool_size, page_size, kv_heads, dim, generator=generator) * 0.5
    v = torch.randn(pool_size, page_size, kv_heads, dim, generator=generator)
    # Up to 8 anchor heads; at least half the heads keep random scores.
    anchors = min(8, max(1, heads // 2))
    q[..., :anchors] = 0
    k[..., :anchors] = 0
    for row, (length, ids) in enumerate(zip(cfg.kv_lengths, pages, strict=True)):
        spread = (length // 2, length // 4, 3 * length // 4, length // 8)
        if cfg.window_left >= 0:
            # The first visible token and the last excluded one.
            start = max(0, length - 1 - cfg.window_left)
            targets = (start, max(0, start - 1), length - 1, 0, *spread)
        else:
            targets = (0, length - 1, *spread, length // 3, length // 5)

        def put(token, head, anchor, value):
            k[ids[token // page_size], token % page_size, head // cfg.group, anchor] = (
                value
            )

        for anchor in range(anchors):
            head = anchor * heads // anchors
            token = targets[anchor]
            q[row, head] = 0
            q[row, head, anchor] = ANCHOR_Q
            if length > 1:
                put((token + 1) % length, head, anchor, CONTRAST_K)
            put(token, head, anchor, ANCHOR_K)
    if poison_padding:
        for tensor in (k, v):
            tensor[nan_page] = float("nan")
            for ids, length in zip(pages, cfg.kv_lengths, strict=True):
                tensor[ids[-1], (length - 1) % page_size + 1 :] = float("nan")
    q_dense = q.to(device=device, dtype=cfg.dtype)
    k_pages = k.to(device=device, dtype=cfg.kv_dtype)
    v_pages = v.to(device=device, dtype=cfg.kv_dtype)
    query = q_dense
    if cfg.q_strided:
        storage = torch.full(
            (len(counts), heads + 3, dim + 8), float("nan"), device=device
        ).to(cfg.dtype)
        query = storage[:, :heads, :dim]
        query.copy_(q_dense)
    meta_device = "cpu" if str(device) == "cpu" else cfg.metadata_device
    indptr = torch.tensor([0] + counts, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return Problem(
        cfg,
        query,
        q_dense,
        _build_cache(k_pages, v_pages, cfg),
        k_pages,
        v_pages,
        tuple(pages),
        indptr.to(meta_device),
        torch.tensor([page for ids in pages for page in ids], dtype=torch.int32).to(
            meta_device
        ),
        torch.tensor(
            [(length - 1) % page_size + 1 for length in cfg.kv_lengths],
            dtype=torch.int32,
        ).to(meta_device),
        nan_page,
    )


def _sm_scale(cfg):
    return cfg.sm_scale if cfg.sm_scale is not None else 1 / math.sqrt(cfg.head_dim)


def _spread(weights, values, out, chunk=512):
    """sum_k weights[h, g, k] * |values[k, h, :] - out[h, g, :]|, in token chunks."""
    total = torch.zeros_like(out)
    for start in range(0, values.shape[0], chunk):
        v = values[start : start + chunk].permute(1, 0, 2)[:, None]
        diff = (v - out[:, :, None]).abs()
        total += torch.einsum(
            "hgk,hgkd->hgd", weights[..., start : start + chunk], diff
        )
    return total


def reference_decode(p):
    """Gather each request from its page list; independent of the wrapper's planning."""
    cfg = p.config
    k_scale, v_scale = cfg.kv_scales or (1.0, 1.0)
    fields = {name: [] for name in Reference.__dataclass_fields__}
    for row, (length, ids) in enumerate(zip(cfg.kv_lengths, p.pages, strict=True)):
        # Query head h reads KV head h // group.
        query = p.q_dense[row].double().view(cfg.num_kv_heads, cfg.group, -1)
        keys, values = (
            pages[list(ids)].reshape(-1, cfg.num_kv_heads, cfg.head_dim)[:length]
            for pages in (p.k_pages, p.v_pages)
        )
        keys, values = keys.double() * k_scale, values.double() * v_scale
        scores = torch.einsum("hgd,khd->hgk", query, keys) * _sm_scale(cfg)
        if cfg.soft_cap > 0:
            scores = cfg.soft_cap * torch.tanh(scores / cfg.soft_cap)
        # Allowed error of each logit (natural-log units).
        delta = TANH_APPROX_ERROR * scores.abs() if cfg.soft_cap > 0 else 0 * scores
        if cfg.window_left >= 0:
            hidden = torch.arange(length, device=scores.device) < (
                length - 1 - cfg.window_left
            )
            scores.masked_fill_(hidden, -float("inf"))
            delta.masked_fill_(hidden, 0)
        probs = scores.softmax(-1)
        out = torch.einsum("hgk,khd->hgd", probs, values)
        # With logit errors e_k, |e_k| <= delta_k, the output moves by at most
        # exp(2 max delta) * sum_k p_k delta_k |v_k - out|, and the LSE by at
        # most exp(max delta) * sum_k p_k delta_k.
        growth = torch.exp(delta.amax(-1))
        fields["soft"].append(
            (growth[..., None] ** 2 * _spread(probs * delta, values, out)).flatten(0, 1)
        )
        fields["lse_soft"].append((growth * (probs * delta).sum(-1)).flatten())
        fields["out"].append(out.flatten(0, 1))
        fields["magnitude"].append(
            torch.einsum("hgk,khd->hgd", probs, values.abs()).flatten(0, 1)
        )
        fields["mass"].append(
            torch.einsum(
                "hgk,khd->hgd", scores.isfinite().double(), values.abs()
            ).flatten(0, 1)
        )
        fields["lse"].append(torch.logsumexp(scores, -1).flatten() / math.log(2))
    return Reference(**{name: torch.stack(v) for name, v in fields.items()})


def _check_numeric(out, lse, p, reference):
    cfg = p.config
    assert torch.isfinite(reference.out).all()
    assert torch.isfinite(out).all(), "non-finite output"
    u = UNIT_ROUNDOFF[cfg.dtype]
    # Subnormal rounding is absolute, eta = tiny * u per value: the output and
    # (on the tensor-core path) each rounded weight of P.
    eta = torch.finfo(cfg.dtype).tiny * u
    error = (out.double() - reference.out).abs()
    bound = (
        BOUND_FACTOR
        * (u * (reference.magnitude + reference.out.abs()) + eta * (1 + reference.mass))
        + reference.soft
    )
    exceeded = error > bound
    if exceeded.any():
        ratio = torch.where(
            bound > 0,
            error / bound,
            torch.where(error == 0, 0.0, float("inf")),
        )
        index = tuple(int(i) for i in torch.unravel_index(ratio.argmax(), ratio.shape))
        raise AssertionError(
            f"{int(exceeded.sum())}/{error.numel()} elements exceed the rounding bound; "
            f"max ratio={float(ratio[index]):.6g} at {index}, "
            f"error={float(error[index]):.6g}, bound={float(bound[index]):.6g}"
        )
    if cfg.return_lse:
        assert torch.isfinite(lse).all(), "non-finite LSE"
        # The tensor-core path sums rounded exp(score - max) weights; the
        # relative sum error is at most u + N * eta (see the MLA fuzzer).
        relative_sum_error = u * (1 + max(cfg.kv_lengths) * torch.finfo(cfg.dtype).tiny)
        tolerance = (
            1e-3
            + (-math.log1p(-relative_sum_error) + reference.lse_soft) / math.log(2)
            + 1e-3 * reference.lse.abs()
        )
        lse_error = (lse.double() - reference.lse).abs()
        if (lse_error > tolerance).any():
            index = tuple(
                int(i)
                for i in torch.unravel_index(
                    (lse_error / tolerance).argmax(), lse_error.shape
                )
            )
            raise AssertionError(
                f"{int((lse_error > tolerance).sum())}/{lse.numel()} LSE values exceed "
                f"the tolerance; at {index}: error={float(lse_error[index]):.6g}, "
                f"tolerance={float(tolerance[index]):.6g}"
            )


def _plan_kwargs(p):
    cfg = p.config
    return dict(
        indptr=p.indptr,
        indices=p.indices,
        last_page_len=p.last_page_len,
        num_qo_heads=cfg.num_qo_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        page_size=cfg.page_size,
        window_left=cfg.window_left,
        logits_soft_cap=cfg.soft_cap,
        q_data_type=cfg.dtype,
        kv_data_type=cfg.kv_dtype,
        sm_scale=cfg.sm_scale,
    )


def _run_kwargs(p, out, lse):
    cfg = p.config
    kwargs = dict(out=out, return_lse=cfg.return_lse)
    if cfg.return_lse:
        kwargs["lse"] = lse
    if cfg.kv_scales is not None:
        kwargs.update(k_scale=cfg.kv_scales[0], v_scale=cfg.kv_scales[1])
    return kwargs


def _new_wrapper(cfg, backend, workspace, graph=None):
    reserves = {}
    if graph is not None:
        batch, max_pages = graph
        reserves = dict(
            paged_kv_indptr_buffer=torch.empty(
                batch + 1, dtype=torch.int32, device=workspace.device
            ),
            paged_kv_indices_buffer=torch.empty(
                max_pages, dtype=torch.int32, device=workspace.device
            ),
            paged_kv_last_page_len_buffer=torch.empty(
                batch, dtype=torch.int32, device=workspace.device
            ),
        )
    return BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        cfg.kv_layout,
        use_cuda_graph=graph is not None,
        use_tensor_cores=cfg.tensor_cores,
        backend=backend,
        **reserves,
    )


def _wrapper(p, backend, workspace, graph=None):
    wrapper = _new_wrapper(p.config, backend, workspace, graph)
    wrapper.plan(**_plan_kwargs(p))
    return wrapper


def _guarded(shape, dtype, device, guard=32):
    storage = torch.full(
        (math.prod(shape) + 2 * guard,), -23, dtype=dtype, device=device
    )
    return storage, storage[guard:-guard].view(shape)


def _crc(tensor):
    return zlib.crc32(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )


def _inputs(p):
    return [
        p.q,
        *_kv_leaves(p.kv_cache),
        p.indptr,
        p.indices,
        p.last_page_len,
    ]


def _run_and_check(p, wrapper, reference=None):
    cfg = p.config
    reference = reference_decode(p) if reference is None else reference
    shape = (len(cfg.kv_lengths), cfg.num_qo_heads)
    out_storage, out = _guarded((*shape, cfg.head_dim), cfg.dtype, p.q.device)
    lse_storage, lse = _guarded(shape, torch.float32, p.q.device)
    input_crc = [_crc(t) for t in _inputs(p)]
    previous = None
    for poison in (float("nan"), 17.0):
        out.fill_(poison)
        lse.fill_(poison)
        returned = wrapper.run(p.q, p.kv_cache, **_run_kwargs(p, out, lse))
        torch.cuda.synchronize()
        if cfg.return_lse:
            assert returned[0] is out and returned[1] is lse
        else:
            assert returned is out
        _check_numeric(out, lse, p, reference)
        for storage in (out_storage, lse_storage):
            assert (storage[:32] == -23).all() and (storage[-32:] == -23).all()
        fingerprint = (_crc(out), _crc(lse) if cfg.return_lse else None)
        if previous is not None:
            assert fingerprint == previous, "non-deterministic output/LSE bytes"
        previous = fingerprint
    assert [_crc(t) for t in _inputs(p)] == input_crc, "input storage was modified"
    return previous


def _first_line(error):
    return (str(error).strip() or type(error).__name__).splitlines()[0]


def _kernel(wrapper):
    return "tensor-core" if wrapper.use_tensor_cores else "cuda-core"


def _skip_if_quarantined(case):
    """Quarantined cases are skipped before any launch, never xfailed."""
    finding = LEDGER.find(case)
    if finding is not None and finding.quarantine:
        pytest.skip(f"[{LEDGER.op}] quarantined, not run: {finding.reason}")


def _require_phase(wrapper):
    if wrapper.resolved_backend not in PHASE_BACKENDS:
        pytest.skip(f"auto resolved to {wrapper.resolved_backend}, not in this phase")


def _require_backend(backend):
    major, _ = get_compute_capability(torch.device("cuda"))
    if major < 8:
        pytest.skip("fa2 paged decode fuzz cases need SM80 or newer (bf16)")


@pytest.fixture(scope="module")
def workspace():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")


@pytest.fixture(autouse=True)
def ieee_reference():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


@contextmanager
def _case(cfg, backend, mutation="valid"):
    try:
        yield
    except BaseException:
        print(f"\n{cfg!r} backend={backend} mutation={mutation}")
        print(
            f"FLASHINFER_PAGED_FUZZ_ONLY_SEED={cfg.seed} pytest -x -s "
            f"tests/attention/test_paged_attention_fuzz.py -k '{backend} and {mutation}'"
        )
        raise
    finally:
        values = torch.arange(128, device="cuda", dtype=torch.int32)
        probe = values * 3 + 1
        torch.cuda.synchronize()
        assert torch.equal(probe.cpu(), torch.arange(128, dtype=torch.int32) * 3 + 1), (
            "device-state probe failed"
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_decode_fuzz_valid(seed, backend, workspace):
    _require_backend(backend)
    cfg = sample_config(seed, backend)
    with _case(cfg, backend):
        control = make_problem(cfg, poison_padding=False)
        try:
            control_wrapper = _wrapper(control, backend, workspace)
        except HOST_ERRORS as error:
            # Capability honesty: a group size the CUDA-core kernel is not
            # compiled for must fail cleanly at plan(); everything else runs.
            uncompiled = not cfg.tensor_cores and cfg.group not in CUDA_CORE_GROUPS
            if not uncompiled or "group_size" not in str(error):
                raise
            return
        _require_phase(control_wrapper)
        _run_and_check(control, control_wrapper)
        _skip_if_quarantined(("poisoned_padding", _kernel(control_wrapper)))
        p = make_problem(cfg)
        wrapper = _wrapper(p, backend, workspace)
        fingerprint = _run_and_check(p, wrapper)
        if backend == "auto":
            # auto must agree bit for bit with the explicit backend it resolved to.
            explicit = _wrapper(p, wrapper.resolved_backend, workspace)
            assert explicit.resolved_backend == wrapper.resolved_backend
            assert _run_and_check(p, explicit) == fingerprint, (
                f"auto differs from explicit {wrapper.resolved_backend}"
            )


@dataclass(frozen=True)
class Mutation:
    phase: str
    message: str
    # The true problem still defines the answer: a correct result is allowed.
    comparable: bool = False
    # A native host-side check (RuntimeError before launch) is the documented guard.
    native: bool = False


HEADS = r"(num_qo_heads|num_kv_heads|heads|group)"
MUTATIONS = {
    # plan(): the metadata contract.
    "indptr_nonmonotonic": Mutation("plan", r"indptr"),
    "indices_short": Mutation("plan", r"\b(indices|indptr)\b"),
    "indptr_length": Mutation("plan", r"\b(indptr|last_page_len|batch)\b"),
    "last_page_len_length": Mutation("plan", r"\b(indptr|last_page_len|batch)\b"),
    "last_page_len_zero": Mutation("plan", r"\blast_page_len\b"),
    "last_page_len_over": Mutation("plan", r"\blast_page_len\b"),
    "last_page_len_negative": Mutation("plan", r"\blast_page_len\b"),
    "heads_not_divisible": Mutation("plan", HEADS),
    "page_size_zero": Mutation("plan", r"\bpage_size\b"),
    # dtype: the native entry's "Inconsistency of Tensor type" check may fire at run().
    **{
        f"metadata_dtype_{field}": Mutation(
            "plan", rf"Tensor type: \w*{field}\b|\b{field}\b.*int32", native=True
        )
        for field in ("indptr", "indices", "last_page_len")
    },
    **{
        f"metadata_rank_{field}": Mutation("plan", rf"\b{field}\b")
        for field in ("indptr", "indices", "last_page_len")
    },
    "window_left_invalid": Mutation("plan", r"\bwindow_left\b", comparable=True),
    "indices_nan_tail": Mutation("plan", r"\bindices\b", comparable=True),
    "indptr_offset": Mutation("plan", r"\bindptr\b", comparable=True),
    # run(): drift from the plan and caller-owned buffers.
    "q_rows_short": Mutation("run", r"\bq\b|batch|q_len_per_req"),
    "q_rows_extra": Mutation("run", r"\bq\b|batch|q_len_per_req"),
    "q_heads_drift": Mutation("run", HEADS + r"|\bq\b"),
    "head_dim_drift": Mutation("run", r"head_dim|head dim"),
    "q_dtype": Mutation("run", r"\bq\b|q_data_type"),
    "kv_dtype": Mutation("run", r"\bk\b|kv_data_type"),
    "v_dtype": Mutation("run", r"\bv\b|v_cache|kv_data_type"),
    "kv_page_size_drift": Mutation("run", r"page_size|page size"),
    "kv_heads_drift": Mutation("run", HEADS),
    "kv_pool_short": Mutation("run", r"\b(pool|page|pages|indices)\b"),
    "v_pool_short": Mutation("run", r"\b(pool|page|pages|indices|v_cache)\b"),
    "kv_strides_differ": Mutation("run", r"stride", comparable=True, native=True),
    "kv_last_dim_strided": Mutation("run", r"stride|contiguous"),
    "q_last_dim_strided": Mutation("run", r"stride|contiguous"),
    "page_id_out_of_pool": Mutation("run", r"\b(pool|page|pages|indices)\b"),
    "page_id_negative": Mutation("run", r"\b(pool|page|pages|indices)\b"),
    "out_shape": Mutation("run", r"\bout\b"),
    "out_dtype": Mutation("run", r"\bout\b"),
    "out_device": Mutation("run", r"\bout\b"),
    "out_noncontiguous": Mutation("run", r"\bout\b|contiguous", comparable=True),
    "lse_shape": Mutation("run", r"\blse\b"),
    "lse_dtype": Mutation("run", r"\blse\b"),
    "lse_noncontiguous": Mutation("run", r"\blse\b|contiguous", comparable=True),
    "lse_unrequested_dtype": Mutation("run", r"\blse\b", comparable=True),
    "lse_unrequested_shape": Mutation(
        "run", r"\blse\b|lse\.size", comparable=True, native=True
    ),
    "window_left_drift": Mutation("run", r"\bwindow_left\b"),
}


def _other_dtype(dtype):
    return torch.bfloat16 if dtype == torch.float16 else torch.float16


def mutate(p, name, plan, run, call):
    """Apply one mutation; returns a check(out, lse) for comparable mutations."""
    cfg = p.config
    batch, heads = len(cfg.kv_lengths), cfg.num_qo_heads
    if name == "indptr_nonmonotonic":
        indptr = plan["indptr"].clone()
        indptr[1], indptr[2] = indptr[2].clone(), indptr[1].clone()
        plan["indptr"] = indptr
    elif name == "indices_short":
        plan["indices"] = plan["indices"][:-1]
    elif name == "indptr_length":
        plan["indptr"] = plan["indptr"][:-1]
    elif name == "last_page_len_length":
        plan["last_page_len"] = plan["last_page_len"][:-1]
    elif name.startswith("last_page_len_"):
        value = {"zero": 0, "over": cfg.page_size + 1, "negative": -1}[
            name.rsplit("_", 1)[1]
        ]
        lengths = plan["last_page_len"].clone()
        # Request 1 spans several pages, so a zero tail is not an empty request.
        lengths[1 if value == 0 else 0] = value
        plan["last_page_len"] = lengths
    elif name == "heads_not_divisible":
        plan["num_qo_heads"] = heads + 1
    elif name == "page_size_zero":
        plan["page_size"] = 0
    elif name.startswith("metadata_"):
        _, change, field = name.split("_", 2)
        tensor = plan[field]
        plan[field] = tensor.long() if change == "dtype" else tensor[None]
    elif name == "window_left_invalid":
        plan["window_left"] = -2
    elif name == "indices_nan_tail":
        _fill_page(p, p.nan_page, float("nan"))
        tail = torch.full(
            (17,), p.nan_page, dtype=torch.int32, device=plan["indices"].device
        )
        plan["indices"] = torch.cat((plan["indices"], tail))
    elif name == "indptr_offset":
        # A shifted CSR window is self-consistent: the leading NaN page is unused.
        _fill_page(p, p.nan_page, float("nan"))
        head = torch.full(
            (1,), p.nan_page, dtype=torch.int32, device=plan["indices"].device
        )
        plan["indices"] = torch.cat((head, plan["indices"]))
        plan["indptr"] = plan["indptr"] + 1
    elif name == "q_rows_short":
        call["q"] = p.q[:-1]
        run.update(out=None, lse=None)
    elif name == "q_rows_extra":
        call["q"] = torch.cat((p.q, p.q[:1]))
        run.update(out=None, lse=None)
    elif name == "q_heads_drift":
        # More query heads than planned: split-KV scratch is sized for the plan.
        call["q"] = torch.cat((p.q, p.q), dim=1)
        run.update(out=None, lse=None)
    elif name == "head_dim_drift":
        # q and KV agree with each other, not with the planned head_dim.
        call["q"] = p.q[..., : cfg.head_dim // 2].contiguous()
        call["kv_cache"] = tuple(
            t[..., : cfg.head_dim // 2].contiguous() for t in _kv_leaves(p.kv_cache)
        )
        run.update(out=None, lse=None)
    elif name == "q_dtype":
        call["q"] = p.q.float()
    elif name == "kv_dtype":
        call["kv_cache"] = tuple(
            t.to(_other_dtype(cfg.dtype)) for t in _kv_leaves(p.kv_cache)
        )
    elif name == "v_dtype":
        k, v = p.kv_cache
        call["kv_cache"] = (k, v.to(_other_dtype(cfg.dtype)))
    elif name == "kv_page_size_drift":
        dim = 1 if cfg.kv_layout == "NHD" else 2
        call["kv_cache"] = tuple(
            t.narrow(dim, 0, cfg.page_size - 1).contiguous()
            for t in _kv_leaves(p.kv_cache)
        )
    elif name == "kv_heads_drift":
        dim = 2 if cfg.kv_layout == "NHD" else 1
        call["kv_cache"] = tuple(
            torch.cat((t, torch.zeros_like(t)), dim=dim) for t in _kv_leaves(p.kv_cache)
        )
    elif name in ("kv_pool_short", "v_pool_short"):
        last = max(max(ids) for ids in p.pages)
        k, v = p.kv_cache
        call["kv_cache"] = (k if name == "v_pool_short" else k[:last], v[:last])
    elif name == "kv_strides_differ":
        k, v = p.kv_cache
        storage = torch.full(
            (*v.shape[:-1], v.shape[-1] + 16), float("nan"), device=v.device
        ).to(v.dtype)
        strided = storage[..., : v.shape[-1]]
        strided.copy_(v)
        call["kv_cache"] = (k, strided)
    elif name == "kv_last_dim_strided":
        leaves = []
        for t in _kv_leaves(p.kv_cache):
            storage = torch.full((*t.shape, 2), float("nan"), device=t.device)
            storage = storage.to(t.dtype)
            storage[..., 0] = t
            leaves.append(storage[..., 0])
        call["kv_cache"] = tuple(leaves)
    elif name == "q_last_dim_strided":
        storage = torch.full((*p.q.shape, 2), float("nan"), device=p.q.device)
        storage = storage.to(p.q.dtype)
        storage[..., 0] = p.q
        call["q"] = storage[..., 0]
    elif name in ("page_id_out_of_pool", "page_id_negative"):
        indices = plan["indices"].clone()
        pool = _kv_leaves(p.kv_cache)[0].shape[0]
        indices[0] = -1 if name == "page_id_negative" else pool
        plan["indices"] = indices
    elif name in ("out_shape", "out_dtype", "out_device", "lse_shape", "lse_dtype"):
        target, change = name.split("_")
        tensor = run[target]
        if change == "shape":
            run[target] = tensor[:-1]
        elif change == "device":
            run[target] = tensor.cpu()
        else:
            run[target] = tensor.to(torch.float16 if target == "lse" else torch.float32)
    elif name in ("out_noncontiguous", "lse_noncontiguous"):
        target = name.split("_")[0]
        dim = (cfg.head_dim,) if target == "out" else ()
        dtype = cfg.dtype if target == "out" else torch.float32
        storage = torch.full(
            (batch, 2 * heads, *dim), float("nan"), dtype=dtype, device=p.q.device
        )
        run[target] = storage[:, :heads]

        def check(out, lse):
            assert torch.isnan(storage[:, heads:]).all(), f"wrote outside {target}"
            _check_numeric(out, lse, p, reference_decode(p))

        return check
    elif name in ("lse_unrequested_dtype", "lse_unrequested_shape"):
        rows = batch - 1 if name.endswith("shape") else batch
        dtype = torch.float16 if name.endswith("dtype") else torch.float32
        # The guard covers an fp32 write into the fp16 buffer.
        storage, unrequested = _guarded((rows, heads), dtype, p.q.device, batch * heads)
        storage.fill_(-23)
        run.update(return_lse=False, lse=unrequested)
        checked = replace(cfg, return_lse=False)

        def check(out, lse):
            assert (storage == -23).all(), "wrote into an LSE buffer return_lse=False"
            _check_numeric(out, None, replace(p, config=checked), reference_decode(p))

        return check
    elif name == "window_left_drift":
        run["window_left"] = 5
    else:
        raise AssertionError(name)

    def check(out, lse):
        _check_numeric(out, lse, p, reference_decode(p))

    return check


def _native_entry(wrapper):
    return "paged_run" if wrapper.use_tensor_cores else "run"


@pytest.mark.parametrize(
    "backend,tensor_cores,mutation",
    [
        (backend, tensor_cores, mutation)
        for backend in BACKENDS
        for tensor_cores in (False, True)
        for mutation in MUTATIONS
    ],
)
@pytest.mark.parametrize("seed", SEEDS[:2])
def test_decode_fuzz_mutation(
    seed, backend, tensor_cores, mutation, workspace, monkeypatch
):
    _require_backend(backend)
    cfg = replace(
        sample_config(seed, backend),
        kv_lengths=(31, 64, 129),
        num_kv_heads=2,
        group=4,
        head_dim=128,
        page_size=16,
        fp8_kv=False,
        tensor_cores=tensor_cores,
        window_left=-1,
        soft_cap=0.0,
        sm_scale=None,
        kv_scales=None,
        return_lse=True,
        kv_form="tuple",
        q_strided=False,
    )
    case = (mutation, "tensor-core" if tensor_cores else "cuda-core")
    _skip_if_quarantined(case)
    finding, failure = LEDGER.find(case), None
    with _case(cfg, backend, mutation):
        # Finite padding isolates contract checks from padding-read failures.
        p = make_problem(cfg, poison_padding=False)
        control = _wrapper(p, backend, workspace)
        _require_phase(control)
        _run_and_check(p, control)
        try:
            _check_mutation(p, mutation, backend, workspace, monkeypatch)
        except (AssertionError, pytest.fail.Exception) as error:
            if finding is None:
                raise
            failure = error
    if finding is not None:
        if failure is None:
            LEDGER.flag_xpass(finding, f"{mutation} on the {case[1]} kernel")
        pytest.xfail(f"[{LEDGER.op}] {finding.reason}: {_first_line(failure)}")


def _check_mutation(p, mutation, backend, workspace, monkeypatch):
    cfg, spec = p.config, MUTATIONS[mutation]
    shape = (len(cfg.kv_lengths), cfg.num_qo_heads)
    out = torch.full(
        (*shape, cfg.head_dim), float("nan"), dtype=cfg.dtype, device="cuda"
    )
    lse = torch.full(shape, float("nan"), dtype=torch.float32, device="cuda")
    plan, run = _plan_kwargs(p), _run_kwargs(p, out, lse)
    call = dict(q=p.q, kv_cache=p.kv_cache)
    check = mutate(p, mutation, plan, run, call)
    candidate = _new_wrapper(cfg, backend, workspace)
    errors = HOST_ERRORS if spec.native or spec.phase == "plan" else CLEAN_ERRORS
    if spec.phase == "plan" and not spec.comparable:
        try:
            candidate.plan(**plan)
        except HOST_ERRORS as error:
            assert re.search(spec.message, str(error)), error
            return
        if not spec.native:
            pytest.fail(f"plan() accepted {mutation}")
    if spec.comparable:
        try:
            candidate.plan(**plan)
            returned = candidate.run(call["q"], call["kv_cache"], **run)
            torch.cuda.synchronize()
        except errors as error:
            assert re.search(spec.message, str(error), re.IGNORECASE), error
            return
        if run.get("return_lse"):
            returned_out, returned_lse = returned
        else:
            returned_out, returned_lse = returned, None
        check(returned_out, returned_lse)
        return
    if spec.phase == "run":
        candidate.plan(**plan)
    if spec.native:
        # The native entry's host checks run before any launch.
        with pytest.raises(HOST_ERRORS, match=spec.message):
            candidate.run(call["q"], call["kv_cache"], **run)
        return

    # Invalid shapes, dtypes and page IDs must fail before a native launch
    # can read or write out of bounds.
    def unexpected_launch(*args, **kwargs):
        pytest.fail(f"{mutation} reached the native kernel without rejection")

    with monkeypatch.context() as patch:
        patch.setattr(
            candidate._cached_module, _native_entry(candidate), unexpected_launch
        )
        with pytest.raises(CLEAN_ERRORS, match=spec.message):
            candidate.run(call["q"], call["kv_cache"], **run)


def _copy_cache(dst, src):
    for target, source in zip(
        _kv_leaves(dst.kv_cache), _kv_leaves(src.kv_cache), strict=True
    ):
        target.fill_(float("nan"))
        target[: source.shape[0]].copy_(source)


class _Capture:
    """A graph-mode wrapper captured once over storage sized for later replans."""

    def __init__(self, cfg, backend, workspace):
        # Each plan's q and KV pages are copied into this captured storage.
        self.storage = make_problem(
            replace(cfg, seed=9000, kv_lengths=(2000, 1, 1)), poison_padding=False
        )
        self.wrapper = _new_wrapper(
            cfg, backend, workspace, graph=(3, len(self.storage.indices) + 16)
        )
        shape = (3, cfg.num_qo_heads)
        self.out = torch.empty((*shape, cfg.head_dim), dtype=cfg.dtype, device="cuda")
        self.lse = torch.empty(shape, dtype=torch.float32, device="cuda")
        first = make_problem(cfg, poison_padding=False)
        self.load(first)
        kwargs = _run_kwargs(first, self.out, self.lse)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.wrapper.run(self.storage.q, self.storage.kv_cache, **kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.wrapper.run(self.storage.q, self.storage.kv_cache, **kwargs)
        self.replay(first)
        self.current = first

    def load(self, problem):
        self.storage.q.copy_(problem.q)
        _copy_cache(self.storage, problem)
        self.wrapper.plan(**_plan_kwargs(problem))
        _require_phase(self.wrapper)
        self.current = problem

    def replay(self, problem=None):
        problem = self.current if problem is None else problem
        self.out.fill_(float("nan"))
        self.lse.fill_(float("nan"))
        self.graph.replay()
        torch.cuda.synchronize()
        _check_numeric(self.out, self.lse, problem, reference_decode(problem))


def _graph_config(tensor_cores, return_lse=True, kv_layout="NHD"):
    return Config(
        9001,
        (31, 65, 129),
        tensor_cores=tensor_cores,
        return_lse=return_lse,
        kv_layout=kv_layout,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tensor_cores", (False, True))
@pytest.mark.parametrize("return_lse", (False, True))
@pytest.mark.parametrize("kv_layout", ("NHD", "HND"))
def test_decode_fuzz_graph_replan(
    backend, tensor_cores, return_lse, kv_layout, workspace
):
    _require_backend(backend)
    cfg = _graph_config(tensor_cores, return_lse, kv_layout)
    with _case(cfg, backend, "graph_replan"):
        capture = _Capture(cfg, backend, workspace)
        capture.storage.q.mul_(0.5)
        capture.current.q.mul_(0.5)
        capture.replay()
        # Same batch, other lengths and pages: the long request moves to row 0,
        # then all-short requests, then a longer request than the capture saw.
        for seed, lengths in (
            (9003, (1100, 15, 32)),
            (9004, (1, 16, 17)),
            (9006, (1800, 31, 16)),
        ):
            capture.load(
                make_problem(
                    replace(cfg, seed=seed, kv_lengths=lengths), poison_padding=False
                )
            )
            capture.replay()
            # A replan rejected before any copy leaves the capture replayable.
            for rejected, message in (
                ((31, 65), r"batch size"),
                ((4096,) * 3, r"indices|buffer"),
            ):
                other = make_problem(replace(cfg, seed=9005, kv_lengths=rejected))
                with pytest.raises(CLEAN_ERRORS, match=message):
                    capture.wrapper.plan(**_plan_kwargs(other))
                capture.replay()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tensor_cores", (False, True))
def test_decode_fuzz_graph_late_rejection(backend, tensor_cores, workspace):
    """A replan rejected after its metadata reached the buffers must not leak."""
    _require_backend(backend)
    cfg = _graph_config(tensor_cores)
    kernel = "tensor-core" if tensor_cores else "cuda-core"
    finding, failure = LEDGER.find(("graph_late_rejection", kernel)), None
    with _case(cfg, backend, "graph_late_rejection"):
        capture = _Capture(cfg, backend, workspace)
        # Same lengths, other pages, plus an argument the plan refuses late.
        late = _plan_kwargs(capture.current)
        late["indices"] = late["indices"].flip(0)
        if tensor_cores:
            late["num_qo_heads"] += 1
        else:
            late["fixed_split_size"] = 4
        with pytest.raises(HOST_ERRORS, match=r"divisible|fixed_split_size"):
            capture.wrapper.plan(**late)
        try:
            capture.replay()
        except AssertionError as error:
            if finding is None:
                raise
            failure = error
    if finding is not None:
        if failure is None:
            LEDGER.flag_xpass(finding, f"graph_late_rejection on the {kernel} kernel")
        pytest.xfail(f"[{LEDGER.op}] {finding.reason}: {_first_line(failure)}")


# Oracle self-tests: CPU only, they check the reference and the bound.


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("length", (128, 4096))
@pytest.mark.parametrize(
    "loss", ("first_page", "last_token", "last_chunk", "window_start", "window_leak")
)
def test_decode_reference_detects_missing_kv(dtype, length, loss):
    window = length // 4 if loss.startswith("window") else -1
    cfg = Config(19, (length,), dtype=dtype, window_left=window, return_lse=False)
    p = make_problem(cfg, "cpu")
    reference = reference_decode(p)
    if loss == "first_page":
        broken = replace(
            p,
            config=replace(cfg, kv_lengths=(length - cfg.page_size,)),
            pages=(p.pages[0][1:],),
        )
    elif loss.startswith("window"):
        shift = -1 if loss == "window_start" else 1
        broken = replace(p, config=replace(cfg, window_left=window + shift))
    else:
        removed = 1 if loss == "last_token" else length // 8
        broken = replace(p, config=replace(cfg, kv_lengths=(length - removed,)))
    with pytest.raises(AssertionError):
        _check_numeric(reference_decode(broken).out.to(dtype), None, p, reference)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("length", (128, 4096))
@pytest.mark.parametrize("soft_cap", (0.0, 30.0))
def test_decode_reference_accepts_rounded_pv(dtype, length, soft_cap):
    cfg = Config(79, (length,), dtype=dtype, soft_cap=soft_cap, return_lse=False)
    p = make_problem(cfg, "cpu")
    keys, values = (
        pages[list(p.pages[0])].reshape(-1, cfg.num_kv_heads, cfg.head_dim)[:length]
        for pages in (p.k_pages, p.v_pages)
    )
    keys = keys.double().repeat_interleave(cfg.group, dim=1)
    values = values.double().repeat_interleave(cfg.group, dim=1)
    scores = torch.einsum("hd,khd->hk", p.q_dense[0].double(), keys) * _sm_scale(cfg)
    if soft_cap:
        # Worst case of the approximate tanh: every capped logit off by the
        # full relative error, alternating in sign.
        signs = torch.where(torch.arange(length) % 2 == 0, 1.0, -1.0).double()
        scores = (
            soft_cap * torch.tanh(scores / soft_cap) * (1 + TANH_APPROX_ERROR * signs)
        )
    probs = scores.softmax(-1).to(dtype).double()
    out = torch.einsum("hk,khd->hd", probs, values)[None].to(dtype)
    _check_numeric(out, None, p, reference_decode(p))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("head_dim", (64, 128, 256))
@pytest.mark.parametrize("soft_cap", (30.0, 50.0))
def test_decode_reference_detects_missing_soft_cap(dtype, head_dim, soft_cap):
    """The contrast tokens make a dropped soft cap visible without LSE."""
    cfg = Config(
        29,
        (128, 4096),
        head_dim=head_dim,
        dtype=dtype,
        soft_cap=soft_cap,
        return_lse=False,
    )
    p = make_problem(cfg, "cpu")
    uncapped = reference_decode(replace(p, config=replace(cfg, soft_cap=0.0)))
    with pytest.raises(AssertionError):
        _check_numeric(uncapped.out.to(dtype), None, p, reference_decode(p))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_decode_reference_zero_bound_allows_only_subnormal_error(dtype):
    p = make_problem(
        Config(0, (1,), num_kv_heads=1, group=1, dtype=dtype, return_lse=False), "cpu"
    )
    p.v_pages.zero_()
    reference = reference_decode(p)
    out = reference.out.to(dtype)
    _check_numeric(out, None, p, reference)
    # One subnormal step passes; the smallest normal value does not.
    out[0, 0, 0] = torch.finfo(dtype).tiny * UNIT_ROUNDOFF[dtype] * 2
    _check_numeric(out, None, p, reference)
    out[0, 0, 0] = torch.finfo(dtype).tiny
    with pytest.raises(AssertionError, match="exceed the rounding bound"):
        _check_numeric(out, None, p, reference)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_decode_reference_accepts_rounded_lse(dtype):
    p = make_problem(
        Config(0, (2,), num_kv_heads=1, group=1, dtype=dtype, return_lse=True), "cpu"
    )
    # Zero values and a score shift make the exact output and LSE both zero.
    # The second exponential lies just below a dtype rounding midpoint.
    weights = torch.tensor(
        [1.0, 0.5 + 0.49 * UNIT_ROUNDOFF[dtype]], dtype=torch.float64
    )
    lse = (weights.to(dtype).double().sum().log() - weights.sum().log()) / math.log(2)
    out = torch.zeros((1, 1, 128), dtype=dtype)
    zero, lse_zero = out.double(), torch.zeros((1, 1), dtype=torch.float64)
    reference = Reference(zero, lse_zero, zero, zero, zero, lse_zero)
    _check_numeric(out, lse.float().reshape(1, 1), p, reference)
    with pytest.raises(AssertionError):
        _check_numeric(out, torch.full((1, 1), 0.1), p, reference)


@pytest.mark.parametrize("window_left", (-1, 0, 1))
@pytest.mark.parametrize("soft_cap", (0.0, 2.0))
def test_decode_reference_hand_values(window_left, soft_cap):
    """GQA head mapping, window, soft cap, base-2 LSE and fp8 scales by hand."""
    cfg = Config(
        7,
        (3,),
        num_kv_heads=2,
        group=2,
        head_dim=4,
        page_size=2,
        window_left=window_left,
        soft_cap=soft_cap,
        sm_scale=1.0,
        fp8_kv=True,
        kv_scales=(0.5, 2.0),
    )
    p = make_problem(cfg, "cpu", poison_padding=True)
    p.q_dense.zero_()
    p.q_dense[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for page in p.pages[0]:
        p.k_pages[page] = 0
        p.v_pages[page] = 0
    for token, (key, value) in enumerate(((2.0, 1.0), (0.0, 2.0), (4.0, 4.0))):
        page, offset = p.pages[0][token // 2], token % 2
        for kv_head in range(2):
            # Head 1 carries different values, so a wrong head mapping shows.
            p.k_pages[page, offset, kv_head, 0] = key * (kv_head + 1)
            p.v_pages[page, offset, kv_head, 0] = value * (kv_head + 1)
    reference = reference_decode(p)
    visible = range(max(0, 2 - window_left) if window_left >= 0 else 0, 3)
    for head in range(4):
        kv_head = head // 2
        q = float(head + 1)
        logits = torch.tensor(
            [q * key * (kv_head + 1) * 0.5 for key in (2.0, 0.0, 4.0)],
            dtype=torch.float64,
        )
        if soft_cap:
            logits = soft_cap * torch.tanh(logits / soft_cap)
        logits = logits[list(visible)]
        values = torch.tensor(
            [value * (kv_head + 1) * 2.0 for value in (1.0, 2.0, 4.0)],
            dtype=torch.float64,
        )[list(visible)]
        torch.testing.assert_close(
            reference.out[0, head, 0], logits.softmax(0) @ values, atol=1e-12, rtol=0
        )
        torch.testing.assert_close(
            reference.lse[0, head],
            torch.logsumexp(logits, 0) / math.log(2),
            atol=1e-12,
            rtol=0,
        )
        assert (reference.out[0, head, 1:] == 0).all()


def test_decode_problem_seed_and_padding():
    cfg = Config(9, (17, 33), kv_layout="HND", kv_form="strided", q_strided=True)
    first = make_problem(cfg, "cpu", poison_padding=True)
    second = make_problem(cfg, "cpu", poison_padding=True)
    assert first.pages == second.pages
    for a, b in zip(_inputs(first), _inputs(second), strict=True):
        assert _crc(a) == _crc(b)
    assert torch.isnan(first.k_pages[first.nan_page]).all()
    assert all(first.nan_page not in ids for ids in first.pages)
    for ids, length in zip(first.pages, cfg.kv_lengths, strict=True):
        tail = (length - 1) % cfg.page_size + 1
        assert torch.isnan(first.v_pages[ids[-1], tail:]).all()
        assert torch.isfinite(first.v_pages[ids[-1], :tail]).all()
    k, v = first.kv_cache
    assert k.stride(-2) == cfg.head_dim + 16 and k.stride() == v.stride()
    assert torch.equal(k.transpose(1, 2).nan_to_num(), first.k_pages.nan_to_num())
    assert torch.equal(first.q, first.q_dense) and not first.q.is_contiguous()
