"""Seeded reject-or-correct checks for the public Batch MLA wrapper (#3605).

FLASHINFER_MLA_FUZZ_NUM_TESTS controls the random cases (default 150).
FLASHINFER_MLA_FUZZ_SEED sets the first seed (default 0).
FLASHINFER_MLA_FUZZ_ONLY_SEED selects comma-separated seeds for reproduction.
Valid cases must run; mutations must reject cleanly or preserve the oracle.
"""

import math
import os
import random
import re
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, replace

import pytest
import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
from flashinfer.utils import get_compute_capability, get_indptr, is_sm90a_supported
from tests.test_helpers.fuzz_ledger import Finding, FuzzLedger


BACKENDS = ("fa2", "fa3", "cutlass", "auto")
NUM_TESTS = int(os.environ.get("FLASHINFER_MLA_FUZZ_NUM_TESTS", "150"))
START_SEED = int(os.environ.get("FLASHINFER_MLA_FUZZ_SEED", "0"))
ONLY_SEED = os.environ.get("FLASHINFER_MLA_FUZZ_ONLY_SEED", "")
SEEDS = (
    [int(seed) for seed in ONLY_SEED.split(",") if seed.strip()]
    if ONLY_SEED
    else list(range(START_SEED, START_SEED + NUM_TESTS))
)
CKV, KPE = 512, 64
CLEAN_ERRORS = (ValueError, TypeError)
LEDGER = FuzzLedger(
    "batch-mla",
    [
        Finding(
            match=lambda mutation: (
                mutation in ("page_id_out_of_pool", "page_id_negative", "kv_pool_short")
            ),
            backend="cutlass",
            reason="#6741: CUTLASS does not validate page indices against the runtime KV pool",
            quarantine=True,
        ),
        Finding(
            match=lambda phase: phase == "poisoned_padding",
            backend="cutlass",
            reason="#6860: CUTLASS MLA propagates NaNs from last-page tails and dense-table padding",
            quarantine=True,
        ),
        Finding(
            match=lambda phase: phase == "poisoned_padding",
            backend="cute-dsl-monolithic",
            reason="#6861: CuTe DSL monolithic MLA propagates padding NaNs into valid output",
            quarantine=True,
        ),
    ],
)
UNIT_ROUNDOFF = {torch.bfloat16: 2**-8, torch.float16: 2**-11}
BOUND_FACTOR = 8


@dataclass(frozen=True)
class Config:
    seed: int
    q_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    heads: int = 16
    page_size: int = 16
    dtype: torch.dtype = torch.bfloat16
    causal: bool = False
    metadata_form: str = "csr"
    metadata_device: str = "cpu"
    query_layout: str = "split"
    kv_layout: str = "split"
    lse_mode: str = "base2"
    fp8_kv: bool = False


@dataclass
class Problem:
    config: Config
    query: object
    kv_cache: object
    q: torch.Tensor
    cache: torch.Tensor
    pages: tuple[tuple[int, ...], ...]
    metadata: object
    nan_page: int


@dataclass
class Reference:
    out: torch.Tensor
    lse: torch.Tensor
    magnitude: torch.Tensor


def sample_config(seed, backend):
    rng = random.Random(seed)
    page = rng.choice((1, 16, 32, 64))
    batch = rng.randint(1, 8)
    kv_lengths = tuple(
        rng.choice((1, max(1, page - 1), page, page + 1, 127, 128, 129, 4096))
        for _ in range(batch)
    )
    q_lengths = tuple(min(rng.choice((1, 8, 33)), length) for length in kv_lengths)
    cfg = Config(
        seed,
        q_lengths,
        kv_lengths,
        heads=rng.choice((16, 32, 64, 128)),
        page_size=page,
        dtype=rng.choice((torch.float16, torch.bfloat16)),
        causal=rng.choice((False, True)),
        metadata_form=rng.choice(("csr", "dense")),
        metadata_device=rng.choice(("cpu", "cuda")),
        query_layout=rng.choice(("packed", "adjacent", "split")),
        kv_layout=rng.choice(("packed", "adjacent", "split")),
        lse_mode=rng.choice(("none", "base2", "basee")),
    )
    if backend == "cutlass":
        cfg = replace(
            cfg,
            q_lengths=(1,) * batch,
            heads=128,
            query_layout="packed",
            kv_layout="packed",
            lse_mode="none",
        )
    return cfg


def _structure(tensor, layout):
    if layout == "packed":
        return tensor
    parts = tensor.split((CKV, KPE), dim=-1)
    return tuple(part.contiguous() for part in parts) if layout == "split" else parts


def make_problem(cfg, device="cuda", *, poison_padding=True):
    generator = torch.Generator().manual_seed(cfg.seed)
    counts = [
        (length + cfg.page_size - 1) // cfg.page_size for length in cfg.kv_lengths
    ]
    pool_size = sum(counts) + 2
    order = torch.randperm(pool_size, generator=generator).tolist()
    pages, cursor = [], 0
    for count in counts:
        pages.append(tuple(order[cursor : cursor + count]))
        cursor += count
    q = torch.randn(sum(cfg.q_lengths), cfg.heads, CKV + KPE, generator=generator) * 0.5
    cache = torch.randn(pool_size, cfg.page_size, CKV + KPE, generator=generator) * 0.5
    # Three heads attend to the first, last and middle token; other heads stay random.
    # Lost pages or split-KV chunks must affect out even when LSE is not requested.
    anchors = min(3, cfg.heads)
    q[..., CKV : CKV + anchors] = 0
    cache[..., CKV : CKV + anchors] = 0
    start = 0
    for q_length, kv_length, ids in zip(
        cfg.q_lengths, cfg.kv_lengths, pages, strict=True
    ):
        for head, token in enumerate((0, kv_length - 1, kv_length // 2)[:anchors]):
            q[start : start + q_length, head] = 0
            q[start : start + q_length, head, CKV + head] = 4
            cache[ids[token // cfg.page_size], token % cfg.page_size, CKV + head] = 48
        start += q_length
    q = q.to(device=device, dtype=cfg.dtype)
    cache = cache.to(
        device=device, dtype=torch.float8_e4m3fn if cfg.fp8_kv else cfg.dtype
    )
    nan_page = order[cursor]
    if poison_padding:
        cache[nan_page].fill_(float("nan"))
        for ids, length in zip(pages, cfg.kv_lengths, strict=True):
            tail = length % cfg.page_size
            if tail:
                cache[ids[-1], tail:].fill_(float("nan"))
    meta_device = "cpu" if str(device) == "cpu" else cfg.metadata_device

    def indptr(lengths):
        return get_indptr(torch.tensor(lengths)).to(
            device=meta_device, dtype=torch.int32
        )

    qptr = indptr(cfg.q_lengths)
    lengths = torch.tensor(cfg.kv_lengths, dtype=torch.int32, device=meta_device)
    if cfg.metadata_form == "csr":
        metadata = MLAPlanMetadata.csr(
            qptr,
            indptr(counts),
            torch.tensor(
                [page for ids in pages for page in ids],
                dtype=torch.int32,
                device=meta_device,
            ),
            lengths,
        )
    else:
        table = torch.full(
            (len(pages), max(counts) + 1),
            nan_page,
            dtype=torch.int32,
            device=meta_device,
        )
        for row, ids in enumerate(pages):
            table[row, : len(ids)] = torch.tensor(
                ids, dtype=torch.int32, device=meta_device
            )
        metadata = MLAPlanMetadata.dense(qptr, table, lengths, max(cfg.q_lengths))
    return Problem(
        cfg,
        _structure(q, cfg.query_layout),
        _structure(cache, cfg.kv_layout),
        q,
        cache,
        tuple(pages),
        metadata,
        nan_page,
    )


def reference_mla(p):
    """Gather logical requests independently of the wrapper's metadata lowering."""
    cfg = p.config
    outputs, lses, magnitudes, start = [], [], [], 0
    for q_length, kv_length, ids in zip(
        cfg.q_lengths, cfg.kv_lengths, p.pages, strict=True
    ):
        query = p.q[start : start + q_length].float()
        kv = p.cache[list(ids)].reshape(-1, CKV + KPE)[:kv_length].float()
        ckv, kpe = kv.split((CKV, KPE), dim=-1)
        if cfg.fp8_kv:
            ckv, kpe = ckv * 0.5, kpe * 2.0
        keys = torch.cat((ckv, kpe), dim=-1)
        scores = torch.einsum("qhd,kd->hqk", query, keys) / math.sqrt(128 + KPE)
        if cfg.causal:
            qpos = torch.arange(q_length, device=query.device)[:, None]
            kpos = torch.arange(kv_length, device=query.device)[None, :]
            scores.masked_fill_(kpos > kv_length - q_length + qpos, -float("inf"))
        lse = torch.logsumexp(scores, dim=-1).T
        probs = scores.softmax(-1)
        outputs.append(torch.einsum("hqk,kd->qhd", probs, ckv))
        magnitudes.append(torch.einsum("hqk,kd->qhd", probs, ckv.abs()))
        lses.append(lse / math.log(2) if cfg.lse_mode == "base2" else lse)
        start += q_length
    return Reference(torch.cat(outputs), torch.cat(lses), torch.cat(magnitudes))


def _plan_kwargs(p):
    cfg = p.config
    return dict(
        metadata=p.metadata,
        num_heads=cfg.heads,
        head_dim_ckv=CKV,
        head_dim_kpe=KPE,
        page_size=cfg.page_size,
        causal=cfg.causal,
        sm_scale=1 / math.sqrt(128 + KPE),
        q_data_type=cfg.dtype,
        kv_data_type=p.cache.dtype,
        query_layout="split" if cfg.query_layout == "split" else "packed",
        kv_cache_layout="split" if cfg.kv_layout == "split" else "packed",
        lse_mode=cfg.lse_mode,
        scale_mode="kv-per-tensor" if cfg.fp8_kv else "default",
    )


def _run_kwargs(p, out, lse):
    kwargs = dict(
        query=p.query,
        kv_cache=p.kv_cache,
        out=out,
        return_lse=p.config.lse_mode != "none",
    )
    if p.config.lse_mode != "none":
        kwargs.update(lse=lse, return_lse_base_on_e=p.config.lse_mode == "basee")
    if p.config.fp8_kv:
        kwargs.update(ckv_scale=0.5, kpe_scale=2.0)
    return kwargs


def _wrapper(p, backend, workspace, graph=False):
    reserves = {}
    if graph:
        batch = len(p.config.q_lengths)
        reserves = dict(
            qo_indptr=torch.empty(batch + 1, dtype=torch.int32, device=p.q.device),
            kv_indptr=torch.empty(batch + 1, dtype=torch.int32, device=p.q.device),
            kv_len_arr=torch.empty(batch, dtype=torch.int32, device=p.q.device),
            kv_indices=torch.empty(
                sum(map(len, p.pages)) + 16, dtype=torch.int32, device=p.q.device
            ),
        )
    wrapper = BatchMLAPagedAttentionWrapper(
        workspace, backend=backend, use_cuda_graph=graph, **reserves
    )
    wrapper.plan(**_plan_kwargs(p))
    return wrapper


def _guarded(shape, dtype, device):
    storage = torch.full((math.prod(shape) + 64,), -23, dtype=dtype, device=device)
    return storage, storage[32:-32].view(shape)


def _crc(tensor):
    return zlib.crc32(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )


def _check_numeric(out, lse, p, reference):
    assert torch.isfinite(reference.out).all()
    assert torch.isfinite(out).all()
    error = (out.float() - reference.out).abs()
    bound = (
        BOUND_FACTOR
        * UNIT_ROUNDOFF[p.config.dtype]
        * (reference.magnitude + reference.out.abs())
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
    if p.config.lse_mode != "none":
        assert torch.isfinite(reference.lse).all() and torch.isfinite(lse).all()
        # FA2 also sums rounded exp(score - max) weights for the normalizer.
        # Their relative sum error is at most u + N * eta: the sum is >= 1,
        # and eta = tiny * u bounds subnormal rounding per weight.
        u = UNIT_ROUNDOFF[p.config.dtype]
        relative_sum_error = u * (
            1 + max(p.config.kv_lengths) * torch.finfo(p.config.dtype).tiny
        )
        rounding_bound = -math.log1p(-relative_sum_error)
        if p.config.lse_mode == "base2":
            rounding_bound /= math.log(2)
        torch.testing.assert_close(
            lse, reference.lse, atol=1e-3 + rounding_bound, rtol=1e-3
        )


def _run_and_check(p, wrapper):
    reference = reference_mla(p)
    shape = (sum(p.config.q_lengths), p.config.heads)
    out_storage, out = _guarded((*shape, CKV), p.config.dtype, p.q.device)
    lse_storage, lse = _guarded(shape, torch.float32, p.q.device)
    inputs = (p.query, p.kv_cache)
    leaves = [
        t for value in inputs for t in (value if isinstance(value, tuple) else (value,))
    ]
    input_crc = [_crc(t) for t in leaves]
    previous = None
    for poison in (float("nan"), 17.0):
        out.fill_(poison)
        lse.fill_(poison)
        returned = wrapper.run(**_run_kwargs(p, out, lse))
        torch.cuda.synchronize()
        if p.config.lse_mode == "none":
            assert returned is out
        else:
            assert returned[0] is out and returned[1] is lse
        _check_numeric(out, lse, p, reference)
        for storage in (out_storage, lse_storage):
            assert (storage[:32] == -23).all() and (storage[-32:] == -23).all()
        fingerprint = (_crc(out), _crc(lse) if p.config.lse_mode != "none" else None)
        if previous is not None:
            assert fingerprint == previous, "non-deterministic output/LSE bytes"
        previous = fingerprint
    assert [_crc(t) for t in leaves] == input_crc, "input storage was modified"


def _require_backend(backend):
    major, _ = get_compute_capability(torch.device("cuda"))
    if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FA3 MLA requires SM90")
    if backend == "cutlass" and major not in (10, 11):
        pytest.skip("CUTLASS MLA requires SM100-family or SM110")
    if major < 8:
        pytest.skip("FA2 MLA requires SM80 or newer")


@pytest.fixture(scope="module")
def workspace():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return torch.empty(128 << 20, dtype=torch.uint8, device="cuda")


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
            f"FLASHINFER_MLA_FUZZ_ONLY_SEED={cfg.seed} pytest -x -s "
            f"tests/attention/test_mla_paged_attention_fuzz.py -k '{backend} and {mutation}'"
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
def test_mla_fuzz_valid(seed, backend, workspace):
    _require_backend(backend)
    cfg = sample_config(seed, backend)
    with _case(cfg, backend):
        control = make_problem(cfg, poison_padding=False)
        control_wrapper = _wrapper(control, backend, workspace)
        _run_and_check(control, control_wrapper)
        LEDGER.xfail_if_quarantined(
            "poisoned_padding", control_wrapper._planned_backend_name
        )
        p = make_problem(cfg)
        _run_and_check(p, _wrapper(p, backend, workspace))


MUTATIONS = {
    "qo_nonzero_start": ("plan", r"\bqo_indptr\b"),
    "qo_nonmonotonic": ("plan", r"\bqo_indptr\b"),
    "kv_indices_short": ("plan", r"\b(kv_indices|kv_indptr)\b"),
    "kv_capacity": ("plan", r"\bkv_len_arr\b"),
    "batch_short": ("plan", r"\b(batch|kv_len_arr)\b"),
    **{
        f"metadata_{change}_{field}": ("plan", rf"\b{field}\b")
        for change in ("dtype", "rank", "device")
        for field in (
            "qo_indptr",
            "kv_indptr",
            "kv_indices",
            "kv_len_arr",
            "cum_seq_lens_q",
            "block_tables",
            "seq_lens",
        )
    },
    "query_rows": ("run", r"\b(query|q_nope|q_nope_pe)\b"),
    "query_heads": ("run", r"\b(query|q_nope|q_nope_pe)\b"),
    "query_dtype": ("run", r"\b(query|q_nope|q_data_type)\b"),
    "kv_page_size": ("run", r"\b(ckv_cache|kpe_cache|page_size|KV cache page size)\b"),
    "kv_pool_short": ("run", r"\bKV cache\b.*\bpage indices\b"),
    "independent_split": ("run", r"\b(packed|split|adjacent)\b"),
    "lse_mode": ("run", r"\b(LSE mode|lse_mode|return_lse|return_lse_base_on_e)\b"),
    "out_shape": ("run", r"\bout\b"),
    "out_dtype": ("run", r"\b(out|output dtype)\b"),
    "out_device": ("run", r"\bout\b"),
    "lse_shape": ("run", r"\blse\b"),
    "lse_dtype": ("run", r"\blse\b"),
    "indices_nan_tail": ("plan", r"\b(kv_indices|kv_indptr)\b"),
    "query_strided": ("run", r"\b(query|q_nope|q_nope_pe)\b"),
    "page_id_out_of_pool": ("run", r"\bKV cache\b.*\bpage indices\b"),
    "page_id_negative": ("run", r"\bKV cache\b.*\bpage indices\b"),
}


def mutate(p, name, plan, run):
    metadata = plan["metadata"]
    if name in ("qo_nonzero_start", "qo_nonmonotonic"):
        qptr = metadata.qo_indptr.clone()
        if name == "qo_nonzero_start":
            qptr[0] = 1
        else:
            qptr[1], qptr[2] = qptr[2].clone(), qptr[1].clone()
        plan["metadata"] = replace(metadata, qo_indptr=qptr)
    elif name == "kv_indices_short":
        plan["metadata"] = replace(metadata, kv_indices=metadata.kv_indices[:-1])
    elif name == "kv_capacity":
        lens = metadata.kv_len_arr.clone()
        lens[0] = len(p.pages[0]) * p.config.page_size + 1
        plan["metadata"] = replace(metadata, kv_len_arr=lens)
    elif name == "batch_short":
        plan["metadata"] = replace(metadata, kv_len_arr=metadata.kv_len_arr[:-1])
    elif name.startswith("metadata_"):
        _, change, field = name.split("_", 2)
        tensor = getattr(metadata, field)
        if change == "dtype":
            value = tensor.long()
        elif change == "rank":
            value = tensor[None]
        else:
            value = torch.empty_like(tensor, device="meta")
        plan["metadata"] = replace(metadata, **{field: value})
    elif name == "query_rows":
        run.update(query=p.q[:-1], out=None, lse=None)
    elif name == "query_heads":
        run.update(query=p.q[:, :-1].contiguous(), out=None, lse=None)
    elif name == "kv_page_size":
        run["kv_cache"] = p.cache[:, :-1].contiguous()
    elif name == "kv_pool_short":
        run["kv_cache"] = p.cache[: max(max(ids) for ids in p.pages)]
    elif name == "query_dtype":
        run["query"] = p.q.float()
    elif name == "independent_split":
        run["query"] = _structure(p.q, "split")
    elif name == "lse_mode":
        run.update(return_lse=True, return_lse_base_on_e=p.config.lse_mode != "basee")
    elif name in ("out_shape", "out_dtype", "out_device", "lse_shape", "lse_dtype"):
        target, change = name.split("_")
        tensor = run[target]
        if change == "shape":
            run[target] = tensor[:-1]
        elif change == "device":
            run[target] = tensor.cpu()
        else:
            run[target] = tensor.to(torch.float16 if target == "lse" else torch.float32)
    elif name == "indices_nan_tail":
        p.cache[p.nan_page].fill_(float("nan"))
        tail = torch.full(
            (17,), p.nan_page, dtype=torch.int32, device=metadata.kv_indices.device
        )
        plan["metadata"] = replace(
            metadata, kv_indices=torch.cat((metadata.kv_indices, tail))
        )
    elif name == "query_strided":
        storage = torch.empty(
            p.q.shape[0],
            p.q.shape[1] * 2,
            p.q.shape[2],
            dtype=p.q.dtype,
            device=p.q.device,
        )
        view = storage[:, : p.q.shape[1]]
        view.copy_(p.q)
        run["query"] = view
    elif name in ("page_id_out_of_pool", "page_id_negative"):
        indices = metadata.kv_indices.clone()
        indices[0] = -1 if name == "page_id_negative" else p.cache.shape[0]
        plan["metadata"] = replace(metadata, kv_indices=indices)
    else:
        raise AssertionError(name)


@pytest.mark.parametrize(
    "backend,mutation",
    [
        (backend, mutation)
        for backend in BACKENDS
        for mutation in MUTATIONS
        if not (backend == "cutlass" and mutation in ("lse_shape", "lse_dtype"))
    ],
)
@pytest.mark.parametrize("seed", SEEDS[:2])
def test_mla_fuzz_mutation(seed, backend, mutation, workspace, monkeypatch):
    _require_backend(backend)
    cfg = replace(
        sample_config(seed, backend),
        q_lengths=(1, 1, 1),
        kv_lengths=(31, 64, 129),
        query_layout="packed",
        kv_layout="packed",
        metadata_form="csr",
        lse_mode="none" if backend == "cutlass" else "base2",
    )
    if mutation.startswith("metadata_") and mutation.split("_", 2)[2] in (
        "cum_seq_lens_q",
        "block_tables",
        "seq_lens",
    ):
        cfg = replace(cfg, metadata_form="dense")
    with _case(cfg, backend, mutation):
        LEDGER.xfail_if_quarantined(mutation, backend)
        # Finite padding isolates contract checks from padding-read failures.
        p = make_problem(cfg, poison_padding=False)
        _run_and_check(p, _wrapper(p, backend, workspace))
        out = torch.full(
            (*p.q.shape[:2], CKV), float("nan"), dtype=cfg.dtype, device=p.q.device
        )
        lse = torch.full(
            p.q.shape[:2], float("nan"), dtype=torch.float32, device=p.q.device
        )
        plan, run = _plan_kwargs(p), _run_kwargs(p, out, lse)
        mutate(p, mutation, plan, run)
        candidate = BatchMLAPagedAttentionWrapper(workspace, backend=backend)
        phase, message = MUTATIONS[mutation]
        if phase == "plan" and mutation != "indices_nan_tail":
            with pytest.raises(CLEAN_ERRORS, match=message):
                candidate.plan(**plan)
            return
        candidate.plan(**plan)
        if mutation in ("indices_nan_tail", "query_strided"):
            try:
                candidate.run(**run)
                torch.cuda.synchronize()
            except CLEAN_ERRORS as error:
                assert re.search(message, str(error), re.IGNORECASE)
            else:
                _check_numeric(out, lse, p, reference_mla(p))
            return

        # Invalid shapes/page IDs must fail before a native launch can corrupt CUDA.
        def unexpected_launch(*args, **kwargs):
            pytest.fail(f"{mutation} reached the native kernel without rejection")

        implementation = candidate._planned_backend
        entry = (
            "cutlass_mla_paged_attention"
            if implementation._backend == "cutlass"
            else "run"
        )
        with monkeypatch.context() as patch:
            patch.setattr(implementation._cached_module, entry, unexpected_launch)
            with pytest.raises(CLEAN_ERRORS, match=message):
                candidate.run(**run)


@pytest.mark.parametrize(
    "backend,lse_mode",
    [
        (backend, mode)
        for backend in ("fa2", "fa3")
        for mode in ("none", "base2", "basee")
    ]
    + [("cutlass", "none")],
)
def test_mla_fuzz_graph_replan(backend, lse_mode, workspace):
    _require_backend(backend)
    cfg = Config(
        9001,
        (1, 1, 1),
        (31, 65, 129),
        lse_mode=lse_mode,
        query_layout="adjacent",
        kv_layout="adjacent",
    )
    if backend == "cutlass":
        cfg = replace(cfg, q_lengths=(1, 1, 1), heads=128)
    with _case(cfg, backend, "graph_replan"):
        p = make_problem(cfg, poison_padding=False)
        wrapper = _wrapper(p, backend, workspace, graph=True)
        out = torch.empty((*p.q.shape[:2], CKV), dtype=cfg.dtype, device=p.q.device)
        lse = torch.empty(p.q.shape[:2], dtype=torch.float32, device=p.q.device)
        kwargs = _run_kwargs(p, out, lse)
        for _ in range(3):
            wrapper.run(**kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(**kwargs)
        graph.replay()
        _check_numeric(out, lse, p, reference_mla(p))
        p.q.mul_(0.5)
        graph.replay()
        _check_numeric(out, lse, p, reference_mla(p))
        if backend in ("fa2", "fa3"):
            updated = make_problem(replace(cfg, seed=9003, kv_lengths=(15, 32, 64)))
            p.q.copy_(updated.q)
            p.cache.fill_(float("nan"))
            p.cache[: updated.cache.shape[0]].copy_(updated.cache)
            wrapper.plan(**_plan_kwargs(updated))
            graph.replay()
            _check_numeric(out, lse, updated, reference_mla(updated))
            p = updated
        other = make_problem(
            replace(cfg, q_lengths=cfg.q_lengths[:2], kv_lengths=(31, 65))
        )
        # Non-FA graph plans reject all replanning as a lifecycle error.
        error = RuntimeError if backend == "cutlass" else CLEAN_ERRORS
        message = (
            r"CUDA graph.*replan"
            if backend == "cutlass"
            else "batch|shape|buffer|size|length|replan"
        )
        with pytest.raises(error, match=message):
            wrapper.plan(**_plan_kwargs(other))
        graph.replay()
        _check_numeric(out, lse, p, reference_mla(p))


@pytest.mark.parametrize("backend", ("fa2", "fa3"))
def test_mla_fuzz_fp8_kv(backend, workspace):
    _require_backend(backend)
    cfg = Config(9002, (1, 8), (65, 129), fp8_kv=True)
    with _case(cfg, backend, "fp8_kv"):
        p = make_problem(cfg)
        if torch.cuda.get_device_capability()[0] != 9:
            with pytest.raises(CLEAN_ERRORS, match="SM90|Hopper"):
                _wrapper(p, backend, workspace)
        else:
            _run_and_check(p, _wrapper(p, backend, workspace))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("length", (128, 4096))
@pytest.mark.parametrize("loss", ("first_page", "last_token", "last_chunk"))
def test_mla_reference_detects_missing_kv(dtype, length, loss):
    cfg = Config(19, (1,), (length,), heads=4, dtype=dtype, lse_mode="none")
    p = make_problem(cfg, "cpu")
    reference = reference_mla(p)
    if loss == "first_page":
        broken = replace(
            p,
            config=replace(cfg, kv_lengths=(length - cfg.page_size,)),
            pages=(p.pages[0][1:],),
        )
    else:
        removed = 1 if loss == "last_token" else length // 8
        broken = replace(p, config=replace(cfg, kv_lengths=(length - removed,)))
    broken_reference = reference_mla(broken)
    with pytest.raises(AssertionError):
        _check_numeric(
            broken_reference.out.to(dtype), broken_reference.lse, p, reference
        )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("length", (128, 4096))
def test_mla_reference_accepts_rounded_pv(dtype, length):
    cfg = Config(79, (1,), (length,), heads=4, dtype=dtype, lse_mode="none")
    p = make_problem(cfg, "cpu")
    kv = p.cache[list(p.pages[0])].reshape(-1, CKV + KPE)[:length].float()
    probs = (
        torch.einsum("qhd,kd->hqk", p.q.float(), kv) / math.sqrt(128 + KPE)
    ).softmax(-1)
    out = torch.einsum(
        "hqk,kd->qhd", probs.to(dtype).float(), kv[:, :CKV].to(dtype).float()
    ).to(dtype)
    _check_numeric(out, None, p, reference_mla(p))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_mla_reference_zero_bound_requires_exact_output(dtype):
    p = make_problem(
        Config(0, (1,), (1,), heads=1, dtype=dtype, lse_mode="none"), "cpu"
    )
    p.cache.zero_()
    reference = reference_mla(p)
    out = reference.out.to(dtype)
    _check_numeric(out, None, p, reference)
    out[0, 0, 0] = torch.finfo(dtype).tiny
    with pytest.raises(AssertionError, match="max ratio=inf"):
        _check_numeric(out, None, p, reference)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("lse_mode", ("base2", "basee"))
def test_mla_reference_accepts_rounded_lse(dtype, lse_mode):
    p = make_problem(
        Config(0, (1,), (2,), heads=1, dtype=dtype, lse_mode=lse_mode), "cpu"
    )
    # Zero values and a score shift make the exact output and LSE both zero.
    # The second exponential lies just below a dtype rounding midpoint.
    weights = torch.tensor(
        [1.0, 0.5 + 0.49 * UNIT_ROUNDOFF[dtype]], dtype=torch.float64
    )
    lse = weights.to(dtype).double().sum().log() - weights.sum().log()
    if lse_mode == "base2":
        lse /= math.log(2)
    out = torch.zeros((1, 1, CKV), dtype=dtype)
    reference = Reference(out.float(), torch.zeros((1, 1)), out.float())
    _check_numeric(out, lse.float().reshape(1, 1), p, reference)
    with pytest.raises(AssertionError):
        _check_numeric(out, torch.full((1, 1), 0.1), p, reference)


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("lse_mode", ("base2", "basee"))
def test_mla_reference_uniform(causal, lse_mode):
    cfg = Config(
        7, (2, 1), (3, 2), heads=1, page_size=2, causal=causal, lse_mode=lse_mode
    )
    p = make_problem(cfg, "cpu", poison_padding=True)
    p.q.zero_()
    for ids, values in zip(p.pages, ((1, 2, 4), (8, 16)), strict=True):
        for token, value in enumerate(values):
            page, offset = ids[token // cfg.page_size], token % cfg.page_size
            p.cache[page, offset].zero_()
            p.cache[page, offset, 0] = value
    reference = reference_mla(p)
    expected_out = torch.zeros_like(reference.out)
    expected_out[:, 0, 0] = torch.tensor([1.5 if causal else 7 / 3, 7 / 3, 12])
    torch.testing.assert_close(reference.out, expected_out, atol=1e-6, rtol=1e-6)
    lengths = [2, 3, 2] if causal else [3, 3, 2]
    expected = torch.tensor(lengths, dtype=torch.float32).log()[:, None]
    if lse_mode == "base2":
        expected /= math.log(2)
    torch.testing.assert_close(reference.lse, expected, atol=1e-6, rtol=1e-6)


def test_mla_reference_pe_and_fp8_scales():
    cfg = Config(8, (1,), (2,), heads=1, page_size=2, fp8_kv=True, lse_mode="basee")
    p = make_problem(cfg, "cpu")
    p.q.zero_()
    p.q[..., CKV] = math.sqrt(128 + KPE)
    page = p.pages[0][0]
    p.cache[page].zero_()
    p.cache[page, 0, :CKV] = 2
    p.cache[page, 1, :CKV] = 4
    p.cache[page, 1, CKV] = 1
    reference = reference_mla(p)
    logits = torch.tensor([0.0, float(p.q[0, 0, CKV]) * 2 / math.sqrt(128 + KPE)])
    value = logits.softmax(0) @ torch.tensor([1.0, 2.0])
    torch.testing.assert_close(
        reference.out, value.expand_as(reference.out), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        reference.lse,
        logits.logsumexp(0).expand_as(reference.lse),
        atol=1e-6,
        rtol=1e-6,
    )


def test_mla_problem_seed_and_padding():
    cfg = Config(9, (1, 8), (17, 33))
    first = make_problem(cfg, "cpu", poison_padding=True)
    second = make_problem(cfg, "cpu", poison_padding=True)
    assert first.pages == second.pages
    assert _crc(first.q) == _crc(second.q) and _crc(first.cache) == _crc(second.cache)
    assert torch.isnan(first.cache[first.nan_page]).all()
    assert all(first.nan_page not in ids for ids in first.pages)
    for ids, length in zip(first.pages, cfg.kv_lengths, strict=True):
        assert torch.isnan(first.cache[ids[-1], length % cfg.page_size :]).all()
