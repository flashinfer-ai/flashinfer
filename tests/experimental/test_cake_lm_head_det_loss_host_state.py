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

Host state of the deterministic chunked LM-head + loss backend across calls: the binding cache of the
log-probability backward across training steps, and the per-thread state (the hidden valid-row count's pinned cell
and event, its int32 constant, the dW side stream) under two threads on one device.
"""

import threading

import pytest
import torch

import flashinfer.chunked_lm_head as public
from tests.test_helpers.cake_lm_head_loss_reference import make_inputs
from tests.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake_lm_head_det_loss", "cake_backend")
cake_jit = cake_package("cake_lm_head_det_loss", "cake_jit")

SEED = 20261031
CUDA = torch.device("cuda")
_TIMEOUT_S = 120


def _require_hidden_count():
    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)
        in cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
    ):
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
    for entry in ("loss", "logprob"):
        if not cake_backend.generated_program_available(CUDA, entry=entry):
            pytest.skip(
                f"generated chunked LM-head program ({entry} entry) not registered for this device"
            )
    if not cake_jit.toolchain_runs_hidden_count(cake_backend.arch_for(CUDA)):
        pytest.skip("the hidden valid-row count does not run with this toolchain")


@pytest.fixture(scope="module")
def weight():
    _require_hidden_count()
    return make_inputs(1, seed=SEED, device=CUDA, ignore_frac=0.0).W


def _bits(t):
    t = t.detach().reshape(-1).contiguous()
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _logprob_step(inp, chunk):
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    logp = public.chunked_lm_head_det_logprob(X, W, inp.labels, chunk_size=chunk)
    logp.backward(inp.dlogp)
    return dict(logp=logp.detach(), dX=X.grad, dW=W.grad)


@pytest.mark.parametrize(
    "T, chunk", [(4097, 4096), (1000, 256)]
)  # the host count (two chunks), the hidden count (four chunks)
def test_device_logprob_steps_hit_one_backward_binding(weight, T, chunk):
    """Training steps of the log-probability entry (the previous step's outputs alive while the next runs) reuse one
    forward and one backward binding: the backward's ``lse`` and ``dlogp`` are fresh tensors every step, so no step
    after the first misses, every step allocates the same number of blocks, and the bits are the uncached call's."""
    inp = make_inputs(T, seed=SEED + T, device=CUDA, W=weight)
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    out = _logprob_step(inp, chunk)  # binds (misses once per direction)
    out = _logprob_step(inp, chunk)
    torch.cuda.synchronize()
    misses, allocations = cache.misses, []
    for _ in range(6):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = _logprob_step(inp, chunk)
        torch.cuda.synchronize()
        allocations.append(
            torch.cuda.memory_stats()["allocation.all.allocated"] - before
        )
    assert cache.misses == misses, (cache.misses - misses, allocations)
    assert len(set(allocations)) == 1, allocations
    was = cache.enabled
    cache.enabled = False
    try:
        fresh = _logprob_step(inp, chunk)
    finally:
        cache.enabled = was
    for key, value in fresh.items():
        assert torch.equal(_bits(out[key]), _bits(value)), key


def _run_threads(first, second):
    """Run ``first`` and ``second`` on the threads named ``first`` / ``second``, each on its own stream of the device;
    re-raise the first error."""
    errors = []

    def wrap(fn):
        def run():
            try:
                with torch.cuda.device(CUDA), torch.cuda.stream(torch.cuda.Stream()):
                    fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
                errors.append(exc)

        return run

    threads = [
        threading.Thread(target=wrap(fn), name=fn.__name__) for fn in (first, second)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(_TIMEOUT_S)
    assert not any(t.is_alive() for t in threads), "a thread did not finish"
    torch.cuda.synchronize()
    if errors:
        raise errors[0]


def test_hidden_count_state_is_per_thread():
    """Two threads get their own pinned count cell, event, int32 constant, side stream and fork event of one device;
    each thread gets the same ones on every call."""
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    index = torch.cuda.current_device()
    got = {}

    def state():
        cell = cake_backend._hidden_count_cell(index)
        ar = cake_backend._hidden_count_arange(index, 8)
        side = cake_backend._dw_stream(index)
        assert cake_backend._hidden_count_cell(index) is cell
        assert cake_backend._hidden_count_arange(index, 8) is ar
        assert cake_backend._dw_stream(index) is side
        got[threading.current_thread().name] = dict(
            cell=cell[0], event=cell[1], arange=ar, stream=side[0], fork=side[1]
        )

    def first():
        state()

    def second():
        state()

    _run_threads(first, second)
    a, b = got["first"], got["second"]
    assert a["cell"].data_ptr() != b["cell"].data_ptr()
    assert a["arange"].data_ptr() != b["arange"].data_ptr()
    assert a["event"] is not b["event"] and a["fork"] is not b["fork"]
    assert a["stream"] != b["stream"]
    assert torch.equal(
        a["arange"][:8].cpu(), torch.arange(1, 9, dtype=torch.int32)
    ) and torch.equal(b["arange"][:8].cpu(), torch.arange(1, 9, dtype=torch.int32))


def test_hidden_count_arange_records_each_stream_once():
    """The int32 constant is recorded once per stream it is used on, again on a new stream, and afresh after it grew."""
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    index = torch.cuda.current_device()
    got = {}

    def run():
        recorded = cake_backend._thread_state("arange_streams")
        ar = cake_backend._hidden_count_arange(index, 8)
        cake_backend._hidden_count_arange(index, 8)
        got["one"] = len(recorded[index])
        side = torch.cuda.Stream(index)
        with torch.cuda.stream(side):
            assert cake_backend._hidden_count_arange(index, 8) is ar
            cake_backend._hidden_count_arange(index, 8)
        got["two"] = len(recorded[index])
        grown = cake_backend._hidden_count_arange(index, 2 * ar.numel() + 1)
        got["grown"] = (grown is not ar, len(recorded[index]))
        got["values"] = torch.equal(
            grown.cpu(), torch.arange(1, grown.numel() + 1, dtype=torch.int32)
        )

    _run_threads(run, lambda: None)
    assert got["one"] == 1 and got["two"] == 2
    assert got["grown"] == (True, 1) and got["values"]


def _policy_step(inp, chunk):
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    loss, logp = public.chunked_lm_head_det_loss(
        X,
        W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=chunk,
        return_logp=True,
    )
    loss.backward()
    return dict(loss=loss.detach(), logp=logp.detach(), dX=X.grad, dW=W.grad)


def test_device_zero_weight_rows_concurrent_threads(weight, monkeypatch):
    """Policy steps with zero-weight and ignored rows on two threads, each on its own stream (four chunks: the hidden
    count and the dW side stream run).  The host order is forced: the first thread's call waits between its count's
    copy and its read until the second thread's whole step returned.  Both give the single-thread bits (one shared
    count cell would hand the first call the second's count)."""
    C, T = 256, 1000
    inputs = {}
    for name, seed, ignore in (("first", SEED + 5, 0.05), ("second", SEED + 6, 0.40)):
        inp = make_inputs(
            T, objective="policy", seed=seed, device=CUDA, W=weight, ignore_frac=ignore
        )
        inp.loss_weights[::3] = 0.0
        inputs[name] = inp
    counts = {name: int(i.labels.ge(0).sum()) for name, i in inputs.items()}
    assert counts["second"] < counts["first"]  # the other count never indexes past
    ref = {
        name: {k: _bits(v).clone() for k, v in _policy_step(inp, C).items()}
        for name, inp in inputs.items()
    }
    torch.cuda.synchronize()
    begun, read = threading.Barrier(2), threading.Barrier(2)
    finish = cake_backend._HiddenCount.finish
    paused = []

    def finish_after_the_other_step(self):
        if threading.current_thread().name == "first" and not paused:
            paused.append(True)
            begun.wait(_TIMEOUT_S)
            read.wait(_TIMEOUT_S)
        return finish(self)

    monkeypatch.setattr(
        cake_backend._HiddenCount, "finish", finish_after_the_other_step
    )
    got = {}

    def first():
        got["first"] = _policy_step(inputs["first"], C)

    def second():
        begun.wait(_TIMEOUT_S)
        got["second"] = _policy_step(inputs["second"], C)
        torch.cuda.current_stream().synchronize()
        read.wait(_TIMEOUT_S)

    _run_threads(first, second)
    assert paused == [True]
    wrong = [
        (name, key)
        for name in ref
        for key in ref[name]
        if not torch.equal(_bits(got[name][key]), ref[name][key])
    ]
    assert not wrong, wrong


def test_device_hidden_count_arange_two_threads_two_streams(weight, monkeypatch):
    """The second thread's call extends the count constant on its stream behind queued work (a GPU sleep); then the
    first thread counts on its own, idle stream.  Each call reads its own constant and gives the single-thread bits
    (one shared constant would hand the first call the second's, not yet written by its stream)."""
    C = 256  # every call below has three or more chunks: the hidden count runs
    index = torch.cuda.current_device()
    a = make_inputs(1000, seed=SEED + 7, device=CUDA, W=weight, ignore_frac=0.05)
    ref_a = _bits(cake_backend.forward_logprob(a.X, a.W, a.labels, chunk_size=C).logp)
    held = {}

    def probe():
        held["rows"] = int(cake_backend._hidden_count_arange(index, 1).numel())

    _run_threads(probe, lambda: None)
    # more rows than the constant a new thread sees, so the second call extends the constant it gets
    rows_b = max(2 * a.T, held["rows"] + 1)
    b = make_inputs(rows_b, seed=SEED + 8, device=CUDA, W=weight, ignore_frac=0.40)
    torch.cuda.synchronize()
    grown = threading.Event()
    arange = cake_backend._hidden_count_arange

    def arange_then_signal(i, T):
        ar = arange(i, T)
        if threading.current_thread().name == "second":
            grown.set()
        return ar

    monkeypatch.setattr(cake_backend, "_hidden_count_arange", arange_then_signal)
    got = {}

    def first():
        assert grown.wait(_TIMEOUT_S)
        got["first"] = cake_backend.forward_logprob(
            a.X, a.W, a.labels, chunk_size=C
        ).logp

    def second():
        # zeros in this stream's free memory: the extended constant's block, unwritten until the sleep ends
        stale = torch.zeros(
            (max(rows_b, 2 * held["rows"]),), dtype=torch.int32, device=CUDA
        )
        torch.cuda.current_stream().synchronize()
        del stale
        torch.cuda._sleep(2_000_000_000)
        got["second"] = cake_backend.forward_logprob(
            b.X, b.W, b.labels, chunk_size=C
        ).logp

    _run_threads(first, second)
    monkeypatch.undo()
    # the second call's reference after the threads: before them it would extend the constant itself
    ref_b = _bits(cake_backend.forward_logprob(b.X, b.W, b.labels, chunk_size=C).logp)
    wrong = [
        name
        for name, ref in (("first", ref_a), ("second", ref_b))
        if not torch.equal(_bits(got[name]), ref)
    ]
    assert not wrong, wrong
