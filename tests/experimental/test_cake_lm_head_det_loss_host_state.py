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
log-probability backward across training steps.
"""

import pytest
import torch

import flashinfer.chunked_lm_head as public
from tests.test_helpers.cake_lm_head_loss_reference import make_inputs
from tests.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake_lm_head_det_loss", "cake_backend")
cake_jit = cake_package("cake_lm_head_det_loss", "cake_jit")

SEED = 20261031
CUDA = torch.device("cuda")


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
