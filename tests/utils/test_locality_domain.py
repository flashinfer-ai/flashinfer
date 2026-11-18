# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
from pathlib import Path
import subprocess
import sys

from packaging.version import Version
import pytest
import torch


def _mock_compute_capability(monkeypatch, locality_domain, capability) -> None:
    monkeypatch.setattr(
        locality_domain, "get_compute_capability", lambda device: capability
    )


def _mock_allocator_apis(monkeypatch, locality_domain) -> None:
    _mock_compute_capability(monkeypatch, locality_domain, (10, 7))
    monkeypatch.setattr(locality_domain, "_cuda_pluggable_allocator", lambda: object)
    monkeypatch.setattr(locality_domain, "_mem_pool", lambda: object)
    monkeypatch.setattr(locality_domain, "_use_mem_pool", lambda: object)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0)])
def test_preflight_rejects_non_sm107_before_compiling(monkeypatch, capability):
    import flashinfer.locality_domain as locality_domain

    _mock_compute_capability(monkeypatch, locality_domain, capability)

    def _unexpected_toolkit_query():
        raise AssertionError("architecture check must run before toolkit checks")

    monkeypatch.setattr(locality_domain, "_get_cuda_version", _unexpected_toolkit_query)
    major, minor = capability
    with pytest.raises(
        locality_domain.LocalityDomainError,
        match=f"only supported on SM107 .* is SM{major}{minor}",
    ):
        locality_domain._preflight(torch.device("cuda:0"))


def test_preflight_rejects_missing_pluggable_allocator(monkeypatch):
    import flashinfer.locality_domain as locality_domain

    _mock_compute_capability(monkeypatch, locality_domain, (10, 7))
    monkeypatch.setattr(locality_domain, "_cuda_pluggable_allocator", lambda: None)
    with pytest.raises(
        locality_domain.LocalityDomainError,
        match="CUDAPluggableAllocator",
    ):
        locality_domain._preflight(torch.device("cuda:0"))


def test_preflight_rejects_cuda_malloc_async(monkeypatch):
    import flashinfer.locality_domain as locality_domain

    _mock_allocator_apis(monkeypatch, locality_domain)
    monkeypatch.setattr(
        locality_domain, "_allocator_backend", lambda: "cudaMallocAsync"
    )
    with pytest.raises(
        locality_domain.LocalityDomainError,
        match="native CUDA allocator",
    ):
        locality_domain._preflight(torch.device("cuda:0"))


def test_preflight_rejects_pre_134_toolkit(monkeypatch):
    import flashinfer.locality_domain as locality_domain

    _mock_allocator_apis(monkeypatch, locality_domain)
    monkeypatch.setattr(locality_domain, "_allocator_backend", lambda: "native")
    monkeypatch.setattr(locality_domain, "_get_cuda_version", lambda: Version("13.0"))
    with pytest.raises(
        locality_domain.LocalityDomainError,
        match="CUDA 13.4 toolkit headers",
    ):
        locality_domain._preflight(torch.device("cuda:0"))


def test_domain_count_query_failure_is_not_reported_as_topology(monkeypatch):
    import flashinfer.locality_domain as locality_domain

    class _FailedDomainCountModule:
        @staticmethod
        def domain_count(device_index):
            del device_index
            return -1

        @staticmethod
        def last_error():
            return "driver query failed"

    monkeypatch.setattr(
        locality_domain,
        "_load_locality_module",
        lambda device_index: (_FailedDomainCountModule(), Path("unused")),
    )
    locality_domain._get_localized_domains.cache_clear()
    try:
        with pytest.raises(
            locality_domain.LocalityDomainError,
            match="failed to query CUDA locality-domain count: driver query failed",
        ):
            locality_domain._get_localized_domains(0, "backfill")
    finally:
        locality_domain._get_localized_domains.cache_clear()


def test_flashinfer_and_benchmark_imports_are_locality_lazy():
    root = Path(__file__).resolve().parents[2]
    code = """
import sys
from pathlib import Path

import flashinfer

sys.path.insert(0, str(Path.cwd() / "benchmarks"))
import bench_trtllm_gen_fused_moe_autotuner

assert "flashinfer.locality_domain" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _require_locality(sm_split="backfill"):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from flashinfer.locality_domain import (
        get_localized_domains,
        locality_domain_unsupported_reason,
    )

    reason = locality_domain_unsupported_reason("cuda:0", sm_split=sm_split)
    if reason is not None:
        pytest.skip(f"CUDA locality domains are unavailable: {reason}")
    return get_localized_domains("cuda:0", sm_split=sm_split)


def test_localized_pool_pointer_placement():
    domains = _require_locality()
    for domain in range(2):
        tensor = domains.empty(domain, (2 * 1024**2,), torch.uint8)
        assert domains.pointer_domain(tensor) == domain


def test_localized_pool_multi_gib_pointer_placement():
    domains = _require_locality()
    num_bytes = 2 * 1024**3
    free_bytes, _ = torch.cuda.mem_get_info(domains.device)
    if free_bytes < 3 * num_bytes:
        pytest.skip("multi-GiB locality test requires at least 6 GiB free")
    for domain in range(2):
        tensor = domains.empty(domain, (num_bytes,), torch.uint8)
        assert domains.pointer_domain(tensor) == domain
        del tensor
        torch.cuda.empty_cache()


def test_backfill_and_strict_stream_topology_share_pools():
    from flashinfer.locality_domain import get_localized_domains
    from flashinfer.prims_ts.moe.partition import MoePartitionResources

    backfill = _require_locality("backfill")
    # Once the basic locality-domain preflight succeeds, strict split failures
    # are regressions rather than a reason to skip this topology check.
    strict = get_localized_domains("cuda:0", sm_split="strict")
    total_sms = torch.cuda.get_device_properties(backfill.device).multi_processor_count

    assert backfill.sm_counts == (total_sms // 2, total_sms // 2)
    assert backfill.remainder_sm_count == 0
    assert strict.sm_counts[0] == strict.sm_counts[1]
    assert strict.sm_coscheduled_alignment >= 2
    assert all(
        count % strict.sm_coscheduled_alignment == 0 for count in strict.sm_counts
    )
    assert sum(strict.sm_counts) + strict.remainder_sm_count == total_sms
    assert backfill.pools[0] is strict.pools[0]
    assert backfill.pools[1] is strict.pools[1]
    assert MoePartitionResources(backfill.streams).fork_join is not None
    assert MoePartitionResources(strict.streams).fork_join is not None


def test_localized_pool_releases_driver_memory():
    domains = _require_locality()
    # Exclude one-time CUDA/PyTorch kernel and runtime initialization from the
    # baseline. Otherwise the first zero_() can leave persistent driver state
    # that looks like a leaked pool segment.
    warmup = domains.empty(0, (1,), torch.uint8)
    warmup.zero_()
    torch.cuda.synchronize(domains.device)
    del warmup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(domains.device)
    free_before, _ = torch.cuda.mem_get_info(domains.device)
    tensor = domains.empty(0, (256 * 1024**2,), torch.uint8)
    tensor.zero_()
    torch.cuda.synchronize(domains.device)
    del tensor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(domains.device)
    free_after, _ = torch.cuda.mem_get_info(domains.device)
    # Context bookkeeping may retain a small amount, but the 256 MiB segment
    # itself must be returned to the driver.
    assert free_after >= free_before - 16 * 1024**2


def test_localized_streams_cuda_graph_replay():
    from flashinfer.prims_ts.moe.partition import MoePartitionResources

    domains = _require_locality()
    resources = MoePartitionResources(domains.streams)
    assert resources.fork_join is not None
    sources = [domains.empty(domain, (256,), torch.float32) for domain in range(2)]
    for source in sources:
        source.copy_(torch.arange(256, device=domains.device, dtype=torch.float32))
    outputs = [torch.empty_like(source) for source in sources]

    def enqueue():
        caller = torch.cuda.current_stream(domains.device)
        resources.fork_join.fork(caller)
        for domain, stream in enumerate(domains.streams):
            with torch.cuda.stream(stream):
                torch.mul(sources[domain], domain + 2, out=outputs[domain])
        resources.fork_join.join(caller)

    enqueue()
    torch.cuda.current_stream(domains.device).synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        enqueue()
    for source in sources:
        source.add_(1)
    graph.replay()
    torch.cuda.current_stream(domains.device).synchronize()
    for domain in range(2):
        torch.testing.assert_close(outputs[domain], sources[domain] * (domain + 2))
