"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Regression coverage for opt-in call-boundary CUDA graph dump snapshots."""

import importlib
import inspect
import json
import sys
from unittest.mock import Mock

import pytest
import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.utils import register_custom_op


@flashinfer_api
@register_custom_op(
    "flashinfer::test_logging_graph_aliases",
    mutates_args=("x", "duplicate", "even", "odd"),
)
def _graph_snapshot_alias_op(
    x: torch.Tensor,
    duplicate: torch.Tensor,
    even: torch.Tensor,
    odd: torch.Tensor,
) -> None:
    """Exercise aliases inside the callable resolved by actual dump replay."""
    assert x is duplicate
    assert x.untyped_storage().data_ptr() == even.untyped_storage().data_ptr()
    assert x.untyped_storage().data_ptr() == odd.untyped_storage().data_ptr()
    assert even.stride() == (x.stride(0), 2)
    assert odd.storage_offset() - x.storage_offset() == 1
    x.add_(1)
    even.mul_(2)
    odd.add_(3)


@flashinfer_api
@register_custom_op(
    "flashinfer::test_logging_graph_readonly",
    mutates_args=("x",),
)
def _graph_snapshot_readonly_op(x: torch.Tensor, w: torch.Tensor) -> None:
    """Declared mutation plus a read-only stride-0 argument."""
    x.add_(w[:, 0])


def _snapshot_review_identity(x, repeated, distinct):
    assert x is repeated
    assert x is not distinct
    assert x.untyped_storage().data_ptr() == distinct.untyped_storage().data_ptr()
    return {"value": [x + 1, (x + 2,)]}


def _snapshot_review_result(x):
    return x + 1


def _snapshot_review_none(x):
    return None


@register_custom_op(
    "flashinfer::test_logging_mutate_none",
    mutates_args=("x",),
)
def _snapshot_review_mutate_none(x: torch.Tensor) -> None:
    """Exercise a completed None-return call with recorded post-call state."""
    x.add_(1)


@pytest.fixture
def snapshot_logging(tmp_path, monkeypatch, request):
    """Reload only logging after its import-time snapshot opt-in is configured."""
    monkeypatch.setenv("FLASHINFER_LOGLEVEL", "10")
    monkeypatch.setenv("FLASHINFER_LOGDEST", str(tmp_path / "logging.txt"))
    monkeypatch.setenv("FLASHINFER_DUMP_DIR", str(tmp_path / "dumps"))
    enabled = getattr(request, "param", True)
    monkeypatch.setenv("FLASHINFER_DUMP_GRAPH_SNAPSHOTS", "1" if enabled else "0")
    monkeypatch.setenv("FLASHINFER_DUMP_SAFETENSORS", "0")
    monkeypatch.setenv("FLASHINFER_DUMP_MAX_COUNT", "1000")
    monkeypatch.setenv("FLASHINFER_DUMP_MAX_SIZE_GB", "1")
    monkeypatch.setenv(
        "FLASHINFER_DUMP_INCLUDE",
        "_graph_snapshot_alias_op,_graph_snapshot_readonly_op,"
        "_snapshot_review_*,apply_rope_inplace,"
        "fused_add_rmsnorm,BatchDecodeWithPagedKVCacheWrapper.run",
    )
    monkeypatch.delenv("FLASHINFER_DUMP_EXCLUDE", raising=False)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_graph_snapshot_alias_op",
        inspect.unwrap(_graph_snapshot_alias_op),
    )
    previous = sys.modules.pop("flashinfer.api_logging", None)
    if previous is not None:
        previous._restore_cuda_graph_hooks()
    logging = importlib.import_module("flashinfer.api_logging")
    try:
        yield logging, tmp_path / "dumps"
    finally:
        logging.clear_graph_dumps()
        logging._restore_cuda_graph_hooks()
        sys.modules.pop("flashinfer.api_logging", None)
        if previous is not None:
            sys.modules["flashinfer.api_logging"] = previous
            sys.modules["flashinfer"].api_logging = previous
            previous._install_cuda_graph_hooks()


@pytest.fixture
def cuda_device():
    """Probe CUDA at test execution rather than during module import."""
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    return torch.device("cuda", torch.cuda.current_device())


def _metadata(path):
    return json.loads((path / "metadata.jsonl").read_text().splitlines()[-1])


def _calls(root, name):
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "metadata.jsonl").exists()
        and _metadata(path)["function_name"] == name
    )


def _load(path, kind):
    return torch.load(path / f"{kind}.pt", weights_only=False)


def _assert_none_replay(logging, path, device):
    replay = logging.replay_from_dump(
        str(path), device=str(device), run=True, compare_outputs=True
    )
    assert "execution_error" not in replay, replay.get("execution_error")
    assert "execution_result" in replay, "The dump callable was not resolved/executed"
    assert replay["execution_result"] is None
    assert replay["comparison_match"] is True
    return replay


def _warmup(device, call):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)


@pytest.mark.cpu_only
def test_graph_snapshot_aliases_cpu(snapshot_logging, monkeypatch):
    """CPU mocked-capture unit test; this does not execute a CUDA graph."""
    logging, root = snapshot_logging
    logged_op = logging.flashinfer_api(inspect.unwrap(_graph_snapshot_alias_op))
    backing = torch.arange(50, dtype=torch.float32).reshape(5, 10)
    x = backing[1:4, 1:9]
    even, odd = x[:, ::2], x[:, 1::2]
    before = x.clone()
    with monkeypatch.context() as capture:
        capture.setattr(logging, "_is_current_stream_capturing", lambda: True)
        assert logged_op(x, x, even, odd) is None
    after = x.clone()

    # A later graph node can overwrite every original reference. Saved outputs
    # must retain the op's post-call values, rather than these end-of-graph values.
    backing.fill_(-1000)
    logging.flush_graph_dumps(synchronize=False)
    call = _calls(root, "_graph_snapshot_alias_op")[0]
    snapshot = call / "graph_flushes" / "flush_0001"
    inputs, outputs = _load(snapshot, "inputs"), _load(snapshot, "outputs")
    metadata = _metadata(snapshot)
    torch.testing.assert_close(inputs["arg_0"], before, rtol=0, atol=0)
    torch.testing.assert_close(inputs["arg_1"], before, rtol=0, atol=0)
    torch.testing.assert_close(inputs["arg_2"], before[:, ::2], rtol=0, atol=0)
    torch.testing.assert_close(inputs["arg_3"], before[:, 1::2], rtol=0, atol=0)
    torch.testing.assert_close(outputs["mutated_input__arg_0"], after, rtol=0, atol=0)
    assert set(metadata["mutated_outputs"].values()) == {
        "arg_0",
        "arg_1",
        "arg_2",
        "arg_3",
    }
    base_key = metadata["input_tensor_aliases"]["arg_0"]["base"]
    assert inputs[base_key].numel() < backing.numel()

    replay = _assert_none_replay(logging, snapshot, "cpu")
    # The toy also asserts these identities inside actual replay execution.
    assert replay["args"][0] is replay["args"][1]
    assert replay["args"][2].stride() == x.stride()[:1] + (2,)


@pytest.mark.cpu_only
def test_eager_snapshot_aliases_safetensors_cpu(snapshot_logging, monkeypatch):
    safetensors = pytest.importorskip("safetensors.torch")
    logging, root = snapshot_logging
    monkeypatch.setattr(logging, "_DUMP_SAFETENSORS", True)
    logged_op = logging.flashinfer_api(inspect.unwrap(_graph_snapshot_alias_op))
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    before = x.clone()
    assert logged_op(x, x, x[:, ::2], x[:, 1::2]) is None
    after = x.clone()
    call = _calls(root, "_graph_snapshot_alias_op")[0]
    inputs = safetensors.load_file(str(call / "inputs.safetensors"))
    outputs = safetensors.load_file(str(call / "outputs.safetensors"))
    assert _metadata(call)["execution_status"] == "completed"
    torch.testing.assert_close(inputs["arg_0"], before, rtol=0, atol=0)
    torch.testing.assert_close(outputs["mutated_input__arg_0"], after, rtol=0, atol=0)
    _assert_none_replay(logging, call, "cpu")
    outputs["mutated_input__arg_0"].add_(10)
    safetensors.save_file(outputs, str(call / "outputs.safetensors"))
    replay = logging.replay_from_dump(
        str(call), device="cpu", run=True, compare_outputs=True
    )
    assert "execution_error" not in replay, replay.get("execution_error")
    assert "execution_result" in replay and replay["execution_result"] is None
    assert replay["comparison_match"] is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("snapshot_logging", [False], indirect=True)
def test_graph_snapshot_default_off_cpu(snapshot_logging, monkeypatch):
    """Opt-out keeps legacy references and never allocates snapshot buffers."""
    logging, root = snapshot_logging
    stage = Mock(side_effect=AssertionError("Opt-out must not stage snapshot copies"))
    monkeypatch.setattr(logging, "_stage_graph_tensor_plan", stage)
    logged_op = logging.flashinfer_api(_graph_snapshot_alias_op)
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    with monkeypatch.context() as capture:
        capture.setattr(logging, "_is_current_stream_capturing", lambda: True)
        logged_op(x, x, x[:, ::2], x[:, 1::2])
    x.fill_(-777)
    logging.flush_graph_dumps(synchronize=False)
    stage.assert_not_called()
    call = _calls(root, "_graph_snapshot_alias_op")[0]
    metadata = _metadata(call)
    for key in ("mutated_input_keys", "mutated_outputs", "input_tensor_aliases"):
        assert key not in metadata
    inputs = _load(call, "inputs")
    assert torch.equal(inputs["arg_0"], torch.full_like(x, -777))
    assert not any(key.startswith("_snapshot_storage_") for key in inputs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("admission", ["excluded", "count", "size"])
def test_graph_snapshot_admission_cpu(snapshot_logging, monkeypatch, admission):
    """Rejected dumps cannot allocate the graph snapshot copy buffers."""
    logging, root = snapshot_logging
    if admission == "excluded":
        monkeypatch.setattr(
            logging, "_DUMP_EXCLUDE_PATTERNS", ["_graph_snapshot_alias_op"]
        )
    elif admission == "count":
        monkeypatch.setattr(logging, "_DUMP_MAX_COUNT", 0)
    else:
        monkeypatch.setattr(logging, "_DUMP_MAX_SIZE_GB", 0)
    stage = Mock(
        side_effect=AssertionError("Unadmitted dump allocated snapshot buffers")
    )
    monkeypatch.setattr(logging, "_stage_graph_tensor_plan", stage)
    logged_op = logging.flashinfer_api(_graph_snapshot_alias_op)
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    with monkeypatch.context() as capture:
        capture.setattr(logging, "_is_current_stream_capturing", lambda: True)
        assert logged_op(x, x, x[:, ::2], x[:, 1::2]) is None
    stage.assert_not_called()
    assert not list(root.glob("*/metadata.jsonl"))


@pytest.mark.cpu_only
@pytest.mark.parametrize("captured", [False, True])
def test_snapshot_identity_and_nested_outputs(snapshot_logging, monkeypatch, captured):
    logging, root = snapshot_logging
    x = torch.arange(8, dtype=torch.float32)
    with monkeypatch.context() as capture:
        capture.setattr(logging, "_is_current_stream_capturing", lambda: captured)
        logging.flashinfer_api(_snapshot_review_identity)(x, x, x.view_as(x))
    if captured:
        logging.flush_graph_dumps(synchronize=False)
    call = _calls(root, "_snapshot_review_identity")[0]
    replay = logging.replay_from_dump(
        str(call), device="cpu", run=True, compare_outputs=True
    )
    assert "execution_error" not in replay
    assert replay["args"][0] is replay["args"][1]
    assert replay["args"][0] is not replay["args"][2]
    assert replay["comparison_match"] is True
    outputs = _load(call, "outputs")
    outputs["result__value_0__1__0"].add_(10)
    torch.save(outputs, call / "outputs.pt")
    replay = logging.replay_from_dump(
        str(call), device="cpu", run=True, compare_outputs=True
    )
    assert replay["comparison_match"] is False


@pytest.mark.cpu_only
def test_snapshot_rejects_uncertain_overlap(snapshot_logging):
    logging, _ = snapshot_logging
    backing = torch.arange(16, dtype=torch.float32)
    overlap = backing.as_strided((2, 2), (2, 2))
    assert torch._debug_has_internal_overlap(overlap) == 2
    with pytest.raises(ValueError, match="non-overlapping"):
        logging._graph_tensor_plan({"x": overlap}, ["x"])
    # A strided, non-dense view also reports status 2 but is provably disjoint.
    sliced = backing.reshape(2, 8)[:, ::2]
    assert logging._graph_tensor_plan({"x": sliced}, ["x"])


@pytest.mark.cpu_only
@pytest.mark.parametrize("tensor_format", ["torch", "safetensors"])
@pytest.mark.parametrize("start", [0, 3])
def test_readonly_alias_gaps_are_zero(
    snapshot_logging, monkeypatch, tmp_path, tensor_format, start
):
    logging, _ = snapshot_logging
    if tensor_format == "safetensors":
        safetensors = pytest.importorskip("safetensors.torch")
    backing = torch.full((12,), 9876.0)
    tensors = {"a": backing[start : start + 2], "b": backing[-2:]}
    plan = logging._graph_tensor_plan(tensors, [])
    with monkeypatch.context() as capture:
        capture.setattr(
            torch,
            "zeros",
            Mock(side_effect=AssertionError("Read-only capture allocated a copy")),
        )
        staged, aliases = logging._stage_graph_tensor_plan(tensors, plan)
    # The readonly values must still be deferred until materialization.
    backing[start : start + 2] = 1
    backing[-2:] = 2
    logging._save_dump_tensors(staged, tensor_format, tmp_path, "inputs", aliases)
    saved = (
        _load(tmp_path, "inputs")
        if tensor_format == "torch"
        else safetensors.load_file(str(tmp_path / "inputs.safetensors"))
    )
    base = saved[aliases["a"]["base"]]
    torch.testing.assert_close(
        base, torch.tensor([1.0, 1.0] + [0.0] * (8 - start) + [2.0, 2.0])
    )
    logging._restore_tensor_aliases(saved, aliases)
    assert saved["a"].storage_offset() == 0
    assert saved["b"].storage_offset() == 10 - start
    torch.testing.assert_close(saved["a"], torch.ones(2))
    torch.testing.assert_close(saved["b"], torch.full((2,), 2.0))
    assert (
        saved["a"].untyped_storage().data_ptr()
        == saved["b"].untyped_storage().data_ptr()
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_snapshot_output_completion(snapshot_logging, monkeypatch, captured, complete):
    logging, _ = snapshot_logging
    func = _snapshot_review_mutate_none if complete else _snapshot_review_result
    x = torch.arange(8, dtype=torch.float32)
    with monkeypatch.context() as capture:
        capture.setattr(logging, "_is_current_stream_capturing", lambda: captured)
        call = logging._dump_function_inputs(func, func.__name__, (x,), {})
        if not complete:
            capture.setattr(
                logging, "_DUMP_MAX_SIZE_GB", logging._dump_total_size_bytes / 1024**3
            )
        logging._dump_function_outputs(call, func(x), args=(x,))
    if captured:
        logging.flush_graph_dumps(synchronize=False)
    replay = logging.replay_from_dump(
        call, device="cpu", run=True, compare_outputs=True
    )
    assert "execution_error" not in replay
    if complete:
        assert replay["execution_result"] is None
        assert set(replay["expected_tensors"]) == {"mutated_input__arg_0"}
        torch.testing.assert_close(replay["expected_tensors"]["mutated_input__arg_0"], x)
        assert replay["comparison_match"] is True
    else:
        assert replay["comparison_match"] is None
        assert replay["comparison_error"] == "Output dump is incomplete"


@pytest.mark.cpu_only
@pytest.mark.parametrize("captured", [False, True])
def test_graph_snapshot_readonly_stride0_arg_cpu(snapshot_logging, monkeypatch, captured):
    """A read-only singleton must not veto the whole call's dump."""
    logging, root = snapshot_logging
    logged_op = logging.flashinfer_api(inspect.unwrap(_graph_snapshot_readonly_op))
    x = torch.arange(4, dtype=torch.float32)
    w = torch.ones(1, 64, dtype=torch.float32).expand(4, 64)
    assert w.stride() == (0, 1)
    with monkeypatch.context() as capture:
        capture.setattr(logging, "_is_current_stream_capturing", lambda: captured)
        assert logged_op(x, w) is None
    if captured:
        logging.flush_graph_dumps(synchronize=False)
    call = _calls(root, "_graph_snapshot_readonly_op")[0]
    snapshot = call / "graph_flushes" / "flush_0001" if captured else call
    assert snapshot.exists()
    inputs = _load(snapshot, "inputs")
    assert "arg_1" in inputs
    # Read-only singletons follow the reference path. Materialization may
    # normalize storage; this input must not need an alias reconstruction group.
    assert "arg_1" not in _metadata(snapshot).get("input_tensor_aliases", {})
    torch.testing.assert_close(inputs["arg_1"], w)
    torch.testing.assert_close(inputs["arg_0"], torch.arange(4, dtype=torch.float32))
    _assert_none_replay(logging, snapshot, "cpu")


@pytest.mark.cpu_only
@pytest.mark.parametrize("snapshot_logging", [False, True], indirect=True)
def test_none_return_without_recorded_state_has_distinct_message(snapshot_logging):
    """No post-call state is a distinct status, not an incomplete dump."""
    logging, _ = snapshot_logging
    func = _snapshot_review_none
    x = torch.arange(8, dtype=torch.float32)
    call = logging._dump_function_inputs(func, func.__name__, (x,), {})
    logging._dump_function_outputs(call, func(x), args=(x,))
    replay = logging.replay_from_dump(
        call, device="cpu", run=True, compare_outputs=True
    )
    assert replay["comparison_match"] is None
    assert "No post-call state recorded" in replay["comparison_error"]
    assert (
        "enable FLASHINFER_DUMP_GRAPH_SNAPSHOTS" in replay["comparison_error"]
    ) is not logging._DUMP_GRAPH_SNAPSHOTS


@pytest.mark.cpu_only
def test_public_rope_mutation_declarations(snapshot_logging, monkeypatch):
    import flashinfer.rope as rope

    logging, _ = snapshot_logging
    for name in (
        "apply_rope_inplace",
        "apply_rope_pos_ids_inplace",
        "apply_llama31_rope_inplace",
        "apply_llama31_rope_pos_ids_inplace",
    ):
        assert inspect.unwrap(getattr(rope, name))._flashinfer_mutates_args == (
            "q",
            "k",
        )
    assert inspect.unwrap(
        rope.apply_rope_with_cos_sin_cache_inplace
    )._flashinfer_mutates_args == ("query", "key")
    # Declarations are attached at decoration time, including loglevel zero.
    monkeypatch.setattr(logging, "_API_LOG_LEVEL", 0)
    original = lambda x: None
    assert logging.flashinfer_api(original, mutates_args=("x",)) is original
    assert original._flashinfer_mutates_args == ("x",)


def test_public_rope_inplace_snapshots(cuda_device, snapshot_logging):
    from flashinfer.rope import apply_rope_inplace

    logging, root = snapshot_logging
    raw_rope = inspect.unwrap(apply_rope_inplace)
    logged_rope = logging.flashinfer_api(raw_rope)
    q = torch.randn(4, 2, 64, device=cuda_device, dtype=torch.float16)
    k = torch.randn(4, 1, 64, device=cuda_device, dtype=torch.float16)
    indptr = torch.tensor([0, 4], device=cuda_device, dtype=torch.int32)
    offsets = torch.tensor([3], device=cuda_device, dtype=torch.int32)
    _warmup(cuda_device, lambda: raw_rope(q, k, indptr, offsets))
    q0, k0 = q.clone(), k.clone()
    logged_rope(q, k, indptr, offsets)
    eager = _calls(root, "apply_rope_inplace")[0]
    torch.testing.assert_close(_load(eager, "inputs")["arg_0"], q0.cpu())
    _assert_none_replay(logging, eager, cuda_device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logged_rope(q, k, indptr, offsets)
    capture = _calls(root, "apply_rope_inplace")[-1]
    for i, scale in enumerate((1.0, 2.0), start=1):
        q.copy_(q0 * scale)
        k.copy_(k0 / scale)
        graph.replay()
        snapshot = capture / "graph_flushes" / f"flush_{i:04d}"
        inputs, outputs = _load(snapshot, "inputs"), _load(snapshot, "outputs")
        torch.testing.assert_close(inputs["arg_0"], (q0 * scale).cpu())
        torch.testing.assert_close(inputs["arg_1"], (k0 / scale).cpu())
        torch.testing.assert_close(outputs["mutated_input__arg_0"], q.cpu())
        torch.testing.assert_close(outputs["mutated_input__arg_1"], k.cpu())
        _assert_none_replay(logging, snapshot, cuda_device)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_graph_snapshot_fused_add_rmsnorm(cuda_device, snapshot_logging, dtype):
    """Real in-place norm: pre-inputs, mutated outputs and non-vacuous replay."""
    from flashinfer.norm import fused_add_rmsnorm

    logging, root = snapshot_logging
    raw_norm = inspect.unwrap(fused_add_rmsnorm)
    logged_norm = logging.flashinfer_api(raw_norm)
    generator = torch.Generator(device=cuda_device).manual_seed(7934)
    initial_x = torch.randn(
        2, 512, device=cuda_device, dtype=dtype, generator=generator
    )
    initial_r = torch.randn(
        2, 512, device=cuda_device, dtype=dtype, generator=generator
    )
    weight = torch.randn(512, device=cuda_device, dtype=dtype, generator=generator)
    x, residual = initial_x.clone(), initial_r.clone()
    _warmup(cuda_device, lambda: raw_norm(x, residual, weight, enable_pdl=False))

    eager_x, eager_r = initial_x.clone(), initial_r.clone()
    assert logged_norm(eager_x, eager_r, weight, enable_pdl=False) is None
    eager = _calls(root, "fused_add_rmsnorm")[0]
    eager_inputs, eager_outputs = _load(eager, "inputs"), _load(eager, "outputs")
    torch.testing.assert_close(eager_inputs["arg_0"], initial_x.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(eager_inputs["arg_1"], initial_r.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        eager_outputs["mutated_input__arg_0"], eager_x.cpu(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        eager_outputs["mutated_input__arg_1"], eager_r.cpu(), rtol=0, atol=0
    )
    _assert_none_replay(logging, eager, cuda_device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logged_norm(x, residual, weight, enable_pdl=False)
    capture = _calls(root, "fused_add_rmsnorm")[-1]
    first_saved_input = first_saved_output = None
    for index, scale in enumerate((1.0, 2.0), start=1):
        pre_x, pre_r = initial_x * scale, initial_r / scale
        x.copy_(pre_x)
        residual.copy_(pre_r)
        graph.replay()
        snapshot = capture / "graph_flushes" / f"flush_{index:04d}"
        inputs, outputs = _load(snapshot, "inputs"), _load(snapshot, "outputs")
        torch.testing.assert_close(inputs["arg_0"], pre_x.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(inputs["arg_1"], pre_r.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(
            outputs["mutated_input__arg_0"], x.cpu(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["mutated_input__arg_1"], residual.cpu(), rtol=0, atol=0
        )
        assert set(_metadata(snapshot)["mutated_outputs"].values()) == {
            "arg_0",
            "arg_1",
        }
        _assert_none_replay(logging, snapshot, cuda_device)
        if index == 1:
            first_saved_input = inputs["arg_0"].clone()
            first_saved_output = outputs["mutated_input__arg_0"].clone()
            torch.testing.assert_close(
                first_saved_output,
                eager_outputs["mutated_input__arg_0"],
                rtol=0,
                atol=0,
            )

    first = capture / "graph_flushes" / "flush_0001"
    torch.testing.assert_close(
        _load(first, "inputs")["arg_0"], first_saved_input, rtol=0, atol=0
    )
    torch.testing.assert_close(
        _load(first, "outputs")["mutated_input__arg_0"],
        first_saved_output,
        rtol=0,
        atol=0,
    )
    corrupted = _load(first, "outputs")
    corrupted["mutated_input__arg_0"].add_(10)
    torch.save(corrupted, first / "outputs.pt")
    replay = logging.replay_from_dump(
        str(first), device=str(cuda_device), run=True, compare_outputs=True
    )
    assert "execution_error" not in replay, replay.get("execution_error")
    assert "execution_result" in replay and replay["execution_result"] is None
    assert replay["comparison_match"] is False, "None return must not compare vacuously"


def test_graph_snapshot_paged_decode_pipeline(cuda_device, snapshot_logging):
    """Tiny real FA2 norm→decode dumps agree with eager and graph controls."""
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.norm import fused_add_rmsnorm

    logging, root = snapshot_logging
    raw_norm = inspect.unwrap(fused_add_rmsnorm)
    raw_run = inspect.unwrap(BatchDecodeWithPagedKVCacheWrapper.run)
    logged_norm = logging.flashinfer_api(raw_norm)
    logged_run = logging.flashinfer_api(raw_run)
    dtype = torch.float16
    generator = torch.Generator(device=cuda_device).manual_seed(7935)
    initial_x = torch.randn(
        2, 512, device=cuda_device, dtype=dtype, generator=generator
    )
    initial_r = torch.randn(
        2, 512, device=cuda_device, dtype=dtype, generator=generator
    )
    weight = torch.randn(512, device=cuda_device, dtype=dtype, generator=generator)
    kv = torch.randn(
        2, 2, 16, 2, 128, device=cuda_device, dtype=dtype, generator=generator
    )
    indptr = torch.tensor([0, 1, 2], device=cuda_device, dtype=torch.int32)
    indices = torch.tensor([0, 1], device=cuda_device, dtype=torch.int32)
    last_page_len = torch.tensor([7, 13], device=cuda_device, dtype=torch.int32)

    def planned_wrapper():
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(8 * 1024 * 1024, device=cuda_device, dtype=torch.uint8),
            "NHD",
            use_cuda_graph=True,
            backend="fa2",
            paged_kv_indptr_buffer=indptr.clone(),
            paged_kv_indices_buffer=indices.clone(),
            paged_kv_last_page_len_buffer=last_page_len.clone(),
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            4,
            2,
            128,
            16,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        return wrapper

    wrapper, control_wrapper = planned_wrapper(), planned_wrapper()
    x, residual = initial_x.clone(), initial_r.clone()
    control_x, control_r = initial_x.clone(), initial_r.clone()
    out = torch.empty(2, 4, 128, device=cuda_device, dtype=dtype)
    control_out = torch.empty_like(out)

    def control_call():
        raw_norm(control_x, control_r, weight, enable_pdl=False)
        return raw_run(
            control_wrapper,
            control_x.view(2, 4, 128),
            kv,
            out=control_out,
            enable_pdl=False,
        )

    _warmup(cuda_device, control_call)
    _warmup(
        cuda_device,
        lambda: raw_run(wrapper, x.view(2, 4, 128), kv, out=out, enable_pdl=False),
    )
    control_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(control_graph):
        control_call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logged_norm(x, residual, weight, enable_pdl=False)
        logged_run(wrapper, x.view(2, 4, 128), kv, out=out, enable_pdl=False)

    decode_call = _calls(root, "BatchDecodeWithPagedKVCacheWrapper.run")[-1]
    for index, scale in enumerate((1.0, 2.0), start=1):
        eager_x = initial_x * scale
        eager_r = initial_r * (3.0 - scale)
        raw_norm(eager_x, eager_r, weight, enable_pdl=False)
        eager_out = raw_run(
            control_wrapper,
            eager_x.view(2, 4, 128),
            kv,
            enable_pdl=False,
        )
        for target in (x, control_x):
            target.copy_(initial_x * scale)
        for target in (residual, control_r):
            target.copy_(initial_r * (3.0 - scale))
        control_graph.replay()
        graph.replay()
        torch.testing.assert_close(out, eager_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out, control_out, rtol=1e-3, atol=1e-3)
        snapshot = decode_call / "graph_flushes" / f"flush_{index:04d}"
        metadata = _metadata(snapshot)
        replay = logging.replay_from_dump(
            str(snapshot),
            device=str(cuda_device),
            run=True,
            compare_outputs=True,
            object_registry={(metadata["process_id"], metadata["self_id"]): wrapper},
        )
        assert "execution_error" not in replay, replay.get("execution_error")
        assert "execution_result" in replay, "Public decode method did not execute"
        assert replay["comparison_match"] is True
        torch.testing.assert_close(
            replay["execution_result"], eager_out, rtol=1e-3, atol=1e-3
        )
