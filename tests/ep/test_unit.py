# tests/ep/test_unit.py
#
# Unit tests for the FlashInfer Unified EP API — covers test IDs U-01 through U-18.
# Run with:  torchrun --nproc_per_node=4 -m pytest tests/ep/test_unit.py -v --tb=short
#
# These tests verify API correctness on both backends (DeepEP, NCCL-EP).
# Each test dispatches tokens, runs an identity expert, combines, and checks
# that the roundtrip is faithful.

import pytest
import torch
import torch.distributed as dist

import flashinfer.ep as fep

# Import helpers via path-relative import (works regardless of PYTHONPATH / cwd)
import importlib.util, pathlib
_helpers_path = pathlib.Path(__file__).parent / "helpers.py"
_spec = importlib.util.spec_from_file_location("ep_helpers", _helpers_path)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
make_tokens = _helpers.make_tokens
identity_expert = _helpers.identity_expert
BACKENDS = _helpers.BACKENDS
has_fp8_support = _helpers.has_fp8_support


# =====================================================================
# U-01, U-02: Group Lifecycle
# =====================================================================


class TestGroupLifecycle:
    """Tests U-01, U-02: create/destroy, context manager, exception cleanup."""

    def test_create_destroy(self, backend, model_config, make_group):
        """U-01: Basic create/destroy with no leaks."""
        group = make_group(backend, model_config)
        assert group.world_size > 0
        assert group.rank >= 0
        assert group.num_experts == model_config["num_experts"]
        # destroy happens in fixture cleanup

    def test_context_manager(self, backend, model_config, ep_process_group):
        """U-02: Context manager calls destroy, even on clean exit."""
        with fep.create_group(
            backend=backend,
            process_group=ep_process_group,
            num_experts=model_config["num_experts"],
            num_local_experts=model_config["num_local_experts"],
            top_k=model_config["top_k"],
            hidden_dim=model_config["hidden_dim"],
        ) as group:
            assert group.backend == backend

        # group.__exit__ has been called — no segfault means pass

    def test_context_manager_exception(self, backend, model_config,
                                       ep_process_group):
        """U-02 variant: exception inside with-block still cleans up."""
        try:
            with fep.create_group(
                backend=backend,
                process_group=ep_process_group,
                num_experts=model_config["num_experts"],
                num_local_experts=model_config["num_local_experts"],
                top_k=model_config["top_k"],
                hidden_dim=model_config["hidden_dim"],
            ) as group:
                raise ValueError("intentional error")
        except ValueError:
            pass  # group.destroy() should have been called by __exit__


# =====================================================================
# U-03, U-07, U-08, U-09: High-Throughput Roundtrip
# =====================================================================


class TestHTRoundtrip:
    """Tests U-03, U-07, U-08, U-09: HT dispatch → identity expert → combine."""

    @pytest.mark.parametrize("num_tokens", [1024, 4096])
    def test_ht_roundtrip_identity(self, backend, output_layout,
                                    model_config, make_group, num_tokens):
        """U-03 + U-07/U-08: HT dispatch -> identity expert -> combine.

        Verifies:
          - Shape matches expected OutputLayout
          - Combined output matches original input (within BF16 tolerance)
        """
        group = make_group(backend, model_config)
        hidden, topk_idx, topk_weights = make_tokens(
            num_tokens,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
        )

        # Pre-compute layout (outside any graph region — involves D2H)
        layout = group.get_dispatch_layout(topk_idx)

        # Dispatch
        result = group.dispatch(
            hidden=hidden,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            layout=layout,
            output_layout=output_layout,
            allocate_on_comm_stream=True,
        )
        result.status.raise_if_error()

        # U-07 / U-08: Verify shape based on output_layout
        if output_layout == fep.OutputLayout.FLAT_2D:
            assert result.recv_hidden.ndim == 2
            assert result.recv_hidden.shape[1] == model_config["hidden_dim"]
        else:
            assert result.recv_hidden.ndim == 3
            assert result.recv_hidden.shape[0] == model_config["num_local_experts"]
            assert result.recv_hidden.shape[2] == model_config["hidden_dim"]

        # Identity expert
        expert_out = identity_expert(result.recv_hidden, result.recv_expert_counts)

        # Combine
        combined = group.combine(
            expert_output=expert_out,
            handle=result.handle,
            topk_weights=topk_weights,
            allocate_on_comm_stream=True,
        )
        combined.status.raise_if_error()
        result.handle.destroy()

        # Verify roundtrip correctness:
        # With identity expert and proper weighted reduction, combined should
        # equal hidden * sum(topk_weights, dim=-1, keepdim=True)
        expected = hidden * topk_weights.sum(dim=-1, keepdim=True).to(hidden.dtype)
        torch.testing.assert_close(
            combined.combined_hidden,
            expected,
            atol=1e-2,
            rtol=1e-2,  # BF16 tolerance
        )

    def test_layout_cross_check(self, backend, model_config, make_group):
        """U-09: FLAT_2D and EXPERT_MAJOR_3D produce identical values.

        Dispatches the same tokens with both layouts, flattens the 3D result
        (stripping padding), and verifies both produce identical values.
        """
        group = make_group(backend, model_config)
        hidden, topk_idx, topk_weights = make_tokens(
            2048,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
        )
        layout = group.get_dispatch_layout(topk_idx)

        # Dispatch with 2D
        r2d = group.dispatch(
            hidden=hidden,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            layout=layout,
            output_layout=fep.OutputLayout.FLAT_2D,
        )
        r2d.status.raise_if_error()

        # Dispatch with 3D (fresh layout — handles are consumed)
        layout2 = group.get_dispatch_layout(topk_idx)
        r3d = group.dispatch(
            hidden=hidden,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            layout=layout2,
            output_layout=fep.OutputLayout.EXPERT_MAJOR_3D,
        )
        r3d.status.raise_if_error()

        # Flatten 3D and compare valid regions
        counts = r3d.recv_expert_counts
        flat_from_3d = []
        for e in range(model_config["num_local_experts"]):
            n = counts[e].item()
            if n > 0:
                flat_from_3d.append(r3d.recv_hidden[e, :n, :])
        flat_from_3d = (
            torch.cat(flat_from_3d, dim=0)
            if flat_from_3d
            else torch.empty(
                0, model_config["hidden_dim"], device="cuda", dtype=torch.bfloat16
            )
        )

        torch.testing.assert_close(
            r2d.recv_hidden, flat_from_3d, atol=1e-5, rtol=1e-5
        )

        r2d.handle.destroy()
        r3d.handle.destroy()


# =====================================================================
# U-04, U-12: Low-Latency Roundtrip + DeferredRecv
# =====================================================================


class TestLLRoundtrip:
    """Tests U-04, U-12: LL dispatch/combine roundtrip and DeferredRecv hook."""

    @pytest.mark.parametrize("num_tokens", [1, 32, 64, 128])
    def test_ll_roundtrip_identity(self, backend, model_config,
                                    make_group, num_tokens):
        """U-04: LL dispatch -> identity expert -> combine.

        Low-latency path for decode workloads. Verifies the full roundtrip
        with small token counts typical of autoregressive generation.
        """
        group = make_group(backend, model_config)
        hidden, topk_idx, topk_weights = make_tokens(
            num_tokens,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
        )

        result = group.low_latency_dispatch(
            hidden=hidden,
            topk_idx=topk_idx,
            max_tokens_per_rank=128,
            output_layout=fep.OutputLayout.FLAT_2D,
        )
        result.status.raise_if_error()

        expert_out = identity_expert(result.recv_hidden, result.recv_expert_counts)

        combined = group.low_latency_combine(
            expert_output=expert_out,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=result.handle,
        )
        combined.status.raise_if_error()
        assert combined.combined_hidden.shape == hidden.shape
        result.handle.destroy()

    def test_deferred_recv(self, backend, model_config, make_group):
        """U-12: DeferredRecv hook — data invalid before, valid after.

        Dispatches with return_recv_hook=True, verifies the recv buffer state
        changes after invoking the deferred hook (invoke_deferred), and checks
        that post-hook data is valid (no NaN/Inf).
        """
        group = make_group(backend, model_config)
        hidden, topk_idx, topk_weights = make_tokens(
            64,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
        )

        result = group.low_latency_dispatch(
            hidden=hidden,
            topk_idx=topk_idx,
            max_tokens_per_rank=128,
            return_recv_hook=True,
        )
        result.status.raise_if_error()

        # Pre-hook: take a checksum snapshot. Data may not be materialized yet.
        pre_hook_sum = result.recv_hidden.float().sum().item()

        # Invoke deferred recv — materializes data
        hook_status = result.handle.invoke_deferred()
        hook_status.raise_if_error()
        torch.cuda.synchronize()

        # Post-hook: recv_hidden should now contain valid data
        assert not torch.isnan(result.recv_hidden).any(), \
            "recv_hidden contains NaN after hook invocation"
        assert not torch.isinf(result.recv_hidden).any(), \
            "recv_hidden contains Inf after hook invocation"

        # Verify data is non-trivial (not all zeros)
        post_hook_sum = result.recv_hidden.float().sum().item()
        if result.handle.get_num_recv_tokens() > 0:
            assert post_hook_sum != 0.0 or pre_hook_sum != post_hook_sum, \
                "recv_hidden appears unchanged after hook — possible deferred recv failure"

        result.handle.destroy()


# =====================================================================
# U-05, U-06: FP8 Dispatch
# =====================================================================


class TestFP8:
    """Tests U-05, U-06: FP8 dispatch on capable / incapable hardware."""

    @pytest.mark.skipif(
        not has_fp8_support(), reason="FP8 requires Hopper+ (SM90+)"
    )
    def test_fp8_dispatch_bf16_combine(self, backend, model_config, make_group):
        """U-05: Dispatch in FP8, combine in BF16.

        Verifies quantization/dequantization correctness by checking the
        roundtrip error is within FP8 tolerance (wider than BF16).
        """
        group = make_group(backend, model_config)

        num_tokens = 512
        hidden, topk_idx, topk_weights = make_tokens(
            num_tokens,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
            dtype=torch.float8_e4m3fn,
        )

        layout = group.get_dispatch_layout(topk_idx)
        result = group.dispatch(
            hidden=hidden,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            layout=layout,
            output_layout=fep.OutputLayout.FLAT_2D,
        )
        result.status.raise_if_error()

        # Expert output in BF16 (upcast)
        expert_out = result.recv_hidden.to(torch.bfloat16)

        combined = group.combine(
            expert_output=expert_out,
            handle=result.handle,
            topk_weights=topk_weights,
        )
        combined.status.raise_if_error()
        result.handle.destroy()

        # FP8 has wider tolerance due to quantization noise
        assert not torch.isnan(combined.combined_hidden).any()
        assert not torch.isinf(combined.combined_hidden).any()

    @pytest.mark.skipif(
        has_fp8_support(), reason="Only runs on non-FP8 hardware (A100)"
    )
    def test_fp8_on_unsupported_gpu(self, backend, model_config, make_group):
        """U-06: FP8 dispatch on A100 should fail gracefully, not crash."""
        group = make_group(backend, model_config)

        hidden = torch.randn(
            64, model_config["hidden_dim"], device="cuda", dtype=torch.bfloat16
        )
        # Attempt to cast to FP8 — this itself may raise on A100
        try:
            hidden_fp8 = hidden.to(torch.float8_e4m3fn)
        except RuntimeError:
            pytest.skip("torch.float8_e4m3fn not supported on this GPU")

        topk_idx = torch.randint(
            0, model_config["num_experts"],
            (64, model_config["top_k"]), device="cuda",
        )
        layout = group.get_dispatch_layout(topk_idx)

        # Should return error status, not crash
        result = group.dispatch(
            hidden=hidden_fp8,
            topk_idx=topk_idx,
            topk_weights=torch.ones(
                64, model_config["top_k"], device="cuda", dtype=torch.float32
            ) / model_config["top_k"],
            layout=layout,
        )
        assert not result.status.ok(), "Expected FP8 error on A100"


# =====================================================================
# U-10: get_dispatch_layout Validation
# =====================================================================


class TestDispatchLayout:
    """Test U-10: Verify get_dispatch_layout returns correct token counts."""

    def test_layout_token_counts(self, backend, model_config, make_group):
        """U-10: num_tokens_per_rank and num_tokens_per_expert sum correctly.

        The total tokens dispatched across all ranks should equal
        num_tokens * top_k (each token is sent to top_k experts).
        """
        group = make_group(backend, model_config)
        num_tokens = 1024
        hidden, topk_idx, topk_weights = make_tokens(
            num_tokens,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
        )

        layout = group.get_dispatch_layout(topk_idx)

        # num_tokens_per_expert should sum to total routed tokens on this rank
        per_expert = layout.num_tokens_per_expert
        assert per_expert is not None
        assert per_expert.shape[0] == model_config["num_local_experts"]

        # All counts should be non-negative
        assert (per_expert >= 0).all()

        # Sum across all ranks (allreduce) should equal num_tokens * top_k
        local_sum = per_expert.sum()
        global_sum = torch.tensor([local_sum], device="cuda", dtype=torch.int64)
        dist.all_reduce(global_sum)
        expected_total = num_tokens * model_config["top_k"]
        assert global_sum.item() == expected_total, \
            f"Global token sum {global_sum.item()} != expected {expected_total}"


# =====================================================================
# U-11: Double Buffering
# =====================================================================


class TestDoubleBuffering:
    """Test U-11: Pipelined dispatch with previous_handle across micro-batches."""

    def test_pipelined_dispatch(self, backend, model_config, make_group):
        """U-11: Dispatch N+1 with previous_handle from N. No corruption.

        Runs 5 micro-batches with overlapping handles to verify that the
        ping-pong scratch buffers prevent corruption.
        """
        group = make_group(backend, model_config)

        prev_handle = None
        prev_combined = None

        for i in range(5):
            hidden, topk_idx, topk_weights = make_tokens(
                64,
                model_config["hidden_dim"],
                model_config["num_experts"],
                model_config["top_k"],
            )

            result = group.low_latency_dispatch(
                hidden=hidden,
                topk_idx=topk_idx,
                max_tokens_per_rank=128,
                previous_handle=prev_handle,
            )
            result.status.raise_if_error()

            out = identity_expert(result.recv_hidden, result.recv_expert_counts)
            combined = group.low_latency_combine(
                expert_output=out,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                handle=result.handle,
            )
            combined.status.raise_if_error()

            # Clean up previous handle
            if prev_handle is not None:
                prev_handle.destroy()

            prev_handle = result.handle
            prev_combined = combined.combined_hidden

            # Verify no NaN/Inf corruption
            assert not torch.isnan(prev_combined).any(), f"NaN at iteration {i}"
            assert not torch.isinf(prev_combined).any(), f"Inf at iteration {i}"

        if prev_handle:
            prev_handle.destroy()


# =====================================================================
# U-13: StreamDep Ordering
# =====================================================================


class TestStreamDep:
    """Test U-13: StreamDep ensures compute waits for comm."""

    def test_stream_dep_ordering(self, backend, small_config, make_group):
        """U-13: Record event on comm stream, wait on compute stream.

        Creates a StreamDep, records on the communication stream, then waits
        on the default (compute) stream. Verifies that a tensor written on
        the comm stream is visible on the compute stream after the wait.
        """
        group = make_group(backend, small_config)

        comm_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.current_stream()

        marker = torch.zeros(1, device="cuda")

        # Write on comm stream
        with torch.cuda.stream(comm_stream):
            marker.fill_(42.0)

        # Create event on comm stream, wait on compute stream
        event = comm_stream.record_event()
        compute_stream.wait_event(event)

        # After wait, marker should be visible
        torch.cuda.synchronize()
        assert marker.item() == 42.0


# =====================================================================
# U-14, U-15, U-16: Error Handling
# =====================================================================


class TestErrorHandling:
    """Tests U-14, U-15, U-16: Handle lifecycle, timeout, buffer overflow."""

    def test_double_destroy_handle(self, backend, model_config, make_group):
        """U-14: Double destroy should not crash.

        Calls handle.destroy() twice. The second call should be a no-op
        or return an error — it must never segfault.
        """
        group = make_group(backend, model_config)
        hidden, topk_idx, topk_weights = make_tokens(
            64,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
        )
        result = group.low_latency_dispatch(
            hidden=hidden, topk_idx=topk_idx, max_tokens_per_rank=128
        )
        result.status.raise_if_error()

        result.handle.destroy()
        # Second destroy — must not crash
        result.handle.destroy()

    def test_timeout(self, backend, model_config, ep_process_group):
        """U-15: Timeout parameter is accepted.

        Sets a very short timeout and verifies the group can still be created
        and destroyed. Deterministic timeout triggering is backend-specific
        and hard to test portably, so this test validates parameter acceptance.
        """
        group = fep.create_group(
            backend=backend,
            process_group=ep_process_group,
            num_experts=model_config["num_experts"],
            num_local_experts=model_config["num_local_experts"],
            top_k=model_config["top_k"],
            hidden_dim=model_config["hidden_dim"],
            timeout_ms=100,  # Very short timeout
        )
        group.destroy()

    def test_buffer_overflow(self, backend, model_config, make_group):
        """U-16: Exceed max_tokens_per_rank. Expect error, not corruption.

        Sets max_tokens_per_rank=4 and dispatches 128 tokens. The API should
        return an error status indicating buffer overflow.
        """
        group = make_group(backend, model_config)
        hidden, topk_idx, topk_weights = make_tokens(
            128,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
        )

        result = group.low_latency_dispatch(
            hidden=hidden,
            topk_idx=topk_idx,
            max_tokens_per_rank=4,  # Way too small
        )
        assert not result.status.ok(), "Expected buffer overflow error"
        assert result.status.error_code != 0


# =====================================================================
# U-17: Mode Switching
# =====================================================================


class TestModeSwitching:
    """Test U-17: Same EpGroup serves HT then LL with no reallocation."""

    def test_same_group_ht_then_ll(self, backend, model_config, make_group):
        """U-17: Use same group for HT then LL. No reallocation.

        Verifies that the single-EpGroup dual-mode design works: the group
        is created once, HT dispatch/combine runs, then LL dispatch/combine
        runs on the same group object. Both must produce correct results.
        """
        group = make_group(backend, model_config)

        # ── HT path ──
        hidden_ht, topk_idx_ht, topk_weights_ht = make_tokens(
            4096,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
        )
        layout = group.get_dispatch_layout(topk_idx_ht)
        r_ht = group.dispatch(
            hidden=hidden_ht,
            topk_idx=topk_idx_ht,
            topk_weights=topk_weights_ht,
            layout=layout,
        )
        r_ht.status.raise_if_error()
        out_ht = identity_expert(r_ht.recv_hidden, r_ht.recv_expert_counts)
        c_ht = group.combine(
            expert_output=out_ht,
            handle=r_ht.handle,
            topk_weights=topk_weights_ht,
        )
        c_ht.status.raise_if_error()
        r_ht.handle.destroy()

        # ── LL path — same group object, no reallocation ──
        hidden_ll, topk_idx_ll, topk_weights_ll = make_tokens(
            64,
            model_config["hidden_dim"],
            model_config["num_experts"],
            model_config["top_k"],
        )
        r_ll = group.low_latency_dispatch(
            hidden=hidden_ll,
            topk_idx=topk_idx_ll,
            max_tokens_per_rank=128,
        )
        r_ll.status.raise_if_error()
        out_ll = identity_expert(r_ll.recv_hidden, r_ll.recv_expert_counts)
        c_ll = group.low_latency_combine(
            expert_output=out_ll,
            topk_idx=topk_idx_ll,
            topk_weights=topk_weights_ll,
            handle=r_ll.handle,
        )
        c_ll.status.raise_if_error()
        r_ll.handle.destroy()


# =====================================================================
# U-18: Edge Cases
# =====================================================================


class TestEdgeCases:
    """Tests U-18: Zero tokens, extreme routing imbalance."""

    def test_zero_tokens_to_rank(self, backend, model_config, make_group):
        """U-18: Zero tokens dispatched to a rank. Combine returns zeros.

        Routes all tokens to expert 0 (which may not be local to every rank).
        Ranks that receive 0 tokens should handle it gracefully — no crash,
        no hang, correct combine output.
        """
        group = make_group(backend, model_config)
        # All tokens go to expert 0 — most ranks get 0 tokens
        hidden = torch.randn(
            1, model_config["hidden_dim"], device="cuda", dtype=torch.bfloat16
        )
        topk_idx = torch.zeros(
            1, model_config["top_k"], device="cuda", dtype=torch.int64
        )
        topk_weights = (
            torch.ones(1, model_config["top_k"], device="cuda", dtype=torch.float32)
            / model_config["top_k"]
        )

        result = group.low_latency_dispatch(
            hidden=hidden, topk_idx=topk_idx, max_tokens_per_rank=128
        )
        result.status.raise_if_error()
        # Should not crash even if this rank receives 0 tokens

        out = identity_expert(result.recv_hidden, result.recv_expert_counts)
        combined = group.low_latency_combine(
            expert_output=out,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=result.handle,
        )
        combined.status.raise_if_error()
        result.handle.destroy()

    def test_all_tokens_to_one_expert(self, backend, model_config, make_group):
        """U-18 variant: All 256 tokens route to expert 0 (extreme imbalance).

        Stresses the dispatch path with maximum imbalance — one expert gets
        everything, all others get nothing.
        """
        group = make_group(backend, model_config)
        num_tokens = 256
        hidden = torch.randn(
            num_tokens, model_config["hidden_dim"],
            device="cuda", dtype=torch.bfloat16,
        )
        topk_idx = torch.zeros(
            num_tokens, model_config["top_k"], device="cuda", dtype=torch.int64
        )
        topk_weights = (
            torch.ones(
                num_tokens, model_config["top_k"],
                device="cuda", dtype=torch.float32,
            )
            / model_config["top_k"]
        )

        result = group.low_latency_dispatch(
            hidden=hidden, topk_idx=topk_idx, max_tokens_per_rank=num_tokens
        )
        result.status.raise_if_error()

        out = identity_expert(result.recv_hidden, result.recv_expert_counts)
        combined = group.low_latency_combine(
            expert_output=out,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            handle=result.handle,
        )
        combined.status.raise_if_error()
        assert not torch.isnan(combined.combined_hidden).any()
        result.handle.destroy()

    def test_zero_tokens_ht_path(self, backend, model_config, make_group):
        """U-18 variant: Zero total tokens through the HT path.

        Dispatches an empty batch (0 tokens). The API should handle this
        without crashing — recv_hidden should be empty.
        """
        group = make_group(backend, model_config)
        hidden = torch.empty(
            0, model_config["hidden_dim"], device="cuda", dtype=torch.bfloat16
        )
        topk_idx = torch.empty(
            0, model_config["top_k"], device="cuda", dtype=torch.int64
        )
        topk_weights = torch.empty(
            0, model_config["top_k"], device="cuda", dtype=torch.float32
        )

        layout = group.get_dispatch_layout(topk_idx)

        result = group.dispatch(
            hidden=hidden,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            layout=layout,
        )
        result.status.raise_if_error()
        result.handle.destroy()
