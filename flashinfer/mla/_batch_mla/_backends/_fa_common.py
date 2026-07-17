"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Stable mechanics shared only by the generated FA2 and FA3 MLA backends.

Backend vertical slices are otherwise intentionally compartmentalized.  This
module is the narrow shared-infrastructure exception for mechanics whose launch
contracts are identical: workspace ownership, metadata staging, generated
module plan/run plumbing, and common tensor validation.  Backend selection,
module loading, supported configurations, and FP8/scale policy stay in the
concrete FA2 and FA3 modules; this is not a generic backend abstraction.
"""

import functools
from typing import Any, Callable, Optional, Tuple, Union

import torch

from flashinfer.jit import gen_batch_mla_module

from ....utils import MaskMode, check_shape_dtype_device


@functools.cache
def get_batch_mla_module(backend, *args):
    return gen_batch_mla_module(backend, *args).build_and_load()


class _BatchMLAGeneratedFaWorkspace:
    """Lazily stage generated-FA plans without mutating the live workspace."""

    _WORKSPACE_SIZE_BYTES = 8 * 1024 * 1024

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._live_int_workspace_buffer: Optional[torch.Tensor] = None
        self._staging_int_workspace_buffer: Optional[torch.Tensor] = None
        self._pin_memory_int_workspace_buffer: Optional[torch.Tensor] = None
        self._terminal_invalidation_reason: Optional[str] = None

    @property
    def live_buffer(self) -> Optional[torch.Tensor]:
        return self._live_int_workspace_buffer

    def invalidate_after_partial_metadata_commit(
        self, failed_stage: str, error: BaseException
    ) -> None:
        """Poison the owning wrapper after a potentially partial graph commit.

        Once reserved CUDA-graph metadata or the live scheduler workspace may
        have changed, the previous captured plan is not recoverable if a copy
        fails. Keeping this terminal state on the wrapper-owned workspace
        prevents later use of a potentially mixed plan generation.
        """
        self._terminal_invalidation_reason = (
            "BatchMLAPagedAttentionWrapper was terminally invalidated because "
            "a generated-FA CUDA-graph commit may have been partially applied; "
            f"failed stage: {failed_stage}: "
            f"{type(error).__name__}: {error}"
        )

    def raise_if_invalid(self) -> None:
        if self._terminal_invalidation_reason is not None:
            raise RuntimeError(self._terminal_invalidation_reason)

    def get_buffers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._staging_int_workspace_buffer is None:
            self._staging_int_workspace_buffer = torch.empty(
                (self._WORKSPACE_SIZE_BYTES,),
                dtype=torch.uint8,
                device=self.device,
            )
        if self._pin_memory_int_workspace_buffer is None:
            self._pin_memory_int_workspace_buffer = torch.empty(
                self._staging_int_workspace_buffer.shape,
                dtype=self._staging_int_workspace_buffer.dtype,
                pin_memory=True,
                device="cpu",
            )
        return (
            self._staging_int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
        )

    def commit_buffers(
        self,
        planned_buffer: torch.Tensor,
        *,
        use_cuda_graph: bool,
    ) -> torch.Tensor:
        """Commit a successfully planned staging buffer and return its run buffer."""
        if planned_buffer is not self._staging_int_workspace_buffer:
            raise ValueError("generated FA plan did not use the current staging buffer.")

        if self._live_int_workspace_buffer is None:
            self._live_int_workspace_buffer = planned_buffer
            self._staging_int_workspace_buffer = None
        elif use_cuda_graph:
            self._live_int_workspace_buffer.copy_(planned_buffer, non_blocking=True)
        else:
            self._live_int_workspace_buffer, self._staging_int_workspace_buffer = (
                planned_buffer,
                self._live_int_workspace_buffer,
            )
        return self._live_int_workspace_buffer


class _BatchMLAGeneratedFaMechanics:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        generated_fa_workspace: _BatchMLAGeneratedFaWorkspace,
        use_cuda_graph: bool,
        qo_indptr: Optional[torch.Tensor],
        kv_indptr: Optional[torch.Tensor],
        kv_indices: Optional[torch.Tensor],
        kv_len_arr: Optional[torch.Tensor],
        before_metadata_commit: Optional[Callable[[], None]],
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        if generated_fa_workspace.device != self.device:
            raise ValueError(
                "generated FA workspace must be on the wrapper device "
                f"{self.device}, got {generated_fa_workspace.device}."
            )
        self._generated_fa_workspace = generated_fa_workspace
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr
        self._before_metadata_commit = before_metadata_commit

    def _validate_metadata_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor.")
        if tensor.ndim != 1:
            raise ValueError(
                f"{name} must be a rank-1 tensor, got shape {tensor.shape}."
            )
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype int32, got {tensor.dtype}.")
        if tensor.device != self.device:
            raise ValueError(
                f"{name} must be on wrapper device {self.device}, got {tensor.device}."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")

    def _validate_plan_metadata(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
    ) -> None:
        metadata = (
            ("qo_indptr", qo_indptr),
            ("kv_indptr", kv_indptr),
            ("kv_indices", kv_indices),
            ("kv_len_arr", kv_len_arr),
        )
        for name, tensor in metadata:
            self._validate_metadata_tensor(name, tensor)

        if qo_indptr.shape[0] == 0:
            raise ValueError("qo_indptr must contain at least one element.")
        batch_size = qo_indptr.shape[0] - 1
        if kv_indptr.shape[0] != batch_size + 1:
            raise ValueError(
                "qo_indptr and kv_indptr must describe the same batch: "
                f"got lengths {qo_indptr.shape[0]} and {kv_indptr.shape[0]}."
            )
        if kv_len_arr.shape[0] != batch_size:
            raise ValueError(
                "kv_len_arr must have the plan batch length "
                f"{batch_size}, got {kv_len_arr.shape[0]}."
            )

        if not self._use_cuda_graph:
            return

        reserved = (
            ("qo_indptr", self._qo_indptr_buf),
            ("kv_indptr", self._kv_indptr_buf),
            ("kv_indices", self._kv_indices_buf),
            ("kv_len_arr", self._kv_len_arr_buf),
        )
        missing = [name for name, tensor in reserved if tensor is None]
        if missing:
            raise ValueError(
                "CUDA graph mode requires all four reserved buffers; missing "
                + ", ".join(missing)
                + "."
            )
        for name, tensor in reserved:
            self._validate_metadata_tensor(f"reserved {name} buffer", tensor)

        expected_indptr_shape = (batch_size + 1,)
        expected_len_shape = (batch_size,)
        if self._qo_indptr_buf.shape != expected_indptr_shape:
            raise ValueError(
                "reserved qo_indptr buffer must have exact shape "
                f"{expected_indptr_shape}, got {tuple(self._qo_indptr_buf.shape)}."
            )
        if self._kv_indptr_buf.shape != expected_indptr_shape:
            raise ValueError(
                "reserved kv_indptr buffer must have exact shape "
                f"{expected_indptr_shape}, got {tuple(self._kv_indptr_buf.shape)}."
            )
        if self._kv_len_arr_buf.shape != expected_len_shape:
            raise ValueError(
                "reserved kv_len_arr buffer must have exact shape "
                f"{expected_len_shape}, got {tuple(self._kv_len_arr_buf.shape)}."
            )
        if self._kv_indices_buf.shape[0] < kv_indices.shape[0]:
            raise ValueError(
                "reserved kv_indices buffer capacity must be at least "
                f"{kv_indices.shape[0]}, got {self._kv_indices_buf.shape[0]}."
            )

    def _metadata_copy_pairs(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
    ) -> Tuple[Tuple[str, torch.Tensor, torch.Tensor], ...]:
        return (
            ("qo_indptr", self._qo_indptr_buf, qo_indptr),
            ("kv_indptr", self._kv_indptr_buf, kv_indptr),
            (
                "kv_indices",
                self._kv_indices_buf[: kv_indices.shape[0]],
                kv_indices,
            ),
            ("kv_len_arr", self._kv_len_arr_buf, kv_len_arr),
        )

    @staticmethod
    def _metadata_byte_range(tensor: torch.Tensor) -> Tuple[int, int]:
        start = tensor.data_ptr()
        return start, start + tensor.numel() * tensor.element_size()

    @classmethod
    def _metadata_copy_is_exact_alias(
        cls, destination: torch.Tensor, source: torch.Tensor
    ) -> bool:
        return cls._metadata_byte_range(destination) == cls._metadata_byte_range(
            source
        )

    @classmethod
    def _preflight_cuda_graph_metadata_copies(
        cls,
        pairs: Tuple[Tuple[str, torch.Tensor, torch.Tensor], ...],
    ) -> None:
        regions = []
        for pair_index, (name, destination, source) in enumerate(pairs):
            regions.extend(
                (
                    (
                        pair_index,
                        name,
                        "destination",
                        destination,
                        cls._metadata_byte_range(destination),
                    ),
                    (
                        pair_index,
                        name,
                        "source",
                        source,
                        cls._metadata_byte_range(source),
                    ),
                )
            )

        for left_index, left in enumerate(regions):
            left_pair, left_name, left_role, left_tensor, left_range = left
            for right in regions[left_index + 1 :]:
                right_pair, right_name, right_role, right_tensor, right_range = right
                overlaps = max(left_range[0], right_range[0]) < min(
                    left_range[1], right_range[1]
                )
                if not overlaps or (left_role == right_role == "source"):
                    continue
                if (
                    left_pair == right_pair
                    and left_role != right_role
                    and cls._metadata_copy_is_exact_alias(left_tensor, right_tensor)
                ):
                    continue
                raise ValueError(
                    "CUDA graph metadata copy storage overlap is unsafe: "
                    f"{left_name} {left_role} overlaps "
                    f"{right_name} {right_role}. Exact source/destination aliases "
                    "for the same metadata field are the only allowed overlap."
                )

    def _commit_cuda_graph_metadata(
        self,
        pairs: Tuple[Tuple[str, torch.Tensor, torch.Tensor], ...],
    ) -> None:
        for name, destination, source in pairs:
            if self._metadata_copy_is_exact_alias(destination, source):
                continue
            try:
                destination.copy_(source, non_blocking=True)
            except Exception as error:
                self._generated_fa_workspace.invalidate_after_partial_metadata_commit(
                    name, error
                )
                raise

    def _commit_generated_fa_plan(
        self,
        planning_int_workspace_buffer: torch.Tensor,
        metadata_pairs: Tuple[Tuple[str, torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        if self._use_cuda_graph:
            self._commit_cuda_graph_metadata(metadata_pairs)
            # This is deliberately the final commit operation before publishing
            # backend state. A failed device copy may have partially updated the
            # stable live scheduler buffer, so the wrapper cannot remain usable.
            try:
                return self._generated_fa_workspace.commit_buffers(
                    planning_int_workspace_buffer,
                    use_cuda_graph=True,
                )
            except Exception as error:
                self._generated_fa_workspace.invalidate_after_partial_metadata_commit(
                    "generated_fa_int_workspace", error
                )
                raise

        # Non-graph replans publish by swapping buffers and retain their prior
        # behavior rather than applying the terminal CUDA-graph failure policy.
        return self._generated_fa_workspace.commit_buffers(
            planning_int_workspace_buffer,
            use_cuda_graph=False,
        )

    def _plan_generated_fa(
        self,
        *,
        module_loader: Callable[[], Any],
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool,
    ) -> None:
        self._validate_plan_metadata(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
        )

        metadata_pairs = ()
        if self._use_cuda_graph:
            metadata_pairs = self._metadata_copy_pairs(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_len_arr=kv_len_arr,
            )
            self._preflight_cuda_graph_metadata_copies(metadata_pairs)

        cached_module = module_loader()
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")
        kv_len_arr_host = kv_len_arr.to("cpu")

        if not self._use_cuda_graph:
            qo_indptr_buf = qo_indptr.to(self.device, non_blocking=True)
            kv_indptr_buf = kv_indptr.to(self.device, non_blocking=True)
            kv_indices_buf = kv_indices.to(self.device, non_blocking=True)
            kv_len_arr_buf = kv_len_arr.to(self.device, non_blocking=True)

        planning_int_workspace_buffer, pin_memory_int_workspace_buffer = (
            self._generated_fa_workspace.get_buffers()
        )
        plan_info = cached_module.plan(
            self._float_workspace_buffer,
            planning_int_workspace_buffer,
            pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            num_heads,
            head_dim_ckv,  # head_dim_o
            causal,
        )

        if self._before_metadata_commit is not None:
            self._before_metadata_commit()

        int_workspace_buffer = self._commit_generated_fa_plan(
            planning_int_workspace_buffer,
            metadata_pairs,
        )
        if not self._use_cuda_graph:
            self._qo_indptr_buf = qo_indptr_buf
            self._kv_indptr_buf = kv_indptr_buf
            self._kv_indices_buf = kv_indices_buf
            self._kv_len_arr_buf = kv_len_arr_buf
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = pin_memory_int_workspace_buffer
        self._cached_module = cached_module
        self._plan_info = plan_info
        self._causal = causal
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._use_profiler = use_profiler

    def _validate_run_input_dtypes(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
    ) -> None:
        """Validate generated-FA input dtypes without allocating or synchronizing."""
        if q_nope.dtype != self._q_data_type:
            raise ValueError(
                f"q_nope.dtype={q_nope.dtype} does not match the planned "
                f"q_data_type={self._q_data_type}."
            )
        if q_pe.dtype != self._q_data_type:
            raise ValueError(
                f"q_pe.dtype={q_pe.dtype} does not match the planned "
                f"q_data_type={self._q_data_type}."
            )
        if ckv_cache.dtype != self._kv_data_type:
            raise ValueError(
                f"ckv_cache.dtype={ckv_cache.dtype} does not match the planned "
                f"kv_data_type={self._kv_data_type}."
            )
        if kpe_cache.dtype != self._kv_data_type:
            raise ValueError(
                f"kpe_cache.dtype={kpe_cache.dtype} does not match the planned "
                f"kv_data_type={self._kv_data_type}."
            )

    def _run_generated_fa_after_input_validation(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        ckv_scale: float,
        kpe_scale: float,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if profiler_buffer is None and self._use_profiler:
            raise ValueError("Profiler is enabled, profiler_buffer must be provided")

        if out is None:
            out = torch.empty_like(q_nope)
        else:
            check_shape_dtype_device(
                out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
            )

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    q_nope.shape[:2], dtype=torch.float32, device=self.device
                )
            else:
                check_shape_dtype_device(
                    lse, q_nope.shape[:2], torch.float32, q_nope.device, "lse"
                )

        mask_mode = MaskMode.CAUSAL.value if self._causal else MaskMode.NON_CAUSAL.value
        profiler_args = (profiler_buffer,) if self._use_profiler else ()
        self._cached_module.run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            self._kv_indices_buf,
            out,
            lse,
            mask_mode,
            q_nope.shape[1],
            self._page_size,
            self._sm_scale,
            return_lse_base_on_e,
            ckv_scale,
            kpe_scale,
            *profiler_args,
        )
        return (out, lse) if return_lse else out
