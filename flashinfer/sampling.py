"""
Copyright (c) 2024 by FlashInfer team.

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

import functools
from types import SimpleNamespace
from typing import Optional, Tuple, Union
import torch

from .api_logging import flashinfer_api
from .jit.sampling import gen_sampling_module
from .jit.sampling_hy3 import gen_fused_sampling_hy3_module
from .trace.templates.sampling import (
    chain_speculative_sampling_trace,
    fused_sampling_hy3_trace_dispatch,
    min_p_sampling_trace,
    sampling_from_logits_trace,
    sampling_from_probs_trace,
    softmax_trace,
    top_k_mask_logits_trace,
    top_k_renorm_probs_trace,
    top_k_sampling_trace,
    top_k_top_p_sampling_from_logits_trace,
    top_k_top_p_sampling_trace,
    top_p_renorm_probs_trace,
    top_p_sampling_trace,
)
from .utils import (
    _get_cache_buf,
    device_support_pdl,
    get_default_generators,
    register_custom_op,
    register_fake_op,
)


def get_seed_and_offset(
    increment: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int]:
    if generator is None:
        generator = get_default_generators(device)
    # add mutex if multi-trheading needed
    state = generator.get_state()
    seed, offset = state.view(torch.int64)
    offset += (increment + 3) // 4 * 4
    generator.set_state(
        torch.tensor(
            [seed, offset], dtype=torch.int64, device=torch.device("cpu")
        ).view(torch.uint8)
    )
    return int(seed), int(offset)


@functools.cache
def get_sampling_module():
    module = gen_sampling_module().build_and_load()

    @register_custom_op("flashinfer::softmax", mutates_args=("workspace_buffer",))
    def softmax(
        workspace_buffer: torch.Tensor,
        logits: torch.Tensor,
        maybe_temperature_arr: Optional[torch.Tensor],
        temperature_val: float,
        enable_pdl: bool,
    ) -> torch.Tensor:
        logits = logits.float()
        probs = torch.empty_like(logits, device=logits.device)
        maybe_temperature_arr = (
            maybe_temperature_arr.float() if maybe_temperature_arr is not None else None
        )
        module.softmax(
            workspace_buffer,
            logits,
            probs,
            maybe_temperature_arr,
            temperature_val,
            enable_pdl,
        )
        return probs

    @register_fake_op("flashinfer::softmax")
    def _fake_softmax(
        workspace_buffer: torch.Tensor,
        logits: torch.Tensor,
        maybe_temperature_arr: Optional[torch.Tensor],
        temperature_val: float,
        enable_pdl: bool,
    ) -> torch.Tensor:
        return torch.empty_like(logits, device=logits.device, dtype=torch.float32)

    # torch library for sampling_from_logits
    @register_custom_op("flashinfer::sampling_from_logits", mutates_args=())
    def sampling_from_logits(
        logits: torch.Tensor,
        indices: Optional[torch.Tensor],
        deterministic: bool,
        generator: Optional[torch.Generator],
        seed: Optional[Union[int, torch.Tensor]] = None,
        offset: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        device = logits.device
        # TODO: support more data types in logits to avoid conversion
        # to float32
        logits = logits.float()
        batch_size = indices.size(0) if indices is not None else logits.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        samples = torch.empty(batch_size, dtype=out_dtype, device=device)
        if seed is None or offset is None:
            seed, offset = get_seed_and_offset(
                batch_size * logits.size(1), generator, device
            )

        maybe_seed_arr, seed_val, maybe_offset_arr, offset_val = (
            _validate_and_convert_seed_offset(seed, offset, device, batch_size)
        )

        module.sampling_from_logits(
            logits,
            samples,
            indices,
            deterministic,
            maybe_seed_arr,
            seed_val,
            maybe_offset_arr,
            offset_val,
        )
        return samples

    @register_fake_op("flashinfer::sampling_from_logits")
    def _fake_sampling_from_logits(
        logits: torch.Tensor,
        indices: Optional[torch.Tensor],
        deterministic: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        batch_size = indices.size(0) if indices is not None else logits.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        return torch.empty(batch_size, dtype=out_dtype, device=logits.device)

    # torch library for sampling_from_probs

    @register_custom_op("flashinfer::sampling_from_probs", mutates_args=())
    def sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        deterministic: bool,
        generator: Optional[torch.Generator],
        seed: Optional[Union[int, torch.Tensor]] = None,
        offset: Optional[Union[int, torch.Tensor]] = None,
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = probs.device
        probs = probs.float()
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        samples = torch.empty(batch_size, dtype=out_dtype, device=device)
        valid = torch.empty(batch_size, dtype=torch.bool, device=device)
        if seed is None or offset is None:
            seed, offset = get_seed_and_offset(batch_size, generator, device)

        maybe_seed_arr, seed_val, maybe_offset_arr, offset_val = (
            _validate_and_convert_seed_offset(seed, offset, device, batch_size)
        )

        module.sampling_from_probs(
            probs,
            samples,
            valid,
            indices,
            deterministic,
            maybe_seed_arr,
            seed_val,
            maybe_offset_arr,
            offset_val,
        )
        if return_valid:
            return samples, valid
        return samples

    # torch library for sampling_from_probs

    @register_fake_op("flashinfer::sampling_from_probs")
    def _fake_sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        deterministic: bool,
        generator: Optional[torch.Generator],
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        if return_valid:
            return (
                torch.empty(batch_size, dtype=out_dtype, device=probs.device),
                torch.empty(batch_size, dtype=torch.bool, device=probs.device),
            )
        return torch.empty(batch_size, dtype=out_dtype, device=probs.device)

    # torch library for top_p_sampling_from_probs

    @register_custom_op("flashinfer::top_p_sampling_from_probs", mutates_args=())
    def top_p_sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        maybe_top_p_arr: Optional[torch.Tensor],
        top_p_val: float,
        deterministic: bool,
        generator: Optional[torch.Generator],
        seed: Optional[Union[int, torch.Tensor]] = None,
        offset: Optional[Union[int, torch.Tensor]] = None,
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = probs.device
        probs = probs.float()
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        samples = torch.empty(batch_size, dtype=out_dtype, device=device)
        valid = torch.empty(batch_size, dtype=torch.bool, device=device)
        if seed is None or offset is None:
            seed, offset = get_seed_and_offset(batch_size * 32, generator, device)

        maybe_seed_arr, seed_val, maybe_offset_arr, offset_val = (
            _validate_and_convert_seed_offset(seed, offset, device, batch_size)
        )

        module.top_p_sampling_from_probs(
            probs,
            samples,
            valid,
            indices,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
            maybe_seed_arr,
            seed_val,
            maybe_offset_arr,
            offset_val,
        )
        if return_valid:
            return samples, valid
        return samples

    @register_fake_op("flashinfer::top_p_sampling_from_probs")
    def _fake_top_p_sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        maybe_top_p_arr: Optional[torch.Tensor],
        top_p_val: float,
        deterministic: bool,
        generator: Optional[torch.Generator],
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        if return_valid:
            return (
                torch.empty(batch_size, dtype=out_dtype, device=probs.device),
                torch.empty(batch_size, dtype=torch.bool, device=probs.device),
            )
        return torch.empty(batch_size, dtype=out_dtype, device=probs.device)

    # torch library for top_k_sampling_from_probs

    @register_custom_op("flashinfer::top_k_sampling_from_probs", mutates_args=())
    def top_k_sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        maybe_top_k_arr: Optional[torch.Tensor],
        top_k_val: int,
        deterministic: bool,
        generator: Optional[torch.Generator],
        seed: Optional[Union[int, torch.Tensor]] = None,
        offset: Optional[Union[int, torch.Tensor]] = None,
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = probs.device
        probs = probs.float()
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
        out_dtype = indices.dtype if indices is not None else torch.int32
        samples = torch.empty(batch_size, dtype=out_dtype, device=device)
        valid = torch.empty(batch_size, dtype=torch.bool, device=device)
        if seed is None or offset is None:
            seed, offset = get_seed_and_offset(batch_size * 32, generator, device)

        maybe_seed_arr, seed_val, maybe_offset_arr, offset_val = (
            _validate_and_convert_seed_offset(seed, offset, device, batch_size)
        )

        module.top_k_sampling_from_probs(
            probs,
            samples,
            valid,
            indices,
            maybe_top_k_arr,
            top_k_val,
            deterministic,
            maybe_seed_arr,
            seed_val,
            maybe_offset_arr,
            offset_val,
        )
        if return_valid:
            return samples, valid
        return samples

    @register_fake_op("flashinfer::top_k_sampling_from_probs")
    def _fake_top_k_sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        maybe_top_k_arr: Optional[torch.Tensor],
        top_k_val: int,
        deterministic: bool,
        generator: Optional[torch.Generator],
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        if return_valid:
            return (
                torch.empty(batch_size, dtype=out_dtype, device=probs.device),
                torch.empty(batch_size, dtype=torch.bool, device=probs.device),
            )
        return torch.empty(batch_size, dtype=out_dtype, device=probs.device)

    # torch library for min_p_sampling_from_probs

    @register_custom_op("flashinfer::min_p_sampling_from_probs", mutates_args=())
    def min_p_sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        maybe_min_p_arr: Optional[torch.Tensor],
        min_p_val: float,
        deterministic: bool,
        generator: Optional[torch.Generator],
        seed: Optional[Union[int, torch.Tensor]] = None,
        offset: Optional[Union[int, torch.Tensor]] = None,
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = probs.device
        probs = probs.float()
        maybe_min_p_arr = (
            maybe_min_p_arr.float() if maybe_min_p_arr is not None else None
        )
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        samples = torch.empty(batch_size, dtype=out_dtype, device=device)
        valid = torch.empty(batch_size, dtype=torch.bool, device=device)
        if seed is None or offset is None:
            seed, offset = get_seed_and_offset(batch_size, generator, device)

        maybe_seed_arr, seed_val, maybe_offset_arr, offset_val = (
            _validate_and_convert_seed_offset(seed, offset, device, batch_size)
        )

        module.min_p_sampling_from_probs(
            probs,
            samples,
            valid,
            indices,
            maybe_min_p_arr,
            min_p_val,
            deterministic,
            maybe_seed_arr,
            seed_val,
            maybe_offset_arr,
            offset_val,
        )
        if return_valid:
            return samples, valid
        return samples

    @register_fake_op("flashinfer::min_p_sampling_from_probs")
    def _fake_min_p_sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        maybe_min_p_arr: Optional[torch.Tensor],
        min_p_val: float,
        deterministic: bool,
        generator: Optional[torch.Generator],
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        if return_valid:
            return (
                torch.empty(batch_size, dtype=out_dtype, device=probs.device),
                torch.empty(batch_size, dtype=torch.bool, device=probs.device),
            )
        return torch.empty(batch_size, dtype=out_dtype, device=probs.device)

    # torch library for top_k_top_p_sampling_from_probs
    @register_custom_op("flashinfer::top_k_top_p_sampling_from_probs", mutates_args=())
    def top_k_top_p_sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        maybe_top_k_arr: Optional[torch.Tensor],
        top_k_val: int,
        maybe_top_p_arr: Optional[torch.Tensor],
        top_p_val: float,
        deterministic: bool,
        generator: Optional[torch.Generator],
        seed: Optional[Union[int, torch.Tensor]] = None,
        offset: Optional[Union[int, torch.Tensor]] = None,
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = probs.device
        probs = probs.float()
        maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        samples = torch.empty(batch_size, dtype=out_dtype, device=device)
        valid = torch.empty(batch_size, dtype=torch.bool, device=device)
        if seed is None or offset is None:
            seed, offset = get_seed_and_offset(batch_size * 32, generator, device)

        maybe_seed_arr, seed_val, maybe_offset_arr, offset_val = (
            _validate_and_convert_seed_offset(seed, offset, device, batch_size)
        )

        module.top_k_top_p_sampling_from_probs(
            probs,
            samples,
            valid,
            indices,
            maybe_top_k_arr,
            top_k_val,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
            maybe_seed_arr,
            seed_val,
            maybe_offset_arr,
            offset_val,
        )
        if return_valid:
            return samples, valid
        return samples

    @register_fake_op("flashinfer::top_k_top_p_sampling_from_probs")
    def _fake_top_k_top_p_sampling_from_probs(
        probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        maybe_top_k_arr: Optional[torch.Tensor],
        top_k_val: int,
        maybe_top_p_arr: Optional[torch.Tensor],
        top_p_val: float,
        deterministic: bool,
        generator: Optional[torch.Generator],
        return_valid: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = indices.size(0) if indices is not None else probs.size(0)
        out_dtype = indices.dtype if indices is not None else torch.int32
        if return_valid:
            return (
                torch.empty(batch_size, dtype=out_dtype, device=probs.device),
                torch.empty(batch_size, dtype=torch.bool, device=probs.device),
            )
        return torch.empty(batch_size, dtype=out_dtype, device=probs.device)

    # torch library for top_p_renorm_probs

    @register_custom_op("flashinfer::top_p_renorm_probs", mutates_args=("workspace",))
    def top_p_renorm_probs(
        probs: torch.Tensor,
        maybe_top_p_arr: Optional[torch.Tensor],
        top_p_val: float,
        is_deterministic: bool,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        probs = probs.float()
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        renorm_probs = torch.empty_like(probs)
        module.top_p_renorm_probs(
            probs,
            renorm_probs,
            maybe_top_p_arr,
            top_p_val,
            is_deterministic,
            workspace,
        )
        return renorm_probs

    @register_fake_op("flashinfer::top_p_renorm_probs")
    def _fake_top_p_renorm_probs(
        probs: torch.Tensor,
        maybe_top_p_arr: Optional[torch.Tensor],
        top_p_val: float,
        is_deterministic: bool,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(probs)

    # torch library for top_k_renorm_probs

    @register_custom_op(
        "flashinfer::top_k_renorm_probs", mutates_args=("row_states_buffer",)
    )
    def top_k_renorm_probs(
        probs: torch.Tensor,
        maybe_top_k_arr: Optional[torch.Tensor],
        top_k_val: int,
        row_states_buffer: torch.Tensor,
    ) -> torch.Tensor:
        # Support FP32, FP16, BF16
        assert probs.dtype in [torch.float32, torch.float16, torch.bfloat16], (
            f"Unsupported dtype {probs.dtype}, expected float32, float16, or bfloat16"
        )
        maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
        renorm_probs = torch.empty_like(probs)
        module.top_k_renorm_probs(
            probs,
            renorm_probs,
            maybe_top_k_arr,
            top_k_val,
            row_states_buffer,
        )
        return renorm_probs

    @register_fake_op("flashinfer::top_k_renorm_probs")
    def _fake_top_k_renorm_probs(
        probs: torch.Tensor,
        maybe_top_k_arr: Optional[torch.Tensor],
        top_k_val: int,
        row_states_buffer: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(probs)

    # torch library for top_k_mask_logits

    @register_custom_op(
        "flashinfer::top_k_mask_logits", mutates_args=("row_states_buffer",)
    )
    def top_k_mask_logits(
        logits: torch.Tensor,
        maybe_top_k_arr: Optional[torch.Tensor],
        top_k_val: int,
        row_states_buffer: torch.Tensor,
    ) -> torch.Tensor:
        # Support FP32, FP16, BF16
        assert logits.dtype in [torch.float32, torch.float16, torch.bfloat16], (
            f"Unsupported dtype {logits.dtype}, expected float32, float16, or bfloat16"
        )
        maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
        mask_logits = torch.empty_like(logits)

        module.top_k_mask_logits(
            logits,
            mask_logits,
            maybe_top_k_arr,
            top_k_val,
            row_states_buffer,
        )
        return mask_logits

    @register_fake_op("flashinfer::top_k_mask_logits")
    def _fake_top_k_mask_logits(
        logits: torch.Tensor,
        maybe_top_k_arr: Optional[torch.Tensor],
        top_k_val: int,
        row_states_buffer: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(logits)

    # torch library for chain_speculative_sampling

    @register_custom_op(
        "flashinfer::chain_speculative_sampling",
        mutates_args=(
            "output_accepted_token_num",
            "output_emitted_draft_token_num",
        ),
    )
    def chain_speculative_sampling(
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        output_accepted_token_num: torch.Tensor,
        output_emitted_draft_token_num: torch.Tensor,
        deterministic: bool,
        generator: Optional[torch.Generator],
        seed: Optional[Union[int, torch.Tensor]] = None,
        offset: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        device = draft_probs.device
        draft_probs = draft_probs.float()
        draft_token_ids = draft_token_ids.int()
        target_probs = target_probs.float()
        output_accepted_token_num = output_accepted_token_num.int()
        output_emitted_draft_token_num = output_emitted_draft_token_num.int()
        b, n = draft_token_ids.shape
        output_token_ids = torch.empty((b, n + 1), dtype=torch.int32, device=device)
        batch_size = b
        if seed is None or offset is None:
            seed, offset = get_seed_and_offset(
                draft_probs.size(0) * (draft_probs.size(1) + 1), generator, device
            )

        maybe_seed_arr, seed_val, maybe_offset_arr, offset_val = (
            _validate_and_convert_seed_offset(seed, offset, device, batch_size)
        )

        module.chain_speculative_sampling(
            draft_probs,
            draft_token_ids,
            target_probs,
            output_token_ids,
            output_accepted_token_num,
            output_emitted_draft_token_num,
            deterministic,
            maybe_seed_arr,
            seed_val,
            maybe_offset_arr,
            offset_val,
        )
        return output_token_ids

    @register_fake_op("flashinfer::chain_speculative_sampling")
    def _fake_chain_speculative_sampling(
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        output_accepted_token_num: torch.Tensor,
        output_emitted_draft_token_num: torch.Tensor,
        deterministic: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        b, n = draft_token_ids.shape
        device = draft_token_ids.device
        return torch.empty((b, n + 1), dtype=torch.int32, device=device)

    # Register the module
    return SimpleNamespace(
        softmax=softmax,
        sampling_from_probs=sampling_from_probs,
        sampling_from_logits=sampling_from_logits,
        top_p_sampling_from_probs=top_p_sampling_from_probs,
        top_k_sampling_from_probs=top_k_sampling_from_probs,
        min_p_sampling_from_probs=min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs=top_k_top_p_sampling_from_probs,
        top_p_renorm_probs=top_p_renorm_probs,
        top_k_renorm_probs=top_k_renorm_probs,
        top_k_mask_logits=top_k_mask_logits,
        chain_speculative_sampling=chain_speculative_sampling,
    )


@functools.cache
def get_fused_sampling_hy3_module():
    return gen_fused_sampling_hy3_module().build_and_load()


@register_custom_op(
    "flashinfer::fused_sampling_from_logits_hy3",
    mutates_args=("workspace_buffer", "output", "penalty_mask"),
)
def _fused_sampling_from_logits_hy3(
    workspace_buffer: torch.Tensor,
    output: torch.Tensor,
    logits: torch.Tensor,
    penalty_mask: Optional[torch.Tensor],
    slot_id: Optional[torch.Tensor],
    repetition_penalty: Optional[torch.Tensor],
    repetition_penalty_val: float,
    temperature: Optional[torch.Tensor],
    temperature_val: float,
    softmax_policy: int,
    top_k: Optional[torch.Tensor],
    top_k_val: int,
    top_p: Optional[torch.Tensor],
    top_p_val: float,
    max_top_k: int,
    gumbel_noise: Optional[torch.Tensor],
    draft_token_ids: Optional[torch.Tensor],
    sm_count: int,
    seed: int,
    offset: int,
    temperature_only: bool,
) -> None:
    get_fused_sampling_hy3_module().fused_sampling_from_logits_hy3(
        workspace_buffer,
        logits,
        output,
        penalty_mask,
        slot_id,
        repetition_penalty,
        repetition_penalty_val,
        temperature,
        temperature_val,
        softmax_policy,
        top_k,
        top_k_val,
        top_p,
        top_p_val,
        max_top_k,
        gumbel_noise,
        draft_token_ids,
        sm_count,
        seed,
        offset,
        temperature_only,
    )


@register_fake_op("flashinfer::fused_sampling_from_logits_hy3")
def _fake_fused_sampling_from_logits_hy3(
    workspace_buffer: torch.Tensor,
    output: torch.Tensor,
    logits: torch.Tensor,
    penalty_mask: Optional[torch.Tensor],
    slot_id: Optional[torch.Tensor],
    repetition_penalty: Optional[torch.Tensor],
    repetition_penalty_val: float,
    temperature: Optional[torch.Tensor],
    temperature_val: float,
    softmax_policy: int,
    top_k: Optional[torch.Tensor],
    top_k_val: int,
    top_p: Optional[torch.Tensor],
    top_p_val: float,
    max_top_k: int,
    gumbel_noise: Optional[torch.Tensor],
    draft_token_ids: Optional[torch.Tensor],
    sm_count: int,
    seed: int,
    offset: int,
    temperature_only: bool,
) -> None:
    pass


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


def _validate_and_convert_seed_offset(
    seed: Union[int, torch.Tensor],
    offset: Union[int, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> Tuple[Optional[torch.Tensor], int, Optional[torch.Tensor], int]:
    """Validate and convert seed/offset to tensor/scalar tuples for sampling kernels.

    Parameters
    ----------
    seed : Union[int, torch.Tensor]
        Seed value or tensor.
    offset : Union[int, torch.Tensor]
        Offset value or tensor.
    device : torch.device
        Expected device for tensor inputs.
    batch_size : int
        Expected batch size for tensor length validation.

    Returns
    -------
    Tuple[Optional[torch.Tensor], int, Optional[torch.Tensor], int]
        (maybe_seed_arr, seed_val, maybe_offset_arr, offset_val)

    Raises
    ------
    ValueError
        If seed and offset are not both tensors or both scalars, or if tensor
        properties (device, dtype, ndim, size) are invalid.
    """
    # Validate tensor/scalar consistency
    if isinstance(seed, torch.Tensor) != isinstance(offset, torch.Tensor):
        raise ValueError("seed and offset must both be tensors or both be scalars")

    # Convert to tensor/scalar tuple
    maybe_seed_arr, seed_val = _to_tensor_scalar_tuple(seed)
    maybe_offset_arr, offset_val = _to_tensor_scalar_tuple(offset)

    # Validate tensor properties
    if maybe_seed_arr is not None:
        if maybe_seed_arr.device != device:
            raise ValueError(f"seed tensor must be on {device}")
        if maybe_seed_arr.dtype not in [torch.int64, torch.uint64]:
            raise ValueError("seed tensor must be int64/uint64")
        if maybe_seed_arr.ndim != 1:
            raise ValueError("seed tensor must be 1D")
        if maybe_seed_arr.size(0) not in [1, batch_size]:
            raise ValueError(f"seed tensor length must be 1 or {batch_size}")
    if maybe_offset_arr is not None:
        if maybe_offset_arr.device != device:
            raise ValueError(f"offset tensor must be on {device}")
        if maybe_offset_arr.dtype not in [torch.int64, torch.uint64]:
            raise ValueError("offset tensor must be int64/uint64")
        if maybe_offset_arr.ndim != 1:
            raise ValueError("offset tensor must be 1D")
        if maybe_offset_arr.size(0) not in [1, batch_size]:
            raise ValueError(f"offset tensor length must be 1 or {batch_size}")

    return maybe_seed_arr, seed_val, maybe_offset_arr, offset_val


HY3_SAMPLER_SOFTMAX_NONE = 0
HY3_SAMPLER_SOFTMAX_BEFORE_TOP_K = 1
HY3_SAMPLER_SOFTMAX_AFTER_TOP_K = 2
_HY3_SAMPLER_VOCAB_SIZE = 120832
_HY3_UINT64_MODULUS = 1 << 64
_HY3_INT64_MAX = (1 << 63) - 1


@functools.cache
def _hy3_sampler_device_info(device: torch.device) -> Tuple[bool, int]:
    """Cache capability/SM queries; neither belongs on the per-token hot path."""
    if device.type != "cuda":
        return False, 0
    major, minor = torch.cuda.get_device_capability(device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return (major, minor) == (10, 0), sm_count


def _hy3_sampler_workspace_size(
    batch_size: int,
    sm_count: int,
    temperature_only: bool,
    softmax_policy: int,
) -> int:
    if temperature_only:
        return batch_size * (sm_count * 8 + 4)
    # The accepted B200 dispatch uses 1024 candidate slots/request for B<8 and
    # 512 otherwise. BEFORE_TOP_K additionally stores a max/sum pair per block.
    candidate_count = batch_size * (1024 if batch_size < 8 else 512)
    size = candidate_count * 8
    if softmax_policy == HY3_SAMPLER_SOFTMAX_BEFORE_TOP_K:
        size += batch_size * (32 if batch_size < 8 else 16) * 8
    return size


def _hy3_sampler_row_value(
    value: Union[torch.Tensor, float, int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.full((batch_size,), value, dtype=dtype, device=device)


def _fused_sampling_from_logits_hy3_fallback(
    logits: torch.Tensor,
    *,
    penalty_mask: Optional[torch.Tensor],
    slot_id: Optional[torch.Tensor],
    repetition_penalty: Union[torch.Tensor, float],
    temperature: Union[torch.Tensor, float],
    softmax_policy: int,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    max_top_k: int,
    gumbel_noise: Optional[torch.Tensor],
    draft_token_ids: Optional[torch.Tensor],
    seed: int,
    offset: int,
    temperature_only: bool,
) -> torch.Tensor:
    """Portable HY3 semantics when the optimized SM100 path is unsupported."""
    batch_size, vocab_size = logits.shape
    work = logits.float()
    if gumbel_noise is None:
        fallback_generator = torch.Generator(device=logits.device)
        mixed_seed = (int(seed) + 0x9E3779B97F4A7C15 * int(offset)) & (
            _HY3_UINT64_MODULUS - 1
        )
        fallback_generator.manual_seed(mixed_seed)
        uniform = torch.rand(
            logits.shape,
            dtype=torch.float32,
            device=logits.device,
            generator=fallback_generator,
        ).clamp_min_(1e-20)
        gumbel_noise = -(-uniform.log()).log()

    if temperature_only:
        temp = _hy3_sampler_row_value(
            temperature, batch_size, logits.device, torch.float32
        )
        work = work / torch.where(temp > 0, temp, torch.ones_like(temp))[:, None]
        if draft_token_ids is not None:
            valid = (draft_token_ids >= 0) & (draft_token_ids < vocab_size)
            rows = torch.arange(batch_size, device=logits.device)
            tokens = draft_token_ids.clamp(0, max(vocab_size - 1, 0)).long()
            old = work[rows, tokens]
            work = work.clone()
            work[rows, tokens] = torch.where(
                valid, torch.full_like(old, float("-inf")), old
            )
        return (work + gumbel_noise).argmax(dim=-1).to(torch.int32).view(-1, 1)

    work = work.clone()
    rp = _hy3_sampler_row_value(
        repetition_penalty, batch_size, logits.device, torch.float32
    )
    if penalty_mask is not None and slot_id is not None:
        valid_slot = (slot_id >= 0) & (slot_id < penalty_mask.size(0))
        safe_slot = slot_id.clamp(0, max(penalty_mask.size(0) - 1, 0)).long()
        packed = penalty_mask.index_select(0, safe_slot)
        columns = torch.arange(vocab_size, device=logits.device)
        bits = ((packed[:, columns >> 3] >> (columns & 7)) & 1).bool()
        active = bits & valid_slot[:, None] & (rp > 0)[:, None]
        penalized = torch.where(work > 0, work / rp[:, None], work * rp[:, None])
        work = torch.where(active, penalized, work)

    temp = _hy3_sampler_row_value(temperature, batch_size, logits.device, torch.float32)
    work = work / torch.where(temp > 0, temp, torch.ones_like(temp))[:, None]
    if softmax_policy == HY3_SAMPLER_SOFTMAX_BEFORE_TOP_K:
        work = torch.softmax(work, dim=-1)

    candidate_count = min(max_top_k, vocab_size)
    values, tokens = torch.topk(work, candidate_count, dim=-1, sorted=True)
    requested_k = _hy3_sampler_row_value(top_k, batch_size, logits.device, torch.int64)
    effective_k = torch.where(
        requested_k > 0,
        requested_k.clamp(max=candidate_count),
        torch.full_like(requested_k, candidate_count),
    )
    positions = torch.arange(candidate_count, device=logits.device)[None, :]
    candidate_valid = positions < effective_k[:, None]

    probabilities: Optional[torch.Tensor] = None
    if softmax_policy == HY3_SAMPLER_SOFTMAX_AFTER_TOP_K:
        probabilities = torch.softmax(
            values.masked_fill(~candidate_valid, float("-inf")), dim=-1
        )
        sample_values = torch.where(
            probabilities > 0,
            probabilities.log(),
            torch.full_like(probabilities, float("-inf")),
        )
    elif softmax_policy == HY3_SAMPLER_SOFTMAX_BEFORE_TOP_K:
        probabilities = values
        sample_values = torch.where(
            probabilities > 0,
            probabilities.log(),
            torch.full_like(probabilities, float("-inf")),
        )
    else:
        sample_values = values

    keep = candidate_valid
    if probabilities is not None:
        thresholds = _hy3_sampler_row_value(
            top_p, batch_size, logits.device, torch.float32
        )
        exclusive = probabilities.cumsum(dim=-1) - probabilities
        keep = keep & (
            (thresholds <= 0)[:, None]
            | (positions == 0)
            | (exclusive < thresholds[:, None])
        )
    scores = sample_values + gumbel_noise.gather(1, tokens)
    scores = scores.masked_fill(~keep, float("-inf"))
    maxima = scores.max(dim=-1, keepdim=True).values
    tied_tokens = torch.where(
        scores == maxima, tokens, torch.full_like(tokens, vocab_size)
    )
    sampled = tied_tokens.min(dim=-1).values
    sampled = torch.where(sampled < vocab_size, sampled, torch.zeros_like(sampled))
    output = sampled.to(torch.int32).view(-1, 1)

    if penalty_mask is not None and slot_id is not None:
        valid_slot = (slot_id >= 0) & (slot_id < penalty_mask.size(0))
        safe_slot = slot_id.clamp(0, max(penalty_mask.size(0) - 1, 0)).long()
        token = output[:, 0].long()
        byte = token >> 3
        bit_position = token & 7
        active = valid_slot & (rp > 0)

        # Match the CUDA kernel's atomicOr semantics when multiple rows map to
        # the same slot and byte.  De-duplicating (byte, bit) pairs prevents a
        # repeated bit from carrying during the subsequent integer sum.
        flat_byte = safe_slot * penalty_mask.stride(0) + byte
        encoded_updates = torch.unique(((flat_byte << 3) | bit_position)[active])
        update_bytes = encoded_updates >> 3
        update_bits = (torch.ones_like(encoded_updates) << (encoded_updates & 7)).to(
            torch.int32
        )
        unique_bytes, inverse = torch.unique(update_bytes, return_inverse=True)
        combined_bits = torch.zeros_like(unique_bytes, dtype=torch.int32)
        combined_bits.scatter_add_(0, inverse, update_bits)

        flat_penalty_mask = penalty_mask.view(-1)
        old = flat_penalty_mask.index_select(0, unique_bytes).to(torch.int32)
        flat_penalty_mask[unique_bytes] = (old | combined_bits).to(torch.uint8)
    return output


@flashinfer_api(trace=fused_sampling_hy3_trace_dispatch)
def fused_sampling_from_logits_hy3(
    logits: torch.Tensor,
    *,
    workspace_buffer: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    penalty_mask: Optional[torch.Tensor] = None,
    slot_id: Optional[torch.Tensor] = None,
    repetition_penalty: Union[torch.Tensor, float] = 0.0,
    temperature: Union[torch.Tensor, float] = 0.0,
    softmax_policy: int = HY3_SAMPLER_SOFTMAX_NONE,
    top_k: Union[torch.Tensor, int] = 0,
    top_p: Union[torch.Tensor, float] = 0.0,
    max_top_k: int = 32,
    gumbel_noise: Optional[torch.Tensor] = None,
    draft_token_ids: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    r"""HY3 fused repetition penalty, temperature, top-k/top-p and sampling.

    The optimized path targets SM100/B200 and HY3's 120832-token vocabulary.
    Passing
    external FP32 Gumbel noise provides the deterministic parity boundary.
    ``workspace_buffer`` and ``out`` can be preallocated to keep addresses
    stable during CUDA graph replay. Graph replay with random sampling must
    update an external ``gumbel_noise`` tensor inside the graph; scalar
    ``seed``/``offset`` values are fixed at capture time. Callers should pass
    explicit buffers for graph capture and must not share one workspace across
    concurrently executing streams. The output has shape ``[batch_size, 1]``
    and dtype ``int32``.

    Parameters
    ----------
    logits : torch.Tensor
        Rank-2 floating-point logits with shape ``[batch_size, vocab_size]``.
        The B200 kernel is selected for the HY3 vocabulary size (120832);
        other supported inputs use the portable PyTorch implementation.
    workspace_buffer : Optional[torch.Tensor]
        Optional contiguous 1-D uint8 scratch buffer. Reuse one buffer per
        concurrently executing CUDA stream.
    out : Optional[torch.Tensor]
        Optional contiguous int32 output with shape ``[batch_size, 1]``.
    penalty_mask : Optional[torch.Tensor]
        Packed uint8 repetition mask. Each bit marks one vocabulary entry and
        the sampled entry is set in place.
    slot_id : Optional[torch.Tensor]
        Int32 row selector into ``penalty_mask``, shape ``[batch_size]``.
    repetition_penalty : Union[torch.Tensor, float]
        Per-row float32 values or one scalar. A positive value requires
        ``penalty_mask`` and ``slot_id``.
    temperature : Union[torch.Tensor, float]
        Per-row float32 values or one scalar temperature. Values greater than
        zero scale logits; non-positive values disable temperature scaling.
    softmax_policy : int
        ``0`` disables softmax, ``1`` applies it before top-k, and ``2``
        applies it after top-k.
    top_k : Union[torch.Tensor, int]
        Per-row int32/int64 values or one scalar top-k cutoff.
    top_p : Union[torch.Tensor, float]
        Per-row float32 values or one scalar nucleus threshold.
    max_top_k : int
        Compile-time candidate capacity; must be 32 or 64.
    gumbel_noise : Optional[torch.Tensor]
        Optional contiguous FP32 noise with the same shape as ``logits``.
        Providing it gives a deterministic parity boundary.
    draft_token_ids : Optional[torch.Tensor]
        Optional int64 IDs excluded by the temperature-only path.
    generator : Optional[torch.Generator]
        Generator used to obtain a Philox seed and offset when external noise
        and an explicit seed are absent.
    seed : Optional[int]
        Random seed used by the fused CUDA path. Must be greater than zero
        when ``gumbel_noise`` is not provided.
    offset : Optional[int]
        Random subsequence offset used by the fused CUDA path.

    Returns
    -------
    torch.Tensor
        Sampled int32 IDs with shape ``[batch_size, 1]``.
    """
    if logits.ndim != 2 or not logits.dtype.is_floating_point:
        raise ValueError("logits must be a rank-2 floating-point tensor")
    batch_size, vocab_size = logits.shape
    if batch_size <= 0 or vocab_size <= 0:
        raise ValueError("logits dimensions must be positive")
    if out is not None and (
        out.device != logits.device
        or out.dtype != torch.int32
        or out.shape != (batch_size, 1)
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous int32 [batch_size, 1] on logits.device"
        )
    if workspace_buffer is not None and (
        workspace_buffer.device != logits.device
        or workspace_buffer.dtype != torch.uint8
        or workspace_buffer.ndim != 1
        or not workspace_buffer.is_contiguous()
    ):
        raise ValueError(
            "workspace_buffer must be contiguous 1D uint8 on logits.device"
        )
    if max_top_k not in (32, 64):
        raise ValueError("max_top_k must be 32 or 64")
    if softmax_policy not in (0, 1, 2):
        raise ValueError("softmax_policy must be 0, 1, or 2")

    def check_row_tensor(
        value: object, name: str, dtypes: Tuple[torch.dtype, ...]
    ) -> None:
        if not isinstance(value, torch.Tensor):
            return
        if value.device != logits.device or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous and on logits.device")
        if value.dtype not in dtypes or value.shape != (batch_size,):
            raise ValueError(f"{name} has an invalid dtype or shape")

    check_row_tensor(repetition_penalty, "repetition_penalty", (torch.float32,))
    check_row_tensor(temperature, "temperature", (torch.float32,))
    check_row_tensor(top_k, "top_k", (torch.int32, torch.int64))
    check_row_tensor(top_p, "top_p", (torch.float32,))
    check_row_tensor(slot_id, "slot_id", (torch.int32,))
    check_row_tensor(draft_token_ids, "draft_token_ids", (torch.int64,))
    if (penalty_mask is None) != (slot_id is None):
        raise ValueError("penalty_mask and slot_id must be provided together")
    if penalty_mask is not None:
        if (
            penalty_mask.device != logits.device
            or penalty_mask.dtype != torch.uint8
            or penalty_mask.ndim != 2
            or not penalty_mask.is_contiguous()
            or penalty_mask.size(0) < batch_size
            or penalty_mask.size(1) < (vocab_size + 7) // 8
        ):
            raise ValueError(
                "penalty_mask must be contiguous uint8 [rows>=B, bytes>=ceil(V/8)]"
            )
        if penalty_mask.stride(0) % 4 != 0:
            raise ValueError("penalty_mask row stride must be a multiple of four bytes")
        if penalty_mask.data_ptr() % 4 != 0:
            raise ValueError("penalty_mask address must be aligned to four bytes")
    if gumbel_noise is not None and (
        gumbel_noise.device != logits.device
        or gumbel_noise.dtype != torch.float32
        or gumbel_noise.shape != logits.shape
        or not gumbel_noise.is_contiguous()
    ):
        raise ValueError("gumbel_noise must be contiguous float32 with logits.shape")

    def scalar_zero(x: Union[torch.Tensor, float]) -> bool:
        """Return whether a non-tensor scalar is zero."""
        return not isinstance(x, torch.Tensor) and float(x) == 0.0

    temperature_only = (
        penalty_mask is None
        and scalar_zero(repetition_penalty)
        and scalar_zero(top_p)
        and not isinstance(top_k, torch.Tensor)
        and int(top_k) == 0
        and softmax_policy == HY3_SAMPLER_SOFTMAX_NONE
        and (isinstance(temperature, torch.Tensor) or float(temperature) > 0.0)
    )
    if draft_token_ids is not None and not temperature_only:
        raise ValueError("draft_token_ids requires the temperature-only path")
    has_rp = (
        isinstance(repetition_penalty, torch.Tensor) or float(repetition_penalty) > 0
    )
    has_top_k = isinstance(top_k, torch.Tensor) or int(top_k) > 0
    has_top_p = isinstance(top_p, torch.Tensor) or float(top_p) > 0
    if has_rp and penalty_mask is None:
        raise ValueError("repetition_penalty requires penalty_mask and slot_id")
    if has_top_p and (not has_top_k or softmax_policy == 0):
        raise ValueError("top_p requires top_k and softmax_policy != NONE")
    if softmax_policy != 0 and not has_top_p:
        raise ValueError("softmax_policy != NONE requires top_p")

    if gumbel_noise is None:
        if seed is None:
            seed, generated_offset = get_seed_and_offset(
                batch_size * max_top_k, generator, logits.device
            )
            # CUDA generator state stores the full uint64 seed, while
            # get_seed_and_offset views it as int64.  Recover the original bit
            # pattern before applying the positive-seed API contract.  Zero is
            # valid generator state but reserved by this API, so only the
            # generated value is remapped; an explicit zero remains invalid.
            seed = int(seed) & (_HY3_UINT64_MODULUS - 1)
            if seed == 0:
                seed = 1
            if offset is None:
                offset = generated_offset
        if int(seed) <= 0:
            raise ValueError("seed must be > 0 without external gumbel_noise")
        if int(seed) >= _HY3_UINT64_MODULUS:
            raise ValueError("seed must be less than 2**64")
    else:
        seed = 0 if seed is None else seed
    offset = 0 if offset is None else offset

    is_sm100, sm_count = _hy3_sampler_device_info(logits.device)
    use_hy3_kernel = (
        is_sm100
        and vocab_size == _HY3_SAMPLER_VOCAB_SIZE
        and logits.dtype in (torch.float32, torch.bfloat16)
        and logits.stride(1) == 1
        and (penalty_mask is None or penalty_mask.stride(0) % 4 == 0)
    )
    if not use_hy3_kernel:
        result = _fused_sampling_from_logits_hy3_fallback(
            logits,
            penalty_mask=penalty_mask,
            slot_id=slot_id,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            softmax_policy=softmax_policy,
            top_k=top_k,
            top_p=top_p,
            max_top_k=max_top_k,
            gumbel_noise=gumbel_noise,
            draft_token_ids=draft_token_ids,
            seed=int(seed),
            offset=int(offset),
            temperature_only=temperature_only,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    workspace_size = _hy3_sampler_workspace_size(
        batch_size, sm_count, temperature_only, softmax_policy
    )
    if workspace_buffer is None:
        stream_id = torch.cuda.current_stream(logits.device).cuda_stream
        workspace_buffer = _get_cache_buf(
            f"fused_sampler_hy3_workspace_{stream_id}", workspace_size, logits.device
        )
    elif workspace_buffer.numel() < workspace_size:
        raise ValueError(
            f"workspace_buffer is too small: need {workspace_size} bytes, "
            f"got {workspace_buffer.numel()}"
        )
    elif workspace_buffer.data_ptr() % 4 != 0:
        raise ValueError("workspace_buffer address must be aligned to four bytes")
    if out is None:
        out = torch.empty((batch_size, 1), dtype=torch.int32, device=logits.device)
    rp_tensor, rp_val = _to_tensor_scalar_tuple(repetition_penalty)
    temp_tensor, temp_val = _to_tensor_scalar_tuple(temperature)
    top_k_tensor, top_k_val = _to_tensor_scalar_tuple(top_k)
    top_p_tensor, top_p_val = _to_tensor_scalar_tuple(top_p)
    ffi_seed = int(seed)
    if ffi_seed > _HY3_INT64_MAX:
        # TVM-FFI transports Python integers through signed int64.  C++ then
        # converts this carrier back to uint64_t without changing its bits.
        ffi_seed -= _HY3_UINT64_MODULUS
    _fused_sampling_from_logits_hy3(
        workspace_buffer,
        out,
        logits,
        penalty_mask,
        slot_id,
        rp_tensor,
        float(rp_val),
        temp_tensor,
        float(temp_val),
        int(softmax_policy),
        top_k_tensor,
        int(top_k_val),
        top_p_tensor,
        float(top_p_val),
        max_top_k,
        gumbel_noise,
        draft_token_ids,
        sm_count,
        ffi_seed,
        int(offset),
        temperature_only,
    )
    return out


@flashinfer_api(trace=softmax_trace)
def softmax(
    logits: torch.Tensor,
    temperature: Optional[Union[torch.Tensor, float]] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Fused GPU kernel for `online safe softmax <https://arxiv.org/abs/1805.02867>`_ with temperature scaling.


    Parameters
    ----------
    logits : torch.Tensor
        Input tensor of logits.
    temperature: Optional[Union[torch.Tensor, float]]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the temperature for temperature scaling.
        If a scalar, the same temperature is used for all requests.
        If a tensor, each request has its own temperature.
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch (PDL) for improved performance on supported hardware.
        If None (default), PDL will be automatically enabled on devices with compute capability >= 9.0.
    Returns
    -------
    probs : torch.Tensor
        Tensor of the same shape as input containing the softmax probabilities.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> logits = torch.rand(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[0.8823, 0.9150, 0.3829, 0.9593, 0.3904],
            [0.6009, 0.2566, 0.7936, 0.9408, 0.1332],
            [0.9346, 0.5936, 0.8694, 0.5677, 0.7411],
            [0.4294, 0.8854, 0.5739, 0.2666, 0.6274]], device='cuda:0')
    >>> probs = flashinfer.sampling.softmax(logits, temperature=1.0)
    >>> probs
    tensor([[0.2309, 0.2385, 0.1401, 0.2493, 0.1412],
            [0.2019, 0.1431, 0.2448, 0.2837, 0.1265],
            [0.2401, 0.1707, 0.2249, 0.1664, 0.1979],
            [0.1724, 0.2719, 0.1991, 0.1465, 0.2101]], device='cuda:0')
    """
    workspace_buffer = _get_cache_buf("softmax_workspace", 1024 * 1024, logits.device)
    if temperature is None:
        temperature = 1.0

    # Auto-detect PDL support if not specified
    if enable_pdl is None:
        enable_pdl = device_support_pdl(logits.device)

    return get_sampling_module().softmax(
        workspace_buffer, logits, *_to_tensor_scalar_tuple(temperature), enable_pdl
    )


@flashinfer_api(trace=sampling_from_logits_trace)
def sampling_from_logits(
    logits: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    r"""Fused GPU kernel for category sampling from logits. It's equivalent to sampling
    from :attr:`logits` after applying softmax.
    Parameters
    ----------
    logits: torch.Tensor
        Logits for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of logits. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)``, dtype ``torch.int32`` or ``torch.int64``
        that maps each output to a row in logits. The output tensor will have the same dtype as indices.
        For example, if indices[i] = j, then the i-th output will be sampled from logits[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of logits
        and output dtype defaults to ``torch.int32``.
    deterministic: bool
        Since the sampling doesn't use cub's BlockScan, the sampling is deterministic. We keep this
        argument for compatibility with other sampling functions.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`logits`, default is ``False``.
    seed: Optional[Union[int, torch.Tensor]]
        Random seed value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. Common approaches include:
        - Incrementing offset by the number of random values consumed
        - Updating seed based on the number of calls to the operation
    offset: Optional[Union[int, torch.Tensor]]
        Random offset value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. The offset should be
        incremented based on the number of random values consumed by the operation.
    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape (batch_size,). It's equivalent to sampling from
        :attr:`logits` after applying softmax.
    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> logits = torch.rand(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[0.8823, 0.9150, 0.3829, 0.9593, 0.3904],
            [0.6009, 0.2566, 0.7936, 0.9408, 0.1332],
            [0.9346, 0.5936, 0.8694, 0.5677, 0.7411],
            [0.4294, 0.8854, 0.5739, 0.2666, 0.6274]], device='cuda:0')
    >>> samples = flashinfer.sampling.sampling_from_logits(logits)
    >>> samples
    tensor([0, 1, 1, 1], device='cuda:0', dtype=torch.int32)
    """
    if check_nan:
        if torch.any(torch.isnan(logits)):
            raise ValueError("Input logits contains NaN.")
    return get_sampling_module().sampling_from_logits(
        logits, indices, deterministic, generator, seed, offset
    )


@flashinfer_api(trace=sampling_from_probs_trace)
def sampling_from_probs(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Fused GPU kernel for category sampling from probabilities.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)``, dtype ``torch.int32`` or ``torch.int64``
        that maps each output to a row in probs. The output tensor will have the same dtype as indices.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs
        and output dtype defaults to ``torch.int32``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.
    seed: Optional[Union[int, torch.Tensor]]
        Random seed value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. Common approaches include:
        - Incrementing offset by the number of random values consumed
        - Updating seed based on the number of calls to the operation
    offset: Optional[Union[int, torch.Tensor]]
        Random offset value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. The offset should be
        incremented based on the number of random values consumed by the operation.
    return_valid : bool
        When ``True``, the kernel returns an additional boolean mask
        indicating which rows had a valid (non-degenerate) distribution.
        Defaults to ``False``.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If ``return_valid`` is ``False`` (default), a 1-D ``samples``
        tensor of shape ``(batch_size,)``.  If ``return_valid`` is
        ``True``, ``(samples, valid)`` where ``valid`` is a boolean
        tensor of shape ``(batch_size,)`` indicating which rows had a
        valid distribution.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.sampling_from_probs(norm_prob)
    >>> samples
    tensor([1, 2, 1, 4], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().sampling_from_probs(
        probs,
        indices,
        deterministic,
        generator,
        seed,
        offset,
        return_valid,
    )


@flashinfer_api(trace=top_p_sampling_trace)
def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Fused GPU kernel for top-p sampling (nucleus sampling) from probabilities,
    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_p: Union[torch.Tensor, float]
        Either a float or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a float, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)``, dtype ``torch.int32`` or ``torch.int64``
        that maps each output to a row in probs. The output tensor will have the same dtype as indices.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs
        and output dtype defaults to ``torch.int32``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.
    seed: Optional[Union[int, torch.Tensor]]
        Random seed value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. Common approaches include:
        - Incrementing offset by the number of random values consumed
        - Updating seed based on the number of calls to the operation
    offset: Optional[Union[int, torch.Tensor]]
        Random offset value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. The offset should be
        incremented based on the number of random values consumed by the operation.
    return_valid : bool
        When ``True``, the kernel returns an additional boolean mask
        indicating which rows had a valid (non-degenerate) distribution
        after the renormalization step.  Defaults to ``False``.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If ``return_valid`` is ``False`` (default), a 1-D ``samples``
        tensor of shape ``(batch_size,)``.  If ``return_valid`` is
        ``True``, ``(samples, valid)`` where ``valid`` is a boolean
        tensor of shape ``(batch_size,)`` indicating which rows had a
        valid distribution after renormalization.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = 0.5
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.top_p_sampling_from_probs(norm_prob, top_p)
    >>> samples
    tensor([1, 2, 0, 4], device='cuda:0', dtype=torch.int32)


    Note
    ----
    This function expects float32 inputs, and the output is int32.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_k_sampling_from_probs
    top_p_renorm_probs
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().top_p_sampling_from_probs(
        probs,
        indices,
        *_to_tensor_scalar_tuple(top_p),
        deterministic,
        generator,
        seed,
        offset,
        return_valid,
    )


@flashinfer_api(trace=top_k_sampling_trace)
def top_k_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Fused GPU kernel for top-k sampling from probabilities,
    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)``, dtype ``torch.int32`` or ``torch.int64``
        that maps each output to a row in probs. The output tensor will have the same dtype as indices.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs
        and output dtype defaults to ``torch.int32``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.
    seed: Optional[Union[int, torch.Tensor]]
        Random seed value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. Common approaches include:
        - Incrementing offset by the number of random values consumed
        - Updating seed based on the number of calls to the operation
    offset: Optional[Union[int, torch.Tensor]]
        Random offset value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. The offset should be
        incremented based on the number of random values consumed by the operation.
    return_valid : bool
        When ``True``, the kernel returns an additional boolean mask
        indicating which rows had a valid (non-degenerate) distribution
        after the renormalization step.  Defaults to ``False``.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If ``return_valid`` is ``False`` (default), a 1-D ``samples``
        tensor of shape ``(batch_size,)``.  If ``return_valid`` is
        ``True``, ``(samples, valid)`` where ``valid`` is a boolean
        tensor of shape ``(batch_size,)`` indicating which rows had a
        valid distribution after renormalization.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_k = 1
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.top_k_sampling_from_probs(norm_prob, top_k)
    >>> samples
    tensor([3, 3, 0, 1], device='cuda:0', dtype=torch.int32)


    Note
    ----
    This function expects float32 inputs, and the output is int32.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_p_sampling_from_probs
    top_k_renorm_probs
    """
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().top_k_sampling_from_probs(
        probs,
        indices,
        *_to_tensor_scalar_tuple(top_k),
        deterministic,
        generator,
        seed,
        offset,
        return_valid,
    )


@flashinfer_api(trace=min_p_sampling_trace)
def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Fused GPU kernel for `min_p sampling <https://arxiv.org/abs/2407.01082>`_ from probabilities,

    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    min_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for min-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)``, dtype ``torch.int32`` or ``torch.int64``
        that maps each output to a row in probs. The output tensor will have the same dtype as indices.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs
        and output dtype defaults to ``torch.int32``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.
    seed: Optional[Union[int, torch.Tensor]]
        Random seed value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. Common approaches include:
        - Incrementing offset by the number of random values consumed
        - Updating seed based on the number of calls to the operation
    offset: Optional[Union[int, torch.Tensor]]
        Random offset value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. The offset should be
        incremented based on the number of random values consumed by the operation.
    return_valid : bool
        When ``True``, the kernel returns an additional boolean mask
        indicating which rows had a valid (non-degenerate) distribution
        after the renormalization step.  Defaults to ``False``.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If ``return_valid`` is ``False`` (default), a 1-D ``samples``
        tensor of shape ``(batch_size,)``.  If ``return_valid`` is
        ``True``, ``(samples, valid)`` where ``valid`` is a boolean
        tensor of shape ``(batch_size,)`` indicating which rows had a
        valid distribution after renormalization.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    <torch._C.Generator object at 0x7f8b3db06df0>
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> min_p = torch.full((batch_size,), 0.05).to(0)
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.min_p_sampling_from_probs(norm_prob, min_p)
    >>> samples
    tensor([1, 2, 1, 4], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.
    """

    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return get_sampling_module().min_p_sampling_from_probs(
        probs,
        indices,
        *_to_tensor_scalar_tuple(min_p),
        deterministic,
        generator,
        seed,
        offset,
        return_valid,
    )


# Gating thresholds for the "top_k_first" fast path (parallel top-k, then top-p over only
# the k survivors). For a modest scalar top_k this is far cheaper than masking/renorming the
# full vocab and running rejection sampling across it: the old path's expensive steps (a
# full-vocab softmax for the logits entry point, and a single-CTA full-vocab top-p rejection
# for both) shrink to k-element work, while top-k selection costs about the same.
#
# The win only holds when (a) the vocab is large enough that the avoided full-vocab work
# outweighs the top-k selection cost, AND (b) k is small enough that the survivors stay
# cheap. Outside these thresholds we fall back to the original kernels.
# Thresholds were empirically determined.
_TOP_K_FIRST_FAST_PATH_MAX_K = 256
_TOP_K_FIRST_FAST_PATH_MIN_VOCAB = 65536


def _top_k_first_fast_path_applicable(
    x: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    indices: Optional[torch.Tensor],
) -> bool:
    return (
        indices is None
        and isinstance(top_k, int)
        and 0 < top_k <= _TOP_K_FIRST_FAST_PATH_MAX_K
        and x.size(-1) >= _TOP_K_FIRST_FAST_PATH_MIN_VOCAB
        and top_k < x.size(-1)
    )


def _top_k_first_fast_path(
    x: torch.Tensor,
    top_k: int,
    top_p: Union[torch.Tensor, float],
    *,
    from_logits: bool,
    deterministic: bool,
    generator: Optional[torch.Generator],
    check_nan: bool,
    seed: Optional[Union[int, torch.Tensor]],
    offset: Optional[Union[int, torch.Tensor]],
    return_valid: bool = False,
):
    """Shared "top_k_first" fast path for both the logits and probs entry points.

    Selects the top-k entries with the parallel radix/cluster top-k kernel, then runs
    top-p sampling over only those k entries. ``sorted=True`` gives an identical,
    deterministic ordering for both logits and probs inputs, so the two entry points
    reduce to the same ``probs_k`` and stay sample-aligned. This is distribution-equivalent
    to the masked full-vocab path (validated TV ~0.01) but far cheaper at small batch.
    """
    # Local import avoids a module-level cycle between sampling and topk.
    from .topk import top_k as _radix_top_k

    # deterministic=True makes top-k reproducible (its radix deterministic-collect path is
    # stable even at ties). We do not enforce a tie break that requires 128KB smem/block.
    values, gathered_indices = _radix_top_k(
        x, top_k, sorted=True, deterministic=deterministic
    )
    values = values.float()
    if from_logits:
        # softmax over the k retained logits == top-k-masked softmax over the full vocab.
        probs_k = torch.softmax(values, dim=-1)
    else:
        # renormalizing the k retained probabilities == top_k_renorm over the full vocab.
        probs_k = values / values.sum(dim=-1, keepdim=True)
    result = top_p_sampling_from_probs(
        probs_k,
        top_p,
        None,
        deterministic,
        check_nan=check_nan,
        generator=generator,
        seed=seed,
        offset=offset,
        return_valid=return_valid,
    )

    def _map(local):
        return (
            gathered_indices.gather(1, local.view(-1, 1).long())
            .squeeze(1)
            .to(torch.int32)
        )

    if return_valid:
        local, valid = result
        return _map(local), valid
    return _map(result)


@flashinfer_api(trace=top_k_top_p_sampling_from_logits_trace)
def top_k_top_p_sampling_from_logits(
    logits: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    r"""Fused GPU kernel for top-k and top-p sampling from pre-softmax logits,

    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    logits: torch.Tensor
        Pre-softmax logits for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of logits. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)``, dtype ``torch.int32`` or ``torch.int64``
        that maps each output to a row in probs. The output tensor will have the same dtype as indices.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs
        and output dtype defaults to ``torch.int32``.
    filter_apply_order: str
        The order of applying top-k and top-p sampling, should be either ``"top_k_first"`` or ``"joint"``.
        If ``"top_k_first"``, we first apply top-k filter, then apply top-p sampling on the top-k results.
        If ``"joint"``, we apply top-k and top-p filter simultaneously in each round. Default is ``"top_k_first"``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.
    seed: Optional[Union[int, torch.Tensor]]
        Random seed value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. Common approaches include:
        - Incrementing offset by the number of random values consumed
        - Updating seed based on the number of calls to the operation
    offset: Optional[Union[int, torch.Tensor]]
        Random offset value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. The offset should be
        incremented based on the number of random values consumed by the operation.

    Returns
    -------
    samples: torch.Tensor
        Sampled categories, shape ``(batch_size,)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = 0.5
    >>> top_k = 3
    >>> logits = torch.rand(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[ 1.9269,  1.4873,  0.9007, -2.1055, -0.7581],
            [ 1.0783,  0.8008,  1.6806,  0.3559, -0.6866],
            [-0.4934,  0.2415, -0.2316,  0.0418, -0.2516],
            [ 0.8599, -0.3097, -0.3957,  0.8034, -0.6216]], device='cuda:0')
    >>> samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k, top_p)
    >>> samples
    tensor([0, 2, 1, 3], device='cuda:0', dtype=torch.int32
    >>> probs = torch.softmax(logits, dim=-1)
    >>> probs
    tensor([[0.4788, 0.3085, 0.1716, 0.0085, 0.0327],
        [0.2358, 0.1787, 0.4307, 0.1145, 0.0404],
        [0.1358, 0.2831, 0.1764, 0.2318, 0.1729],
        [0.3613, 0.1122, 0.1029, 0.3415, 0.0821]], device='cuda:0')
    >>> samples
    tensor([0, 2, 1, 3], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.

    See Also
    --------
    top_k_top_p_sampling_from_probs
    top_k_mask_logits
    top_p_sampling_from_probs
    """
    if filter_apply_order == "top_k_first":
        if _top_k_first_fast_path_applicable(logits, top_k, indices):
            return _top_k_first_fast_path(
                logits,
                top_k,
                top_p,
                from_logits=True,
                deterministic=deterministic,
                generator=generator,
                check_nan=check_nan,
                seed=seed,
                offset=offset,
            )
        masked_logits = top_k_mask_logits(logits, top_k)
        probs = torch.softmax(masked_logits, dim=-1)
        return top_p_sampling_from_probs(
            probs,
            top_p,
            indices,
            deterministic,
            check_nan=check_nan,
            generator=generator,
            seed=seed,
            offset=offset,
        )
    elif filter_apply_order == "joint":
        probs = torch.softmax(logits, dim=-1)
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return get_sampling_module().top_k_top_p_sampling_from_probs(
            probs,
            indices,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
            generator,
            seed,
            offset,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


@flashinfer_api(trace=top_k_top_p_sampling_trace)
def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Fused GPU kernel for top-k and top-p sampling from probabilities,

    this operator implements GPU-based rejection sampling without explicit sorting.
    Check the `blog post <https://flashinfer.ai/2025/03/10/sampling.html>`_ for more details.

    The multiple rounds of rejection sampling are implemented in a single CUDA kernel,
    which is more efficient than the naive implementation that launches a series of kernels.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities for sampling. When indices is not provided, shape should be ``(batch_size, num_classes)``
        and the i-th output will be sampled from the i-th row of probabilities. When indices is provided,
        shape should be ``(unique_batch_size, num_classes)`` where unique_batch_size is the number of unique
        probability distributions.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-k sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the threshold for top-p sampling.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
    indices: Optional[torch.Tensor]
        Optional indices tensor of shape ``(batch_size,)``, dtype ``torch.int32`` or ``torch.int64``
        that maps each output to a row in probs. The output tensor will have the same dtype as indices.
        For example, if indices[i] = j, then the i-th output will be sampled from probs[j].
        This allows reusing the same probability distribution for multiple outputs.
        If indices is not provided, the i-th output will be sampled from the i-th row of probs
        and output dtype defaults to ``torch.int32``.
    filter_apply_order: str
        The order of applying top-k and top-p sampling, should be either ``"top_k_first"`` or ``"joint"``.
        If ``"top_k_first"``, we first apply top-k filter, then apply top-p sampling on the top-k results.
        If ``"joint"``, we apply top-k and top-p filter simultaneously in each round. Default is ``"top_k_first"``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    check_nan: bool
        Whether to check nan in :attr:`probs`, default is ``False``.
    seed: Optional[Union[int, torch.Tensor]]
        Random seed value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. Common approaches include:
        - Incrementing offset by the number of random values consumed
        - Updating seed based on the number of calls to the operation
    offset: Optional[Union[int, torch.Tensor]]
        Random offset value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. The offset should be
        incremented based on the number of random values consumed by the operation.
    return_valid : bool
        When ``True``, the kernel returns an additional boolean mask
        indicating which rows had a valid (non-degenerate) distribution
        after the renormalization step.  Defaults to ``False``.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If ``return_valid`` is ``False`` (default), a 1-D ``samples``
        tensor of shape ``(batch_size,)``.  If ``return_valid`` is
        ``True``, ``(samples, valid)`` where ``valid`` is a boolean
        tensor of shape ``(batch_size,)`` indicating which rows had a
        valid distribution after renormalization.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = torch.full((batch_size,), 0.2).to(0)
    >>> top_k = torch.full((batch_size,), 2).to(0)
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> norm_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> norm_prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(norm_prob, top_k, top_p)
    >>> samples
    tensor([3, 3, 0, 1], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function expects float32 inputs, and the output is int32.

    See Also
    --------
    top_k_sampling_from_probs
    top_p_sampling_from_probs
    top_k_renorm_probs
    top_p_renorm_probs
    top_k_mask_logits
    """
    if filter_apply_order == "top_k_first":
        if _top_k_first_fast_path_applicable(probs, top_k, indices):
            return _top_k_first_fast_path(
                probs,
                top_k,
                top_p,
                from_logits=False,
                deterministic=deterministic,
                generator=generator,
                check_nan=check_nan,
                seed=seed,
                offset=offset,
                return_valid=return_valid,
            )
        renorm_probs = top_k_renorm_probs(probs, top_k)
        return top_p_sampling_from_probs(
            renorm_probs,
            top_p,
            indices,
            deterministic,
            check_nan=check_nan,
            generator=generator,
            seed=seed,
            offset=offset,
            return_valid=return_valid,
        )
    elif filter_apply_order == "joint":
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return get_sampling_module().top_k_top_p_sampling_from_probs(
            probs,
            indices,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
            generator,
            seed,
            offset,
            return_valid,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


@flashinfer_api(trace=top_p_renorm_probs_trace)
def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    is_deterministic: bool = False,
) -> torch.Tensor:
    r"""Fused GPU kernel for renormalizing probabilities by top-p thresholding.

    Uses AIR Top-P algorithm (radix-based) for efficient threshold finding.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
    top_p: Union[torch.Tensor, float]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-p threshold for
        re-normalizing probabilities, should be in ``(0, 1)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We mask out the probabilities less than `threshold` where the cumulative sum
        of ``probs[probs >= threshold]`` is `top_p`, and renormalize the probabilities.
    is_deterministic: bool
        If True, use deterministic integer accumulation for reproducible results. Will affect performance.
        Default is False.

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_p = 0.3
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> renormed_probs = flashinfer.sampling.top_p_renorm_probs(prob, top_p)
    >>> renormed_probs
    tensor([[0.0000, 0.4882, 0.0000, 0.5118, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
            [0.5181, 0.0000, 0.4819, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000, 0.0000]], device='cuda:0')

    Note
    ----
    This combination of ``top_p_renorm_probs`` and ``sampling_from_probs`` should be equivalent to
    ``top_p_sampling_from_probs``.

    See Also
    --------
    top_p_sampling_from_probs
    sampling_from_probs
    top_k_renorm_probs
    """
    batch_size = probs.size(0)
    vocab_size = probs.size(1)
    # Workspace size for AIR Top-P radix algorithm.
    # Must match GetAirTopPRenormWorkspaceSize in air_top_p.cuh.
    align256 = lambda x: ((x + 255) // 256) * 256
    counter_size = 384  # sizeof(Counter<float>) with alignas(128) members
    # buf_len = alignTo(vocab_size / (ratio * 8), 256), ratio = 4 for float32
    buf_len = max(align256(vocab_size // 32), 256)
    hist_entry_size = 8 if is_deterministic else 4  # uint64 vs float
    ws_size = (
        align256(counter_size * batch_size)  # counters
        + align256(hist_entry_size * 2048 * batch_size)  # histogram
        + align256(4 * 2048 * batch_size)  # countHistogram
        + align256(4 * buf_len * batch_size)  # buf1
        + align256(4 * buf_len * batch_size)  # buf2
    )
    workspace = torch.empty(ws_size, dtype=torch.uint8, device=probs.device)
    return get_sampling_module().top_p_renorm_probs(
        probs, *_to_tensor_scalar_tuple(top_p), is_deterministic, workspace
    )


top_p_renorm_prob = top_p_renorm_probs


@flashinfer_api(trace=top_k_renorm_probs_trace)
def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    r"""Fused GPU kernel for renormalizing probabilities by top-k thresholding.

    Parameters
    ----------
    probs: torch.Tensor
        Probabilities, shape ``(batch_size, num_classes)``.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-k threshold for
        for re-normalizing probabilities, should be in ``(0, num_classes)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We keep the top-k probabilities, set the rest to zero, and renormalize the probabilities.

    Returns
    -------
    renorm_probs: torch.Tensor
        Renormalized probabilities, shape ``(batch_size, num_classes)``.
        Same dtype as input ``probs``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_k = 3
    >>> pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    >>> prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    >>> prob
    tensor([[0.2499, 0.2592, 0.1085, 0.2718, 0.1106],
            [0.2205, 0.0942, 0.2912, 0.3452, 0.0489],
            [0.2522, 0.1602, 0.2346, 0.1532, 0.2000],
            [0.1543, 0.3182, 0.2062, 0.0958, 0.2255]], device='cuda:0')
    >>> renormed_probs = flashinfer.sampling.top_k_renorm_probs(prob, top_k)
    >>> renormed_probs
    tensor([[0.3201, 0.3319, 0.0000, 0.3480, 0.0000],
            [0.2573, 0.0000, 0.3398, 0.4028, 0.0000],
            [0.3672, 0.0000, 0.3416, 0.0000, 0.2912],
            [0.0000, 0.4243, 0.2750, 0.0000, 0.3007]], device='cuda:0')

    Note
    ----
    This combination of ``top_k_renorm_probs`` and ``sampling_from_probs`` should be equivalent to
    ``top_k_sampling_from_probs``.

    See Also
    --------
    top_k_sampling_from_probs
    sampling_from_probs
    top_p_renorm_probs
    top_k : General-purpose top-k selection (returns indices and values)
    """
    # Allocate row_states buffer for multi-CTA kernel (1MB is enough for any GPU)
    buffer_bytes = 1024 * 1024  # 1MB
    row_states_buffer = _get_cache_buf(
        f"top_k_renorm_probs_row_states_{probs.device}",
        buffer_bytes,
        probs.device,
        zero_init=True,
    )

    return get_sampling_module().top_k_renorm_probs(
        probs, *_to_tensor_scalar_tuple(top_k), row_states_buffer
    )


top_k_renorm_prob = top_k_renorm_probs


@flashinfer_api(trace=top_k_mask_logits_trace)
def top_k_mask_logits(
    logits: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    r"""Fused GPU kernel for masking logits by top-k thresholding.

    Parameters
    ----------
    logits: torch.Tensor
        Logits before softmax, shape ``(batch_size, num_classes)``.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
    top_k: Union[torch.Tensor, int]
        Either a scalar or a tensor of shape ``(batch_size,)``, representing the top-k threshold for
        for masking logits, should be in ``(0, num_classes)``.
        If a scalar, the same threshold is used for all requests.
        If a tensor, each request has its own threshold.
        We keep the top-k logits, set the rest to negative infinity.

    Returns
    -------
    masked_logits: torch.Tensor
        Masked logits, shape ``(batch_size, num_classes)``.
        Same dtype as input ``logits``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 5
    >>> top_k = 3
    >>> logits = torch.randn(batch_size, vocab_size).to(0)
    >>> logits
    tensor([[ 1.9269,  1.4873,  0.9007, -2.1055, -0.7581],
            [ 1.0783,  0.8008,  1.6806,  0.3559, -0.6866],
            [-0.4934,  0.2415, -0.2316,  0.0418, -0.2516],
            [ 0.8599, -0.3097, -0.3957,  0.8034, -0.6216]], device='cuda:0')
    >>> masked_logits = flashinfer.sampling.top_k_mask_logits(logits, top_k)
    >>> masked_logits
    tensor([[ 1.9269,  1.4873,  0.9007,    -inf,    -inf],
            [ 1.0783,  0.8008,  1.6806,    -inf,    -inf],
            [   -inf,  0.2415, -0.2316,  0.0418,    -inf],
            [ 0.8599, -0.3097,    -inf,  0.8034,    -inf]], device='cuda:0')

    Note
    ----
    The combination of ``top_k_mask_logits`` and ``softmax`` should be equivalent to ``top_k_renorm_probs``.

    See Also
    --------
    top_k_renorm_probs
    top_k : General-purpose top-k selection (returns indices and values)
    """
    # Allocate row_states buffer for multi-CTA kernel (1MB is enough for any GPU)
    buffer_bytes = 1024 * 1024  # 1MB
    row_states_buffer = _get_cache_buf(
        f"top_k_mask_logits_row_states_{logits.device}",
        buffer_bytes,
        logits.device,
        zero_init=True,
    )

    # Note: row_states_buffer is zero-initialized on first allocation by _get_cache_buf
    # Kernel will reset arrival_counter to 0 at the end of each launch

    return get_sampling_module().top_k_mask_logits(
        logits, *_to_tensor_scalar_tuple(top_k), row_states_buffer
    )


@flashinfer_api(trace=chain_speculative_sampling_trace)
def chain_speculative_sampling(
    draft_probs,
    draft_token_ids,
    target_probs,
    maybe_output_accepted_token_num: Optional[torch.Tensor] = None,
    maybe_output_emitted_draft_token_num: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    r"""Fused-GPU kernel for speculative sampling for sequence generation (proposed in
    paper `Accelerating Large Language Model Decoding with Speculative Sampling <https://arxiv.org/pdf/2302.01318>`_),
    where the draft model generates a sequence(chain) of tokens for each request.

    Parameters
    ----------
    draft_probs: torch.Tensor
        The probability over vocabulary generated by draft model.
        Shape: ``(batch_size, num_speculate_tokens, vocab_size)``
    draft_token_ids: torch.Tensor
        The draft model's generated token indices.
        Shape: ``(batch_size, num_speculate_tokens)``
    target_probs: torch.Tensor
        The probability over vocabulary generated by target model.
        Compared to input :attr:`draft_probs`, the target model's probability has an additional
        slot at the end because the target model will generate one more token than the draft model.
        Shape: ``(batch_size, num_speculate_tokens + 1, vocab_size)``
    maybe_output_accepted_token_num: Optional[torch.Tensor]
        The number of tokens that can be accepted if each token is considered independently for each request.
        This metric does not consider the fact that rejection sampling will stop at the first token that does not
        satisfy the probability requirement r < p/q.
        It only evaluates the alignment of draft model and target model.
        Shape: ``(batch_size)``
        If specified, the number of accepted token number will be added to this tensor inplace. Default is ``None``.
    maybe_output_emitted_draft_token_num: Optional[torch.Tensor]
        The number of draft tokens that are finally emitted for each request. Does not include
        the bonus token. (Thus the total number of tokens sampled for a given request is
        output_emitted_draft_token_num + 1).
        Shape: ``(batch_size)``
        If specified, the number of emitted token number will be added to this tensor inplace. Default is ``None``.
    deterministic: bool
        Whether to use deterministic kernel implementation, default is ``True``.
    generator: Optional[torch.Generator]
        A random number generator for the operation.
    seed: Optional[Union[int, torch.Tensor]]
        Random seed value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. Common approaches include:
        - Incrementing offset by the number of random values consumed
        - Updating seed based on the number of calls to the operation
    offset: Optional[Union[int, torch.Tensor]]
        Random offset value for the sampling operation. Can be either an integer or a torch.Tensor.
        When provided as a torch.Tensor, it must be int64 or uint64 dtype, 1D, and length 1 or batch_size.
        Using torch.Tensor is required for CUDA graph compatibility.

        Warning: If you provide seed and offset explicitly, you are responsible for updating
        their values between calls to ensure different random samples. The offset should be
        incremented based on the number of random values consumed by the operation.

    Returns
    -------
    output_token_ids: torch.Tensor
        The output token indices verified by the target model, rejected samples are
        padded with ``-1``.
        Compared to input :attr:`draft_token_ids`, the output tensor has an additional
        token index at the end for the final token, if all previous tokens are accepted,
        another "bonus" token will be sampled from the target model's probability.
        Shape: (batch_size, num_speculate_tokens + 1)
    output_accepted_token_num: torch.Tensor
        The number of tokens that can be accepted if each token is considered independently for each request.
        This metric does not consider the fact that rejection sampling will stop at the first token that does not
        satisfy the probability requirement r < p/q.
        It only evaluates the alignment of draft model and target model.
        Shape: ``(batch_size)``
    output_emitted_draft_token_num: torch.Tensor
        The number of draft tokens that are finally emitted for each request. Does not include
        the bonus token. (Thus the total number of tokens sampled for a given request is
        output_emitted_draft_token_num + 1).
        Shape: ``(batch_size)``

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 1
    >>> num_speculate_tokens = 2
    >>> vocab_size = 4
    >>> draft_probs = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1]]]).to(0)
    >>> # token 2 was sampled from draft model for the first token, and
    >>> # token 1 was sampled from draft model for the second token
    >>> draft_token_ids = torch.tensor([[2, 1]], dtype=torch.int32).to(0)
    >>> target_probs = torch.tensor([[[0.0, 0.1, 0.6, 0.3], [1.0, 0.0, 0.0, 0.0], [0.7, 0.1, 0.1, 0.1]]]).to(0)
    >>> output_token_ids, output_accepted_token_num, output_emitted_draft_token_num =\
    ...     flashinfer.sampling.chain_speculative_sampling(
    ...         draft_probs, draft_token_ids, target_probs)
    >>> # the first token is accepted, the second token is rejected and sampled from the difference
    >>> # between the target model and the draft model, the third token is padded with -1
    >>> output_token_ids
    tensor([[ 2,  0, -1]], device='cuda:0', dtype=torch.int32)
    >>> output_accepted_token_num
    tensor([1], device='cuda:0')
    >>> output_emitted_draft_token_num
    tensor([1], device='cuda:0')
    """
    b = draft_probs.size(0)
    dev = draft_probs.device
    if maybe_output_accepted_token_num is None:
        output_accepted_token_num = torch.zeros(b, dtype=torch.int32, device=dev)
    else:
        output_accepted_token_num = maybe_output_accepted_token_num
    if maybe_output_emitted_draft_token_num is None:
        output_emitted_draft_token_num = torch.zeros(b, dtype=torch.int32, device=dev)
    else:
        output_emitted_draft_token_num = maybe_output_emitted_draft_token_num
    output_token_ids = get_sampling_module().chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        output_accepted_token_num,
        output_emitted_draft_token_num,
        deterministic,
        generator,
        seed,
        offset,
    )
    return output_token_ids, output_accepted_token_num, output_emitted_draft_token_num
