"""Device-independent Mamba bring-up path used by the MUSA backend.

This module intentionally uses regular PyTorch operators.  It is a correctness
scaffold for the first FlashInfer-MUSA port: it preserves the public SSU
contract while the native MUSA kernel is being developed.  It must not be used
as a performance implementation; the MUSA kernel will replace this function
behind the same dispatch boundary.
"""

from __future__ import annotations

from typing import Optional

import torch


def _philox_uniform(
    value: torch.Tensor, rand_seed: torch.Tensor, offset: int = 0, rounds: int = 10
) -> torch.Tensor:
    """Generate the requested Philox-4x32 counter stream for reference casts.

    The implementation is intentionally expressed with tensor operations so
    it follows the selected device.  ``offset`` indexes Philox counters; the reference returns the first
    word of each counter, rather than the grouped native SSU stream.
    """
    mask = 0xFFFFFFFF
    flat = torch.arange(value.numel(), device=value.device, dtype=torch.int64) + offset
    c0 = flat & mask
    c1 = (flat >> 32) & mask
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    seed = rand_seed.reshape(-1)[0].to(torch.int64)
    k0 = seed & mask
    k1 = (seed >> 32) & mask
    m0, m1 = 0xD2511F53, 0xCD9E8D57
    w0, w1 = 0x9E3779B9, 0xBB67AE85
    for _ in range(rounds):
        p0 = c0 * m0
        p1 = c2 * m1
        hi0, lo0 = (p0 >> 32) & mask, p0 & mask
        hi1, lo1 = (p1 >> 32) & mask, p1 & mask
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & mask,
            lo1 & mask,
            (hi0 ^ c3 ^ k1) & mask,
            lo0 & mask,
        )
        k0 = (k0 + w0) & mask
        k1 = (k1 + w1) & mask
    return ((c0.to(torch.float32) + 0.5) / 4294967296.0).reshape(value.shape)


def _musa_stream_is_capturing() -> bool:
    musa = getattr(torch, "musa", None)
    query = getattr(musa, "is_current_stream_capturing", None)
    if query is None:
        return False
    try:
        return bool(query())
    except Exception:
        return False


def _musa_mtp_capture_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    z,
    dt_bias,
    dt_softplus,
    state_batch_indices,
    dst_state_batch_indices,
    out,
    disable_state_update,
    intermediate_states_buffer,
    intermediate_state_indices,
    num_accepted_tokens,
):
    """Tensor-only MTP recurrence used while MUSA graph capture is active."""
    if x.dim() == 4:
        batch, steps = x.shape[:2]
        x4, dt4, b4, c4 = x, dt, B, C
        flatten = False
    elif x.dim() == 3:
        batch, steps = 1, x.shape[0]
        x4, dt4 = x.unsqueeze(0), dt.unsqueeze(0)
        b4, c4 = B.unsqueeze(0), C.unsqueeze(0)
        flatten = True
    else:
        raise RuntimeError("capture-safe MTP SSU requires a dense decode window")
    nheads, dim = x4.shape[-2], x4.shape[-1]
    ngroups = b4.shape[-2]
    if nheads % ngroups or state.dim() < 4:
        raise RuntimeError("unsupported capture-safe MTP state layout")
    ratio = nheads // ngroups
    fp = torch.float32
    a_h = (A if A.dim() == 3 else A.unsqueeze(0)).to(fp)
    b_h = b4.to(fp).repeat_interleave(ratio, dim=-2)
    c_h = c4.to(fp).repeat_interleave(ratio, dim=-2)
    x_h, dt_h = x4.to(fp), dt4.to(fp)
    if dt_bias is not None:
        dt_h = dt_h + dt_bias.to(fp).view(1, 1, -1)
    if dt_softplus:
        dt_h = torch.nn.functional.softplus(dt_h)
    if state_batch_indices is None:
        slots = torch.arange(batch, device=state.device, dtype=torch.long)
        slots = slots.remainder(state.shape[0])[:, None].expand(batch, steps)
    else:
        slots = state_batch_indices.to(torch.long)
        if slots.dim() == 1:
            slots = slots[:, None].expand(batch, steps)
    accepted = num_accepted_tokens.to(torch.long).reshape(-1)
    accepted = accepted.clamp(min=1, max=steps) - 1
    running = state.index_select(0, slots.gather(1, accepted[:, None]).squeeze(1)).to(fp)
    if dst_state_batch_indices is None:
        dst = slots
    else:
        dst = dst_state_batch_indices.to(torch.long)
        if dst.dim() == 1:
            dst = dst[:, None].expand(batch, steps)
    dst = dst.clamp(min=0, max=state.shape[0] - 1)
    outputs = []
    for token in range(steps):
        dt_t, x_t = dt_h[:, token], x_h[:, token]
        running = running * torch.exp(a_h[None] * dt_t[:, :, None, None])
        running = running + (dt_t[:, :, None, None] * x_t[:, :, :, None]) * b_h[:, token, :, None, :]
        y = (c_h[:, token, :, None, :] * running).sum(dim=-1)
        if D is not None:
            y = y + D.to(fp).view(1, nheads, dim) * x_t
        if z is not None:
            z_t = z[:, token].to(fp) if z.dim() == 4 else z[token].to(fp)
            y = y * z_t * torch.sigmoid(z_t)
        outputs.append(y)
        if not disable_state_update:
            state.index_copy_(0, dst[:, token], running.to(state.dtype))
    result = torch.stack(outputs, dim=1).to(out.dtype)
    if flatten:
        result = result.squeeze(0)
    out.copy_(result)
    return out


def _stochastic_cast(
    value: torch.Tensor,
    target_dtype: torch.dtype,
    rand_seed: Optional[torch.Tensor],
    philox_rounds: int,
    rng_offset: int,
) -> torch.Tensor:
    if (
        rand_seed is None
        or philox_rounds <= 0
        or target_dtype not in (torch.float16, torch.bfloat16)
    ):
        return value.to(target_dtype)
    mantissa_bits = 10 if target_dtype == torch.float16 else 7
    magnitude = value.abs().clamp_min(torch.finfo(torch.float32).tiny)
    exponent = torch.floor(torch.log2(magnitude))
    step = torch.pow(2.0, exponent - mantissa_bits)
    lower = torch.floor(value / step) * step
    probability = ((value - lower) / step).clamp(0, 1)
    random = _philox_uniform(value, rand_seed, rng_offset, philox_rounds)
    return torch.where(random < probability, lower + step, lower).to(target_dtype)


def _stochastic_round_integer(
    value: torch.Tensor,
    rand_seed: Optional[torch.Tensor],
    philox_rounds: int,
    rng_offset: int,
) -> torch.Tensor:
    if rand_seed is None or philox_rounds <= 0:
        return value.round()
    lower = torch.floor(value)
    random = _philox_uniform(value, rand_seed, rng_offset, philox_rounds)
    return lower + (random < (value - lower)).to(value.dtype)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    # Match the selective-scan kernels' numerically stable branch at large x.
    return torch.where(x <= 20, torch.nn.functional.softplus(x), x)


def _as_head_dim(tensor: Optional[torch.Tensor], nheads: int, dim: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.dim() == 1:
        return tensor.view(1, -1).expand(nheads, dim)
    if tensor.dim() == 2 and tensor.shape == (nheads, 1):
        return tensor.expand(nheads, dim)
    return tensor


def _index_for(
    indices: Optional[torch.Tensor], batch: int, token: int, default: int
) -> int:
    if indices is None:
        return default
    if indices.dim() == 1:
        return int(indices[batch].item())
    if token < 0 or token >= indices.shape[1]:
        raise IndexError(
            f"state index token {token} is outside the metadata width "
            f"{indices.shape[1]}"
        )
    return int(indices[batch, token].item())


def selective_state_update_musa_reference(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    out: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    state_scale: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    philox_rounds: int,
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reference SSU implementation for MUSA bring-up.

    The function keeps recurrence arithmetic in fp32 and mirrors the public
    state/index/replay contract.  Native MUSA kernels will replace this slow
    reference implementation behind the same boundary.
    """
    # MTP's accepted-count metadata is device-resident and the same recurrence
    # is valid in eager mode. Use the tensor-only implementation whenever MTP
    # is selected; this prevents a graph capture from falling into the Python
    # reference (which necessarily reads metadata to the host).
    if num_accepted_tokens is not None:
        return _musa_mtp_capture_update(
            state, x, dt, A, B, C, D, z, dt_bias, dt_softplus,
            state_batch_indices, dst_state_batch_indices, out,
            disable_state_update, intermediate_states_buffer,
            intermediate_state_indices, num_accepted_tokens,
        )
    quantized_state = state.dtype in (torch.int8, torch.int16, torch.float8_e4m3fn)
    if state.dtype not in (
        torch.int8, torch.int16, torch.float8_e4m3fn,
        torch.float16, torch.bfloat16, torch.float32,
    ):
        raise NotImplementedError("unsupported MUSA SSU state dtype")
    if D is not None and D.dtype != dt.dtype:
        raise ValueError("D must have the same dtype as dt")
    if quantized_state and state_scale is None:
        raise ValueError("quantized MUSA SSU state requires state_scale")
    if not quantized_state and state_scale is not None:
        raise ValueError("state_scale is only valid for quantized state")
    if state_scale is not None and state_scale.dim() == 4 and state_scale.shape[-1] == 1:
        state_scale = state_scale.squeeze(-1)
    if (
        intermediate_state_scales is not None
        and intermediate_state_scales.dim() == 5
        and intermediate_state_scales.shape[-1] == 1
    ):
        intermediate_state_scales = intermediate_state_scales.squeeze(-1)
    if (
        state_batch_indices is not None
        and intermediate_state_indices is not None
        and state_batch_indices.dtype != intermediate_state_indices.dtype
    ):
        raise ValueError("state and intermediate index tensors must have the same dtype")

    if state.dim() != 4:
        raise ValueError(f"state must be [slots, heads, dim, dstate], got {state.shape}")
    slots, nheads, dim, dstate = state.shape

    def _counter_uniform(value: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return _philox_uniform(value, rand_seed, offset, philox_rounds)

    def cast_state(
        value: torch.Tensor,
        target_dtype: Optional[torch.dtype] = None,
        rng_offset: int = 0,
    ) -> torch.Tensor:
        target_dtype = state.dtype if target_dtype is None else target_dtype
        if (
            rand_seed is None
            or philox_rounds <= 0
            or target_dtype not in (torch.float16, torch.bfloat16)
        ):
            return value.to(target_dtype)
        # Reference stochastic cast.  Native kernels will use the exact
        # device Philox sequence; this preserves the public seed/rounds
        # contract while keeping the bring-up path device-independent.
        mantissa_bits = 10 if target_dtype == torch.float16 else 7
        magnitude = value.abs().clamp_min(torch.finfo(torch.float32).tiny)
        exponent = torch.floor(torch.log2(magnitude))
        step = torch.pow(2.0, exponent - mantissa_bits)
        lower = torch.floor(value / step) * step
        probability = ((value - lower) / step).clamp(0, 1)
        random = _counter_uniform(value, rng_offset)
        rounded = torch.where(random < probability, lower + step, lower)
        return rounded.to(target_dtype)

    def round_integer(value: torch.Tensor, rng_offset: int = 0) -> torch.Tensor:
        if rand_seed is None or philox_rounds <= 0:
            return value.round()
        random = _counter_uniform(value, rng_offset)
        lower = torch.floor(value)
        return lower + (random < (value - lower)).to(value.dtype)

    def read_state(state_slot: int) -> torch.Tensor:
        if state_slot == pad_slot_id:
            return torch.zeros((nheads, dim, dstate), dtype=torch.float32, device=state.device)
        if state_slot < 0 or state_slot >= slots:
            raise IndexError(f"state slot {state_slot} is outside [0, {slots})")
        value = state[state_slot].to(torch.float32).clone()
        if quantized_state:
            value = value * state_scale[state_slot].to(torch.float32)[..., None]
        return value

    def write_state(state_slot: int, value: torch.Tensor) -> None:
        if disable_state_update or state_slot == pad_slot_id:
            return
        if state_slot < 0 or state_slot >= slots:
            raise IndexError(f"state slot {state_slot} is outside [0, {slots})")
        if quantized_state:
            amax = value.abs().amax(dim=-1)
            qmax = 127 if state.dtype == torch.int8 else 32767 if state.dtype == torch.int16 else 448
            scale = torch.where(amax == 0, torch.ones_like(amax), amax / qmax)
            quantized = value / scale[..., None]
            if state.dtype == torch.float8_e4m3fn:
                state[state_slot].copy_(quantized.to(state.dtype))
            else:
                state[state_slot].copy_(
                    round_integer(value / scale[..., None], state_slot * value.numel())
                    .clamp(-qmax - 1, qmax).to(state.dtype)
                )
            state_scale[state_slot].copy_(scale.to(state_scale.dtype))
        else:
            state[state_slot].copy_(cast_state(value, rng_offset=state_slot * value.numel()))

    is_varlen = cu_seqlens is not None and x.dim() == 3 and dt.dim() == 3
    is_mtp = x.dim() == 4
    if is_varlen:
        if cu_seqlens.dim() != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be a 1D int32 or int64 tensor")
        cu_values = [int(v) for v in cu_seqlens.detach().cpu().tolist()]
        if not cu_values or cu_values[0] != 0:
            raise ValueError("cu_seqlens must start at zero")
        if any(
            end < start for start, end in zip(cu_values, cu_values[1:], strict=False)
        ):
            raise ValueError("cu_seqlens must be monotonically nondecreasing")
        if cu_values[-1] != x.shape[0]:
            raise ValueError(
                "cu_seqlens final value must equal the packed token count "
                f"{x.shape[0]}, got {cu_values[-1]}"
            )
        batch = int(cu_seqlens.numel() - 1)
        steps = [
            (cu_values[b], cu_values[b + 1])
            for b in range(batch)
        ]
    elif is_mtp:
        batch, nsteps = x.shape[:2]
        steps = [(0, nsteps) for _ in range(batch)]
    else:
        batch = x.shape[0]
        steps = [(0, 1) for _ in range(batch)]

    is_spec_decoding = num_accepted_tokens is not None
    if is_spec_decoding:
        if not (is_varlen or is_mtp):
            raise ValueError("num_accepted_tokens requires varlen or MTP input")
        if state_batch_indices is None or state_batch_indices.dim() != 2:
            raise ValueError(
                "speculative varlen state_batch_indices must be a 2D tensor"
            )
        if num_accepted_tokens.dim() != 1 or num_accepted_tokens.shape[0] != batch:
            raise ValueError(
                "num_accepted_tokens must have one entry per packed sequence"
            )
        if num_accepted_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("num_accepted_tokens must be int32 or int64")
        accepted_values = [
            max(int(v), 0)
            for v in num_accepted_tokens.detach().cpu().tolist()
        ]
        if state_batch_indices.shape[0] < batch:
            raise ValueError("state_batch_indices has fewer rows than cu_seqlens")
        max_len = max((end - start for start, end in steps), default=0)
        required_width = max(max_len, max(accepted_values, default=0))
        if state_batch_indices.shape[1] < required_width:
            raise ValueError(
                "state_batch_indices metadata width is smaller than the packed "
                f"sequence/accepted-token requirement ({required_width})"
            )
        if dst_state_batch_indices is not None:
            if dst_state_batch_indices.dim() != 2:
                raise ValueError(
                    "speculative varlen dst_state_batch_indices must be 2D"
                )
            if (
                dst_state_batch_indices.shape[0] < batch
                or dst_state_batch_indices.shape[1] < max_len
            ):
                raise ValueError(
                    "dst_state_batch_indices metadata is smaller than the packed "
                    "sequences"
                )
        else:
            # Upstream Triton uses source slots as destinations by default in
            # speculative mode; preserve that cache ownership contract.
            dst_state_batch_indices = state_batch_indices
    else:
        accepted_values = []
        if is_varlen and state_batch_indices is not None:
            if state_batch_indices.dim() not in (1, 2):
                raise ValueError("state_batch_indices must be 1D or 2D")
            if state_batch_indices.shape[0] < batch:
                raise ValueError("state_batch_indices has fewer rows than cu_seqlens")
        if is_varlen and dst_state_batch_indices is not None:
            if dst_state_batch_indices.dim() not in (1, 2):
                raise ValueError("dst_state_batch_indices must be 1D or 2D")
            if dst_state_batch_indices.shape[0] < batch:
                raise ValueError("dst_state_batch_indices has fewer rows than cu_seqlens")

    if is_varlen:
        if B.dim() != 3 or C.dim() != 3:
            raise ValueError("varlen B/C must be [tokens, groups, dstate]")
        ngroups = B.shape[1]
    elif is_mtp:
        if B.dim() != 4 or C.dim() != 4:
            raise ValueError("MTP B/C must be [batch, T, groups, dstate]")
        ngroups = B.shape[2]
    else:
        if B.dim() != 3 or C.dim() != 3:
            raise ValueError("single-token B/C must be [batch, groups, dstate]")
        ngroups = B.shape[1]
    if nheads % ngroups != 0:
        raise ValueError("nheads must be divisible by ngroups")
    group_ratio = nheads // ngroups

    A_h = A if A.dim() == 3 else A.unsqueeze(0)
    if A_h.shape != (nheads, dim, dstate):
        raise ValueError(f"A must be [{nheads}, {dim}, {dstate}], got {A.shape}")
    D_h = _as_head_dim(D, nheads, dim)
    bias_h = _as_head_dim(dt_bias, nheads, dim)
    if D_h is not None and D_h.shape != (nheads, dim):
        raise ValueError(f"D must broadcast to [{nheads}, {dim}], got {D.shape}")
    if bias_h is not None and bias_h.shape != (nheads, dim):
        raise ValueError(
            f"dt_bias must broadcast to [{nheads}, {dim}], got {dt_bias.shape}"
        )

    def update_one(
        batch_idx: int,
        token_idx: int,
        state_slot: int,
        running_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        running = read_state(state_slot) if running_override is None else running_override

        if is_varlen:
            x_t = x[token_idx].to(torch.float32)
            dt_t = dt[token_idx].to(torch.float32)
            b_t = B[token_idx].to(torch.float32)
            c_t = C[token_idx].to(torch.float32)
            z_t = z[token_idx].to(torch.float32) if z is not None else None
        elif is_mtp:
            x_t = x[batch_idx, token_idx].to(torch.float32)
            dt_t = dt[batch_idx, token_idx].to(torch.float32)
            b_t = B[batch_idx, token_idx].to(torch.float32)
            c_t = C[batch_idx, token_idx].to(torch.float32)
            z_t = z[batch_idx, token_idx].to(torch.float32) if z is not None else None
        else:
            x_t = x[batch_idx].to(torch.float32)
            dt_t = dt[batch_idx].to(torch.float32)
            b_t = B[batch_idx].to(torch.float32)
            c_t = C[batch_idx].to(torch.float32)
            z_t = z[batch_idx].to(torch.float32) if z is not None else None

        b_h = b_t.repeat_interleave(group_ratio, dim=0)
        c_h = c_t.repeat_interleave(group_ratio, dim=0)
        if bias_h is not None:
            dt_t = dt_t + bias_h
        if dt_softplus:
            dt_t = _softplus(dt_t)
        if dt_t.dim() == 1:
            dt_state = dt_t[:, None, None]
            dt_x = dt_t[:, None]
        else:
            dt_state = dt_t[:, :, None]
            dt_x = dt_t
        running.copy_(
            running * torch.exp(A_h * dt_state)
            + (dt_x * x_t)[:, :, None] * b_h[:, None, :]
        )
        y_t = torch.sum(c_h[:, None, :] * running, dim=-1)
        if D_h is not None:
            y_t = y_t + D_h.to(torch.float32) * x_t
        if z_t is not None:
            y_t = y_t * z_t * torch.sigmoid(z_t)

        if is_varlen:
            out[token_idx].copy_(y_t.to(out.dtype))
        elif is_mtp:
            out[batch_idx, token_idx].copy_(y_t.to(out.dtype))
        else:
            out[batch_idx].copy_(y_t.to(out.dtype))
        return running

    for b, (start, end) in enumerate(steps):
        if end <= start:
            # Empty packed sequences have no initial read or final write.
            continue
        accepted = (
            max(accepted_values[b] - 1, 0)
            if is_spec_decoding
            else 0
        )
        read_slot = _index_for(state_batch_indices, b, accepted, b)
        running = None
        for token in range(end - start):
            token_idx = start + token if is_varlen else token
            running = update_one(
                b,
                token_idx,
                read_slot,
                running_override=running if token > 0 else None,
            )
            if is_spec_decoding or (not is_mtp and not is_varlen):
                write_slot = _index_for(dst_state_batch_indices, b, token, read_slot)
                write_state(write_slot, running)
                read_slot = write_slot
            if intermediate_states_buffer is not None:
                cache_slot = (
                    int(intermediate_state_indices[b].item())
                    if intermediate_state_indices is not None
                    else b
                )
                if intermediate_states_buffer.dtype == torch.int16:
                    if intermediate_state_scales is None:
                        raise ValueError("int16 intermediate state requires scales")
                    amax = running.abs().amax(dim=-1)
                    scale = torch.where(amax == 0, torch.ones_like(amax), amax / 32767)
                    intermediate_states_buffer[cache_slot, token].copy_(
                        round_integer(
                            running / scale[..., None],
                            (cache_slot * cache_steps + token) * running.numel(),
                        ).clamp(-32768, 32767).to(torch.int16)
                    )
                    intermediate_state_scales[cache_slot, token].copy_(scale.to(intermediate_state_scales.dtype))
                else:
                    intermediate_states_buffer[cache_slot, token].copy_(
                        cast_state(
                            running,
                            intermediate_states_buffer.dtype,
                            (cache_slot * cache_steps + token) * running.numel(),
                        )
                    )
        if not disable_state_update and not is_mtp and is_varlen and not is_spec_decoding:
            final_slot = _index_for(dst_state_batch_indices, b, 0, read_slot)
            write_state(final_slot, running)
        elif (
            not disable_state_update
            and is_mtp
            and dst_state_batch_indices is None
            and intermediate_states_buffer is None
        ):
            write_state(read_slot, running)
    return out


def ssd_combined_fwd_musa_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_final_states: bool = True,
    checkpoint_token_indices: Optional[torch.Tensor] = None,
    checkpoint_state_slots: Optional[torch.Tensor] = None,
    checkpoint_states: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference Mamba2 SSD forward for the initial MUSA port.

    This implements the same recurrence as SSDCombined without materializing
    the chunk matrices.  It is intentionally a correctness path; a native
    MUSA implementation will replace it after the API and reference tests are
    established.
    """
    if x.dim() != 4 or dt.dim() != 3:
        raise ValueError("x must be [batch, seqlen, heads, headdim] and dt [batch, seqlen, heads]")
    batch, seqlen, nheads, headdim = x.shape
    if dt.shape != (batch, seqlen, nheads):
        raise ValueError("dt shape does not match x")
    if B.shape != C.shape or B.shape[:2] != (batch, seqlen):
        raise ValueError("B and C must have shape [batch, seqlen, groups, dstate]")
    ngroups, dstate = B.shape[2:]
    if nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    if A.shape != (nheads,):
        raise ValueError(f"A must have shape [{nheads}]")
    if z is not None and z.shape != x.shape:
        raise ValueError("z must have the same shape as x")

    state_dtype = initial_states.dtype if initial_states is not None else torch.bfloat16
    if state_dtype not in (
        torch.int8, torch.int16, torch.float8_e4m3fn,
        torch.float16, torch.bfloat16, torch.float32,
    ):
        raise NotImplementedError("unsupported MUSA SSD state dtype")
    if initial_states is None:
        num_sequences = batch
        if seq_idx is not None:
            num_sequences = max(num_sequences, int(seq_idx.max().item()) + 1)
        state = torch.zeros(
            batch, nheads, headdim, dstate, dtype=torch.float32, device=x.device
        )
        initial = None
    else:
        if initial_states.dim() != 4 or initial_states.shape[1:] != (
            nheads,
            headdim,
            dstate,
        ):
            raise ValueError("initial_states must be [num_sequences, heads, headdim, dstate]")
        num_sequences = initial_states.shape[0]
        initial = initial_states.to(torch.float32)
        state = initial[:batch].clone()
        if state.shape[0] != batch:
            state = torch.zeros(
                batch, nheads, headdim, dstate, dtype=torch.float32, device=x.device
            )

    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape:
        raise ValueError("out must have the same shape as x")
    if checkpoint_states is not None:
        if checkpoint_token_indices is None or checkpoint_state_slots is None:
            raise ValueError("checkpoint token indices, slots, and states are required together")
        if checkpoint_token_indices.shape != (batch,) or checkpoint_state_slots.shape != (batch,):
            raise ValueError("checkpoint metadata must have one entry per batch sequence")
        if checkpoint_states.ndim != 4 or checkpoint_states.shape[1:] != (nheads, headdim, dstate):
            raise ValueError("checkpoint_states has incompatible state shape")
        if torch.any(checkpoint_state_slots >= checkpoint_states.shape[0]):
            raise ValueError("checkpoint_state_slots contains an out-of-bounds slot")

    bias = None if dt_bias is None else dt_bias.reshape(1, 1, nheads).to(torch.float32)
    d_head = None
    if D is not None:
        d_head = D.to(torch.float32)
        if d_head.dim() == 1:
            d_head = d_head[:, None].expand(nheads, headdim)
        if d_head.shape != (nheads, headdim):
            raise ValueError("D must have shape [heads] or [heads, headdim]")

    # Direct recurrence over tokens.  All state arithmetic remains fp32; only
    # the externally visible output and final state use the requested dtypes.
    final = torch.empty(
        num_sequences, nheads, headdim, dstate, dtype=state_dtype, device=x.device
    )
    state_scales = (
        torch.empty(num_sequences, nheads, headdim, device=x.device, dtype=torch.float32)
        if state_dtype in (torch.int8, torch.int16, torch.float8_e4m3fn)
        else None
    )
    seen = torch.zeros(num_sequences, dtype=torch.bool, device=x.device)
    previous_seq = None
    ratio = nheads // ngroups
    dt_f = dt.to(torch.float32)
    for token in range(seqlen):
        if seq_idx is not None:
            ids = seq_idx[:, token].to(torch.int64)
            if ids.numel() != batch:
                raise ValueError("seq_idx must have shape [batch, seqlen]")
            changed = torch.ones(batch, dtype=torch.bool, device=x.device)
            if previous_seq is not None:
                changed = ids != previous_seq
            if initial is not None:
                replacement = initial.index_select(0, ids.clamp_min(0))
                state = torch.where(changed[:, None, None, None], replacement, state)
            else:
                state = torch.where(
                    changed[:, None, None, None],
                    torch.zeros_like(state),
                    state,
                )
            previous_seq = ids

        delta = dt_f[:, token]
        if bias is not None:
            delta = delta + bias[:, 0]
        if dt_softplus:
            delta = _softplus(delta)
        delta = delta.clamp(dt_limit[0], dt_limit[1])
        decay = torch.exp(A.to(torch.float32)[None, :, None, None] * delta[:, :, None, None])
        state = state * decay
        b_h = B[:, token].to(torch.float32).repeat_interleave(ratio, dim=1)
        c_h = C[:, token].to(torch.float32).repeat_interleave(ratio, dim=1)
        state += (
            delta[:, :, None, None]
            * x[:, token].to(torch.float32)[:, :, :, None]
            * b_h[:, :, None, :]
        )
        y = torch.empty(batch, nheads, headdim, dtype=torch.float32, device=x.device)
        y.copy_(torch.sum(c_h[:, :, None, :] * state, dim=-1))
        if d_head is not None:
            y = y + x[:, token].to(torch.float32) * d_head[None]
        if z is not None:
            z_t = z[:, token].to(torch.float32)
            y = y * z_t * torch.sigmoid(z_t)
        out[:, token].copy_(y.to(out.dtype))

        if checkpoint_states is not None:
            for sequence in range(batch):
                if int(checkpoint_token_indices[sequence].item()) == token + 1:
                    slot = int(checkpoint_state_slots[sequence].item())
                    if slot >= 0:
                        checkpoint_states[slot].copy_(state[sequence].to(checkpoint_states.dtype))

        if seq_idx is None:
            if state_scales is None:
                final.copy_(state.to(state_dtype))
            else:
                qmax = (
                    127
                    if state_dtype == torch.int8
                    else 32767
                    if state_dtype == torch.int16
                    else 448
                )
                state_amax = state.abs().amax(dim=-1)
                scale = torch.where(
                    state_amax == 0, torch.ones_like(state_amax), state_amax / qmax
                )
                final.copy_((state / scale[..., None]).to(state_dtype))
                state_scales.copy_(scale)
        else:
            for sequence in torch.unique(ids).tolist():
                selected = state[ids == sequence][0]
                if state_scales is None:
                    final[sequence].copy_(selected.to(state_dtype))
                else:
                    qmax = (
                        127
                        if state_dtype == torch.int8
                        else 32767
                        if state_dtype == torch.int16
                        else 448
                    )
                    selected_amax = selected.abs().amax(dim=-1)
                    scale = torch.where(
                        selected_amax == 0,
                        torch.ones_like(selected_amax),
                        selected_amax / qmax,
                    )
                    final[sequence].copy_((selected / scale[..., None]).to(state_dtype))
                    state_scales[sequence].copy_(scale)
                seen[sequence] = True

    if seq_idx is not None and not bool(seen.all()):
        missing = (~seen).nonzero(as_tuple=False).flatten().tolist()
        if initial is not None:
            final[missing] = initial[missing].to(state_dtype)
        else:
            final[missing] = 0
    return out, final if return_final_states else None


def ssd_combined_fwd_varlen_musa_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor,
    cu_chunk_seqlens: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    seq_idx: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    return_intermediate_states: bool = False,
    state_dtype: Optional[torch.dtype] = None,
    checkpoint_token_indices: Optional[torch.Tensor] = None,
    checkpoint_state_slots: Optional[torch.Tensor] = None,
    checkpoint_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Packed/varlen SSD API matching vLLM's Mamba2 prefill contract."""
    if x.dim() != 3 or dt.dim() != 2 or B.dim() != 3 or C.dim() != 3:
        raise ValueError("varlen SSD expects packed x/dt/B/C tensors")
    tokens, nheads, headdim = x.shape
    nchunks = cu_chunk_seqlens.numel() - 1
    if dt.shape != (tokens, nheads) or B.shape != C.shape:
        raise ValueError("packed SSD tensor shapes do not match")
    ngroups, dstate = B.shape[1:]
    if nheads % ngroups or A.shape != (nheads,):
        raise ValueError("A/groups are incompatible with the packed head count")
    if cu_seqlens.numel() != last_chunk_indices.numel() + 1:
        raise ValueError("cu_seqlens and last_chunk_indices do not describe the same batch")
    if seq_idx.numel() != nchunks:
        raise ValueError("seq_idx must contain one sequence id per physical chunk")
    if torch.any(cu_chunk_seqlens[1:] < cu_chunk_seqlens[:-1]) or int(cu_chunk_seqlens[-1]) != tokens:
        raise ValueError("cu_chunk_seqlens must be monotonic and end at x.shape[0]")

    if out is None:
        out = torch.empty_like(x)
    if initial_states is not None:
        if initial_states.dim() != 4 or initial_states.shape[1:] != (
            nheads,
            headdim,
            dstate,
        ):
            raise ValueError("initial_states must be [num_sequences, heads, headdim, dstate]")
        state_dtype = state_dtype or initial_states.dtype
        initial = initial_states.to(torch.float32)
    else:
        state_dtype = state_dtype or C.dtype
        initial = None
    checkpoint_args = (checkpoint_token_indices, checkpoint_state_slots, checkpoint_states)
    if any(value is not None for value in checkpoint_args) and not all(
        value is not None for value in checkpoint_args
    ):
        raise ValueError("varlen SSD checkpoint arguments must be provided together")
    if checkpoint_states is not None:
        if checkpoint_token_indices.shape != (last_chunk_indices.numel(),):
            raise ValueError("checkpoint token metadata must have one entry per sequence")
        if checkpoint_state_slots.shape != checkpoint_token_indices.shape:
            raise ValueError("checkpoint state slots must match token metadata")
        if checkpoint_states.ndim != 4 or checkpoint_states.shape[1:] != (
            nheads,
            headdim,
            dstate,
        ):
            raise ValueError("checkpoint_states has incompatible varlen state shape")
        if torch.any(checkpoint_state_slots >= checkpoint_states.shape[0]):
            raise ValueError("checkpoint_state_slots contains an out-of-bounds slot")

    if D is not None:
        D_f = D.to(torch.float32)
        if D_f.dim() == 1:
            D_f = D_f[:, None].expand(nheads, headdim)
        if D_f.shape != (nheads, headdim):
            raise ValueError("D must have shape [heads] or [heads, headdim]")
    else:
        D_f = None
    bias = None if dt_bias is None else dt_bias.to(torch.float32)
    if bias is not None and bias.shape not in ((nheads,), (nheads, headdim)):
        raise ValueError("dt_bias must have shape [heads] or [heads, headdim]")
    ratio = nheads // ngroups
    A_f = A.to(torch.float32)
    if A_f.dim() == 1:
        A_state = A_f[:, None, None]
    elif A_f.shape == (nheads, headdim, dstate):
        A_state = A_f
    else:
        raise ValueError("A must have shape [heads] or [heads, headdim, dstate]")
    states = torch.empty(
        nchunks, nheads, headdim, dstate, dtype=state_dtype, device=x.device
    )
    state_by_sequence: dict[int, torch.Tensor] = {}
    seen_sequences: set[int] = set()
    sequence_token_counts: dict[int, int] = {}

    for chunk in range(nchunks):
        sequence = int(seq_idx[chunk].item())
        start = int(cu_chunk_seqlens[chunk].item())
        end = int(cu_chunk_seqlens[chunk + 1].item())
        if end - start > chunk_size:
            raise ValueError("a physical chunk exceeds chunk_size")
        if sequence not in seen_sequences:
            if initial is None:
                running = torch.zeros(
                    nheads, headdim, dstate, dtype=torch.float32, device=x.device
                )
            else:
                # Keep the recurrent accumulator in fp32 across a sequence.
                # The Triton path materializes the requested state dtype at
                # chunk boundaries; rounding every token would accumulate a
                # different error profile and make final/intermediate states
                # disagree with the native provider.
                running = initial[sequence].clone()
            seen_sequences.add(sequence)
        else:
            running = state_by_sequence[sequence]

        for token in range(start, end):
            delta = dt[token].to(torch.float32)
            if bias is not None:
                delta = delta + bias
            if dt_softplus:
                delta = _softplus(delta)
            delta = delta.clamp(dt_limit[0], dt_limit[1])
            x_t = x[token].to(torch.float32)
            b_h = B[token].to(torch.float32).repeat_interleave(ratio, dim=0)
            c_h = C[token].to(torch.float32).repeat_interleave(ratio, dim=0)
            if delta.dim() == 1:
                delta_state = delta[:, None, None]
                delta_x = delta[:, None]
            else:
                delta_state = delta[:, :, None]
                delta_x = delta
            running.copy_(
                running * torch.exp(A_state * delta_state)
                + (delta_x * x_t)[:, :, None] * b_h[:, None, :]
            )
            y = torch.sum(c_h[:, None, :] * running, dim=-1)
            if D_f is not None:
                y = y + D_f * x_t
            if z is not None:
                z_t = z[token].to(torch.float32)
                y = y * z_t * torch.sigmoid(z_t)
            out[token].copy_(y.to(out.dtype))
            sequence_token_counts[sequence] = sequence_token_counts.get(sequence, 0) + 1
            if checkpoint_states is not None and sequence < checkpoint_token_indices.numel():
                if sequence_token_counts[sequence] == int(checkpoint_token_indices[sequence].item()):
                    checkpoint_slot = int(checkpoint_state_slots[sequence].item())
                    if checkpoint_slot >= 0:
                        if checkpoint_slot >= checkpoint_states.shape[0]:
                            raise IndexError("varlen checkpoint state slot is out of bounds")
                        checkpoint_states[checkpoint_slot].copy_(running.to(checkpoint_states.dtype))
        state_by_sequence[sequence] = running
        states[chunk].copy_(running.to(state_dtype))

    if return_intermediate_states:
        return states
    return states.index_select(0, last_chunk_indices.to(torch.int64))


def replayssm_materialize_musa_reference(
    dependency_inputs: list[torch.Tensor],
    dependency_outputs: list[torch.Tensor],
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    ring_start: torch.Tensor,
    replay_prefix_len: torch.Tensor,
    active_request_indices: torch.Tensor,
    *,
    heads_per_group: int,
    max_window: int,
    ring_buffer_len: int,
    pad_slot_id: int,
    rand_seed: Optional[torch.Tensor],
    philox_rounds: int,
    state_scale_ptrs: torch.Tensor,
    state_dtype: torch.dtype,
) -> None:
    """Reference ReplaySSM materialization for MUSA.

    Pointer tables are opaque on Python, so the public MUSA path requires the
    same dependency tensors that the CUDA graph path keeps alive. Inputs are
    ordered as ``[x_cache, B_cache, dt_cache, A]``. Outputs contain one state
    tensor per layer; quantized state additionally appends one scale tensor per
    layer, matching the MUSA dependency-anchor convention.
    """
    if len(dependency_inputs) % 4 != 0 or not dependency_outputs:
        raise ValueError("MUSA ReplaySSM dependency lists have invalid layout")
    layers = len(dependency_inputs) // 4
    if len(dependency_outputs) not in (layers, 2 * layers):
        raise ValueError(
            "ReplaySSM requires one state output per layer; quantized state "
            "also requires one scale output per layer"
        )
    if src_slots.shape[0] != layers or dst_slots.shape != src_slots.shape:
        raise ValueError("slot tables must have shape [layers, batch]")
    batch = src_slots.shape[1]
    active_values = active_request_indices.to(torch.int64).flatten().tolist()
    active = []
    for index in active_values:
        if index < 0:
            break
        active.append(index)
    if active and max(active) >= batch:
        raise ValueError("active request index is outside the slot table")
    for layer in range(layers):
        x_cache = dependency_inputs[layer]
        b_cache = dependency_inputs[layers + layer]
        dt_cache = dependency_inputs[2 * layers + layer]
        a = dependency_inputs[3 * layers + layer].to(torch.float32)
        state = dependency_outputs[layer]
        state_scale = (
            dependency_outputs[layers + layer] if len(dependency_outputs) == 2 * layers else None
        )
        if x_cache.dim() < 4 or b_cache.dim() < 4 or dt_cache.dim() < 3:
            raise ValueError("ReplaySSM cache tensors have invalid rank")
        heads = state.shape[1]
        dim, dstate = state.shape[-2:]
        groups = b_cache.shape[1]
        if heads != groups * heads_per_group or a.shape[0] != heads:
            raise ValueError("ReplaySSM head/group dimensions are inconsistent")
        for request in active:
            src = int(src_slots[layer, request].item())
            dst = int(dst_slots[layer, request].item())
            if src == pad_slot_id or dst == pad_slot_id:
                continue
            start = int(ring_start[request].item())
            prefix = int(replay_prefix_len[request].item())
            if not (0 <= start < ring_buffer_len and 0 <= prefix <= max_window):
                raise ValueError("ReplaySSM ring metadata is outside its declared bounds")
            if prefix == 0:
                state[dst].copy_(state[src])
                if state_scale is not None:
                    state_scale[dst].copy_(state_scale[src])
                continue
            running = state[src].to(torch.float32).clone()
            if state.dtype in (torch.int8, torch.int16, torch.float8_e4m3fn):
                if state_scale is None:
                    raise ValueError(
                        "quantized MUSA ReplaySSM state requires scale backing tensors "
                        "in dependency_outputs"
                    )
                scale = state_scale[src].to(torch.float32)
                if scale.shape[-1] == 1:
                    scale = scale.squeeze(-1)
                running = running * scale[..., None]
            for offset in range(prefix):
                ring = (start + offset) % ring_buffer_len
                for head in range(heads):
                    group = head // heads_per_group
                    delta = dt_cache[src, head, ring].to(torch.float32)
                    decay = torch.exp(a[head] * delta)
                    x_t = x_cache[src, head, ring].to(torch.float32)
                    b_t = b_cache[src, group, ring].to(torch.float32)
                    running[head] = running[head] * decay
                    running[head] += (delta * x_t)[:, None] * b_t[None, :]
            if state.dtype in (torch.int8, torch.int16, torch.float8_e4m3fn):
                if state_scale is None:
                    raise ValueError("quantized MUSA ReplaySSM state requires scales")
                qmax = 127 if state.dtype == torch.int8 else 32767
                if state.dtype == torch.float8_e4m3fn:
                    qmax = 448
                amax = running.abs().amax(dim=-1)
                scale = torch.where(amax == 0, torch.ones_like(amax), amax / qmax)
                quantized = running / scale[..., None]
                if state.dtype == torch.float8_e4m3fn:
                    state[dst].copy_(quantized.to(state.dtype))
                else:
                    quantized = _stochastic_round_integer(
                        quantized,
                        rand_seed,
                        philox_rounds,
                        layer * state.numel() + dst * running.numel(),
                    )
                    state[dst].copy_(quantized.clamp(-qmax - 1, qmax).to(state.dtype))
                if state_scale[dst].shape[-1] == 1:
                    state_scale[dst].copy_(scale.to(state_scale.dtype).unsqueeze(-1))
                else:
                    state_scale[dst].copy_(scale.to(state_scale.dtype))
            else:
                state[dst].copy_(
                    _stochastic_cast(
                        running,
                        state_dtype,
                        rand_seed,
                        philox_rounds,
                        layer * state.numel() + dst * running.numel(),
                    )
                )


def checkpointing_ssu_musa_reference(
    state: torch.Tensor,
    x_cache: torch.Tensor,
    b_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    ring_start: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    *,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    state_scale: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    philox_rounds: int,
) -> torch.Tensor:
    """Reference checkpointing SSU for MUSA ring-cache bring-up."""
    if x.dim() != 4 or dt.dim() not in (3, 4):
        raise ValueError(
            "checkpointing SSU expects x [batch,T,H,D] and dt [batch,T,H] "
            "or tied-dim dt [batch,T,H,D]"
        )
    batch, predicted, heads, dim = x.shape
    slots, state_heads, state_dim, dstate = state.shape
    if state_heads != heads or state_dim != dim:
        raise ValueError("checkpointing state and x head dimensions differ")
    if dt.shape[:3] != (batch, predicted, heads) or (
        dt.dim() == 4 and dt.shape[3] not in (1, dim)
    ):
        raise ValueError("checkpointing dt shape is incompatible with x")
    if state.dtype in (torch.int8, torch.int16, torch.float8_e4m3fn) and state_scale is None:
        raise ValueError("quantized checkpointing state requires state_scale")
    if x_cache.shape[1:] != (heads, x_cache.shape[2], dim):
        raise ValueError("x_cache must be [slots, heads, ring, dim]")
    groups = B.shape[2] if B.dim() == 4 else B.shape[1]
    if heads % groups:
        raise ValueError("heads must be divisible by groups")
    ratio = heads // groups
    ring_len = x_cache.shape[2]
    if ring_start.numel() != batch or prev_num_accepted_tokens.numel() != batch:
        raise ValueError("checkpointing ring metadata must have one entry per request")
    if state_batch_indices is not None and state_batch_indices.numel() != batch:
        raise ValueError("state_batch_indices must have one entry per request")
    out.zero_()
    A_f = A.to(torch.float32)
    bias = None if dt_bias is None else dt_bias.to(torch.float32)
    for request in range(batch):
        slot = int(state_batch_indices[request].item()) if state_batch_indices is not None else request
        if slot == pad_slot_id:
            running = torch.zeros(heads, dim, dstate, dtype=torch.float32, device=x.device)
        else:
            if slot < 0 or slot >= slots:
                raise IndexError(f"state slot {slot} is outside [0, {slots})")
            running = state[slot].to(torch.float32).clone()
            if state.dtype in (torch.int8, torch.int16, torch.float8_e4m3fn) and state_scale is not None:
                scale = state_scale[slot].to(torch.float32)
                if scale.shape[-1] == 1:
                    scale = scale.squeeze(-1)
                running = running * scale[..., None]
        start = int(ring_start[request].item())
        accepted = int(prev_num_accepted_tokens[request].item())
        if start < 0 or start >= ring_len or accepted < 0 or accepted > ring_len:
            raise ValueError("checkpointing ring metadata is outside cache bounds")
        for offset in range(accepted + predicted):
            ring = (start + offset) % ring_len
            if offset < accepted:
                if slot == pad_slot_id:
                    x_t = torch.zeros(heads, dim, dtype=torch.float32, device=x.device)
                    dt_t = torch.zeros(heads, dtype=torch.float32, device=x.device)
                    b_t = torch.zeros(groups, dstate, dtype=torch.float32, device=x.device)
                else:
                    x_t = x_cache[slot, :, ring].to(torch.float32)
                    dt_t = dt_cache[slot, :, ring].to(torch.float32)
                    b_t = b_cache[slot, :, ring].to(torch.float32)
            else:
                token = offset - accepted
                x_t = x[request, token].to(torch.float32)
                dt_t = dt[request, token].to(torch.float32)
                b_t = B[request, token].to(torch.float32)
                if slot != pad_slot_id:
                    x_cache[slot, :, ring].copy_(x[request, token])
                    b_cache[slot, :, ring].copy_(B[request, token])
            if bias is not None:
                dt_t = dt_t + bias
            if dt_softplus:
                dt_t = _softplus(dt_t)
            if offset >= accepted and slot != pad_slot_id:
                # Replay consumes the processed delta, after bias and
                # softplus, exactly as the recurrence did.
                if dt_t.dim() in (0, 1):
                    dt_cache[slot, :, ring].copy_(dt_t)
                elif dt_t.shape[-1] in (1, dim):
                    # The public 4-D form is tied across the head dimension;
                    # the replay ring stores one processed delta per head.
                    dt_cache[slot, :, ring].copy_(dt_t[..., 0])
                else:
                    raise ValueError(
                        "dt_cache is [slot,head,ring] and cannot store tied-dim dt"
                    )
            for head in range(heads):
                group = head // ratio
                dt_head = dt_t[head]
                if dt_head.dim() == 0 or dt_head.numel() == 1:
                    dt_head = dt_head.expand(dim)
                running[head] = running[head] * torch.exp(A_f[head] * dt_head[..., None])
                running[head] += (dt_head * x_t[head])[:, None] * b_t[group][None, :]
            if offset >= accepted:
                token = offset - accepted
                for head in range(heads):
                    group = head // ratio
                    y = torch.sum(C[request, token, group].to(torch.float32)[None, :] * running[head], dim=-1)
                    if D is not None:
                        y = y + D[head].to(torch.float32) * x[request, token].to(torch.float32)
                    if z is not None:
                        z_t = z[request, token, head].to(torch.float32)
                        y = y * z_t * torch.sigmoid(z_t)
                    out[request, token, head].copy_(y.to(out.dtype))
        if slot != pad_slot_id:
            if state.dtype in (torch.int8, torch.int16, torch.float8_e4m3fn):
                qmax = (
                    127
                    if state.dtype == torch.int8
                    else 32767
                    if state.dtype == torch.int16
                    else 448
                )
                amax = running.abs().amax(dim=-1)
                scale = torch.where(amax == 0, torch.ones_like(amax), amax / qmax)
                quantized = running / scale[..., None]
                if state.dtype == torch.float8_e4m3fn:
                    state[slot].copy_(quantized.to(state.dtype))
                else:
                    quantized = _stochastic_round_integer(
                        quantized,
                        rand_seed,
                        philox_rounds,
                        slot * running.numel(),
                    )
                    state[slot].copy_(quantized.clamp(-qmax - 1, qmax).to(state.dtype))
                if state_scale[slot].shape[-1] == 1:
                    state_scale[slot].copy_(scale.to(state_scale.dtype).unsqueeze(-1))
                else:
                    state_scale[slot].copy_(scale.to(state_scale.dtype))
            else:
                state[slot].copy_(
                    _stochastic_cast(
                        running,
                        state.dtype,
                        rand_seed,
                        philox_rounds,
                        slot * running.numel(),
                    )
                )
    return out
