"""Small eager oracle independent of FlashInfer/Triton kernels."""

import torch


def ssu_one_token(
    state, x, dt, A, B, C, D, dt_bias, state_slots, *, state_scale=None, z=None, dt_softplus=True
):
    state_ref = state.clone()
    scale_ref = state_scale.clone() if state_scale is not None else None
    output = torch.empty_like(x)
    heads, dim, dstate = state.shape[1:]
    groups = B.shape[1]
    ratio = heads // groups
    for batch in range(x.shape[0]):
        slot = int(state_slots[batch].item())
        running = state_ref[slot].to(torch.float32).clone()
        if state_ref.dtype == torch.int16:
            scale = scale_ref[slot].to(torch.float32)
            if scale.shape[-1] == 1:
                scale = scale.squeeze(-1)
            running = running * scale[..., None]
        delta = dt[batch].to(torch.float32) + dt_bias
        if dt_softplus:
            delta = torch.nn.functional.softplus(delta)
        for head in range(heads):
            group = head // ratio
            running[head] *= torch.exp(
                A[head].to(torch.float32) * delta[head, :, None]
            )
            running[head] += (
                delta[head] * x[batch, head].to(torch.float32)
            )[:, None] * B[batch, group].to(torch.float32)[None, :]
            output[batch, head] = torch.sum(
                C[batch, group].to(torch.float32)[None, :] * running[head], dim=-1
            )
            if D is not None:
                output[batch, head] += D[head].to(torch.float32) * x[batch, head].to(torch.float32)
            if z is not None:
                z_t = z[batch, head].to(torch.float32)
                output[batch, head] *= z_t * torch.sigmoid(z_t)
        if state_ref.dtype == torch.int16:
            amax = running.abs().amax(dim=-1)
            scale = torch.where(amax == 0, torch.ones_like(amax), amax / 32767)
            state_ref[slot].copy_((running / scale[..., None]).round().clamp(-32768, 32767).to(torch.int16))
            if scale_ref[slot].shape[-1] == 1:
                scale_ref[slot].copy_(scale.to(scale_ref.dtype).unsqueeze(-1))
            else:
                scale_ref[slot].copy_(scale.to(scale_ref.dtype))
        else:
            state_ref[slot].copy_(running.to(state_ref.dtype))
    return output, state_ref, scale_ref
