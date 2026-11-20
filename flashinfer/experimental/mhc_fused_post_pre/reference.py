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
"""

# Composable PyTorch implementation of the fused mHC post -> pre (+ input RMSNorm):
# the fallback of flashinfer.mhc.mhc_fused_post_pre and the reference of its tests.

import torch


@torch.no_grad()
def mhc_fused_post_pre_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float = 1e-5,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-5,
):
    """Post map of layer i, then pre map + input RMSNorm of layer i + 1 (FP32 math).

    ``residual`` is ``[M, HC, H]``; ``x``, ``post_layer_mix`` and ``comb_res_mix`` hold
    ``M`` tokens in any shape that reshapes to ``[M, H]``, ``[M, HC]`` and
    ``[M, HC, HC]``.

    - post map: ``residual_cur[m, j] = bf16(x[m] * post_layer_mix[m, j]
      + sum_i residual[m, i] * comb_res_mix[m, i, j])``;
    - pre map: the 24 mixes are the FP32 projection of the flattened BF16
      ``residual_cur`` onto ``fn``, scaled by ``rsqrt(mean(residual_cur^2) + rms_eps)``,
      then by ``hc_scale`` (pre / post / comb) plus ``hc_base``;
      ``pre_mix = sigmoid(pre) + hc_pre_eps``,
      ``post_mix_cur = sigmoid(post) * hc_post_mult_value``,
      ``comb_mix_cur`` = ``sinkhorn_repeat`` Sinkhorn iterations of
      ``softmax(comb) + hc_sinkhorn_eps``;
    - ``layer_input_cur = bf16(bf16(sum_j pre_mix[j] * residual_cur[j]) * rstd * w)``
      with the RMS statistic taken from the pre-rounding FP32 weighted sum.

    Returns ``(residual_cur [M, HC, H] bf16, post_mix_cur [M, HC, 1] fp32,
    comb_mix_cur [M, HC, HC] fp32, layer_input_cur [M, H] bf16)``.
    """
    num_tokens, hc, hidden = residual.shape
    x2 = x.reshape(num_tokens, hidden).float()
    post = post_layer_mix.reshape(num_tokens, hc).float()
    comb = comb_res_mix.reshape(num_tokens, hc, hc).float()
    res = residual.float()
    new = post.unsqueeze(-1) * x2.unsqueeze(1)
    for i in range(hc):
        new = new + comb[:, i, :].unsqueeze(-1) * res[:, i, :].unsqueeze(1)
    residual_cur = new.to(torch.bfloat16)
    flat = residual_cur.reshape(num_tokens, hc * hidden).float()
    mixes = flat @ fn.float().t()
    sqrsum = flat.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc * hidden) + rms_eps)
    scale = hc_scale.float()
    base = hc_base.float()
    pre_mix = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_pre_eps
    post_mix_cur = (
        torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
        * hc_post_mult_value
    )
    comb_logits = mixes[:, 2 * hc :].reshape(num_tokens, hc, hc) * scale[2] + base[
        2 * hc :
    ].reshape(1, hc, hc)
    comb_mix_cur = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix_cur = comb_mix_cur / (
        comb_mix_cur.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
    )
    for _ in range(sinkhorn_repeat - 1):
        comb_mix_cur = comb_mix_cur / (
            comb_mix_cur.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps
        )
        comb_mix_cur = comb_mix_cur / (
            comb_mix_cur.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
        )
    weighted = (residual_cur.float() * pre_mix.unsqueeze(-1)).sum(dim=1)
    rstd = torch.rsqrt(weighted.square().mean(dim=-1, keepdim=True) + norm_eps)
    layer_input_cur = (
        weighted.to(torch.bfloat16).float() * rstd * norm_weight.float()
    ).to(torch.bfloat16)
    return (
        residual_cur,
        post_mix_cur.unsqueeze(-1),
        comb_mix_cur,
        layer_input_cur,
    )


__all__ = ["mhc_fused_post_pre_reference"]
