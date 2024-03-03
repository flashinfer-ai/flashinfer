"""
Attention with Linear Biases (ALiBi) reference implementation.

Code adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations

Licensed under MIT, you may obtain a copy of the License at

  https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license

Source:
- https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/285cb3735bde02fbc8c19ddeb24d0ae7e77135c1/labml_nn/transformers/mha.py
- https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/285cb3735bde02fbc8c19ddeb24d0ae7e77135c1/labml_nn/transformers/alibi/__init__.py
"""

import torch
import math
from torch import nn
from typing import Optional, List


def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head

    * `n_heads` is the number of heads in the attention layer $n$

    The slope for first head is

    $$\frac{1}{2^{\frac{8}{n}}} = 2^{-\frac{8}{n}}$$

    The slopes for the rest of the heads are in a geometric series with a ratio same as above.

    For instance when the number of heads is $8$ the slopes are
    $$\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$$
    """

    # Get the closest power of 2 to `n_heads`.
    # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
    # and then add the remaining slopes.
    n = 2 ** math.floor(math.log2(n_heads))
    # $2^{-\frac{8}{n}}$
    m_0 = 2.0 ** (-8.0 / n)
    # $2^{-1\frac{8}{n}}, 2^{-2 \frac{8}{n}}, 2^{-3 \frac{8}{n}}, \dots$
    m = torch.pow(m_0, torch.arange(1, 1 + n))

    # If `n_heads` is not a power of 2, then we add the remaining slopes.
    # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
    # And pick the slopes upto `n_heads`.
    if n < n_heads:
        # $2^{-\frac{8}{2n}}$
        m_hat_0 = 2.0 ** (-4.0 / n)
        # $2^{-1\frac{8}{2n}}, 2^{-3 \frac{8}{2n}}, 2^{-5 \frac{8}{2n}}, \dots$
        # Note that we take steps by $2$ to avoid slopes added previously.
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        # Concatenate the slopes with the remaining slopes.
        m = torch.cat([m, m_hat])

    return m


@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    """
    ## Calculate the attention biases matrix

    * `n_heads` is the number of heads in the attention layer
    * `mask` is the attention mask of shape `[seq_len_q, seq_len_k]`

    This returns a matrix of shape `[seq_len_q, seq_len_k, n_heads, ]` with ALiBi attention biases.
    """

    # Get slopes $m$ for each head
    m = get_slopes(n_heads).to(mask.device)

    # Calculate distances $[0, 1, \dots, N]$
    # Here we calculate the distances using the mask.
    #
    # Since it's causal mask we can just use $[0, 1, \dots, N]$ too.
    distance = torch.arange(mask.shape[1], dtype=torch.long, device=mask.device)[
        None, :
    ]

    # Multiply them pair-wise to get the AliBi bias matrix
    return distance[:, :, None] * m[None, None, :]


def alibi_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    """
    query: [q_len, num_heads, head_dim]
    key: [kv_len, num_heads, head_dim]
    value: [kv_len, num_heads, head_dim]
    mask: [q_len, kv_len]
    """
    q_len, num_heads, head_dim = query.shape
    kv_len = key.shape[0]

    scores = torch.einsum("qhd,khd->qkh", query.float(), key.float())
    # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
    scores *= 1.0 / math.sqrt(head_dim)

    # Create AliBi biases if it's not cached
    alibi_biases = get_alibi_biases(num_heads, mask)

    # Add AliBi biases to attention scores.
    # ALiBi biases has shape `[seq_len, seq_len, n_heads]`
    # and `scores` has shape `[seq_len, seq_len, batch_size, n_heads]`
    scores += alibi_biases

    # Apply mask
    scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))

    # $softmax$ attention along the key sequence dimension
    # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
    attn = torch.softmax(scores, dim=1)

    # Multiply by values
    # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
    return torch.einsum("ovh,vhd->ohd", attn, value.float()).to(query)
