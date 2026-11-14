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

from typing import Optional

import einops
import torch


def sink_softmax(logits, sink):
    sink = einops.repeat(sink, "h -> b h m 1", b=logits.shape[0], m=logits.shape[2])
    # (b, h, m, (n + 1))
    logits = torch.cat([logits, sink], dim=-1)
    # (s_1, s_2, ..., s_n)
    # (s_1, s_2, ..., s_n, log(sink))
    # (exp(s_1), exp(s_2), ..., exp(s_n), sink)
    # (exp(s_1) / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink),
    #  exp(s_2) / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink),
    #  ...,
    #  exp(s_n) / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink))
    #  sink / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink)
    score = torch.softmax(logits, dim=-1)[..., :-1].contiguous()
    return score


def sink_attention_unified(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
    batch_size: Optional[int] = None,
    mode: str = "auto",
    qo_indptr: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Unified sink attention implementation supporting prefill, incremental, chunk prefill, and variable-length scenarios.

    Args:
        q: Query tensor. Format depends on mode:
           - Regular Prefill: [total_q_len, num_qo_heads, head_dim] where q_len == kv_len
           - Incremental: [batch_size, num_qo_heads, head_dim] where q_len == 1
           - Chunk Prefill: [total_q_len, num_qo_heads, head_dim] where q_len != kv_len and q_len > 1
           - Variable Length: [total_q_len, num_qo_heads, head_dim] with different q_len per request
        k: Key tensor. Format depends on mode:
           - Regular Prefill: [total_kv_len, num_kv_heads, head_dim]
           - Incremental: [batch_size, kv_len, num_kv_heads, head_dim]
           - Chunk Prefill: [total_kv_len, num_kv_heads, head_dim]
           - Variable Length: [total_kv_len, num_kv_heads, head_dim]
        v: Value tensor, same format as k
        sink: Sink values [num_qo_heads]
        window_left: Sliding window size (-1 for no window)
        causal: Whether to apply causal masking
        sm_scale: Scaling factor for attention
        batch_size: Required for prefill/chunk modes, auto-detected for incremental
        mode: Processing mode:
            - "auto": Auto-detect based on tensor shapes and dimensions
            - "prefill": Regular prefill (q_len == kv_len)
            - "incremental": Incremental generation (q_len == 1)
            - "chunk": Chunk prefill (q_len != kv_len and q_len > 1)
            - "varlen": Variable length sequences within batch
        qo_indptr: Optional[torch.Tensor] - Query sequence length pointers for variable length mode.
                  Shape: [batch_size + 1]. qo_indptr[i+1] - qo_indptr[i] gives the query length for request i.
                  Only used when mode="varlen".
        kv_indptr: Optional[torch.Tensor] - Key/Value sequence length pointers for variable length mode.
                  Shape: [batch_size + 1]. kv_indptr[i+1] - kv_indptr[i] gives the kv length for request i.
                  Only used when mode="varlen".

    Returns:
        Output tensor. Format depends on mode:
        - Regular Prefill: [total_q_len, num_qo_heads, head_dim]
        - Incremental: [batch_size, num_qo_heads, head_dim]
        - Chunk Prefill: [total_q_len, num_qo_heads, head_dim]
        - Variable Length: [total_q_len, num_qo_heads, head_dim]
    """

    # Auto-detect mode if not specified
    if mode == "auto":
        # Check if variable length mode is indicated by presence of indptr
        if qo_indptr is not None or kv_indptr is not None:
            mode = "varlen"
        elif len(q.shape) == 3 and len(k.shape) == 4:
            # q: [batch_size, num_heads, head_dim], k: [batch_size, kv_len, num_heads, head_dim]
            # This is incremental mode
            mode = "incremental"
        elif len(q.shape) == 3 and len(k.shape) == 3:
            # Both q and k are flattened: [total_len, num_heads, head_dim]
            if batch_size is None:
                raise ValueError(
                    "batch_size is required for auto-detection in prefill/chunk modes"
                )

            qo_len = q.shape[0] // batch_size
            kv_len = k.shape[0] // batch_size

            if qo_len == kv_len:
                mode = "prefill"
            elif qo_len == 1:
                mode = "incremental"  # Special case: single token with flattened format
            elif qo_len > 1 and qo_len != kv_len:
                mode = "chunk"
            else:
                raise ValueError(
                    f"Cannot auto-detect mode: qo_len={qo_len}, kv_len={kv_len}"
                )
        else:
            raise ValueError(
                f"Cannot auto-detect mode from tensor shapes: q={q.shape}, k={k.shape}"
            )

    # Process based on detected/specified mode
    if mode == "incremental":
        # Incremental generation mode: q_len=1, kv_len from cache
        batch_size = q.shape[0]
        qo_len = 1
        kv_len = k.shape[1]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[2]

        # Handle GQA
        if num_qo_heads != num_kv_heads:
            k = torch.repeat_interleave(
                k, num_qo_heads // num_kv_heads, dim=2
            ).contiguous()
            v = torch.repeat_interleave(
                v, num_qo_heads // num_kv_heads, dim=2
            ).contiguous()
            num_kv_heads = num_qo_heads

        head_dim_qk = q.shape[2]
        head_dim_vo = v.shape[3]

        # Compute logits: [batch_size, num_heads, 1, kv_len]
        logits = (
            torch.einsum(
                "bhd,blhd->bhl",
                q.float(),
                k.float(),
            ).unsqueeze(2)  # Add seq_len=1 dimension
            * sm_scale
        )

    elif mode in ["prefill", "chunk"]:
        # Prefill or Chunk prefill mode: q and k are flattened tensors
        if batch_size is None:
            raise ValueError(f"batch_size is required for {mode} mode")

        qo_len = q.shape[0] // batch_size
        kv_len = k.shape[0] // batch_size
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]

        # Handle GQA
        if num_qo_heads != num_kv_heads:
            k = torch.repeat_interleave(
                k, num_qo_heads // num_kv_heads, dim=1
            ).contiguous()
            v = torch.repeat_interleave(
                v, num_qo_heads // num_kv_heads, dim=1
            ).contiguous()

        head_dim_qk = q.shape[2]
        head_dim_vo = v.shape[2]

        # Compute logits: [batch_size, num_heads, qo_len, kv_len]
        logits = (
            torch.einsum(
                "bmhd,bnhd->bhmn",
                q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
                k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
            )
            * sm_scale
        )

    elif mode == "varlen":
        # Variable length sequences mode
        if qo_indptr is None or kv_indptr is None:
            raise ValueError("qo_indptr and kv_indptr are required for varlen mode")

        batch_size = qo_indptr.shape[0] - 1
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim_qk = q.shape[2]
        head_dim_vo = v.shape[2]

        # Handle GQA
        if num_qo_heads != num_kv_heads:
            k = torch.repeat_interleave(
                k, num_qo_heads // num_kv_heads, dim=1
            ).contiguous()
            v = torch.repeat_interleave(
                v, num_qo_heads // num_kv_heads, dim=1
            ).contiguous()
            num_kv_heads = num_qo_heads

        # Process each request in the batch separately
        output_list = []

        for i in range(batch_size):
            # Extract tensors for current request
            qo_start, qo_end = qo_indptr[i].item(), qo_indptr[i + 1].item()
            kv_start, kv_end = kv_indptr[i].item(), kv_indptr[i + 1].item()

            q_i = q[qo_start:qo_end]  # [qo_len_i, num_heads, head_dim]
            k_i = k[kv_start:kv_end]  # [kv_len_i, num_heads, head_dim]
            v_i = v[kv_start:kv_end]  # [kv_len_i, num_heads, head_dim]

            qo_len_i = qo_end - qo_start
            kv_len_i = kv_end - kv_start

            # Compute logits for current request: [1, num_heads, qo_len_i, kv_len_i]
            logits_i = (
                torch.einsum(
                    "qhd,khd->hqk",
                    q_i.float(),
                    k_i.float(),
                ).unsqueeze(0)  # Add batch dimension
                * sm_scale
            )

            # Build attention mask for current request
            if causal:
                # Create causal mask for this specific request
                row_idx = torch.arange(qo_len_i, dtype=torch.int32, device=q.device)[
                    :, None
                ]
                col_idx = torch.arange(kv_len_i, dtype=torch.int32, device=q.device)[
                    None, :
                ]

                # Default causal mask: position i can attend to positions 0 to i in the kv sequence
                # Assuming queries correspond to the last qo_len_i positions in the kv sequence
                query_positions = kv_len_i - qo_len_i + row_idx
                mask_i = query_positions >= col_idx

                if window_left >= 0:
                    mask_i &= query_positions - window_left <= col_idx
            else:
                # Non-causal mask
                mask_i = torch.ones(
                    qo_len_i, kv_len_i, device=q.device, dtype=torch.bool
                )
                if window_left >= 0:
                    row_idx = torch.arange(
                        qo_len_i, dtype=torch.int32, device=q.device
                    )[:, None]
                    col_idx = torch.arange(
                        kv_len_i, dtype=torch.int32, device=q.device
                    )[None, :]
                    query_positions = kv_len_i - qo_len_i + row_idx
                    mask_i = query_positions - window_left <= col_idx

            # Apply mask
            logits_i = logits_i.masked_fill(
                mask_i.unsqueeze(0).unsqueeze(0) == 0, float("-inf")
            )

            # Apply sink softmax
            p_i = sink_softmax(logits_i, sink)  # [1, num_heads, qo_len_i, kv_len_i]

            # Compute output for current request
            o_i = (
                torch.einsum(
                    "bhmn,nhd->bmhd",
                    p_i,  # [1, num_heads, qo_len_i, kv_len_i]
                    v_i.float(),  # [kv_len_i, num_heads, head_dim]
                )
                .contiguous()
                .view(qo_len_i, num_qo_heads, head_dim_vo)
                .to(q)
            )

            output_list.append(o_i)

        # Concatenate outputs from all requests
        o_ref = torch.cat(output_list, dim=0)

        return o_ref

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Supported modes: 'auto', 'prefill', 'incremental', 'chunk', 'varlen'"
        )

    # Build attention mask (unified for all modes)
    if causal:
        if mode == "incremental":
            # For incremental: new token can attend to all previous tokens
            mask = torch.ones(1, kv_len, device=q.device, dtype=torch.bool)
            if window_left >= 0:
                col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)
                mask = (kv_len - 1 - window_left) <= col_idx
        elif mode == "prefill":
            # For regular prefill: standard causal mask
            mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
                1
            ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
            if window_left >= 0:
                row_idx = torch.arange(qo_len, dtype=torch.int32, device=q.device)[
                    :, None
                ]
                col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)[
                    None, :
                ]
                mask &= row_idx - window_left <= col_idx
        elif mode == "chunk":
            # For chunk prefill: each query position can attend to all previous KV positions
            # Current chunk positions are at the end: [kv_len - qo_len : kv_len]
            current_chunk_start = kv_len - qo_len
            row_idx = torch.arange(qo_len, dtype=torch.int32, device=q.device)[
                :, None
            ]  # Positions within chunk
            col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)[
                None, :
            ]  # All KV positions

            # Each position can attend to: all historical + positions up to itself in current chunk
            abs_row_positions = (
                current_chunk_start + row_idx
            )  # Absolute positions in full sequence
            mask = abs_row_positions >= col_idx  # Standard causal mask

            if window_left >= 0:
                mask &= abs_row_positions - window_left <= col_idx
    else:
        # Non-causal mask
        if mode == "incremental":
            mask = torch.ones(1, kv_len, device=q.device, dtype=torch.bool)
            if window_left >= 0:
                col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)
                mask = (kv_len - 1 - window_left) <= col_idx
        else:  # prefill or chunk
            mask = torch.ones(qo_len, kv_len, device=q.device, dtype=torch.bool)
            if window_left >= 0:
                if mode == "chunk":
                    # For chunk mode, apply window relative to absolute positions
                    current_chunk_start = kv_len - qo_len
                    row_idx = torch.arange(qo_len, dtype=torch.int32, device=q.device)[
                        :, None
                    ]
                    col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)[
                        None, :
                    ]
                    abs_row_positions = current_chunk_start + row_idx
                    mask = abs_row_positions - window_left <= col_idx
                else:  # prefill
                    row_idx = torch.arange(qo_len, dtype=torch.int32, device=q.device)[
                        :, None
                    ]
                    col_idx = torch.arange(kv_len, dtype=torch.int32, device=q.device)[
                        None, :
                    ]
                    mask = row_idx - window_left <= col_idx

    # Apply mask
    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

    # Apply sink softmax
    p = sink_softmax(logits, sink)

    # Compute output
    if mode == "incremental":
        # Incremental mode output
        o_ref = (
            torch.einsum(
                "bhml,blhd->bhd",
                p,  # [batch_size, num_heads, 1, kv_len]
                v.float(),  # [batch_size, kv_len, num_heads, head_dim]
            )
            .contiguous()
            .to(q)
        )
    else:  # prefill or chunk mode
        # Prefill/Chunk mode output
        o_ref = (
            torch.einsum(
                "bhmn,bnhd->bmhd",
                p,  # [batch_size, num_heads, qo_len, kv_len]
                v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
            )
            .contiguous()
            .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
            .to(q)
        )

    return o_ref
