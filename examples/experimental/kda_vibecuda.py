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

# Run the experimental VibeCUDA backend for recurrent-KDA prefill on SM100a/SM103a.

import argparse

import torch

from flashinfer.kda import recurrent_kda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 512])
    parser.add_argument("--num-heads", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")
    total_tokens = sum(args.seq_lens)
    num_seqs = len(args.seq_lens)
    shape = (1, total_tokens, args.num_heads, 128)

    offsets = [0]
    for length in args.seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)
    initial_state = torch.randn(
        (num_seqs, args.num_heads, 128, 128), dtype=torch.bfloat16, device=device
    )

    output, final_state = recurrent_kda(
        q=torch.randn(shape, dtype=torch.bfloat16, device=device),
        k=torch.randn(shape, dtype=torch.bfloat16, device=device),
        v=torch.randn(shape, dtype=torch.bfloat16, device=device),
        g=(0.1 * torch.randn(shape, device=device)).to(torch.bfloat16),
        beta=torch.randn(
            (1, total_tokens, args.num_heads), dtype=torch.bfloat16, device=device
        ),
        A_log=0.1 * torch.randn(args.num_heads, device=device),
        dt_bias=0.1 * torch.randn((args.num_heads, 128), device=device),
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
        beta_is_logit=True,
        backend="vibecuda",
    )
    torch.cuda.synchronize()
    print(f"output shape={tuple(output.shape)}, dtype={output.dtype}")
    print(f"final state shape={tuple(final_state.shape)}, dtype={final_state.dtype}")


if __name__ == "__main__":
    main()
