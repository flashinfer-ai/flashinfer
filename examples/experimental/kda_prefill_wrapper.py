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

# Run the experimental plan/run wrapper for packed recurrent-KDA prefill.
#
# Requires CC 10.0 or 10.3. The wrapper stages sequence metadata once in
# ``plan`` so that ``run`` keeps fixed buffer addresses across CUDA graph
# replays; ``run`` is otherwise equivalent to calling
# ``flashinfer.recurrent_kda`` with ``backend="cute-dsl"``.

import argparse

import torch

from flashinfer import RecurrentKDAPrefillWrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[7, 29, 13])
    parser.add_argument("--num-heads", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda")
    total_tokens = sum(args.seq_lens)
    shape = (1, total_tokens, args.num_heads, 128)

    offsets = [0]
    for length in args.seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)

    wrapper = RecurrentKDAPrefillWrapper(device)
    wrapper.plan(cu_seqlens)

    output, final_state = wrapper.run(
        q=torch.randn(shape, dtype=torch.bfloat16, device=device),
        k=torch.randn(shape, dtype=torch.bfloat16, device=device),
        v=torch.randn(shape, dtype=torch.bfloat16, device=device),
        g=(0.1 * torch.randn(shape, device=device)).to(torch.bfloat16),
        beta=torch.randn(
            (1, total_tokens, args.num_heads), dtype=torch.bfloat16, device=device
        ),
        A_log=0.1 * torch.randn(args.num_heads, device=device),
        dt_bias=0.1 * torch.randn((args.num_heads, 128), device=device),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
        output_final_state=True,
    )
    torch.cuda.synchronize()
    print(f"output shape={tuple(output.shape)}, dtype={output.dtype}")
    print(f"final state shape={tuple(final_state.shape)}, dtype={final_state.dtype}")


if __name__ == "__main__":
    main()
