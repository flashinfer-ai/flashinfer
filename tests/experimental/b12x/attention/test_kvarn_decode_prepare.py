"""Sanitizer repro: kvarn decode kernel on server-probe tensors.

Mirrors vLLM B12xMLASparseImpl._make_prepare_call: q=empty, selected=zeros,
topk_lengths=ones, cache compiled from caps. Run under compute-sanitizer on
GPU 2:

  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/opt/cuda compute-sanitizer --tool memcheck \
      ~/lil/vllm/.venv/bin/python tests/attention/test_kvarn_decode_prepare.py
"""

import os
import sys

REPO = os.environ.get("B12X_REPO", os.path.expanduser("~/.cache/b12x-kvarn-wt"))
for p in (REPO,):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from b12x.attention.sparse_mla import Caps  # noqa: E402
from b12x.attention.sparse_mla._preparation import (  # noqa: E402
    compile_sparse_mla,
)
from b12x.attention._shared.mla.kernel import run_prepared_unified_decode  # noqa: E402

ORDINAL = int(os.environ.get("REPRO_DEVICE", "2"))
ROWS = int(os.environ.get("REPRO_ROWS", "32"))
HEADS = 16
HEAD_DIM = 512
TOPK = 2051
RECORD = int(os.environ.get("REPRO_RECORD", "536"))
PAGE = 256
PTW = 34


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", ORDINAL)
    caps = Caps(
        device=device,
        num_q_heads=HEADS,
        max_q_rows=ROWS,
        max_width=TOPK,
        softmax_scale=0.088388,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=HEAD_DIM,
        v_head_dim=512,
        model_type=4,  # ModelType.KVARN_MLA
        scale_format=3,  # ScaleFormat.KVARN_DUAL
        cache_record_bytes=RECORD,
        fp8_rope=False,
        kvarn_k_bits=int(os.environ.get("REPRO_K_BITS", "4")),
        kvarn_v_bits=int(os.environ.get("REPRO_V_BITS", "4")),
        kvarn_group=256,
        latent_scale_per_token=False,
        mode="decode",
        max_batch=ROWS,
        max_chunks_per_row=max(1, (TOPK + 63) // 64),
        page_size=PAGE,
        max_page_table_width=PTW,
        return_lse=False,
        lse_scale="natural",
    )
    prepared = compile_sparse_mla(caps, ORDINAL)
    print("compiled; running prepared decode launch")

    from b12x.attention.sparse_mla._scratch import (
        _materialize_sparse_mla_scratch,
        _sparse_mla_scratch_layout,
    )

    layout = _sparse_mla_scratch_layout(caps)
    scratch = torch.empty((layout.nbytes,), dtype=torch.uint8, device=device)
    scratch_t = _materialize_sparse_mla_scratch(caps, scratch, layout)
    q = torch.empty((ROWS, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    q.fill_(1)
    cache = torch.zeros((PTW, PAGE, RECORD), dtype=torch.uint8, device=device)
    selected = torch.zeros((ROWS, TOPK), dtype=torch.int32, device=device)
    import os as _os

    _lm = _os.environ.get("REPRO_LEN_MODE", "ones")
    if _lm == "full":
        topk_lengths = torch.full((ROWS,), TOPK, dtype=torch.int32, device=device)
    elif _lm == "zeros":
        topk_lengths = torch.zeros((ROWS,), dtype=torch.int32, device=device)
    else:
        topk_lengths = torch.ones((ROWS,), dtype=torch.int32, device=device)
    print(
        f"BASES q={hex(q.data_ptr())} cache={hex(cache.data_ptr())} "
        f"selected={hex(selected.data_ptr())} lengths={hex(topk_lengths.data_ptr())} "
        f"scratch={hex(scratch.data_ptr())}"
    )

    out = run_prepared_unified_decode(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=selected,
        swa_topk_lengths=topk_lengths,
        workspace=scratch_t,
        sm_scale=0.088388,
        swa_page_size=PAGE,
        attn_sink=None,
        return_lse=False,
        lse_scale="natural",
        prepared=prepared,
    )
    torch.cuda.synchronize()
    print("decode run OK:", out.shape if hasattr(out, "shape") else type(out))


if __name__ == "__main__":
    main()
