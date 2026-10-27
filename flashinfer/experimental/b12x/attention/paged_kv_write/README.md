# Prepared paged K/V writes

This experimental b12x operation writes BF16 projection rows into BF16 or
FP8-E4M3 paged caches on SM120/SM121. Importing `b12x.attention.paged_kv_write`
opts into the operation. Preparation compiles only the cache writer; attention
head dimensions and attention algorithm selection do not constrain it.

Sources have shape `[tokens, kv_heads, head_dim]`; caches have shape
`[pages, page_size, kv_heads, head_dim]`. BF16 head dimensions must be multiples
of eight, and FP8 dimensions multiples of sixteen. Inner strides are one;
addresses and outer byte strides must be 16-byte aligned. Strided hybrid-cache
views are supported. K/V may have different head dimensions.

Every int64 slot-map entry is `page * page_size + offset`. Negative and
out-of-capacity destinations are skipped. Source storage must remain valid
even for skipped rows. Live destinations must be unique, and sources must not
overlap destination elements. FP8 uses independent positive finite float32
K/V scales and stores saturating `E4M3(source / scale)`. BF16 copies are exact.

The caller retains all tensors and the prepared plan until captured graphs
have drained. Slot maps and FP8 scales may change in place between replays.
Tensor shapes, strides, addresses, and graph token capacity remain fixed.
Correctness and capture are tested; this operation has no throughput claim.

The following example writes a two-head, 256-wide row after graph capture:

```python
import torch
from b12x.attention import paged_kv_write
from b12x.preparation import PreparationSession, PreparedCall

device = torch.device("cuda", torch.cuda.current_device())
key = torch.randn((1, 2, 256), dtype=torch.bfloat16, device=device)
value = torch.randn_like(key)
k_cache = torch.zeros((2, 64, 2, 256), dtype=torch.bfloat16, device=device)
v_cache = torch.zeros_like(k_cache)
slots = torch.full((1,), -1, dtype=torch.int64, device=device)
args = dict(key=key, value=value, k_cache=k_cache, v_cache=v_cache, slot_mapping=slots)
plan = paged_kv_write.plan(
    paged_kv_write.Caps(
        device=device,
        num_kv_heads=2,
        head_dim_k=256,
        head_dim_v=256,
        page_size=64,
    )
)

with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
    session.prepare(
        (
            plan.request(
                name="cache_write",
                prepare_call=lambda state: PreparedCall(run=lambda: state.run(**args)),
            ),
        )
    )
    session.freeze()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_kv_write.run(plan, **args)
    slots.fill_(65)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(k_cache[1, 1], key[0], rtol=0, atol=0)
    torch.testing.assert_close(v_cache[1, 1], value[0], rtol=0, atol=0)
    graph.reset()
```
