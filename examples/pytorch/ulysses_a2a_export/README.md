# Ulysses all-to-all example

This portfolio exercises the native `flashinfer.comm.UlyssesCommunicator` API.
The generated NVLink implementation targets Blackwell SM100/SM103; the communicator
retains its existing topology selection, output ownership and NCCL fallback.

```python
from flashinfer.comm import UlyssesCommunicator

with UlyssesCommunicator(group, max_elems=q.numel(), dtype=q.dtype) as comm:
    q_global = comm.scatter_heads(q)
    k_global = comm.scatter_heads(k)
    v_global = comm.scatter_heads(v)
    output = comm.gather_heads(attention_output)
```

`out=` and `workspace=` retain the normal public API semantics. A prepared
benchmark allocates input/output/workspace once; every measured call includes
three head scatters and one gather with independently supplied input, together
with the four staging-to-output copies. No attention operation is timed.

`shapes.json` contains 36 small correctness rows and four large BF16 rows,
covering worlds 2/4/6/8, fp16/bf16/fp32, batch dimensions and scalar tails.
`interface.py` provides the reusable prepared fixture and independent
all-gather references.

After installing this FlashInfer checkout, run correctness on an eight-GPU
SM100 or SM103 NVLink node:

```bash
python -m pytest tests/comm/test_cake_ulysses_a2a.py -v
```

Run a reproducible native NVLink/NCCL comparison for each desired world:

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/comm/bench_cake_ulysses_a2a.py --repeat-iters 30 --json results.json
```

Add `--all-shapes` to measure all ten rows for that world. GPU timing requires
CUPTI 13 or newer, uses cold L2 and reports the rank maximum before taking the
median. Missing CUPTI is an error. The JSON retains individual samples,
absolute milliseconds, NCCL speedup, GPU identity and benchmark wall time.
This benchmark does not itself assert parity with a separate source checkout;
the exported implementation's paired validation results supply that comparison.
