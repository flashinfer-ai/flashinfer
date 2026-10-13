# Reproducing the automatic-dispatch comparison

Save the Python block as `reproduce_topk.py` and run each case/seed/cache regime in a fresh process. Use the FlashInfer branch containing the optional `cudnn` backend and a cuDNN Frontend build exposing `IndexerTopKVarlen` ([frontend PR](https://github.com/NVIDIA/cudnn-frontend/pull/1298)); CUDA, PyTorch, NumPy, and CUDA Python runtime bindings are required. SM103 needs CuTe DSL >=4.7; SM107 needs >=4.8. The measured builds used DSL4.8.0 and TVM-FFI0.1.14.post1.

This reproduces the measured input generation, graph depths, changed-input checks, seeded arm order, 9 samples and 100 replays. It prints the automatic backend choice and the feature flag for each arm. It uses BF16 scores generated in FP32 then converted, K512, next_n=1, compress_ratio=1, and physical valid lengths. Full/ragged profiles and seeds20261211/20261213/20261217 are compared separately. No baseline exception is allowed.

Warm graphs use the same 2GiB depth formula, capped at16; their event interval contains100 replays. Cold graphs contain one call: every event interval follows a same-stream read/write sweep of randomized bytes spanning at least3 times the queried L2 size. The sweep is outside the timed interval; its cost is never subtracted. This supplies capacity eviction pressure, not a hardware cache-miss guarantee. Every timed block/cold call is checked for exact selected-value multiset, bounds, uniqueness, padding and unmodified inputs. The checks are outside timing.

The snippet is a portable extraction of the measured driver, not an independently rerun benchmark. Deployment identity attestation and actual internal-helper observation remain in the archived campaign; this public version prints the dispatcher choice. Extreme lengths, aliases and independent-graph concurrency are covered separately by `tests/experimental/test_cudnn_topk_varlen.py`.

```bash
# One profile; repeat with --regime cold_l2 for the paired cache regime.
CUDA_VISIBLE_DEVICES=0 python reproduce_topk.py --rows 512 --cols 16384 \
  --pattern full --seed 20261211 --regime warm --output profile.json
```

Use `(rows,cols)=(512,16384),(512,32768),(512,131072)` on SM103; use `(512,16384),(512,32768),(256,131072)` on SM107. For each shape run both patterns, all3 listed seeds and both cache regimes. A shape meets the reported bounded threshold only when all6 profiles in **each** regime have speedup>=1.05. This does not imply model-serving or full-matrix qualification.

```python
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import torch
import flashinfer

ARMS=('disabled_auto','enabled_auto')
GATE='FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS'
SAMPLES=9
REPLAYS=100


def depth(case,regime):
    return 1 if regime=='cold_l2' else max(1,min(16,2048*1024**2//
        (8*case['rows']*case['cols']+4*case['rows']*case['k'])))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def effective_lengths(torch, lengths, case):
    offsets = torch.arange(case["next_n"], device=lengths.device, dtype=torch.int64)
    raw = lengths.to(torch.int64)[:, None] - case["next_n"] + offsets[None, :] + 1
    return torch.div(raw, case["compress_ratio"], rounding_mode="floor").clamp(0, case["cols"]).flatten()

def reference_values(torch, scores, lengths, case):
    """Value-only oracle: bounds checks distinguish valid -inf from padding."""
    lens = effective_lengths(torch, lengths, case)
    masked = scores.masked_fill(torch.arange(case["cols"], device=scores.device)[None, :] >= lens[:, None], -torch.inf)
    values = masked.topk(min(case["k"], case["cols"]), dim=1, sorted=True).values
    if case["cols"] < case["k"]:
        pad = torch.full((case["rows"], case["k"] - case["cols"]), -torch.inf,
                         dtype=scores.dtype, device=scores.device)
        values = torch.cat((values, pad), dim=1)
    return lens, values

class OutputCheckError(AssertionError):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind

def check_output(torch, scores, lengths, expected_scores, expected_lengths, output, case, phase, reference):
    for name, actual, expected in (("scores", scores, expected_scores), ("lengths", lengths, expected_lengths)):
        if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
            raise OutputCheckError("mutation", f"{phase}: backend mutated read-only {name}")
    if output.shape != (case["rows"], case["k"]) or output.dtype != torch.int32 or output.device != scores.device:
        raise OutputCheckError("shape", f"{phase}: output shape/dtype/device mismatch")
    lens, expected_values = reference
    keep = torch.arange(case["k"], device=scores.device)[None, :] < lens.clamp(max=case["k"])[:, None]
    bounds = torch.where(keep, (output >= 0) & (output < lens[:, None]), output == -1)
    if not bool(bounds.all().item()):
        raise OutputCheckError("bounds", f"{phase}: invalid prefix index, count, or -1 suffix")
    ordered = output.sort(dim=1).values
    if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any().item()):
        raise OutputCheckError("duplicate", f"{phase}: duplicate selected index")
    values = expected_scores.gather(1, output.to(torch.int64).clamp(min=0))
    values = values.masked_fill(~keep, -torch.inf).sort(dim=1, descending=True).values
    if not torch.equal(values, expected_values):
        raise OutputCheckError("values", f"{phase}: selected value multiset differs from exact top-K")

def query_l2_cache_size(device_index):
    """Query this visible CUDA device through the runtime; never guess by model.

    cudaDevAttrL2CacheSize reports bytes:
    https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html
    """
    unavailable = []
    for module_name in ("cuda.bindings.runtime", "cuda.cudart"):
        try:
            runtime = importlib.import_module(module_name)
        except ImportError as exc:
            unavailable.append(f"{module_name}: {exc}")
            continue
        error, size = runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize,
                                                     device_index)
        if error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{module_name}.cudaDeviceGetAttribute(L2CacheSize, {device_index}) failed: {error}")
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"CUDA runtime reported invalid L2 cache size {size!r} for device {device_index}")
        return {"l2_cache_bytes": size, "device_index": device_index,
                "query": f"{module_name}.cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)"}
    raise RuntimeError("cold_l2 requires a CUDA runtime binding to query actual L2 size; " + "; ".join(unavailable))

class ColdL2Eviction:
    """Capacity eviction on the caller stream, never part of the timed graph.

    Randomized bytes avoid a constant-fill/compressible buffer. A full read-
    modify-write sweep of three times L2 supplies eviction pressure; this is
    not a hardware cache-invalidate instruction or a profiler proof of misses.
    """

    def __init__(self, torch, device):
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        self.metadata = {"cache_regime": "cold_l2", **query_l2_cache_size(device_index)}
        flush_bytes = ((3 * self.metadata["l2_cache_bytes"] + 255) // 256) * 256
        self.torch = torch
        generator = torch.Generator(device=device).manual_seed(938417)
        self.buffer = torch.empty(flush_bytes, dtype=torch.uint8, device=device)
        self.buffer.random_(0, 256, generator=generator)
        self.metadata.update({"eviction_buffer_bytes": flush_bytes, "minimum_l2_capacity_multiple": 3,
                              "eviction_before_timed_calls": True,
                              "eviction_method": "randomized uint8 buffer, full in-place add(1) read/write sweep",
                              "eviction_stream": "operator capture/caller stream",
                              "eviction_inside_timed_interval": False, "calls_per_graph": 1,
                              "calls_per_cuda_event_interval": 1,
                              "timing_subtraction": False,
                              "scope": "capacity-eviction pressure; hardware cache miss rate is not measured"})

    def flush(self, stream):
        with self.torch.cuda.stream(stream):
            self.buffer.add_(1)

def time_cold_call(torch, operation, stream, start, end, eviction):
    """One flush, then one graph replay between events, on the same stream."""
    with torch.cuda.stream(stream):
        eviction.flush(stream)
        start.record(stream)
        operation.graph.replay()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000

class Operation:
    def __init__(self, torch, flashinfer, arm, scores, lengths, case, stream):
        self.torch,self.flashinfer,self.arm,self.case,self.stream=torch,flashinfer,arm,case,stream
        self.scores,self.lengths=scores.clone(),lengths.clone()
        self.destination=torch.empty((case['rows'],512),dtype=torch.int32,device=scores.device)
        self.graph=None
        self.routes=[]

    def call(self,phase):
        with feature_gate(self.arm):
            output,values=self.flashinfer.top_k_varlen(self.scores,self.lengths,512,
                out_indices=self.destination,backend='auto')
            selected=list(self.flashinfer.top_k_varlen.suitable_auto_backends)
        assert output is self.destination and values is None
        assert (selected[0]=='cudnn')==(self.arm=='enabled_auto'),selected
        self.routes.append({'phase':phase,'chosen_auto_backend':selected[0]})

    def capture(self,inner):
        self.graph=self.torch.cuda.CUDAGraph()
        with self.torch.cuda.graph(self.graph,stream=self.stream):
            for index in range(inner):self.call(f'capture/{index}')

    def close(self):
        if self.graph is not None:self.graph.reset();self.graph=None


@contextmanager
def feature_gate(arm):
    previous = os.environ.get(GATE)
    os.environ[GATE] = '1' if arm=='enabled_auto' else '0'
    try:
        yield
    finally:
        if previous is None: os.environ.pop(GATE,None)
        else: os.environ[GATE]=previous

def inputs(torch,case,seed,device):
    generator=torch.Generator(device=device).manual_seed(seed)
    scores=torch.randn((case['rows'],case['cols']),generator=generator,device=device).to(torch.bfloat16)
    lengths=(torch.full((case['rows'],),case['cols'],dtype=torch.int32,device=device)
             if case['pattern']=='full' else torch.randint(513,case['cols']+1,(case['rows'],),
                 generator=generator,dtype=torch.int32,device=device))
    return scores,lengths

def tensor_sha(torch,tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def run_case(torch,flashinfer,case,seed,regime,device):
    scores,lengths=inputs(torch,case,seed,device)
    assert bool(((lengths>=0)&(lengths<=case['cols'])).all())
    stream=torch.cuda.Stream(device=device)
    ops={arm:Operation(torch,flashinfer,arm,scores,lengths,case,stream) for arm in ARMS}
    stream.wait_stream(torch.cuda.current_stream(device))
    inner=depth(case,regime)
    rng=random.Random(seed ^ int(digest(case)[:8],16))
    checks={arm:[] for arm in ARMS}
    eviction=ColdL2Eviction(torch,device) if regime=='cold_l2' else None
    policy=eviction.metadata if eviction else {'cache_regime':'warm'}

    def observe(arm,phase,expected_scores,expected_lengths,reference):
        op=ops[arm]
        check_output(torch,op.scores,op.lengths,expected_scores,expected_lengths,
            op.destination,case,phase,reference)
        checks[arm].append({'phase':phase,'status':'pass'})

    def verify(phase,x,lens,eager=False):
        assert bool(((lens>=0)&(lens<=case['cols'])).all()),'Comparison lengths must be physically valid'
        reference=reference_values(torch,x,lens,case)
        order=list(ARMS);rng.shuffle(order)
        stream.wait_stream(torch.cuda.current_stream(device))
        for arm in order:
            op=ops[arm]
            with torch.cuda.stream(stream):
                op.scores.copy_(x);op.lengths.copy_(lens);op.destination.fill_(-12345)
                op.call(phase) if eager else op.graph.replay()
            stream.synchronize()
            observe(arm,phase,x,lens,reference)

    try:
        ref=reference_values(torch,scores,lengths,case)
        for arm in ARMS:
            with torch.cuda.stream(stream):
                for index in range(3):ops[arm].call(f'warmup/{index}')
            stream.synchronize();observe(arm,'warmup',scores,lengths,ref)
        verify('eager',scores,lengths,eager=True)
        for op in ops.values():op.capture(inner)
        verify('graph_initial',scores,lengths)
        fresh,_=inputs(torch,case,seed+17011,device)
        shrunk=lengths//2
        grown=(lengths.to(torch.int64)+max(1,case['cols']//3)).clamp(max=case['cols']).to(torch.int32)
        verify('graph_changed_scores',fresh,lengths)
        verify('graph_changed_scores_and_shrunk_lengths',-fresh,shrunk)
        verify('graph_changed_scores_and_grown_lengths',fresh,grown)
        verify('graph_restored_inputs',scores,lengths)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        rounds=[]
        reference=reference_values(torch,scores,lengths,case)
        for sample in range(SAMPLES):
            units=[]
            for repeat in range(REPLAYS if regime=='cold_l2' else 1):
                order=list(ARMS);rng.shuffle(order);times={}
                for arm in order:
                    stream.wait_stream(torch.cuda.current_stream(device))
                    if regime=='cold_l2':
                        elapsed=time_cold_call(torch,ops[arm],stream,start,end,eviction)
                        phase=f'timed_block/{sample}/call/{repeat}'
                    else:
                        with torch.cuda.stream(stream):
                            for _ in range(3):ops[arm].graph.replay()
                            start.record(stream)
                            for _ in range(REPLAYS):ops[arm].graph.replay()
                            end.record(stream)
                        end.synchronize()
                        elapsed=start.elapsed_time(end)*1000/(inner*REPLAYS)
                        phase=f'timed_block/{sample}'
                    assert math.isfinite(elapsed) and elapsed>0
                    observe(arm,phase,scores,lengths,reference)
                    times[arm]=elapsed
                units.append({'repetition':repeat,'order':order,'microseconds':times})
            rounds.append({'round':sample,'units':units,'microseconds':{
                arm:statistics.median(unit['microseconds'][arm] for unit in units) for arm in ARMS}})
        verify('post_timing_changed_inputs',-fresh,grown)
        stats={arm:{'samples_us':[row['microseconds'][arm] for row in rounds],
            'median_us':statistics.median(row['microseconds'][arm] for row in rounds)} for arm in ARMS}
        return {'event':'case_result','case':case,'seed':seed,'regime':regime,'status':'pass',
            'checks':checks,'routes':{arm:op.routes for arm,op in ops.items()},'rounds':rounds,'timing':stats,
            'speedup':stats['disabled_auto']['median_us']/stats['enabled_auto']['median_us'],
            'inner_calls_per_graph':inner,'replays_per_sample':REPLAYS,'cache_policy':policy,
            'input_sha256':{'scores':tensor_sha(torch,scores),'lengths':tensor_sha(torch,lengths)},
            'length_min':int(lengths.min()),'length_max':int(lengths.max()),
            'comparison_length_policy':'all_phases_clamped_to_physical_width',
            'baseline_failure_exemptions':False,'promotion_eligible':False}
    finally:
        stream.synchronize()
        for op in ops.values():op.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,default=512)
    parser.add_argument('--cols',type=int,default=16384)
    parser.add_argument('--pattern',choices=('full','ragged'),default='full')
    parser.add_argument('--seed',type=int,choices=(20261211,20261213,20261217),default=20261211)
    parser.add_argument('--regime',choices=('warm','cold_l2'),default='warm')
    parser.add_argument('--output',help='Optional JSON path for all checks, orders and event samples')
    args=parser.parse_args()
    torch.cuda.set_device(0)
    cap=torch.cuda.get_device_capability(0)
    shapes={(10,3):{(512,16384),(512,32768),(512,131072)},
            (10,7):{(512,16384),(512,32768),(256,131072)}}
    assert cap in shapes and (args.rows,args.cols) in shapes[cap]
    case={'group':'integration_valid_lengths','rows':args.rows,'cols':args.cols,'k':512,
          'next_n':1,'compress_ratio':1,'pattern':args.pattern}
    result=run_case(torch,flashinfer,case,args.seed,args.regime,torch.device('cuda',0))
    if args.output:
        with open(args.output,'x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'case':case,'seed':args.seed,'cache_regime':args.regime,
        'actual_auto_gate_values':{'disabled_auto':'0','enabled_auto':'1'},
        'chosen_backends':{arm:sorted({r['chosen_auto_backend'] for r in result['routes'][arm]}) for arm in ARMS},
        'median_microseconds':{arm:result['timing'][arm]['median_us'] for arm in ARMS},
        'speedup':result['speedup'],'checks_per_arm':{arm:len(result['checks'][arm]) for arm in ARMS},
        'calls_per_graph':result['inner_calls_per_graph'],'cache_policy':result['cache_policy']},indent=2))


if __name__=='__main__':main()

```
