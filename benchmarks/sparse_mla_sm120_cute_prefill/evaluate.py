"""CuTe C++ experimental prefill: independent numerical gates and paired graph timings."""
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import tempfile
import torch
from tests.attention.dspark_mla_utils import dspark_fixture, numerical_metrics

ROOT = Path(os.getenv('CUTE_PREFILL_ARTIFACT_DIR', str(Path(tempfile.gettempdir()) / 'flashinfer-cute-prefill')))
class Params(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in (
        'q', 'kv', 'extra_kv', 'indices', 'extra_indices', 'lengths', 'extra_lengths',
        'positions', 'cos_sin', 'sink', 'output', 'scales', 'lse')]
    _fields_ += [(name, C.c_int) for name in ('tokens', 'heads', 'topk', 'extra_topk', 'kv_stride', 'extra_stride')]
    _fields_ += [('sm_scale', C.c_float)]

def candidate(f, library_path=None):
    if f['cache_format'] != 'v41_fp8':
        raise ValueError('This prototype supports only V41 inline FP8 caches')
    lib = C.CDLL(str(library_path or ROOT / 'kernel.so'))
    lib.launch.argtypes = [C.POINTER(Params), C.c_void_p]
    lib.launch.restype = C.c_int
    lse = torch.empty(f['q'].shape[:2], device='cuda', dtype=torch.float32)
    tensors = [f['q'], *f['caches'], *f['indices'], *f['lengths'], f['positions'], f['cs'], f['sink'], f['out'], f['scales'], lse]
    args = Params(*[t.data_ptr() if t is not None else 0 for t in tensors],
                  *f['q'].shape[:2], f['indices'][0].shape[1], f['indices'][1].shape[1],
                  528, 528, 512**-0.5)
    def run():
        rc = lib.launch(C.byref(args), torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f'CUDA launch failed: {rc}')
        return lse
    run.lse = lse
    run.params = args
    return run

def graph(fn):
    fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(3):
            fn()
    return g

def paired(functions, repeats=12):
    graphs = {k: graph(v) for k, v in functions.items()}
    for _ in range(5):
        for g in graphs.values():
            g.replay()
    torch.cuda.synchronize()
    samples = {k: [] for k in functions}
    names = list(graphs)
    for i in range(repeats):
        for name in (names if i % 2 == 0 else names[::-1]):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / 3)
    (ROOT / 'timing_samples.json').write_text(json.dumps(samples, indent=2) + '\n')
    return {k: statistics.median(v) for k,v in samples.items()}

def check(f, run, rows):
    lse = run()
    torch.cuda.synchronize()
    metrics = numerical_metrics(f, lse, rows)
    assert metrics['finite_output'], metrics
    assert metrics['relative_l2'] < 0.07, metrics
    assert metrics['lse_max_abs'] < 0.06, metrics
    return metrics

def main():
    mode = os.getenv('MODE', 'small')
    sizes = [4096] if mode == 'bench' else [1, 65]
    heads_list = [64] if mode == 'bench' else [8, 32, 64]
    records = []
    for tokens in sizes:
        for heads in heads_list:
            f = dspark_fixture(1, 'prefill', 'v41_fp8', heads=heads,
                               prefill_tokens=tokens, prefill_cp_size=4,
                               prefill_cp_rank=0, history_tokens=0, with_sink=heads != 8)
            if mode == 'bench' and os.getenv('CUTE_PREFILL_BASELINE_MODULE'):
                import tvm_ffi
                from benchmarks.bench_sparse_mla_sm120_mega import use_baseline_module
                baseline = tvm_ffi.load_module(os.environ['CUTE_PREFILL_BASELINE_MODULE'])
                use_baseline_module(f, baseline)
            run = candidate(f)
            rows = torch.unique(torch.linspace(0, tokens-1, min(16, tokens), device='cuda').long())
            baseline_lse = f['direct_mega']()
            baseline_metrics = numerical_metrics(f, baseline_lse, rows)
            record = dict(tokens=tokens, heads=heads, baseline=baseline_metrics, workload=f['workload'], gpu=torch.cuda.get_device_name())
            record['candidate'] = check(f, run, rows)
            if mode != 'bench':
                g = graph(run)
                for step in range(3):
                    for lengths, ids in zip(f['lengths'], f['indices'], strict=True):
                        lengths[0] = 0
                        if tokens > 1:
                            lengths[1] = min(ids.shape[1], [1, 33, 65][step])
                            ids[1, ::7] = -1
                    f['positions'].add_(1)
                    g.replay()
                    torch.cuda.synchronize()
                    # Check graph-produced output before another candidate launch.
                    metrics = numerical_metrics(f, run.lse, rows)
                    assert metrics['finite_output'] and metrics['relative_l2'] < .07 and metrics['lse_max_abs'] < .06, metrics
            else:
                record['ms'] = paired({'original_pipeline': f['baseline'], 'ldsm_mega': f['direct_mega'], 'cute_splitk':run})
                record['performance_pass'] = record['ms']['cute_splitk'] <= record['ms']['original_pipeline']
            records.append(record)
            print(json.dumps(record),flush=True)
    (ROOT / (mode + '.json')).write_text(json.dumps(records,indent=2)+'\n')
    assert all(r.get('performance_pass', True) for r in records), records

if __name__ == '__main__':
    main()
