# FlashInfer Profiling Examples

This directory contains practical examples for profiling and optimizing FlashInfer kernels.

## Examples

### 01_decode_profiling.py
Basic decode attention profiling with detailed metrics.

```bash
python examples/profiling/01_decode_profiling.py
```

Demonstrates:
- Simple profiling workflow
- Collecting latency and throughput metrics
- Statistical analysis (median, P95, P99, CV)
- Scaling analysis across different context lengths

### 02_backend_comparison.py
Compare performance across different FlashInfer backends.

```bash
python examples/profiling/02_backend_comparison.py
```

Demonstrates:
- Benchmarking multiple backends (FlashInfer, cuDNN, CUTLASS)
- Identifying the best backend for different scenarios
- Performance visualization and recommendations

## Requirements

```bash
pip install flashinfer-python torch cupti-python
```

## Related Documentation

- [Performance Optimization Tutorial](../../docs/tutorials/performance_optimization.rst)
- [Profiling Guide](../../benchmarks/PROFILING_GUIDE.md)
- [Benchmark Framework README](../../benchmarks/README.md)

## Contributing

Have optimization tips or profiling examples? Submit a PR!

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
