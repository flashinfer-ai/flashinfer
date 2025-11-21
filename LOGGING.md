# FlashInfer Logging

FlashInfer provides a  logging feature to help debug issues, and reproduce crashes. This document describes all available logging levels and their features.

## Quick Start

Enable logging using two environment variables:

```bash
# Set logging level (0-5)
export FLASHINFER_LOGLEVEL=3

# Set log destination (default is stdout)
export FLASHINFER_LOGDEST=stdout  # or stderr, or a file path like "flashinfer.log"

# Run your code
python train.py
```

## Logging Levels

| Level | Name | Features | Use Case |
|-------|------|----------|----------|
| **0** | Disabled (Default) | No logging (zero overhead) | Production |
| **1** | Function Names | Function names only | Basic tracing |
| **3** | Inputs/Outputs | Function names + arguments + outputs with metadata | Standard debugging |
| **5** | Statistics | Level 3 + tensor statistics (min, max, mean, NaN/Inf counts) | Numerical analysis |


## Environment Variables

### Main Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FLASHINFER_LOGLEVEL` | int | 0 | Logging level (0, 1, 3, 5) |
| `FLASHINFER_LOGDEST` | str | `stdout` | Log destination: `stdout`, `stderr`, or file path |

### Process ID Substitution

Use `%i` in file paths for automatic process ID substitution (useful for multi-GPU training):

```bash
export FLASHINFER_LOGDEST="flashinfer_log_%i.txt"  # → flashinfer_log_12345.txt
```

This works for:
- `FLASHINFER_LOGDEST`

## Miscellaneous Notes and Examples
### CUDA Graph Compatibility

Level 5 statistics are **automatically skipped during CUDA graph capture** to avoid synchronization issues.

```python
# This works correctly - no synchronization errors
with torch.cuda.graph(cuda_graph):
    result = mm_fp4(a, b, scales)  # Level 5 logging active
    # Statistics automatically skipped during capture
```

Output shows: `[statistics skipped: CUDA graph capture in progress]`

### Process IDs for Multi-GPU Environments

```bash
# Use %i for process ID substitution
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST="logs/flashinfer_api_%i.log"

torchrun --nproc_per_node=8 awesome_script_that_uses_FlashInfer.py

# Creates separate logs:
# logs/flashinfer_api_12345.log (rank 0)
# logs/flashinfer_api_12346.log (rank 1)
# ...
```

## Frequently Asked Questions

### Q: Does Level 0 really have zero overhead?

**A: Yes.** At Level 0, the decorator returns the original function unchanged. No wrapper, no checks, no overhead.
