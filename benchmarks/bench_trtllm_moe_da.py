#!/usr/bin/env python3
"""Compatibility entry point for the backend-aware DA MoE benchmark."""

if __package__:
    from . import bench_moe_da as _impl
    from .bench_moe_da import *  # noqa: F403
else:
    import bench_moe_da as _impl

    from bench_moe_da import *  # type: ignore[no-redef] # noqa: F403

# Preserve the established imported helper surface used by user-facing tests and AC tooling.
main = _impl.main
_benchmark_precision = _impl._benchmark_precision
_canonical_inputs = _impl._canonical_inputs
_capture = _impl._capture
_matching_diagnostic = _impl._matching_diagnostic
_prepare_precision = _impl._prepare_precision
_realization = _impl._realization
_temporary_environment = _impl._temporary_environment


if __name__ == "__main__":
    main()
