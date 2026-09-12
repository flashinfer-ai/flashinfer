# kda_prefill_wrapper

Planning implementation behind the experimental `flashinfer.RecurrentKDAPrefillWrapper`.

- **Owner:** @kahyunnam
- **Tracking issue:** [#5069](https://github.com/flashinfer-ai/flashinfer/issues/5069)

`RecurrentKDAPrefillPlanner` owns the fixed-address buffers, plan validation and
graph-replay invariants for packed recurrent-KDA prefill, then hands off to the
stable `flashinfer.recurrent_kda` facade with `backend="cute-dsl"`. It contains
no kernels; the kernels it dispatches to are the stable AOT-registered ones.

The public entry point is the thin `RecurrentKDAPrefillWrapper` class in
`flashinfer/kda.py`, whose `plan` and `run` are marked
`@flashinfer_experimental_api`. Calling either is the opt-in and needs no
environment variable.

Tests live in `tests/experimental/test_kda_prefill_wrapper.py`; a runnable
example is at `examples/experimental/kda_prefill_wrapper.py`. Requires compute
capability 10.0 or 10.3.
