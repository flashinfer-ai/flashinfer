# Legacy NVFP4 GIN support and qualification

The fixed legacy GIN case below is qualified by the recorded correctness
matrix. Broader geometry and mode combinations are not implied.
The explicit experimental `internode=True, ranks_per_node=2` path is wired
through the formal backend, and the fixed case passes the numerical/public-runtime checks below. It is
default-off; broader production support requires separate qualification.

| Mode | Configuration | Status |
| --- | --- | --- |
| Existing intranode | All F_BASE-supported NVFP4 configs, default behavior | EP1/EP2/EP4, BF16/MXFP8 compatibility and existing precision checks pass |
| First internode qualification target | N2R2, 4 ranks, 2 distinct NVLink domains; default PG | Runtime, strong protocol/Torch/byte oracles and relaunch pass |
| Target geometry | T=32/rank, H=7168, 12 experts/rank (48 total), top-k=4 | Implemented static allowlist; donor and FI strong-oracle passes |
| Target gate-up/down width | Runner/shim `intermediate=3072`; FC2 input width=1536 | Public intermediate_size=1536 maps to runner intermediate=3072 |
| Target tile/cluster | MMA (128,128,256), cluster (1,1,1), static scheduling | Compiles and runs finite relaunch cases |
| Target math/finalize | Standard SwiGLU, NVFP4×NVFP4, BF16 combine, separate TopkReduce | Strong protocol/Torch oracle passes |
| Routing | Balanced, local-only/remote-only/mixed and partial/empty cases | Matrix-first/balanced/partial/empty/sparse-poison numeric and byte checks pass |
| N2R4, N4R8 and other geometries | No FI numerical evidence | Not enabled |
| 48 or 64 experts **per rank** | Historical boundary not recovered/reproduced | Unqualified; 48 **total** above is unrelated |
| Subgroup | All default-group ranks must join preflight | Host error aggregation tested; real runtime rejection passed; public backend checks implemented |
| CUDA Graph capture | Reject internode capture; preserve existing intranode graphs | Public capture rejection/recovery passes |
| Custom activation/scale/finalize, autotune, experimental comm options | Preserve off behavior; reject unqualified on combinations | Five config rejections, including epilogue scales, and single-rank metadata errors/recovery pass |

On mode resolves the donor `group_hint=one` to 536870912, uses epilogue
publication `(1,4)` and reuses dispatch warps for token-back. The public
`intermediate_size=1536` maps to gate-up width 3072 and FC2 input width 1536.
Those defaults do not alter off-mode configuration.

A runnable [example](../../../examples/nvfp4_gin.py) uses the formal layer API.
Launch on both domains with matching master endpoint and distinct node ranks:

```bash
torchrun --nnodes=2 --nproc-per-node=2 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 examples/nvfp4_gin.py
```

The example is a finite/shape smoke. Strong numerical acceptance belongs to
`test_nvfp4_cutedsl_gin_oracles.py`, which checks the actual BF16 term tree,
independent Torch math, transport message bytes and physical DATA/SF pools.

Hardware acceptance requires actual NCCL device API support, railed RDMA,
exactly N LSA teams and reachable same-domain global peers. Two hosts do not
establish two NVLink domains. The mapper contract rejects R>16; this ABI bound
does not certify every R<=16 topology.

Advance a row only with full-rank logs, fixed versions and the numerical,
relaunch and lifetime gates in [validation](validation.md).

## User scope clarification

On 2026-11-02 the user stated: “我们的这个feature不涉及rubin环境，可以不用在本地验证”.
Rubin/local performance measurements are therefore not a completion gate for
this delivery. Existing Blackwell correctness checks continue. Performance is
not measured, rather than passed; no hardware restriction is silently bypassed.
