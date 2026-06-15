# `flashinfer.moe_ep` — Current Design

Class diagram and module map for the **current** implementation under
`flashinfer/flashinfer/moe_ep/` (split EP path: `nccl_ep` + `nixl_ep`).
This document reflects the code as it exists today, not the planned Mega MoE
restructure described in
[`moe_ep_deep_gemm_mega_moe.md`](moe_ep_deep_gemm_mega_moe.md).

---

## Package layout

```
flashinfer/moe_ep/
├── __init__.py              # Public exports, build probes, backend import side-effects
├── layer.py                 # MoEEpLayer (nn.Module entry point)
├── fleet.py                 # Fleet ABC + create_fleet() + _BACKEND_REGISTRY
├── handle.py                # Handle ABC
├── config.py                # Enums + frozen dataclass I/O envelopes
├── tensors.py               # MoEEpTensors input bundle
├── algo_knobs.py            # AlgoKnob hierarchy + _index_knobs()
├── _validators.py           # Backend-specific FleetParams / arch checks
├── split_backends/
│   ├── __init__.py
│   ├── nccl_ep_comm.py      # NcclEpConfig
│   └── nixl_ep_comm.py      # NvepConfig
├── nccl_ep/
│   ├── __init__.py          # libnccl / libnccl_ep lazy loaders
│   ├── fleet.py             # NcclEpFleet
│   ├── handle.py            # NcclEpHandle
│   └── ndtensor.py          # NDTensor + get_nccl_lib()
└── nixl_ep/
    ├── __init__.py          # libnixl / nixl_ep_cpp lazy loaders
    ├── fleet.py             # NixlEpFleet
    └── handle.py            # NixlEpHandle
```

---

## End-to-end flow

```mermaid
sequenceDiagram
    participant User
    participant MoEEpLayer
    participant create_fleet
    participant Fleet
    participant Handle

    User->>MoEEpLayer: forward(MoEEpTensors)
    MoEEpLayer->>create_fleet: _ensure_fleet() [lazy, once]
    create_fleet->>Fleet: _BACKEND_REGISTRY[backend](...)
    MoEEpLayer->>Fleet: create_handle(HandleParams, handle_knobs)
    Fleet->>Handle: NcclEpHandle | NixlEpHandle
    MoEEpLayer->>Handle: dispatch(DispatchInputParams)
    Handle-->>MoEEpLayer: DispatchOutput
    MoEEpLayer->>MoEEpLayer: inner_compute (identity stub)
    MoEEpLayer->>Handle: combine(CombineInputParams)
    Handle-->>MoEEpLayer: CombineOutput
    MoEEpLayer->>Handle: complete()
    MoEEpLayer-->>User: torch.Tensor
```

---

## Complete class diagram

### Public API & layer

```mermaid
classDiagram
    direction TB

    class nn_Module {
        <<PyTorch>>
    }

    class MoEEpLayer {
        -BootstrapConfig _bootstrap
        -FleetParams _fleet_params
        -list~AlgoKnob~ _fleet_knobs
        -Union~str, object~ _backend
        -Fleet _fleet
        +__init__(bootstrap, fleet_params, fleet_knobs, backend)
        +forward(t: MoEEpTensors) torch.Tensor
        +destroy() void
        -_ensure_fleet() Fleet
        -_inner_compute_identity(expert_tensors, num_tokens) torch.Tensor
    }

    class MoEEpTensors {
        +torch.Tensor hidden_states
        +torch.Tensor topk_ids
        +torch.Tensor topk_weights
        +Optional~torch.Tensor~ scales
        +Optional~torch.Tensor~ recv_count
        +Optional~torch.Tensor~ num_tokens_per_expert
    }

    nn_Module <|-- MoEEpLayer
    MoEEpLayer ..> MoEEpTensors : forward input
    MoEEpLayer ..> Fleet : owns (lazy)
    MoEEpLayer ..> BootstrapConfig : uses
    MoEEpLayer ..> FleetParams : uses
    MoEEpLayer ..> AlgoKnob : fleet + handle knobs
```

### Fleet / Handle abstraction & backends

```mermaid
classDiagram
    direction TB

    class Fleet {
        <<abstract>>
        +__init__(bootstrap, params, algo_knobs)*
        +create_handle(params, algo_knobs)* Handle
        +update_topology(bootstrap, algo_knobs)*
        +destroy()*
    }

    class Handle {
        <<abstract>>
        +dispatch(params: DispatchInputParams)* DispatchOutput
        +combine(params: CombineInputParams)* CombineOutput
        +complete()*
        +dispatch_send_only(params) DispatchOutput
        +dispatch_recv_only() DispatchOutput
    }

    class NcclEpFleet {
        -FleetParams _params
        -dict _fleet_knobs
        -BootstrapConfig _bootstrap
        -int _stream
        -NCCLLibrary _lib
        -int _comm
        -ncclEpGroupConfig_t _cfg
        -ncclEpGroup_t _group
        -bool _destroyed
        +use_fp8 bool
        +use_ue8m0 bool
        +group c_void_p
        +stream int
        +params FleetParams
        +bootstrap BootstrapConfig
        -_build_group_config() ncclEpGroupConfig_t
        -_knob_or_auto(knob_cls, field) int
    }

    class NcclEpHandle {
        -NcclEpFleet _fleet
        -dict _handle_knobs
        -int _stream
        -bool _staged
        -NCCLLibrary _lib
        -NDTensor _topk_idx_nd
        -list~NDTensor~ _handle_local_nds
        -ncclEpHandle_t _handle
        -torch.Tensor _recv_count_t
        -NDTensor _recv_count_nd
        +dispatch(params) DispatchOutput
        +combine(params) CombineOutput
        +complete() void
        -_knob_stream() int
        -_build_handle_local_tensors() list~NDTensor~
        -_sync_stream() void
    }

    class NixlEpFleet {
        -FleetParams _params
        -dict _fleet_knobs
        -BootstrapConfig _bootstrap
        -int _capacity
        -nixl_ep.Buffer _buffer
        -bool _destroyed
        +use_fp8 bool
        +use_ue8m0 bool
        +buffer nixl_ep.Buffer
        +params FleetParams
    }

    class NixlEpHandle {
        -NixlEpFleet _fleet
        -dict _handle_knobs
        -bool _staged
        -torch.Tensor _topk_ids
        -Any _nixl_handle
        -Any _event
        -Callable _recv_hook
        +dispatch(params) DispatchOutput
        +combine(params) CombineOutput
        +complete() void
    }

    class NDTensor {
        -c_void_p _group
        -ncclNDTensor_t _handle
        -bool _owns
        -torch.dtype _dtype
        -tuple _shape
        -int _tag
        -torch.Tensor _keepalive
        -NCCLLibrary _lib
        +from_torch(group, tensor, tag)$ NDTensor
        +allocate(group, dtype, shape, tag)$ NDTensor
        +as_torch() torch.Tensor
        +handle c_void_p
        +tag int
        +shape tuple
        +dtype torch.dtype
    }

    Fleet <|-- NcclEpFleet
    Fleet <|-- NixlEpFleet
    Handle <|-- NcclEpHandle
    Handle <|-- NixlEpHandle

    NcclEpFleet ..> Handle : create_handle →
    NixlEpFleet ..> Handle : create_handle →
    NcclEpHandle --> NcclEpFleet : _fleet
    NixlEpHandle --> NixlEpFleet : _fleet
    NcclEpHandle ..> NDTensor : wraps tensors
```

### Config, enums & I/O envelopes

```mermaid
classDiagram
    direction TB

    class EpAlgorithm {
        <<enumeration>>
        LOW_LATENCY = 0
        HIGH_THROUGHPUT = 1
    }

    class QuantType {
        <<enumeration>>
        FP8E4M3
        FP8E5M2
        NVFP8
        UE8M0
    }

    class BootstrapConfig {
        <<frozen dataclass>>
        +int world_size
        +int rank
        +int stream
        +Optional~int~ nccl_comm
        +Optional~TCPStore~ tcp_store
    }

    class FleetParams {
        <<frozen dataclass>>
        +int num_experts
        +int max_tokens_per_rank
        +int token_hidden_size
        +int dtype_bytes
        +EpAlgorithm algorithm
    }

    class HandleParams {
        <<frozen dataclass>>
        +torch.Tensor topk_ids
    }

    class DispatchInputParams {
        <<frozen dataclass>>
        +Sequence~torch.Tensor~ x
    }

    class DispatchOutput {
        <<frozen dataclass>>
        +torch.Tensor expert_tensors
        +int num_tokens
    }

    class CombineInputParams {
        <<frozen dataclass>>
        +Sequence~torch.Tensor~ x
        +Optional~torch.Tensor~ out
    }

    class CombineOutput {
        <<frozen dataclass>>
        +torch.Tensor x
    }

    class NcclEpConfig {
        <<dataclass>>
        +str backend_name = "nccl_ep"
    }

    class NvepConfig {
        <<dataclass>>
        +str backend_name = "nixl_ep"
    }

    FleetParams --> EpAlgorithm
    Fleet ..> BootstrapConfig : constructed with
    Fleet ..> FleetParams : constructed with
    Handle ..> HandleParams : constructed with
    Handle ..> DispatchInputParams : dispatch()
    Handle ..> DispatchOutput : dispatch()
    Handle ..> CombineInputParams : combine()
    Handle ..> CombineOutput : combine()
    MoEEpLayer ..> NcclEpConfig : backend selector
    MoEEpLayer ..> NvepConfig : backend selector
```

### AlgoKnob hierarchy

```mermaid
classDiagram
    direction TB

    class AlgoKnob {
        <<marker base>>
    }

    class FleetAlgoKnobQuantization {
        <<frozen dataclass>>
        +FrozenSet~QuantType~ quants
    }

    class FleetAlgoKnobNumChannelsPerRank {
        <<frozen dataclass>>
        +int n
    }

    class FleetAlgoKnobNumQpsPerRank {
        <<frozen dataclass>>
        +int n
    }

    class FleetAlgoKnobRdmaBufferSize {
        <<frozen dataclass>>
        +int bytes_
    }

    class FleetAlgoKnobTopologyCapacity {
        <<frozen dataclass>>
        +int n
    }

    class HandleAlgoKnobUserStream {
        <<frozen dataclass>>
        +int stream
    }

    class HandleAlgoKnobSplitOperation {
        <<frozen dataclass, marker>>
    }

    class HandleAlgoKnobTopKWeights {
        <<frozen dataclass>>
        +torch.Tensor weights
    }

    class HandleAlgoKnobNumReceivedTokens {
        <<frozen dataclass>>
        +torch.Tensor target
    }

    AlgoKnob <|-- FleetAlgoKnobQuantization
    AlgoKnob <|-- FleetAlgoKnobNumChannelsPerRank
    AlgoKnob <|-- FleetAlgoKnobNumQpsPerRank
    AlgoKnob <|-- FleetAlgoKnobRdmaBufferSize
    AlgoKnob <|-- FleetAlgoKnobTopologyCapacity
    AlgoKnob <|-- HandleAlgoKnobUserStream
    AlgoKnob <|-- HandleAlgoKnobSplitOperation
    AlgoKnob <|-- HandleAlgoKnobTopKWeights
    AlgoKnob <|-- HandleAlgoKnobNumReceivedTokens

    FleetAlgoKnobQuantization ..> QuantType : quants
    NcclEpFleet ..> FleetAlgoKnobQuantization : reads at init
    NcclEpFleet ..> FleetAlgoKnobRdmaBufferSize : reads at init
    NcclEpFleet ..> FleetAlgoKnobNumQpsPerRank : reads at init
    NcclEpFleet ..> FleetAlgoKnobNumChannelsPerRank : reads at init
    NixlEpFleet ..> FleetAlgoKnobTopologyCapacity : reads at init
    NcclEpHandle ..> HandleAlgoKnobUserStream : per forward
    NcclEpHandle ..> HandleAlgoKnobTopKWeights : per forward
    NcclEpHandle ..> HandleAlgoKnobSplitOperation : optional
    NcclEpHandle ..> HandleAlgoKnobNumReceivedTokens : optional
    NixlEpHandle ..> HandleAlgoKnobTopKWeights : per forward
    NixlEpHandle ..> HandleAlgoKnobSplitOperation : optional
```

### Exceptions & validation

```mermaid
classDiagram
    direction TB

    class ValueError {
        <<stdlib>>
    }

    class RuntimeError {
        <<stdlib>>
    }

    class MoEEpConfigError {
        +validation failures
    }

    class MoEEpArchError {
        +unsupported GPU arch
    }

    class MoEEpNotBuiltError {
        +missing native libs
    }

    ValueError <|-- MoEEpConfigError
    MoEEpConfigError <|-- MoEEpArchError
    RuntimeError <|-- MoEEpNotBuiltError

    note for MoEEpConfigError "Raised by validate_fleet_params()\nand backend Fleet __init__"
    note for MoEEpNotBuiltError "Raised by _require_built(),\nget_nccl_lib(), _load_nixl_ep()"
```

---

## Backend registry & factory

Backends self-register at import time by assigning into `_BACKEND_REGISTRY`:

| Backend key | Fleet class | Native library | Bootstrap requirement |
|-------------|-------------|----------------|----------------------|
| `"nccl_ep"` | `NcclEpFleet` | `libnccl_ep.so` + `libnccl.so.2` (wheel) | `nccl_comm` or default PG |
| `"nixl_ep"` | `NixlEpFleet` | `nixl_ep_cpp*.so` + `libnixl.so` (wheel) | `tcp_store` required |

```mermaid
flowchart LR
    subgraph init ["__init__.py import side-effects"]
        A[import nccl_ep.fleet]
        B[import nixl_ep.fleet]
    end

    subgraph registry ["fleet._BACKEND_REGISTRY"]
        R1["nccl_ep → NcclEpFleet"]
        R2["nixl_ep → NixlEpFleet"]
    end

    subgraph factory ["create_fleet()"]
        F[resolve backend name]
    end

    A --> R1
    B --> R2
    F --> R1
    F --> R2
    MoEEpLayer --> factory
```

`create_fleet(bootstrap, params, algo_knobs, backend)` accepts either a string
(`"nccl_ep"` / `"nixl_ep"`) or a config object with a `.backend_name` attribute
(`NcclEpConfig`, `NvepConfig`).

---

## Module dependency graph

```mermaid
flowchart TB
    subgraph public ["Public surface"]
        init["__init__.py"]
        layer["layer.py"]
    end

    subgraph core ["Core abstractions"]
        fleet["fleet.py"]
        handle["handle.py"]
        config["config.py"]
        tensors["tensors.py"]
        knobs["algo_knobs.py"]
        validators["_validators.py"]
    end

    subgraph split ["split_backends/"]
        nccl_cfg["NcclEpConfig"]
        nvep_cfg["NvepConfig"]
    end

    subgraph nccl ["nccl_ep/"]
        nccl_fleet["NcclEpFleet"]
        nccl_hdl["NcclEpHandle"]
        ndtensor["NDTensor"]
    end

    subgraph nixl ["nixl_ep/"]
        nixl_fleet["NixlEpFleet"]
        nixl_hdl["NixlEpHandle"]
    end

    init --> fleet
    init --> handle
    init --> layer
    init --> config
    init --> knobs
    init --> validators
    init --> split
    init --> nccl_fleet
    init --> nixl_fleet

    layer --> fleet
    layer --> config
    layer --> knobs
    layer --> tensors

    fleet --> handle
    nccl_fleet --> fleet
    nccl_fleet --> handle
    nccl_fleet --> validators
    nccl_fleet --> knobs
    nccl_fleet --> config
    nccl_hdl --> handle
    nccl_hdl --> ndtensor

    nixl_fleet --> fleet
    nixl_fleet --> validators
    nixl_fleet --> knobs
    nixl_fleet --> config
    nixl_hdl --> handle

    knobs --> config
    validators --> config
```

---

## Class inventory (quick reference)

| Class | Module | Role |
|-------|--------|------|
| `MoEEpLayer` | `layer.py` | Public `nn.Module`; lazy Fleet + per-forward Handle |
| `Fleet` | `fleet.py` | ABC for durable EP transport (group / buffer) |
| `Handle` | `handle.py` | ABC for one dispatch/combine iteration |
| `NcclEpFleet` | `nccl_ep/fleet.py` | NCCL-EP `ncclEpGroup_t` owner |
| `NcclEpHandle` | `nccl_ep/handle.py` | NCCL-EP `ncclEpHandle_t` per forward |
| `NDTensor` | `nccl_ep/ndtensor.py` | `ncclNDTensor_t` ↔ torch bridge |
| `NixlEpFleet` | `nixl_ep/fleet.py` | NIXL-EP `Buffer` owner |
| `NixlEpHandle` | `nixl_ep/handle.py` | NIXL per-dispatch handle tuple |
| `BootstrapConfig` | `config.py` | Rank/world + NCCL comm or TCPStore |
| `FleetParams` | `config.py` | Expert count, token sizing, algorithm |
| `HandleParams` | `config.py` | Per-iteration `topk_ids` |
| `DispatchInputParams` / `DispatchOutput` | `config.py` | Dispatch I/O envelope |
| `CombineInputParams` / `CombineOutput` | `config.py` | Combine I/O envelope |
| `MoEEpTensors` | `tensors.py` | Layer forward input bundle |
| `AlgoKnob` + 9 subclasses | `algo_knobs.py` | Typed fleet/handle tuning knobs |
| `NcclEpConfig` / `NvepConfig` | `split_backends/` | Backend selector objects |
| `MoEEpConfigError` / `MoEEpArchError` | `_validators.py` | Config / arch validation |
| `MoEEpNotBuiltError` | `__init__.py` | Missing native build artifacts |
| `EpAlgorithm` / `QuantType` | `config.py` | Enums |

### Module-level functions (not classes)

| Function | Module | Role |
|----------|--------|------|
| `create_fleet()` | `fleet.py` | Factory via `_BACKEND_REGISTRY` |
| `available_backends()` | `__init__.py` | List built backend names |
| `have_nccl_ep()` / `have_nixl_ep()` | `__init__.py` | Probe staged `.so` files |
| `validate_fleet_params()` | `_validators.py` | Backend-specific sizing checks |
| `validate_arch_for_backend()` | `_validators.py` | sm_90+ gate |
| `_index_knobs()` | `algo_knobs.py` | `Sequence[AlgoKnob]` → `dict[type, AlgoKnob]` |
| `get_nccl_lib()` | `nccl_ep/ndtensor.py` | Singleton `NCCLLibrary` |
| `_load_libnccl_ep()` | `nccl_ep/__init__.py` | Lazy dlopen of EP plugin |
| `_load_nixl_ep_cpp()` | `nixl_ep/__init__.py` | Lazy dlopen of NIXL EP extension |

---

## Ownership & lifetimes

```mermaid
stateDiagram-v2
    [*] --> MoEEpLayerCreated: MoEEpLayer.__init__
    MoEEpLayerCreated --> FleetAlive: first forward → create_fleet
    FleetAlive --> HandlePerForward: create_handle (each forward)
    HandlePerForward --> HandlePerForward: dispatch → combine → complete
    HandlePerForward --> FleetAlive: Handle GC / __del__
    FleetAlive --> [*]: MoEEpLayer.destroy() → Fleet.destroy()
```

- **Fleet** — one per `MoEEpLayer` instance (lazy); owns NCCL EP group or NIXL
  buffer until `destroy()`.
- **Handle** — one per forward pass; short-lived; destroyed in `__del__`
  (`NcclEpHandle`) or stateless after combine (`NixlEpHandle`).
- **NDTensor** — per tensor slot in NCCL dispatch/combine; borrows torch storage
  (`from_torch`) or owns library allocation (`allocate`).

---

## External native dependencies

| Backend | Python binding | Staged in package | From pip wheel |
|---------|----------------|-------------------|----------------|
| NCCL-EP | `nccl_ep.NCCLLibrary` | `nccl_ep/_libs/libnccl_ep.so` | `nvidia-nccl-cu13` → `libnccl.so.2` |
| NIXL-EP | `nixl_ep.Buffer` | `nixl_ep/_libs/nixl_ep_cpp*.so` | `nixl-cu13` → `libnixl.so` + siblings |

Build gate: `BUILD_NVEP=1` (or per-backend `BUILD_NCCL_EP` / `BUILD_NIXL_EP`).
