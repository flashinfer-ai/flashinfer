class CustomAllReduceHelper:
    """
        Globally visible class to help usage of custom_all_reduce plugin.
        Provides the following utilities:

        workspace: Tensor
            When using CUSTOM or AUTO mode, a tensor containing pointers to memory
            visible to all GPUs. It should be 3 pointers per TP rank -
            ptr to data buffer, ptr to barriers in, ptr to barriers out.
            It must be initialized using IpcMemory class.

        Usage:
            - Set custom_all_reduce_helper.workspace with the required tensor.
              Then, each instance of allreduce will reference that tensor automatically.
    """
    POINTERS_PER_RANK = 7
    POINTERS_OF_COUNTER = 2

    def __init__(self) -> None:
        self.workspace: Optional[Tensor] = None

    def set_workspace_tensor(self,
                             mapping: Mapping,
                             num_profiles: Optional[int] = None):
        from ..functional import Tensor
        workspace_size = self.POINTERS_PER_RANK * mapping.tp_size + self.POINTERS_OF_COUNTER

        dim_range = None
        if num_profiles is not None:
            dim_range = OrderedDict([('all_reduce_size',
                                      [workspace_size] * num_profiles)])

        self.workspace = Tensor(
            name='all_reduce_workspace',
            dtype=trt.int64,
            shape=[workspace_size],
            dim_range=dim_range,
        )

    @staticmethod
    def max_workspace_size_auto(tp_size: int,
                                support_deterministic=True) -> int:
        if force_all_reduce_deterministic() and support_deterministic:
            workspace_size = os.getenv("FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE",
                                       "1000000000")
            return int(workspace_size)
        if tp_size <= 2:
            return 16_000_000
        return 8_000_000

    @staticmethod
    def allocate_workspace(mapping: Mapping,
                           size: int) -> Tuple[List[IpcMemory], "torch.tensor"]:
        import torch

        # Force pull mode and disable lamport when force deterministic is enabled, for reducing device memory usage.
        force_deterministic = force_all_reduce_deterministic()
        is_p2p_supported = can_access_peer(mapping)
        ipc_buffers_size = size if force_deterministic else size * mapping.tp_size
        ipc_buffers_ping = IpcMemory(mapping, ipc_buffers_size,
                                     is_p2p_supported)
        ipc_buffers_pong = IpcMemory(mapping, ipc_buffers_size,
                                     is_p2p_supported)
        ipc_barriers_in = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size * 2,
            is_p2p_supported)
        ipc_barriers_out = IpcMemory(
            mapping, IpcMemory.IPC_BARRIERS_SIZE_PER_GPU * mapping.tp_size * 2,
            is_p2p_supported)
        lamport_buffers_size = 1 if force_deterministic else size * mapping.tp_size
        lamport_buffers_0 = IpcMemory(mapping, lamport_buffers_size,
                                      is_p2p_supported)
        lamport_buffers_1 = IpcMemory(mapping, lamport_buffers_size,
                                      is_p2p_supported)
        lamport_buffers_2 = IpcMemory(mapping, lamport_buffers_size,
                                      is_p2p_supported)
        # TODO: it seems we may need to initialize lamport buffers for all tp groups
        # just like its cpp counterpart (AllReduceBuffers::AllReduceBuffers()) does.
        if is_p2p_supported:
            lamport_initialize_all(
                lamport_buffers_0.local_ptr,
                lamport_buffers_1.local_ptr,
                lamport_buffers_2.local_ptr,
                lamport_buffers_size,
            )
        buffers = [
            ipc_buffers_ping, ipc_buffers_pong, ipc_barriers_in,
            ipc_barriers_out, lamport_buffers_0, lamport_buffers_1,
            lamport_buffers_2
        ]

        return buffers, torch.tensor(
            ipc_buffers_ping.serialize() + ipc_buffers_pong.serialize() +
            ipc_barriers_in.serialize() + ipc_barriers_out.serialize() +
            lamport_buffers_0.serialize() + lamport_buffers_1.serialize() +
            lamport_buffers_2.serialize() + [0] + [0],
            dtype=torch.int64,
            device="cpu")

    @staticmethod
    def allocate_allreduce_fusion_workspace(
            mapping: Mapping,
            size: int) -> Tuple[List[IpcMemory], "torch.tensor"]:
        import torch
        is_p2p_supported = can_access_peer(mapping)
        ipc_buffers_size = size * mapping.tp_size
        ipc_buffers = IpcMemory(mapping, ipc_buffers_size, is_p2p_supported)
        ipc_barriers = IpcMemory(mapping, 256 * mapping.tp_size,
                                 is_p2p_supported)
        lamport_buffers_size = size * mapping.tp_size
        lamport_buffers = IpcMemory(mapping, 3 * lamport_buffers_size,
                                    is_p2p_supported)
        if is_p2p_supported:
            lamport_initialize(
                lamport_buffers.local_ptr,
                3 * lamport_buffers_size,
            )
        flag_buffer = torch.tensor([0, 0, 0, lamport_buffers_size, 0],
                                   dtype=torch.int,
                                   device="cuda")
        buffers = [ipc_buffers, ipc_barriers, lamport_buffers, flag_buffer]

        return buffers, torch.tensor(
            ipc_buffers.serialize() + ipc_barriers.serialize() +
            lamport_buffers.serialize() + [flag_buffer.data_ptr()],
            dtype=torch.int64,
            device="cuda")


custom_all_reduce_helper = None


def init_all_reduce_helper():
    global custom_all_reduce_helper
    custom_all_reduce_helper = CustomAllReduceHelper()


def current_all_reduce_helper():
    global custom_all_reduce_helper
    assert custom_all_reduce_helper is not None, "You must call `init_all_reduce_helper` first"
    return custom_all_reduce_helper