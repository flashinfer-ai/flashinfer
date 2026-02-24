import logging
from typing import ClassVar, Dict, List, Optional, Type

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


class AttnParallelConfig:
    """Base configuration class for ring and Ulysses parallelism with singleton pattern.

    Attributes:
        ulysses_size (int): Ulysses parallel degree, defaults to 1
        ring_size (int): Ring attention parallel degree, defaults to 1
    """

    _instances: ClassVar[Dict[Type["AttnParallelConfig"], "AttnParallelConfig"]] = {}
    _initialized: ClassVar[Dict[Type["AttnParallelConfig"], bool]] = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._initialized[cls] = False
            logger.debug(f"[{cls.__name__}] Created new {cls.__name__} instance")
        return cls._instances[cls]

    def __init__(self, *args, **kwargs):
        if not self._initialized[self.__class__]:
            super().__init__(*args, **kwargs)
            self._initialized[self.__class__] = True
            logger.debug(
                f"[{self.__class__.__name__}] Initialized {self.__class__.__name__}"
            )
            self._ulysses_size = 1
            self._ring_size = 1
            self._device_mesh = None
            self._cached_device_mesh = {}
            self.set_device_mesh()


    def set_device_mesh(
        self, device_type: str = "cuda"
    ) -> torch.distributed.DeviceMesh:
        """Initialize the device mesh for distributed training.

        Args:
            device_type (str): The type of device to use, defaults to "cuda"

        Returns:
            torch.distributed.DeviceMesh: The initialized device mesh

        Note:
            The device mesh is created in the order: ring -> ulysses
        """
        if self._device_mesh is not None:
            logger.debug(
                f"[{self.__class__.__name__}] Device mesh already initialized"
            )
            return self._device_mesh

        total_parallel_size = self.get_total_parallel_size()
        world_size = self.get_world_size()
        if world_size % total_parallel_size != 0:
            raise ValueError(
                f"World size ({world_size}) is not divisible by "
                f"total parallel size ({total_parallel_size})"
            )

        logger.debug(
            f"[{self.__class__.__name__}] Setting up device mesh with "
            f"total parallel size: {total_parallel_size}"
        )

        if total_parallel_size == 1:
            logger.debug(
                f"[{self.__class__.__name__}] No parallelism needed, "
                f"skipping device mesh setup"
            )
            self._device_mesh = None
            return self._device_mesh

        mesh_dims = []
        mesh_sizes = []

        if world_size != total_parallel_size:
            mesh_dims.append("redundant")
            mesh_sizes.append(world_size // total_parallel_size)
            logger.debug(
                f"[{self.__class__.__name__}] Added redundant dimension: "
                f"{world_size // total_parallel_size}, "
                f"world_size={world_size}, "
                f"total_parallel_size={total_parallel_size}"
            )

        if str(self) in self._cached_device_mesh:
            self._device_mesh = self._cached_device_mesh[str(self)]
            return self._device_mesh

        if self._ring_size > 1:
            mesh_dims.append("ring")
            mesh_sizes.append(self._ring_size)
            logger.debug(
                f"[{self.__class__.__name__}] Added Ring dimension: {self._ring_size}"
            )
        if self._ulysses_size > 1:
            mesh_dims.append("ulysses")
            mesh_sizes.append(self._ulysses_size)
            logger.debug(
                f"[{self.__class__.__name__}] Added Ulysses dimension: "
                f"{self._ulysses_size}"
            )

        if not mesh_dims:
            logger.debug(
                f"[{self.__class__.__name__}] No mesh dimensions needed"
            )
            self._device_mesh = None
        else:
            logger.info(
                f"[{self.__class__.__name__}] Creating device mesh: "
                f"dims={mesh_dims}, sizes={mesh_sizes}"
            )
            self._device_mesh = init_device_mesh(
                device_type,
                tuple(mesh_sizes),
                mesh_dim_names=tuple(mesh_dims),
            )
            logger.info(
                f"[{self.__class__.__name__}] Device mesh created successfully"
            )

        if str(self) not in self._cached_device_mesh:
            self._cached_device_mesh[str(self)] = self._device_mesh

        return self._device_mesh



    @classmethod
    def get_group(cls, group_name: str) -> Optional[torch.distributed.ProcessGroup]:
        """Get the process group.

        Args:
            group_name (str): The name of the process group
        """
        instance = cls.get_instance()
        if instance._device_mesh is None:
            return None
        return instance._device_mesh.get_group(group_name)

    @classmethod
    def ring_group(cls) -> Optional[torch.distributed.ProcessGroup]:
        """Get the ring attention process group.

        Returns:
            Optional[torch.distributed.ProcessGroup]: The ring attention process group,
                or None if ring_size=1
        """
        instance = cls.get_instance()
        if instance._ring_size <= 1 or instance._device_mesh is None:
            return None
        group = cls.get_group("ring")
        logger.debug(f"[{cls.__name__}] Retrieved Ring group: {group}")
        return group

    @classmethod
    def ulysses_group(cls) -> Optional[torch.distributed.ProcessGroup]:
        """Get the Ulysses process group.

        Returns:
            Optional[torch.distributed.ProcessGroup]: The Ulysses process group,
                or None if ulysses_size=1
        """
        instance = cls.get_instance()
        if instance._ulysses_size <= 1 or instance._device_mesh is None:
            return None
        group = cls.get_group("ulysses")
        logger.debug(f"[{cls.__name__}] Retrieved Ulysses group: {group}")
        return group

    @classmethod
    def get_local_rank(cls, group_name: str) -> int:
        """Get the local rank within a named group.

        Returns:
            int: local rank within the group
        """
        instance = cls.get_instance()
        if instance._device_mesh is None:
            raise RuntimeError("Device mesh not initialized")
        if group_name not in instance._device_mesh.mesh_dim_names:
            raise RuntimeError(f"Group {group_name} not found in device mesh")
        return instance._device_mesh.get_local_rank(group_name)

    @classmethod
    def ring_rank(cls) -> int:
        """Get the local rank in the ring attention group.

        Returns:
            int: Local rank in the ring attention group (0 to ring_size-1)
        """
        instance = cls.get_instance()
        if instance._ring_size <= 1:
            return 0
        rank = cls.get_local_rank("ring")
        logger.debug(f"[{cls.__name__}] Ring rank: {rank}")
        return rank

    @classmethod
    def ring_ranks(cls) -> List[int]:
        """Get all the local ranks in the ring attention group."""
        instance = cls.get_instance()
        return instance._device_mesh["ring"].mesh.flatten().tolist()

    @classmethod
    def ulysses_rank(cls) -> int:
        """Get the local rank in the Ulysses group.

        Returns:
            int: Local rank in the Ulysses group (0 to ulysses_size-1)
        """
        instance = cls.get_instance()
        if instance._ulysses_size <= 1:
            return 0
        rank = cls.get_local_rank("ulysses")
        logger.debug(f"[{cls.__name__}] Ulysses rank: {rank}")
        return rank

    @classmethod
    def ulysses_ranks(cls) -> List[int]:
        """Get all the local ranks in the Ulysses group."""
        instance = cls.get_instance()
        return instance._device_mesh["ulysses"].mesh.flatten().tolist()

    @classmethod
    def get_instance(cls) -> "AttnParallelConfig":
        """Get the singleton instance of parallel configuration.

        Returns:
            AttnParallelConfig: The singleton instance of the configuration

        Note:
            This method ensures that each subclass has its own singleton instance.
            For example, VAEParallelConfig.get_instance() and
            DiTParallelConfig.get_instance() will return different instances.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._initialized[cls] = False
            instance = cls._instances[cls]
            instance.__init__()
        return cls._instances[cls]

    @classmethod
    def set_config(
        cls,
        ulysses_size: int = 1,
        ring_size: int = 1,
    ) -> None:
        """Set the configuration values for parallelism and reinitialize device mesh.

        Args:
            ulysses_size: Ulysses parallel degree, defaults to 1
            ring_size: Ring attention parallel degree, defaults to 1

        Raises:
            ValueError: If parallel configuration is invalid
        """
        if not isinstance(ulysses_size, int) or ulysses_size < 1:
            raise ValueError(
                f"ulysses_size must be a positive integer, got {ulysses_size}"
            )
        if not isinstance(ring_size, int) or ring_size < 1:
            raise ValueError(
                f"ring_size must be a positive integer, got {ring_size}"
            )

        total_size = ulysses_size * ring_size
        if total_size > torch.cuda.device_count():
            raise ValueError(
                f"Total parallel size ({total_size}) exceeds available "
                f"GPU count ({torch.cuda.device_count()})"
            )

        instance = cls.get_instance()

        instance._ulysses_size = ulysses_size
        instance._ring_size = ring_size
        if total_size > 1:
            instance._device_mesh = None
            instance.set_device_mesh()
        instance.check_parallel_size()

    def get_world_size(self) -> int:
        """Get the world size.

        Returns:
            int: World size
        """
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    def check_parallel_size(self) -> bool:
        """Check if the total parallel size is compatible with the world size.

        Returns:
            bool: True if the total parallel size is compatible with the world size
        """
        total_parallel_size = self.get_total_parallel_size()
        if total_parallel_size == 1:
            logger.debug(
                f"[{self.__class__.__name__}] Total parallel size is 1, "
                f"skipping check"
            )
            return True
        if self.get_world_size() % total_parallel_size != 0:
            par_size = (
                f"ULYSSES: {self._ulysses_size}, RING: {self._ring_size}"
            )
            raise ValueError(
                f"Parallel size {par_size} does not match "
                f"world size ({self.get_world_size()})"
            )
        return True

    @classmethod
    def get_total_parallel_size(cls) -> int:
        """Calculate the total parallel size by multiplying all parallel degrees.

        Returns:
            int: Total parallel size
        """
        return cls.ulysses_size() * cls.ring_size()

    def __str__(self) -> str:
        """String representation of the parallel configuration."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  ulysses_size={self._ulysses_size},\n"
            f"  ring_size={self._ring_size}\n"
            f")"
        )

    @classmethod
    def ulysses_size(cls) -> int:
        """Get the Ulysses parallel size.

        Returns:
            int: Ulysses parallel size
        """
        return cls.get_instance()._ulysses_size

    @classmethod
    def ring_size(cls) -> int:
        """Get the ring attention parallel size.

        Returns:
            int: Ring attention parallel size
        """
        return cls.get_instance()._ring_size


class UnevenCPConfig:
    """Configuration for uneven context parallelism.

    Handles the case where the total sequence length is not evenly divisible
    by the number of ranks. Each rank may hold a different number of tokens,
    and the last rank typically gets fewer tokens (the remainder).

    This config gathers per-rank sequence lengths via ``all_gather`` so that
    the parallel wrappers know how to truncate padding and zero out extra
    output positions.

    Attributes:
        seq_len: Actual (unpadded) total sequence length.
        seq_len_padded: Padded total sequence length (divisible by world_size).
        seq_len_all_ranks: Tensor of per-rank sequence lengths for all ranks.
        seq_len_cur_ring_group: Tensor of per-rank sequence lengths within
            the current ring group.
    """

    def __init__(self):
        self.seq_len = None
        self.seq_len_padded = None
        self.seq_len_all_ranks = None
        self.seq_len_cur_ring_group = None

    def set_uneven_cp_config(
        self, seq_len, seq_len_padded, seq_len_cur_rank, attn_parallel_config
    ):
        """Gather per-rank sequence lengths and compute the current ring group sequence length.

        Args:
            seq_len: Actual (unpadded) total sequence length.
            seq_len_padded: Padded total sequence length (divisible by world_size).
            seq_len_cur_rank: Number of real (non-padding) tokens on this rank. for example, if the total sequence 
            length is 1023 and the world size is 8, and the rank is 0, then seq_len_cur_rank is 128. If the rank is 7, then seq_len_cur_rank is 127.
            attn_parallel_config: The :class:`AttnParallelConfig` instance that provides ring/ulysses group information.
        """
        self.seq_len = seq_len
        self.seq_len_padded = seq_len_padded

        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")

        seq_len_cur_rank = torch.tensor(
            [seq_len_cur_rank], dtype=torch.int32, device=device
        )
        gather_list = [
            torch.empty_like(seq_len_cur_rank)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gather_list, seq_len_cur_rank)
        self.seq_len_all_ranks = torch.cat(gather_list, dim=0).cpu()

        if attn_parallel_config.ring_size() == 1:
            return
        if attn_parallel_config.ulysses_size() == 1:
            self.seq_len_cur_ring_group = self.seq_len_all_ranks[
                torch.tensor(attn_parallel_config.ring_ranks())
            ]
            return

        seq_len_cur_ulysses_group = self.seq_len_all_ranks[
            torch.tensor(attn_parallel_config.ulysses_ranks())
        ]
        ring_seq_cur_ring_rank = torch.sum(
            seq_len_cur_ulysses_group, dtype=torch.int32
        ).to(device)
        gather_list = [
            torch.empty(1, dtype=torch.int32, device=device)
            for _ in range(attn_parallel_config.ring_size())
        ]

        torch.distributed.all_gather(
            gather_list,
            ring_seq_cur_ring_rank,
            group=attn_parallel_config.ring_group(),
        )
        self.seq_len_cur_ring_group = torch.cat(gather_list, dim=0)

    def reset(self):
        self.seq_len = None
        self.seq_len_padded = None
        self.seq_len_all_ranks = None
        self.seq_len_cur_ring_group = None


class VarlenCPConfig:
    """Configuration for variable-length context parallelism.

    Handles the case where multiple sequences of different lengths are packed
    together (varlen). Cumulative sequence length arrays
    (``cu_seqlens``) are computed so that the attention kernel can correctly
    identify sequence boundaries.

    Supports two modes (mutually exclusive):

    - **Ulysses-only** (``ring_size == 1``): The packed sequences are treated
      as a whole and split across heads via all-to-all. No per-sequence
      splitting is needed — only the overall ``cu_seqlens`` are stored so the
      attention kernel knows where each sequence starts and ends.
    - **Ring-only** (``ulysses_size == 1``): Each individual sequence is split
      across ranks along the sequence dimension. ``cu_seqlens`` are stored as
      a 2D tensor of shape ``[ring_size, num_seqs + 1]``, one row per rank,
      because each rank holds a different slice of every sequence.

    Attributes:
        cu_seqlens_q_cur_ulysses_group: Cumulative query sequence lengths for
            the current ulysses group (shared across all ulysses ranks).
        cu_seqlens_kv_cur_ulysses_group: Cumulative key/value sequence lengths
            for the current ulysses group.
        max_seq_len_q_cur_ulysses_group: Max query sequence length in the
            current ulysses group.
        max_seq_len_kv_cur_ulysses_group: Max key/value sequence length in the
            current ulysses group.
        cu_seqlens_q_cur_ring_group: Cumulative query sequence lengths for all
            ranks in the current ring group, shape ``[ring_size, num_seqs + 1]``.
        cu_seqlens_kv_cur_ring_group: Cumulative key/value sequence lengths for
            all ranks in the current ring group.
        max_seq_len_q_cur_ring_group: Max query sequence length across all
            ranks in the ring group (per-rank padded length).
        max_seq_len_kv_cur_ring_group: Max key/value sequence length across all
            ranks in the ring group (per-rank padded length).
    """

    def __init__(self):
        self.cu_seqlens_q_cur_ulysses_group = None
        self.cu_seqlens_kv_cur_ulysses_group = None
        self.max_seq_len_q_cur_ulysses_group = None
        self.max_seq_len_kv_cur_ulysses_group = None
        self.cu_seqlens_q_cur_ring_group = None
        self.cu_seqlens_kv_cur_ring_group = None
        self.max_seq_len_q_cur_ring_group = None
        self.max_seq_len_kv_cur_ring_group = None

    def set_varlen_cp_config(
        self,
        cu_seqlens_q_all_ranks,
        cu_seqlens_kv_all_ranks,
        max_seq_len_q,
        max_seq_len_kv,
        attn_parallel_config,
    ):
        if attn_parallel_config.ring_size() == 1:
            self.cu_seqlens_q_cur_ulysses_group = cu_seqlens_q_all_ranks
            self.cu_seqlens_kv_cur_ulysses_group = cu_seqlens_kv_all_ranks
            self.max_seq_len_q_cur_ulysses_group = max_seq_len_q
            self.max_seq_len_kv_cur_ulysses_group = max_seq_len_kv
            return

        if attn_parallel_config.ulysses_size() == 1:
            self.cu_seqlens_q_cur_ring_group = cu_seqlens_q_all_ranks
            self.cu_seqlens_kv_cur_ring_group = cu_seqlens_kv_all_ranks
            self.max_seq_len_q_cur_ring_group = max_seq_len_q
            self.max_seq_len_kv_cur_ring_group = max_seq_len_kv
            return

        raise NotImplementedError(
            "Varlen CP only supported when ulysses_size == 1 or ring_size == 1"
        )

    def set_ulysses_varlen_config(
        self, seq_lens_q, seq_lens_kv, attn_parallel_config
    ):
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")

        cu_seqlens_q = torch.cat([
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(torch.tensor(seq_lens_q, dtype=torch.int32), dim=0),
        ]).to(device).to(torch.int32)

        cu_seqlens_kv = torch.cat([
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(torch.tensor(seq_lens_kv, dtype=torch.int32), dim=0),
        ]).to(device).to(torch.int32)

        max_seqlen_q = max(seq_lens_q)
        max_seqlen_k = max(seq_lens_kv)

        self.set_varlen_cp_config(
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_k,
            attn_parallel_config,
        )

    def set_ring_varlen_config(
        self, seq_lens_q, seq_lens_kv, attn_parallel_config
    ):
        if not isinstance(seq_lens_q, torch.Tensor):
            seq_lens_q = torch.tensor(seq_lens_q, dtype=torch.int32)
        if not isinstance(seq_lens_kv, torch.Tensor):
            seq_lens_kv = torch.tensor(seq_lens_kv, dtype=torch.int32)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")

        padded_seq_lens_q = (
            (seq_lens_q + world_size - 1) // world_size * world_size
        )
        padded_seq_lens_kv = (
            (seq_lens_kv + world_size - 1) // world_size * world_size
        )

        padded_seq_len_q_cur_rank = padded_seq_lens_q // world_size
        padded_seq_len_kv_cur_rank = padded_seq_lens_kv // world_size

        max_seq_len_q = padded_seq_len_q_cur_rank.max()
        max_seq_len_kv = padded_seq_len_kv_cur_rank.max()

        cu_seqlens_q_all_ranks = []
        cu_seqlens_kv_all_ranks = []

        for i in range(world_size):
            if i == world_size - 1:
                seq_len_q_cur_rank = padded_seq_len_q_cur_rank - (
                    padded_seq_lens_q - seq_lens_q
                )
                seq_len_kv_cur_rank = padded_seq_len_kv_cur_rank - (
                    padded_seq_lens_kv - seq_lens_kv
                )
            else:
                seq_len_q_cur_rank = padded_seq_len_q_cur_rank
                seq_len_kv_cur_rank = padded_seq_len_kv_cur_rank

            cu_seqlens_q = torch.cat([
                torch.zeros(1),
                torch.cumsum(seq_len_q_cur_rank, dim=0),
            ]).to(device).to(torch.int32)
            cu_seqlens_q_all_ranks.append(cu_seqlens_q)

            cu_seqlens_kv = torch.cat([
                torch.zeros(1),
                torch.cumsum(seq_len_kv_cur_rank, dim=0),
            ]).to(device).to(torch.int32)
            cu_seqlens_kv_all_ranks.append(cu_seqlens_kv)

        self.set_varlen_cp_config(
            torch.stack(cu_seqlens_q_all_ranks),
            torch.stack(cu_seqlens_kv_all_ranks),
            max_seq_len_q,
            max_seq_len_kv,
            attn_parallel_config,
        )

    def reset(self):
        self.cu_seqlens_q_cur_ulysses_group = None
        self.cu_seqlens_kv_cur_ulysses_group = None
        self.max_seq_len_q_cur_ulysses_group = None
        self.max_seq_len_kv_cur_ulysses_group = None
        self.cu_seqlens_q_cur_ring_group = None
        self.cu_seqlens_kv_cur_ring_group = None
        self.max_seq_len_q_cur_ring_group = None
        self.max_seq_len_kv_cur_ring_group = None
