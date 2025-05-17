import ctypes
import multiprocessing as mp
import random
import socket
import unittest
from typing import Any, List, Optional
import torch

import flashinfer.trtllm_all_reduce as trtllm_ops

# temp test for compilation
if __name__ == "__main__":
    params = trtllm_ops.AllReduceFusionParams(
        workspace=torch.zeros(1), # todo
        allreduce_in=torch.zeros(1), # todo
        nranks=2,
        rank=0,
        dtype=trtllm_ops.DataType.FP16,
        size=1024,
        hidden_dim=1024,
    )
    trtllm_ops.allreduce_fusion_op(params)
    trtllm_ops.moereduction_allreduce_fusion_op(params)
