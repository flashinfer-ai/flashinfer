import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm

if __name__ == "__main__":
    mod = comm.get_comm_module()
