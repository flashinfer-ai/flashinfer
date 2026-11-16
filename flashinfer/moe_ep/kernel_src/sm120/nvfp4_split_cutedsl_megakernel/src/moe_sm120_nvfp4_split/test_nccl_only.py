"""Host policy regressions; GPU dependency audits are in the campaign evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from .transport.gin_runtime import select_side

@pytest.mark.parametrize('tokens',[1,8,15,16,17,256,640,24575,24576,24577,32768,65536])
def test_gin_policy_has_no_token_whitelist(tokens,monkeypatch):
 monkeypatch.delenv('MEGA_GIN_SIDE',raising=False)
 monkeypatch.delenv('MEGAMOE_NETWORK_BACKEND',raising=False)
 assert select_side(SimpleNamespace(num_tokens_per_rank=tokens),8)=='both'

def test_no_empty_input_or_split_backend(monkeypatch):
 monkeypatch.delenv('MEGA_GIN_SIDE',raising=False)
 monkeypatch.delenv('MEGAMOE_NETWORK_BACKEND',raising=False)
 with pytest.raises(ValueError):select_side(SimpleNamespace(num_tokens_per_rank=0),8)
 for value in ['nvshmem','dispatch','none']:
  monkeypatch.setenv('MEGAMOE_NETWORK_BACKEND',value)
  with pytest.raises(ValueError):select_side(SimpleNamespace(num_tokens_per_rank=256),8)
 monkeypatch.setenv('MEGAMOE_NETWORK_BACKEND','gin')
 monkeypatch.setenv('MEGA_GIN_SIDE','dispatch')
 with pytest.raises(ValueError):select_side(SimpleNamespace(num_tokens_per_rank=256),8)

def test_no_nvshmem_python_import_dependency():
 for path in Path(__file__).parent.rglob('*.py'):
  tree=ast.parse(path.read_text())
  for node in ast.walk(tree):
   names=([node.module] if isinstance(node,ast.ImportFrom) else
          [n.name for n in node.names] if isinstance(node,ast.Import) else [])
   assert not any(n and n.split('.')[0]=='nvshmem' for n in names),path


def test_reject_framework_with_nvshmem_library(monkeypatch):
 from .transport.gin_runtime import require_nccl_only_process
 monkeypatch.setattr(Path,'read_text',lambda self: '123-456 r-xp 0 00:00 0 /opt/torch/lib/libtorch_nvshmem.so\n')
 with pytest.raises(RuntimeError,match='qualified clean Torch'):
  require_nccl_only_process()

def test_allow_nccl_only_process(monkeypatch):
 from .transport.gin_runtime import require_nccl_only_process
 monkeypatch.setattr(Path,'read_text',lambda self: '123-456 r-xp 0 00:00 0 /opt/lib/libnccl.so.2\n')
 require_nccl_only_process()


def test_bounded_byte_comparison_preserves_exact_count():
 import torch
 from .combine_reference import byte_difference
 a=torch.arange(41,dtype=torch.uint8);b=a.clone()
 for i in [0,7,8,15,16,31,40]:b[i]^=1
 for chunk in [1,7,8,16,100]:
  assert byte_difference(a,b,chunk_bytes=chunk)==7
  assert byte_difference(a,a,chunk_bytes=chunk)==0
 assert byte_difference(a[:0],b[:0],chunk_bytes=8)==0
 with pytest.raises(ValueError):byte_difference(a,b,chunk_bytes=0)
