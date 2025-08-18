import torch

from ..xqa import get_xqa_module

def xqa_decode(
    use_fp16 : bool,
    token_per_page : int,
    head_size : int,
    head_grp_size : int,
    multiProcessorCount : int,
    nbKHeads : int,
    qScale : float, 
    output : torch.Tensor,
    q : torch.Tensor,
    attentionSinks : torch.Tensor,
    pool : torch.Tensor,
    kvCachePageList : torch.Tensor,
    maxSeqLen : int, 
    seqLen : torch.Tensor,
    batchSize : int,
    kvCacheScale : torch.Tensor,
    semaphores : torch.Tensor, 
    scratch : torch.Tensor
) -> None:
    xqa_module = get_xqa_module(use_fp16, token_per_page, head_size, head_grp_size)
    xqa_module.xqa(
            multiProcessorCount,
            nbKHeads,
            qScale, 
            output,
            q,
            attentionSinks,
            pool,
            kvCachePageList,
            maxSeqLen, 
            seqLen,
            batchSize,
            kvCacheScale,
            semaphores, 
            scratch)