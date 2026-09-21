"""Frozen tensor binding expressions from the selected native preparation boundary."""

def make_bindings(layout, views, weights, y):
    import torch
    config = layout.config
    T, H, I = config.num_tokens, config.hidden, config.intermediate
    sms, shared = layout.num_sms, bool(config.num_shared_experts)
    b1, b2, sb1, sb2 = (weights[name] for name in ("B1", "B2", "SB1", "SB2"))
    sfb1, sfb2, ssfb1, ssfb2 = (weights[name] for name in ("SFB1", "SFB2", "SSFB1", "SSFB2"))
    shared_a2 = views["shared_l2_acts"] if shared else views["l2_acts"]
    shared_sf1 = views["shared_l1_sf"] if shared else views["l1_sf"]
    shared_sf2 = views["shared_l2_sf"] if shared else views["l2_sf"]
    bindings = dict(
            A1=views["l1_acts"].view(torch.uint8), A2=views["l2_acts"].view(torch.uint8),
            SA1=views["x"].view(torch.uint8), SA2=shared_a2.view(torch.uint8),
            B1=b1, B2=b2, SB1=sb1, SB2=sb2,
            SFA1=views["l1_sf"].view(torch.uint32), SFA2=views["l2_sf"].view(torch.uint32),
            SSFA1=shared_sf1.view(torch.uint32), SSFA2=shared_sf2.view(torch.uint32),
            SFB1=sfb1, SFB2=sfb2, SSFB1=ssfb1, SSFB2=ssfb2,
            L1Output=views["l2_acts"].view(torch.uint8), SharedL1Output=shared_a2.view(torch.uint8),
            X=views["x"].view(torch.uint8).reshape(-1), XSF=views["x_sf"].view(torch.uint32).reshape(-1),
            TopK=views["topk_idx"].reshape(-1), Weights=views["topk_weights"].reshape(-1),
            L1Acts=views["l1_acts"].view(torch.uint8).reshape(-1), L1SF=views["l1_sf"].view(torch.uint32).reshape(-1),
            L1Weights=views["l1_topk_weights"], L2SF=views["l2_sf"].view(torch.uint8).reshape(-1),
            SharedL2SF=shared_sf2.view(torch.uint8).reshape(-1), SourceIndices=views["src_token_topk"].reshape(-1),
            TokenMetadata=views["token_src_metadata"].reshape(-1),
            GridCounters=views["grid_sync_count"], NvlCounter=views["nvl_barrier_counter"],
            NvlSignals=views["nvl_barrier_signals"].view(torch.uint32),
            PeerGrid=views["peer_grid_idx"], ReadyGrid=views["combine_ready_grid_idx"],
            SendCounts=views["expert_send_count"], RecvCounts=views["expert_recv_count"], RecvSum=views["expert_recv_count_sum"],
            L1Full=views["l1_full_count"], L1Empty=views["l1_empty_count"], L2Mask=views["l2_full_mask"], L2Empty=views["l2_empty_count"],
            SharedFull=views["shared_l2_full_count"], L1Counter=views["l1_task_count"], L2Counter=views["l2_task_count"],
            SharedL1Counter=views["shared_l1_task_count"], SharedL2Counter=views["shared_l2_task_count"],
            Combine=views["combine"].view(torch.uint32).reshape(-1), CombineBytes=views["combine"].view(torch.uint8).reshape(-1),
            Y=y.view(torch.uint8).reshape(-1), num_tokens=T,
        )
    return bindings
