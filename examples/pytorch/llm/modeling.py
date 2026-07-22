"""FlashInfer end-to-end LLM example: Llama / Qwen dense decoder.

Reference/verification code, not a serving engine: it strings FlashInfer's
serving-path ops together the way an inference framework would — paged KV
cache (`append_paged_kv_cache`), batched prefill/decode attention wrappers,
RoPE (`apply_rope_pos_ids_inplace`), RMSNorm (`rmsnorm` /
`fused_add_rmsnorm`), and SwiGLU (`silu_and_mul`) — in plain PyTorch, loading
weights directly from a Hugging Face checkpoint's safetensors. See
`docs/design_docs/e2e_pytorch_llm_examples.md` for the rationale.

Supported HF `model_type`s: `llama`, `qwen2`, `qwen3` (dense). The three
share one skeleton (RMSNorm -> GQA attention with RoPE -> SwiGLU FFN); the
deltas are Qwen3's per-head q/k norm, Qwen2's QKV bias, and Llama-3.1-style
RoPE scaling.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch

import flashinfer

SUPPORTED_MODEL_TYPES = ("llama", "qwen2", "qwen3")


@dataclasses.dataclass
class ModelConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    rope_scaling: Optional[dict]
    eos_token_ids: List[int]

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "ModelConfig":
        model_type = cfg["model_type"]
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type {model_type!r}; this example supports "
                f"{SUPPORTED_MODEL_TYPES}"
            )
        rope_scaling = cfg.get("rope_scaling")
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type not in ("llama3", "default"):
                raise ValueError(
                    f"Unsupported rope_scaling {rope_scaling!r}; this example "
                    "supports null, 'default', and 'llama3' scaling"
                )
        eos = cfg.get("eos_token_id")
        if eos is None:
            eos_token_ids = []
        elif isinstance(eos, int):
            eos_token_ids = [eos]
        else:
            eos_token_ids = list(eos)
        return cls(
            model_type=model_type,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg.get(
                "num_key_value_heads", cfg["num_attention_heads"]
            ),
            head_dim=cfg.get(
                "head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]
            ),
            vocab_size=cfg["vocab_size"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg.get("rope_theta", 1e4),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            rope_scaling=rope_scaling,
            eos_token_ids=eos_token_ids,
        )


def resolve_checkpoint_dir(model_id_or_path: str) -> Path:
    """Return a local directory containing config.json + safetensors.

    Accepts either a local path or a Hugging Face model id (downloads
    config/tokenizer/safetensors via huggingface_hub, honoring HF_HOME).
    """
    local = Path(model_id_or_path)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id_or_path,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"],
        )
    )


def _load_state_dict(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index_file = ckpt_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            shard_names = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shard_names = [p.name for p in sorted(ckpt_dir.glob("*.safetensors"))]
    if not shard_names:
        raise FileNotFoundError(f"No safetensors found under {ckpt_dir}")
    state_dict: Dict[str, torch.Tensor] = {}
    for shard in shard_names:
        state_dict.update(load_file(ckpt_dir / shard))
    return state_dict


class DecoderLayerWeights:
    def __init__(self, sd: Dict[str, torch.Tensor], i: int, device, dtype):
        p = f"model.layers.{i}."

        def get(name: str, required: bool = True) -> Optional[torch.Tensor]:
            key = p + name
            if key not in sd:
                if required:
                    raise KeyError(f"Missing weight {key}")
                return None
            return sd[key].to(device=device, dtype=dtype)

        self.input_layernorm = get("input_layernorm.weight")
        self.post_attention_layernorm = get("post_attention_layernorm.weight")
        self.q_proj = get("self_attn.q_proj.weight")
        self.k_proj = get("self_attn.k_proj.weight")
        self.v_proj = get("self_attn.v_proj.weight")
        self.o_proj = get("self_attn.o_proj.weight")
        # Qwen2-family checkpoints carry QKV bias; Llama/Qwen3 do not.
        self.q_bias = get("self_attn.q_proj.bias", required=False)
        self.k_bias = get("self_attn.k_proj.bias", required=False)
        self.v_bias = get("self_attn.v_proj.bias", required=False)
        # Qwen3 per-head q/k RMSNorm over head_dim.
        self.q_norm = get("self_attn.q_norm.weight", required=False)
        self.k_norm = get("self_attn.k_norm.weight", required=False)
        self.gate_proj = get("mlp.gate_proj.weight")
        self.up_proj = get("mlp.up_proj.weight")
        self.down_proj = get("mlp.down_proj.weight")


class FlashInferLLM:
    """Dense Llama/Qwen-family decoder over a paged KV cache.

    The model itself is stateless w.r.t. sequences: all per-request state
    (page tables, sequence lengths) lives in the caller-provided
    :class:`PagedKVCache` and the planned attention wrapper.
    """

    def __init__(
        self,
        config: ModelConfig,
        state_dict: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.embed_tokens = state_dict["model.embed_tokens.weight"].to(
            device=device, dtype=dtype
        )
        self.final_norm = state_dict["model.norm.weight"].to(device=device, dtype=dtype)
        if config.tie_word_embeddings or "lm_head.weight" not in state_dict:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = state_dict["lm_head.weight"].to(device=device, dtype=dtype)
        self.layers = [
            DecoderLayerWeights(state_dict, i, device, dtype)
            for i in range(config.num_hidden_layers)
        ]

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FlashInferLLM":
        ckpt_dir = resolve_checkpoint_dir(model_id_or_path)
        with open(ckpt_dir / "config.json") as f:
            config = ModelConfig.from_hf_config(json.load(f))
        state_dict = _load_state_dict(ckpt_dir)
        return cls(config, state_dict, device, dtype)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor):
        cfg = self.config
        scaling = cfg.rope_scaling
        if (
            scaling is not None
            and scaling.get("rope_type", scaling.get("type")) == "llama3"
        ):
            flashinfer.apply_llama31_rope_pos_ids_inplace(
                q,
                k,
                pos_ids,
                interleave=False,
                rope_scale=scaling["factor"],
                rope_theta=cfg.rope_theta,
                low_freq_factor=scaling["low_freq_factor"],
                high_freq_factor=scaling["high_freq_factor"],
                old_context_len=scaling["original_max_position_embeddings"],
            )
        else:
            flashinfer.apply_rope_pos_ids_inplace(
                q, k, pos_ids, interleave=False, rope_theta=cfg.rope_theta
            )

    def forward(
        self,
        input_ids: torch.Tensor,  # (nnz,) int64, tokens packed over the batch
        pos_ids: torch.Tensor,  # (nnz,) int32, absolute position per token
        attn_wrapper,  # planned prefill or decode wrapper
        kv_cache: "PagedKVCache",
        batch_indices: torch.Tensor,  # (nnz,) int32, request of each token
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        last_token_rows: torch.Tensor,  # (batch,) int64 rows to compute logits for
    ) -> torch.Tensor:
        """One forward step (prefill or decode). Returns (batch, vocab) fp32 logits."""
        cfg = self.config
        nnz = input_ids.shape[0]
        num_q, num_kv, hd = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        )
        x = torch.nn.functional.embedding(input_ids, self.embed_tokens)
        residual: Optional[torch.Tensor] = None
        for i, w in enumerate(self.layers):
            if residual is None:
                residual = x
                x = flashinfer.rmsnorm(x, w.input_layernorm, cfg.rms_norm_eps)
            else:
                flashinfer.fused_add_rmsnorm(
                    x, residual, w.input_layernorm, cfg.rms_norm_eps
                )
            q = torch.nn.functional.linear(x, w.q_proj, w.q_bias).view(nnz, num_q, hd)
            k = torch.nn.functional.linear(x, w.k_proj, w.k_bias).view(nnz, num_kv, hd)
            v = torch.nn.functional.linear(x, w.v_proj, w.v_bias).view(nnz, num_kv, hd)
            if w.q_norm is not None:
                q = flashinfer.rmsnorm(q, w.q_norm, cfg.rms_norm_eps)
                k = flashinfer.rmsnorm(k, w.k_norm, cfg.rms_norm_eps)
            self._apply_rope(q, k, pos_ids)
            flashinfer.append_paged_kv_cache(
                k,
                v,
                batch_indices,
                pos_ids,
                kv_cache.layer_view(i),
                kv_indices,
                kv_indptr,
                kv_last_page_len,
                kv_layout="NHD",
            )
            attn = attn_wrapper.run(q, kv_cache.layer_view(i))
            attn = torch.nn.functional.linear(attn.reshape(nnz, num_q * hd), w.o_proj)
            flashinfer.fused_add_rmsnorm(
                attn, residual, w.post_attention_layernorm, cfg.rms_norm_eps
            )
            gate_up = torch.cat(
                [
                    torch.nn.functional.linear(attn, w.gate_proj),
                    torch.nn.functional.linear(attn, w.up_proj),
                ],
                dim=-1,
            )
            x = torch.nn.functional.linear(
                flashinfer.silu_and_mul(gate_up), w.down_proj
            )
        # Final residual add + norm, only then project the rows we sample from.
        flashinfer.fused_add_rmsnorm(x, residual, self.final_norm, cfg.rms_norm_eps)
        hidden = x[last_token_rows]
        return torch.nn.functional.linear(hidden, self.lm_head).float()


class PagedKVCache:
    """Paged KV cache + page tables for a fixed batch of requests.

    Layout is NHD: per layer ``[max_pages, 2, page_size, num_kv_heads,
    head_dim]``. Pages are handed out from a bump allocator; all index
    tensors are int32 as required by the FlashInfer kernels.
    """

    def __init__(
        self,
        config: ModelConfig,
        max_pages: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.page_size = page_size
        self.device = device
        self.kv_data = torch.zeros(
            config.num_hidden_layers,
            max_pages,
            2,
            page_size,
            config.num_key_value_heads,
            config.head_dim,
            device=device,
            dtype=dtype,
        )
        self.max_pages = max_pages
        self._next_free_page = 0
        self.page_tables: List[List[int]] = []
        self.seq_lens: List[int] = []

    def layer_view(self, layer: int) -> torch.Tensor:
        return self.kv_data[layer]

    def add_request(self) -> int:
        self.page_tables.append([])
        self.seq_lens.append(0)
        return len(self.page_tables) - 1

    def _alloc_page(self) -> int:
        if self._next_free_page >= self.max_pages:
            raise RuntimeError(
                f"Out of KV pages (max_pages={self.max_pages}); increase the "
                "page budget for this prompt/generation length"
            )
        page = self._next_free_page
        self._next_free_page += 1
        return page

    def extend(self, request: int, num_new_tokens: int) -> None:
        """Grow request's page table to hold num_new_tokens more tokens."""
        self.seq_lens[request] += num_new_tokens
        needed_pages = math.ceil(self.seq_lens[request] / self.page_size)
        table = self.page_tables[request]
        while len(table) < needed_pages:
            table.append(self._alloc_page())

    def page_table_tensors(self):
        """Return (kv_indptr, kv_indices, kv_last_page_len) int32 tensors."""
        indptr = [0]
        for t in self.page_tables:
            indptr.append(indptr[-1] + len(t))
        kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=self.device)
        kv_indices = torch.tensor(
            [p for t in self.page_tables for p in t],
            dtype=torch.int32,
            device=self.device,
        )
        kv_last_page_len = torch.tensor(
            [(sl - 1) % self.page_size + 1 for sl in self.seq_lens],
            dtype=torch.int32,
            device=self.device,
        )
        return kv_indptr, kv_indices, kv_last_page_len

    def seq_lens_tensor(self) -> torch.Tensor:
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)
