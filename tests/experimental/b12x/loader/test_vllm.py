"""The adapter preserves vLLM's indexed source selection and owned inputs."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("vllm")

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from b12x.integration.vllm.loader import B12xModelLoader
from b12x.loader._checkpoint import DirectWeightSession

from b12x.loader._progress import CheckpointDisplay


def test_direct_loader_reports_missing_host_hooks_when_selected(monkeypatch):
    from vllm.model_executor.model_loader import weight_utils

    monkeypatch.delattr(weight_utils, "file_source_tensor", raising=False)
    with pytest.raises(RuntimeError, match="requires vLLM file-source hooks.*file_source_tensor"):
        B12xModelLoader(LoadConfig(load_format="b12x"))


@pytest.mark.parametrize("show_progress", [True, False])
@pytest.mark.parametrize("read_mode", ["auto", "bounce"])
def test_draft_iterator_uses_index_and_retained_tensors_keep_their_bytes(
    tmp_path, capsys, show_progress, read_mode
):
    """Draft loading must not open unrelated target shards or reuse buffers."""
    save_file({"mtp.weight": torch.arange(16)}, tmp_path / "draft.safetensors")
    save_file({"mtp.bias": torch.arange(4) + 100}, tmp_path / "bias.safetensors")
    (tmp_path / "target.safetensors").write_bytes(b"must not be opened")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.weight": "draft.safetensors",
                    "mtp.bias": "bias.safetensors",
                    "model.weight": "target.safetensors",
                }
            }
        )
    )
    config = LoadConfig(
        load_format="b12x",
        use_tqdm_on_load=show_progress,
        model_loader_extra_config={"read_mode": read_mode},
    )
    loader = B12xModelLoader(config)
    source = DefaultModelLoader.Source(
        str(tmp_path), revision=None, prefix="draft.", weight_name_prefixes=("mtp.",)
    )
    with (

        DirectWeightSession(read_mode=loader.read_mode) as session,
        CheckpointDisplay(enabled=show_progress) as display,
    ):
        loader._session = session
        loader._progress = display
        retained = dict(loader._get_weights_iterator(source))
        assert set(retained) == {"draft.mtp.weight", "draft.mtp.bias"}
        values = {}
        for name, descriptor in retained.items():
            values[name] = torch.empty_like(descriptor, device="cuda")
            assert session(values[name], descriptor)
        loader._session = None
        loader._progress = None
    torch.testing.assert_close(values["draft.mtp.weight"].cpu(), torch.arange(16))
    torch.testing.assert_close(values["draft.mtp.bias"].cpu(), torch.arange(4) + 100)
    assert config.load_format == "b12x"
    assert config.model_loader_extra_config == {"read_mode": read_mode}
    progress = capsys.readouterr().err
    if show_progress:
        assert "b12x routing checkpoint shards" in progress
        assert "2/2 shards routed" in progress
        assert "0.00 GB selected on rank 0" in progress
        assert "it/s" not in progress
    else:
        assert "shards routed" not in progress


def test_gdn_convolution_shards_read_into_final_parameter_slices(tmp_path):
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        mamba_v2_sharded_weight_loader,
    )
    from vllm.model_executor.weight_transfer import weight_transfer

    path = tmp_path / "conv.safetensors"
    checkpoint = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    save_file({"conv.weight": checkpoint}, path)
    loader = mamba_v2_sharded_weight_loader(
        [(8, 0, 0), (4, 2, 1)], tp_size=2, tp_rank=1
    )
    with (

        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        source = dict(session.weights([path]))["conv.weight"]
        target = torch.full((6, 2), -1.0, device="cuda")
        loader(target, source)
    torch.testing.assert_close(target.cpu(), checkpoint[4:])


def test_hyperconnection_weights_load_without_allocator_hooks(tmp_path, monkeypatch):
    """Loading the norm must preserve separate workspace contents."""
    from vllm.model_executor.weight_transfer import copy_weight, weight_transfer
    from vllm.models.qwen4_exp.nvidia.hyperconnection import (
        GroupedGemmaRMSNorm,
        HyperConnectionConfig,
        HyperConnectionWorkspace,
    )

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.hyperconnection.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    expected = torch.arange(128, dtype=torch.bfloat16)
    path = tmp_path / "norm.safetensors"
    save_file({"weight": expected}, path)
    with (

        DirectWeightSession() as session,
        weight_transfer(session),
        torch.device("cuda"),
    ):
        norm = GroupedGemmaRMSNorm(128, eps=1e-6, group_size=32, dtype=expected.dtype)
        workspace = HyperConnectionWorkspace(
            HyperConnectionConfig(
                hc_count=4,
                hidden_size=32,
                params_dtype=expected.dtype,
                hc_lowrank=16,
                rms_norm_eps=1e-6,
                hc_per_branch_norm=True,
            ),
            8,
        )
        source = dict(session.weights([path]))["weight"]
        copy_weight(norm.weight, source)
    torch.testing.assert_close(norm.weight.cpu(), expected)
    for buffer in workspace.buffers():
        buffer.fill_(7)
    torch.cuda.synchronize()
    torch.testing.assert_close(norm.weight.cpu(), expected)


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("quantization", ["mxfp8", "block_fp8"])
def test_glm_attention_dequantization_reads_owned_checkpoint_inputs(
    tmp_path, scale_first, quantization
):
    """Numerical projection transforms must consume payloads, not meta views."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.weight_transfer import weight_transfer
    from vllm.models.glm5next.nvidia.model import Glm5NextModel

    prefix = "layers.3.self_attn"
    if quantization == "mxfp8":
        projection = target = "indexer.weights_proj"
        scale_name = "weight_scale"
        scale = torch.full((1, 1), 128, dtype=torch.uint8)
    else:
        projection, target = "q_a_proj", "fused_qkv_a_proj"
        scale_name = "weight_scale_inv"
        scale = torch.full((1, 1), 2.0, dtype=torch.float32)
    parameter_name = f"{prefix}.{target}.weight"
    weight = torch.arange(32).reshape(1, 32).to(torch.float8_e4m3fn)
    expected = weight.float() * 2
    path = tmp_path / "attention.safetensors"
    save_file(
        {
            f"{prefix}.{projection}.weight": weight,
            f"{prefix}.{projection}.{scale_name}": scale,
        },
        path,
    )

    class Projection(torch.nn.Module):
        config = SimpleNamespace(
            is_moe=False, is_linear_attn=True, mla_nope=False, qk_rope_head_dim=0
        )

        def named_parameters(self):
            return iter([(parameter_name, param)])

    with (

        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        param = torch.nn.Parameter(
            torch.empty((1, 32), dtype=torch.float32, device="cuda")
        )
        param.weight_loader = lambda p, value, shard_id=None: default_weight_loader(
            p, value
        )
        sources = sorted(
            session.weights([path]),
            key=lambda pair: pair[0].endswith(".weight") == scale_first,
        )
        loaded = Glm5NextModel.load_weights(Projection(), iter(sources))
        assert loaded == {parameter_name}
    torch.testing.assert_close(param.cpu(), expected)


@pytest.mark.parametrize("rank", [0, 1])
def test_kda_convolution_loads_each_tp_shard_into_fused_weights(tmp_path, rank):
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        _make_fused_conv1d_weight_loader,
    )
    from vllm.model_executor.weight_transfer import weight_transfer

    path = tmp_path / "kda.safetensors"
    weights = {
        name: (torch.arange(32).reshape(8, 1, 4) + i * 64).to(torch.bfloat16)
        for i, name in enumerate(("q", "k", "v"))
    }
    save_file(weights, path)
    with (

        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        param = torch.empty((12, 1, 4), device="cuda")
        loader = _make_fused_conv1d_weight_loader([8, 8, 8], 2, rank)
        sources = dict(session.weights([path]))
        for i, name in enumerate(("q", "k", "v")):
            loader(param, sources[name], i)
    expected = torch.cat([weights[name][rank * 4 : (rank + 1) * 4] for name in weights])
    torch.testing.assert_close(param.cpu(), expected.float())


@pytest.mark.parametrize(
    "tp_size,rank", [(tp, rank) for tp in (2, 4) for rank in range(tp)]
)
def test_deepseek_sink_shards_are_flushed_before_derived_weights(
    tmp_path, monkeypatch, tp_size, rank
):
    """Padded sinks keep -inf and model post-load hooks consume completed reads."""
    from vllm.model_executor.weight_transfer import weight_transfer
    from vllm.models.deepseek_v4.nvidia import model as ds4

    monkeypatch.setattr(ds4, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(ds4, "get_tensor_model_parallel_rank", lambda: rank)
    expected = torch.arange(64, dtype=torch.float32) / 4
    path = tmp_path / "sinks.safetensors"
    name = "layers.0.attn.attn_sink"
    save_file({name: expected}, path)

    class Model(torch.nn.Module):
        config = SimpleNamespace(num_attention_heads=64)
        quant_config = None
        use_sequence_parallel = False

        def __init__(self):
            super().__init__()
            self.sink = torch.nn.Parameter(
                torch.full((64,), -torch.inf, device="cuda"),
                requires_grad=False,
            )
            self.derived = None

        def named_parameters(self):
            return iter([(name, self.sink)])

        def get_expert_mapping(self):
            return []

        def finalize_mega_moe_weights(self):
            pass

        def finalize_mhc_broadcast_weights(self):
            self.derived = self.sink[: 64 // tp_size] * 2

        def process_b12x_weights_after_loading(self):
            pass

    with (

        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        model = Model()
        assert ds4.DeepseekV4Model.load_weights(model, session.weights([path])) == {
            name
        }
        ds4.DeepseekV4ForCausalLM.process_weights_after_loading(
            SimpleNamespace(model=model)
        )
    width = 64 // tp_size
    torch.testing.assert_close(
        model.derived.cpu(), expected[rank * width : (rank + 1) * width] * 2
    )
    assert torch.isneginf(model.sink[width:]).all()


@pytest.mark.parametrize("scale_first", [False, True])
def test_dsa_indexer_dequantization_owns_inputs_across_checkpoint_shards(
    tmp_path, scale_first
):
    """Full GLM's fused WK projection reads real FP8 values before dequantizing."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk
    from vllm.model_executor.weight_transfer import weight_transfer

    prefix = "layers.0.self_attn.indexer"
    weight = (torch.arange(128 * 256).reshape(128, 256) % 64).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 2.0]])
    paths = [tmp_path / "weight.safetensors", tmp_path / "scale.safetensors"]
    save_file({f"{prefix}.wk.weight": weight}, paths[0])
    save_file({f"{prefix}.wk.weight_scale_inv": scale}, paths[1])
    if scale_first:
        paths.reverse()
    with (

        DirectWeightSession() as session,
        weight_transfer(session),
    ):
        param = torch.nn.Parameter(
            torch.zeros((160, 256), device="cuda", dtype=torch.bfloat16
            ),
            requires_grad=False,
        )
        param.weight_loader = lambda p, value, shard_id: default_weight_loader(
            p[:128], value
        )
        name = f"{prefix}.wk_weights_proj.weight"
        pending, loaded = {}, set()
        for source_name, source in session.weights(paths):
            assert _try_load_fp8_indexer_wk(
                source_name, source, pending, {name: param}, loaded, []
            )
        assert not pending
        assert loaded == {name}
    expected = (weight.float() * scale.repeat_interleave(128, dim=1)).bfloat16()
    torch.testing.assert_close(param[:128].cpu(), expected, rtol=0, atol=0)
    assert torch.count_nonzero(param[128:]) == 0


def test_dspark_markov_embedding_reads_checkpoint_into_weight_storage(
    tmp_path, monkeypatch
):
    """A plain nn.Embedding loads without allocation hooks."""
    from vllm import envs
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
    from vllm.model_executor.weight_transfer import weight_transfer

    monkeypatch.setattr(envs, "VLLM_MXFP8_LM_HEAD", False)
    expected = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8)
    path = tmp_path / "markov.safetensors"
    save_file({"markov_w1.weight": expected}, path)
    with (

        DirectWeightSession() as session,
        weight_transfer(session),
        torch.device("cuda"),
    ):
        head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
        source = dict(session.weights([path]))["markov_w1.weight"]
        default_weight_loader(head.markov_w1.weight, source)
        session.flush()
        result = head.embed(torch.tensor([0, 17, 127]))
    torch.testing.assert_close(result.cpu(), expected[[0, 17, 127]])


def test_glm_mtp_projection_loads_from_main_shard_without_sharing_runtime_buffers(
    tmp_path, monkeypatch
):
    """MTP's plain Linear loads without a target-model allocation scope."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.weight_transfer import weight_transfer
    from vllm.models.glm5next.nvidia import mtp

    class Decoder(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.topk_indices_buffer = kwargs["topk_indices_buffer"]
            self.pool_topk_indices_buffer = kwargs["pool_topk_indices_buffer"]

    monkeypatch.setattr(mtp, "Glm5NextDecoderLayer", Decoder)
    monkeypatch.setattr(mtp, "SharedHead", lambda **kwargs: torch.nn.Identity())
    config = SimpleNamespace(
        hidden_size=8, rms_norm_eps=1e-6, index_topk=16, index_kpool=4
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
    )
    name = "model.language_model.layers.45.eh_proj.weight"
    expected = torch.arange(128, dtype=torch.float32).reshape(8, 16)
    path = tmp_path / "model-00001-of-00001.safetensors"
    save_file(
        {name: expected, "model.language_model.layers.0.weight": torch.ones(8)}, path
    )
    with (
        set_current_vllm_config(VllmConfig()),

        DirectWeightSession() as session,
        weight_transfer(session),
        torch.device("cuda"),
    ):
        layer = mtp.Glm5NextMultiTokenPredictorLayer(vllm_config, "model.layers.45")
        source = dict(
            session.weights([path], prefixes=("model.language_model.layers.45.",))
        )
        assert set(source) == {name}
        default_weight_loader(layer.eh_proj.weight, source[name])
        session.flush()
        result = layer.eh_proj(torch.ones(1, 16))
    torch.testing.assert_close(layer.eh_proj.weight.cpu(), expected)
    torch.testing.assert_close(result.cpu(), expected.sum(dim=1).unsqueeze(0))
