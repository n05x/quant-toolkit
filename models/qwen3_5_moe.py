from .base import ModelQuantConfig


class _Qwen35MoeConfig(ModelQuantConfig):
    def get_model_cls(self):
        from transformers import Qwen3_5MoeForConditionalGeneration
        return Qwen3_5MoeForConditionalGeneration

    def register_moe(self):
        from moe_registry import register_qwen35_moe_for_quantization
        register_qwen35_moe_for_quantization()


Qwen35MoeConfig = _Qwen35MoeConfig(
    model_id="Qwen/Qwen3.5-397B-A17B",
    trust_remote_code=False,
    streaming=True,
    extra_quant_overrides={
        "*visual*": {"enable": False},
        "*linear_attn*weight_quantizer": {"enable": False},
        "*linear_attn*input_quantizer": {"enable": False},
        "*shared_expert_gate*weight_quantizer": {"enable": False},
        "*shared_expert_gate*input_quantizer": {"enable": False},
        "*mlp.gate.weight_quantizer": {"enable": False},
        "*mlp.gate.input_quantizer": {"enable": False},
        "*mtp*weight_quantizer": {"enable": False},
        "*mtp*input_quantizer": {"enable": False},
    },
)
