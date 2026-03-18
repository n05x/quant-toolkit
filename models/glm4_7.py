from .base import ModelQuantConfig


Glm4_7Config = _Glm4_7Config(
    model_id="zai-org/GLM-4.7",
    trust_remote_code=True,
    streaming=True,
    extra_quant_overrides={
        "*mlp.gate*": {"enable": False},
        "*mtp*": {"enable": False},
    },
)


class _Glm4_7Config(ModelQuantConfig):
    def register_moe(self):
        from moe_registry import register_glm4_7_moe_for_quantization
        register_glm4_7_moe_for_quantization()


