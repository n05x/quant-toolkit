from .base import ModelQuantConfig


Glm5Config = ModelQuantConfig(
    model_id="zai-org/GLM-5",
    trust_remote_code=True,
    streaming=False,
    extra_quant_overrides={
        "*indexer*": {"enable": False},
    },
)


def _register_moe():
    from moe_registry import register_glm5_moe_for_quantization
    register_glm5_moe_for_quantization()


Glm5Config.register_moe = _register_moe
