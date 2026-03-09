from .base import ModelQuantConfig


MinimaxM25Config = ModelQuantConfig(
    model_id="/data/models/transformers/MiniMax-M2.5-dequantized",
    trust_remote_code=True,
    streaming=True,
)
