"""delta_encoder 包: Visual Delta Encoder + Qwen2.5-Omni Thinker 前向补丁.

    delta_encoder/resnet.py             ResNet 视觉前端
    delta_encoder/delta_encoder.py      SVT / DeltaStream25fps / DeltaEncoder
    delta_encoder/qwen_omni_forward.py  Thinker.forward 猴子补丁 (apply_forward_patch)
"""
from .delta_encoder import (
    DeltaEncoder,
    DeltaEncoderBackbone,
    DeltaStream25fps,
    StructuredVisualTokenizer,
    TokenProj,
)
from .qwen_omni_forward import apply_forward_patch

__all__ = [
    "DeltaEncoder",
    "DeltaEncoderBackbone",
    "DeltaStream25fps",
    "StructuredVisualTokenizer",
    "TokenProj",
    "apply_forward_patch",
]
