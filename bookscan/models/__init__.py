"""검출 모델 — 내장 알고리즘과 선택 설치형 신경망을 같은 인터페이스로 묶는다."""

from .catalog import DEFAULT_MODEL, MODELS, ModelInfo, get, weights_for
from .runtime import (
    download_weights,
    install_packages,
    ready,
    segment,
    status,
    status_all,
    torch_installed,
    unload,
)

__all__ = [
    "DEFAULT_MODEL", "MODELS", "ModelInfo", "get", "weights_for",
    "download_weights", "install_packages", "ready", "segment",
    "status", "status_all", "torch_installed", "unload",
]
