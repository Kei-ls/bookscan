"""원근 보정. 검출된 사각형을 정면에서 본 직사각형으로 편다."""

from __future__ import annotations

import cv2
import numpy as np

from .detect import order_corners

#: 가로/세로 비 프리셋. None이면 네 변 길이에서 추정.
ASPECT_PRESETS: dict[str, float | None] = {
    "auto": None,
    "a4": 210.0 / 297.0,
    "letter": 8.5 / 11.0,
    "b5": 176.0 / 250.0,
}


def target_size(quad: np.ndarray, aspect: str = "auto", max_long_edge: int = 3500) -> tuple[int, int]:
    """워프 결과의 (폭, 높이).

    마주보는 변 중 긴 쪽을 취한다. 원근 때문에 가까운 쪽 변이 길게 찍히는데,
    그 쪽에 맞춰야 해상도 손실이 없다.
    """
    tl, tr, br, bl = quad
    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    w, h = float(max(w, 1.0)), float(max(h, 1.0))

    ratio = ASPECT_PRESETS.get(aspect)
    if ratio is not None:
        # 면적을 유지하면서 지정 비율에 맞춘다
        area = w * h
        h = float(np.sqrt(area / ratio))
        w = h * ratio

    long_edge = max(w, h)
    if long_edge > max_long_edge:
        s = max_long_edge / long_edge
        w, h = w * s, h * s
    return max(1, round(w)), max(1, round(h))


def warp_transform(
    corners: np.ndarray | list,
    aspect: str = "auto",
    max_long_edge: int = 3500,
) -> tuple[np.ndarray, int, int]:
    """워프에 쓸 호모그래피와 출력 크기. (M, w, h)

    마스크처럼 이미지와 같은 변환을 따로 적용해야 하는 것들이 있어서 분리해 뒀다.
    """
    quad = order_corners(np.asarray(corners, dtype=np.float32))
    w, h = target_size(quad, aspect, max_long_edge)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(quad, dst), w, h


def warp_page(
    bgr: np.ndarray,
    corners: np.ndarray | list,
    aspect: str = "auto",
    max_long_edge: int = 3500,
) -> np.ndarray:
    """corners(입력 이미지 좌표계)로 지정된 사각형 영역을 펴서 돌려준다."""
    M, w, h = warp_transform(corners, aspect, max_long_edge)
    # 축소가 크게 일어나면 INTER_AREA가 낫지만 warpPerspective는 지원하지 않아
    # 확대/유지에 안전한 CUBIC을 쓴다.
    return cv2.warpPerspective(bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def warp_mask(mask: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """마스크를 이미지와 같은 변환으로 옮긴다. 바깥은 배경(0)으로 채운다."""
    w, h = size
    out = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (out > 127).astype(np.uint8) * 255


def clamp_corners(corners: np.ndarray | list, shape: tuple[int, int]) -> np.ndarray:
    """코너를 이미지 범위 안으로 (약간의 여유를 두고) 밀어넣는다."""
    h, w = shape[:2]
    q = np.asarray(corners, dtype=np.float32).reshape(4, 2).copy()
    q[:, 0] = np.clip(q[:, 0], -w * 0.02, w * 1.02)
    q[:, 1] = np.clip(q[:, 1], -h * 0.02, h * 1.02)
    return q
