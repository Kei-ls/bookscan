"""화질 보정.

핵심은 조명 얼룩 제거다. 카메라로 찍은 종이는 조명이 균일하지 않아 한쪽이
누렇거나 어둡다. 조명은 곱셈으로 작용하므로 배경 조명장을 추정해 **나눗셈**으로
정규화한다(뺄셈이 아니다). 채널별로 나누면 화이트밸런스가 덤으로 따라와서
누런 종이가 자연스럽게 흰색이 된다.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Mode


def _background_field(bgr: np.ndarray, ksize_frac: float = 0.13, work: int = 320) -> np.ndarray:
    """조명장 추정.

    작은 해상도에서 morphological closing으로 어두운 구조(글자)를 지워버리면
    남는 건 종이에 떨어진 빛의 분포다.
    """
    h, w = bgr.shape[:2]
    scale = work / max(h, w)
    sw, sh = max(8, round(w * scale)), max(8, round(h * scale))
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    k = max(3, int(max(sw, sh) * ksize_frac))
    k = k + 1 if k % 2 == 0 else k
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), max(1.0, k / 3.0))

    # 큰 사진/도판처럼 커널보다 넓은 어두운 영역이 있으면 조명장이 과소추정되어
    # 그 부분이 하얗게 날아간다. 전역 종이 밝기의 일정 비율 아래로는 안 내려가게 막는다.
    paper = np.percentile(bg.reshape(-1, bg.shape[-1]), 90, axis=0)
    floor = (paper * 0.55).astype(np.float32)
    bg = np.maximum(bg.astype(np.float32), floor[None, None, :])

    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def flatten_illumination(bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """조명 얼룩·그림자 제거 + 자동 화이트밸런스."""
    if strength <= 0:
        return bgr
    bg = _background_field(bgr)
    src = bgr.astype(np.float32)
    norm = np.divide(src, np.maximum(bg, 1.0)) * 255.0
    if strength < 1.0:
        norm = src * (1.0 - strength) + norm * strength
    return np.clip(norm, 0, 255).astype(np.uint8)


def _tone_lut(L: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """L 채널 히스토그램에 맞춘 256엔트리 LUT.

    a/b 채널은 건드리지 않으므로 색상이 틀어지지 않는다.
    """
    lo = float(np.percentile(L, 1.0))
    hi = float(np.percentile(L, 98.0))
    # 글자가 거의 없는 페이지에서 검은점이 과하게 올라가는 걸 막는다
    lo = float(np.clip(lo, 0.0, 110.0))
    hi = float(np.clip(hi, lo + 25.0, 255.0))

    x = np.arange(256, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)

    gamma = 1.0 + (contrast / 140.0)  # >1이면 중간톤이 어두워져 글자가 진해진다
    y = np.power(y, gamma)
    y = y * 255.0 + brightness * 1.6
    return np.clip(y, 0, 255).astype(np.uint8)


def _apply_tone(bgr: np.ndarray, brightness: int, contrast: int, saturation: float) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = cv2.LUT(L, _tone_lut(L, brightness, contrast))
    if abs(saturation - 1.0) > 1e-3:
        # 조명 정규화 후엔 색이 옅어 보여서 살짝 올려준다
        a = np.clip((a.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)
        b = np.clip((b.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def unsharp(bgr: np.ndarray, amount: float = 0.6, radius: float = 1.2) -> np.ndarray:
    if amount <= 0:
        return bgr
    blurred = cv2.GaussianBlur(bgr, (0, 0), radius)
    return cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)


def to_bw(gray: np.ndarray) -> np.ndarray:
    """Sauvola 국소 이진화. 문서에는 전역 Otsu보다 훨씬 강하다."""
    h, w = gray.shape[:2]
    block = int(max(h, w) * 0.02)
    block = max(15, block + 1 if block % 2 == 0 else block)
    try:
        binary = cv2.ximgproc.niBlackThreshold(
            gray, 255, cv2.THRESH_BINARY, block, 0.2,
            binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA, r=128.0,
        )
    except Exception:  # ximgproc이 없거나 실패하면 표준 적응형 이진화로
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, 12)
    return binary


def enhance(
    bgr: np.ndarray,
    mode: Mode = "color",
    brightness: int = 0,
    contrast: int = 0,
    sharpen: float = 1.0,
    flatten: bool = True,
) -> np.ndarray:
    """워프된 페이지를 스캔 품질로 마무리한다."""
    img = flatten_illumination(bgr) if flatten else bgr

    if mode == "color":
        out = _apply_tone(img, brightness, contrast + 10, saturation=1.12)
        return unsharp(out, amount=0.55 * sharpen)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        gray = cv2.LUT(gray, _tone_lut(gray, brightness, contrast + 15))
        gray = unsharp(gray, amount=0.5 * sharpen)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if mode == "bw":
        gray = cv2.LUT(gray, _tone_lut(gray, brightness, contrast))
        binary = to_bw(gray)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"알 수 없는 모드: {mode}")
