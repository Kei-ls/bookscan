"""펼침면 처리 — 두 페이지가 한 장에 찍힌 사진에서 책등을 찾아 분할한다.

책등 위치를 알려주는 신호는 두 가지다.
  1) 그림자 골짜기 — 책이 접히는 곳은 빛이 덜 들어 어둡다
  2) 잉크 공백 — 좌우 본문 사이에 글자가 전혀 없는 세로 띠가 있다
평평하게 눌러 찍어 그림자가 거의 없는 책도 있고, 여백이 좁은 책도 있어서
두 신호를 합쳐서 본다. 신호가 약하면 **분할하지 않는다** — 단면 사진을
잘못 쪼개는 쪽이 안 쪼개는 쪽보다 훨씬 나쁜 실패다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: 이 정도는 가로로 길어야 펼침면 후보로 본다 (단면 세로 페이지는 보통 0.7 근처)
MIN_SPREAD_ASPECT = 1.05
#: 책등은 가운데 이 구간 안에서만 찾는다
SEARCH_LO, SEARCH_HI = 0.34, 0.66
#: 합산 점수가 이보다 낮으면 단면으로 판정
SPLIT_THRESHOLD = 0.28


@dataclass
class SpreadResult:
    is_spread: bool
    gutter: float | None  # 폭 기준 0~1로 정규화한 책등 위치
    confidence: float
    reason: str


def _column_profiles(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(밝기 프로파일, 잉크밀도 프로파일). 둘 다 열 단위."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # 위아래 끝은 페이지 경계 잔재가 섞이므로 가운데 80%만 본다
    band = gray[int(h * 0.10): int(h * 0.90), :]

    brightness = band.mean(axis=0).astype(np.float32)

    # 잉크: 국소 배경보다 뚜렷이 어두운 픽셀
    blur = cv2.GaussianBlur(band, (0, 0), max(2.0, w * 0.01))
    ink = (band.astype(np.float32) < blur.astype(np.float32) - 18).astype(np.float32)
    ink_density = ink.mean(axis=0)

    sigma = max(2.0, w * 0.006)
    brightness = cv2.GaussianBlur(brightness.reshape(1, -1), (0, 0), sigma).ravel()
    ink_density = cv2.GaussianBlur(ink_density.reshape(1, -1), (0, 0), sigma).ravel()
    return brightness, ink_density


def detect_gutter(bgr: np.ndarray, force: str = "auto") -> SpreadResult:
    """책등 위치를 추정한다. force로 자동판정을 덮어쓸 수 있다."""
    h, w = bgr.shape[:2]
    aspect = w / max(h, 1)

    if force == "single":
        return SpreadResult(False, None, 1.0, "사용자 지정: 단면")
    if force == "split":
        res = detect_gutter(bgr, "auto")
        gutter = res.gutter if res.gutter is not None else 0.5
        return SpreadResult(True, gutter, 1.0, "사용자 지정: 분할")

    if aspect < MIN_SPREAD_ASPECT:
        return SpreadResult(False, None, 0.9, f"가로세로비 {aspect:.2f} — 단면 형태")

    brightness, ink = _column_profiles(bgr)
    lo, hi = int(w * SEARCH_LO), int(w * SEARCH_HI)
    if hi - lo < 8:
        return SpreadResult(False, None, 0.5, "폭이 너무 좁음")

    # 그림자 신호: 페이지 평균 밝기 대비 얼마나 어두운가
    baseline = float(np.median(brightness))
    if baseline < 1.0:
        return SpreadResult(False, None, 0.3, "밝기 정보 없음")
    shadow = np.clip((baseline - brightness) / baseline, 0, 1)

    # 잉크 공백 신호: 본문 평균 잉크량 대비 얼마나 비어 있는가
    ink_ref = float(np.percentile(ink, 75))
    gap = np.clip((ink_ref - ink) / ink_ref, 0, 1) if ink_ref > 1e-4 else np.zeros_like(ink)

    score = 0.55 * shadow + 0.45 * gap
    idx = int(np.argmax(score[lo:hi])) + lo
    peak = float(score[idx])

    if peak < SPLIT_THRESHOLD:
        return SpreadResult(False, None, 1.0 - peak,
                            f"책등 신호 약함 ({peak:.2f}) — 단면으로 처리")

    # 양쪽에 본문이 비슷하게 있어야 진짜 펼침면이다. 오른쪽 여백이 넓은
    # 단면 사진은 '잉크 공백' 신호만 보면 책등처럼 보이는데, 이 검사가 걸러낸다.
    ink_left = float(ink[:idx].sum())
    ink_right = float(ink[idx:].sum())
    balance = min(ink_left, ink_right) / max(ink_left, ink_right, 1e-6)
    if balance < 0.35:
        return SpreadResult(False, None, 0.8,
                            f"한쪽에만 본문이 있음 (균형 {balance:.2f}) — 단면으로 처리")

    # 그림자가 있으면 책등이 거의 확실하다. 그림자가 없다면(평평하게 눌러 찍은
    # 책) 잉크 공백만으로 판단해야 하는데, 그건 2단 조판 같은 것과 구별이 안 되므로
    # 확실히 두 페이지 폭인 경우에만 인정한다.
    has_shadow = float(shadow[idx]) >= 0.08
    has_gap = float(gap[idx]) >= 0.85
    if not has_shadow and not (has_gap and aspect >= 1.25):
        return SpreadResult(False, None, 0.7,
                            f"책등 그림자 없음 (그림자 {shadow[idx]:.2f}, 공백 {gap[idx]:.2f}, "
                            f"비율 {aspect:.2f}) — 단면으로 처리")

    return SpreadResult(True, idx / w, min(1.0, peak * 1.6),
                        f"책등 검출 (점수 {peak:.2f}, 그림자 {shadow[idx]:.2f}, "
                        f"공백 {gap[idx]:.2f}, 균형 {balance:.2f})")


def split_spread(bgr: np.ndarray, gutter: float, overlap: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """책등에서 좌/우로 자른다. overlap만큼 겹쳐 잘라 글자가 잘리는 걸 막는다."""
    h, w = bgr.shape[:2]
    gx = int(np.clip(gutter, 0.05, 0.95) * w)
    pad = int(w * overlap)
    left = bgr[:, : min(w, gx + pad)]
    right = bgr[:, max(0, gx - pad):]
    return np.ascontiguousarray(left), np.ascontiguousarray(right)
