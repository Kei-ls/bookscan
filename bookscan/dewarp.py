"""곡면 펴기 — 책등 근처가 휘어 들어간 페이지를 평평하게 만든다.

원리: 원래 수평이어야 할 본문 텍스트 줄들이 실제로는 휘어 있다. 그 휘어짐을
측정하면 종이 표면의 형상을 역으로 알 수 있다.

  1) Sauvola 이진화 → 가로 방향으로 뭉쳐서 '글자 줄' 덩어리를 만든다
  2) 각 줄의 중심선에 2차 다항식을 맞춘다
  3) 줄들의 평균 휘어짐 d(x)를 구해 행을 되돌린다 (수직 보정)
  4) d(x)의 호 길이로 가로 압축을 되돌린다 (수평 보정) — 책등 근처는
     비스듬히 누워 있어서 가로로 눌려 찍히기 때문

안전장치가 이 모듈의 핵심이다. 잘못 편 페이지는 안 편 페이지보다 훨씬 나쁘다.
조건이 하나라도 안 맞으면 **아무것도 하지 않고 원본을 그대로 돌려준다.**
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_WIDTH = 1000  # 곡률 측정용 작업 해상도

MIN_LINES = 4  # 이보다 적으면 신뢰할 수 없다
MIN_CURVATURE_FRAC = 0.006  # 높이 대비 이 정도는 휘어야 손댈 가치가 있다
MAX_CURVATURE_FRAC = 0.18  # 이보다 크면 측정 실패로 보고 포기
MAX_FIT_RESIDUAL = 0.020  # 다항식 잔차(높이 대비)
MAX_LINE_DISAGREE = 0.055  # 줄들끼리 휘어짐이 얼마나 일치하는가


@dataclass
class DewarpResult:
    image: np.ndarray
    applied: bool
    reason: str
    n_lines: int = 0
    curvature: float = 0.0


def _binarize_text(gray: np.ndarray) -> np.ndarray:
    """글자를 255로 만든 이진 이미지."""
    h, w = gray.shape[:2]
    block = int(max(h, w) * 0.025)
    block = max(15, block + 1 if block % 2 == 0 else block)
    try:
        th = cv2.ximgproc.niBlackThreshold(
            gray, 255, cv2.THRESH_BINARY_INV, block, 0.2,
            binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA, r=128.0,
        )
    except Exception:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 12)
    return th


def find_text_lines(gray: np.ndarray) -> list[np.ndarray]:
    """본문 줄들의 중심선을 찾는다.

    각 줄마다 (N,2) 배열(x, y)을 돌려준다. 글자가 있는 열에서만 샘플링한다.
    """
    h, w = gray.shape[:2]
    binary = _binarize_text(gray)

    # 글자를 가로로 뭉쳐 한 줄을 하나의 덩어리로 만든다
    kw = max(9, int(w * 0.035))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (kw // 2 or 1, 3)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    # 1단계: 모양만으로 후보를 고른다 (넓고 납작하고 속이 찬 덩어리 = 글자 줄)
    cands: list[tuple[int, int, int, int, int]] = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if not (h * 0.004 < ch < h * 0.07):  # 너무 얇거나 두꺼운 건 줄이 아님
            continue
        if cw / max(ch, 1) < 6:
            continue
        if area < cw * ch * 0.15:  # 속이 너무 빈 덩어리 제외
            continue
        if cw < w * 0.08:  # 아주 짧은 파편만 배제
            continue
        cands.append((i, x, y, cw, ch))

    if not cands:
        return []

    # 2단계: 길이 기준을 이미지 폭이 아니라 '찾아낸 줄들'에 맞춘다.
    # 이미지 폭 대비 고정 비율로 자르면, 펼침면처럼 한 줄이 전체 폭의 일부만
    # 차지하는 경우에 본문 줄이 전부 걸러져 버린다.
    ref = float(np.percentile([c[3] for c in cands], 75))
    lines: list[np.ndarray] = []
    for i, x, y, cw, ch in cands:
        if cw < ref * 0.5:
            continue
        # 열마다 글자 픽셀의 무게중심 y = 그 줄의 중심선
        sub = (labels[y: y + ch, x: x + cw] == i)
        cols = np.where(sub.any(axis=0))[0]
        if len(cols) < cw * 0.5:
            continue
        ys = np.array([sub[:, c].nonzero()[0].mean() for c in cols], dtype=np.float32)
        lines.append(np.stack([cols + x, ys + y], axis=1))

    return lines


def _fit_curves(lines: list[np.ndarray], w: int, h: int) -> tuple[np.ndarray | None, float, float]:
    """줄들의 공통 휘어짐 d(x)를 뽑는다. (d, 잔차, 불일치도) 반환."""
    xs = np.arange(w, dtype=np.float32)
    curves = []
    residuals = []
    # 짧은 줄은 2차 피팅이 불안정하다. 기준을 이미지 폭이 아니라 실제로 찾은
    # 줄들의 길이에 맞춘다 (find_text_lines와 같은 이유).
    spans = [float(p[:, 0].max() - p[:, 0].min()) for p in lines]
    min_span = max(w * 0.10, float(np.percentile(spans, 75)) * 0.5) if spans else w * 0.25
    for pts in lines:
        x, y = pts[:, 0], pts[:, 1]
        if x.max() - x.min() < min_span:
            continue
        try:
            coef = np.polyfit(x, y, 2)
        except Exception:
            continue
        pred = np.polyval(coef, x)
        residuals.append(float(np.sqrt(np.mean((pred - y) ** 2))))
        # 상수항(줄의 높이)을 없애 '모양'만 남긴다
        curve = np.polyval(coef, xs)
        curves.append(curve - curve.mean())

    if len(curves) < MIN_LINES:
        return None, 0.0, 1.0

    stack = np.stack(curves)  # (줄 수, w)
    d = stack.mean(axis=0)
    # 줄들이 서로 얼마나 다른 모양인지 — 크면 측정을 믿을 수 없다
    disagree = float(np.mean(stack.std(axis=0))) / max(h, 1)
    residual = float(np.mean(residuals)) / max(h, 1)
    return d.astype(np.float32), residual, disagree


def _build_maps(d: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """수직 변위 d(x)로부터 remap용 좌표맵을 만든다."""
    xs = np.arange(w, dtype=np.float32)

    # 수평 보정: 휘어진 표면의 호 길이만큼 가로로 늘려줘야 원래 간격이 된다
    dp = np.gradient(d)
    arc = np.sqrt(1.0 + dp * dp)
    cum = np.cumsum(arc)
    cum -= cum[0]
    if cum[-1] > 1e-6:
        stretched = cum / cum[-1] * (w - 1)
    else:
        stretched = xs
    # 출력 x가 주어졌을 때 원본의 어느 x에서 가져올지 (역함수)
    src_x = np.interp(xs, stretched, xs).astype(np.float32)

    # 그 위치에서의 수직 변위
    d_at = np.interp(src_x, xs, d).astype(np.float32)

    map_x = np.tile(src_x, (h, 1))
    map_y = np.arange(h, dtype=np.float32)[:, None] + d_at[None, :]
    return map_x, map_y.astype(np.float32)


def dewarp(bgr: np.ndarray, enabled: bool = True) -> DewarpResult:
    """페이지를 평평하게 편다. 확신이 없으면 원본을 그대로 돌려준다."""
    if not enabled:
        return DewarpResult(bgr, False, "사용자가 끔")

    h0, w0 = bgr.shape[:2]
    scale = WORK_WIDTH / max(w0, 1)
    if scale < 1.0:
        work = cv2.resize(bgr, (WORK_WIDTH, max(1, round(h0 * scale))), interpolation=cv2.INTER_AREA)
    else:
        work, scale = bgr, 1.0
    wh, ww = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    lines = find_text_lines(gray)
    if len(lines) < MIN_LINES:
        return DewarpResult(bgr, False, f"본문 줄 {len(lines)}개 — {MIN_LINES}개 미만이라 생략",
                            n_lines=len(lines))

    d, residual, disagree = _fit_curves(lines, ww, wh)
    if d is None:
        return DewarpResult(bgr, False, "곡선 피팅 실패", n_lines=len(lines))

    curvature = float(np.ptp(d)) / max(wh, 1)
    if residual > MAX_FIT_RESIDUAL:
        return DewarpResult(bgr, False, f"피팅 잔차 큼 ({residual:.3f}) — 생략",
                            n_lines=len(lines), curvature=curvature)
    if disagree > MAX_LINE_DISAGREE:
        return DewarpResult(bgr, False, f"줄마다 휘어짐이 제각각 ({disagree:.3f}) — 생략",
                            n_lines=len(lines), curvature=curvature)
    if curvature < MIN_CURVATURE_FRAC:
        return DewarpResult(bgr, False, f"거의 평평함 ({curvature:.3f}) — 그대로 둠",
                            n_lines=len(lines), curvature=curvature)
    if curvature > MAX_CURVATURE_FRAC:
        return DewarpResult(bgr, False, f"휘어짐이 비정상적으로 큼 ({curvature:.3f}) — 측정 실패로 보고 생략",
                            n_lines=len(lines), curvature=curvature)

    # 원본 해상도로 확대 적용
    d_full = np.interp(
        np.linspace(0, ww - 1, w0), np.arange(ww, dtype=np.float32), d
    ).astype(np.float32) / scale
    # 안전을 위해 변위 크기를 한 번 더 제한
    limit = h0 * MAX_CURVATURE_FRAC
    d_full = np.clip(d_full, -limit, limit)

    map_x, map_y = _build_maps(d_full, w0, h0)
    out = cv2.remap(bgr, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return DewarpResult(out, True, f"적용됨 (줄 {len(lines)}개, 곡률 {curvature:.3f})",
                        n_lines=len(lines), curvature=curvature)


def estimate_rotation(bgr: np.ndarray) -> int:
    """본문이 세로로 서 있으면 90도 회전이 필요하다고 알려준다.

    상하 반전(180도)은 OCR 없이는 못 가리므로 UI의 수동 버튼에 맡긴다.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = 800 / max(gray.shape)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    horiz = len(find_text_lines(gray))
    vert = len(find_text_lines(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)))
    if vert > horiz + 2:
        return 90
    return 0
