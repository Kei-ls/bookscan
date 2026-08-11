"""책 영역 마스크 — 사진에서 책이 실제로 차지하는 모양을 따낸다.

펼친 책은 사각형이 아니다. 바깥 테두리가 바깥으로 불룩하고, 종이 뭉치의 두께
때문에 위아래 경계가 휘어 있다. 그래서 직선 네 변으로 감싸면 모서리 쪽에
책상·바닥이 딸려 들어온다. 펼침면을 반으로 쪼개면 그 배경이 한쪽 페이지에
몰려서 더 눈에 띈다.

검출된 사각형을 씨앗으로 GrabCut을 돌려 실제 실루엣을 얻는다. 사각형이 이미
"대충 맞는" 상태라 씨앗으로 쓰기에 충분하고, GrabCut이 색 모델을 학습해
경계를 실제 책 가장자리로 끌어당긴다.

  안쪽 88%  → 확실한 전경
  사각형 안 → 전경일 가능성 높음
  110%까지  → 배경일 가능성 높음
  그 바깥   → 확실한 배경

마스크가 이상하면(면적이 크게 어긋나거나 조각나면) None을 돌려주고 호출부는
기존 동작을 그대로 쓴다.
"""

from __future__ import annotations

import cv2
import numpy as np

#: GrabCut을 돌릴 해상도. 실루엣만 있으면 되므로 크게 볼 필요가 없다.
WORK_LONG_EDGE = 700
ITERATIONS = 5

#: 씨앗 사각형을 얼마나 넓혀 "배경일 가능성" 띠로 둘 것인가.
#: 넉넉히 열어주면(예: 1.10) GrabCut이 어두운 책상을 책의 어두운 부분과 같은 색으로
#: 보고 바깥으로 자라버린다. 실제로 그 실패를 겪었다. 우리가 원하는 건 사각형이
#: 과하게 잡은 부분을 **안쪽으로 깎는** 것이므로 바깥은 거의 열어두지 않는다.
SEED_OUTER = 1.02
#: 이 안쪽은 확실한 전경으로 못박는다.
SEED_INNER = 0.85

#: 결과 마스크 면적이 씨앗 사각형 대비 이 범위를 벗어나면 실패로 본다.
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.55, 1.45
#: 가장 큰 덩어리가 전체 마스크의 이 비율은 되어야 한다 (조각나면 못 믿는다).
MIN_SOLIDITY = 0.85


def _scaled(quad: np.ndarray, factor: float) -> np.ndarray:
    """사각형을 자기 중심 기준으로 확대/축소."""
    c = quad.mean(axis=0)
    return ((quad - c) * factor + c).astype(np.int32)


def book_mask(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """책이 차지하는 영역을 255로 채운 마스크. 실패하면 None.

    입력 이미지와 같은 크기로 돌려준다.
    """
    h0, w0 = bgr.shape[:2]
    scale = WORK_LONG_EDGE / max(h0, w0)
    if scale < 1.0:
        small = cv2.resize(bgr, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                           interpolation=cv2.INTER_AREA)
        q = np.asarray(quad, np.float32) * scale
    else:
        small, q, scale = bgr, np.asarray(quad, np.float32), 1.0

    h, w = small.shape[:2]
    seed_area = float(cv2.contourArea(q.astype(np.float32)))
    if seed_area < h * w * 0.05:
        return None

    gm = np.full((h, w), cv2.GC_BGD, np.uint8)
    cv2.fillPoly(gm, [_scaled(q, SEED_OUTER)], cv2.GC_PR_BGD)
    cv2.fillPoly(gm, [_scaled(q, 1.00)], cv2.GC_PR_FGD)
    cv2.fillPoly(gm, [_scaled(q, SEED_INNER)], cv2.GC_FGD)
    if not (gm == cv2.GC_FGD).any() or not (gm == cv2.GC_BGD).any():
        return None  # 씨앗이 한쪽으로 쏠리면 GrabCut이 의미가 없다

    try:
        cv2.grabCut(small, gm, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), ITERATIONS, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return None

    mask = np.where((gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = max(3, int(min(h, w) * 0.012)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    total = float((mask > 0).sum())
    if total <= 0:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    big_area = float(stats[big, cv2.CC_STAT_AREA])
    if big_area / total < MIN_SOLIDITY:
        return None  # 여러 조각으로 갈라졌다 = 분할 실패
    mask = np.where(labels == big, 255, 0).astype(np.uint8)

    ratio = big_area / seed_area
    if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
        return None  # 사각형과 너무 다르면 뭔가 잘못 잡은 것

    # 구멍 메우기 — 책 안쪽의 어두운 도판이 배경으로 빠지는 걸 막는다
    filled = mask.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)

    if scale < 1.0:
        filled = cv2.resize(filled, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return (filled > 127).astype(np.uint8) * 255
