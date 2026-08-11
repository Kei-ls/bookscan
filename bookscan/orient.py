"""페이지 방향 판별 — 글줄이 가로로 눕도록 90도 단위 회전을 정한다.

책을 옆으로 눕혀 찍으면 글줄이 사진 안에서 세로로 흐른다. 그 상태로는
곡면 펴기(글줄 곡률 측정)도, 펼침면 분할(세로 책등 탐색)도 전제가 깨져서
전부 헛돈다. 그래서 워프 직후에 축을 바로잡는다.

**상하(180도)는 판별하지 않는다.** 판별하려고 시도했고 반증했다:

  글줄 안에서 잉크가 위/아래로 쏠린 정도(라틴 문자의 어센더가 디센더보다
  많다는 성질)를 재봤다. 실제 사진 3장에서는 정방향에서 일관되게 음수
  (-0.058 / -0.057 / -0.049)가 나왔지만, 폰트를 바꿔 렌더한 합성 페이지
  11종에서는 정방향에서 양수(Times +0.017, Georgia +0.028, Verdana +0.023)로
  **부호가 반대로** 나왔다. 규칙 적중 1/11.

  즉 이 신호는 책·폰트·촬영 조건에 따라 부호가 뒤집힌다. 틀린 자동 회전은
  안 한 것보다 나쁘므로 채택하지 않았다. 대신 한 번에 찍은 묶음은 상하가
  모두 같다는 점을 이용해 UI에서 **배치 전체를 한 번에** 뒤집게 했다.
  (제대로 하려면 OCR이 필요한데, 그러자고 tesseract를 끌어들이지는 않는다.)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .dewarp import find_text_lines

#: 방향 판별용 작업 해상도. 글줄 덩어리만 보면 되므로 크게 볼 필요가 없다.
WORK_LONG_EDGE = 900
#: 한쪽 방향의 글줄 수가 다른 쪽의 이 배 이상이어야 회전을 결정한다.
DOMINANCE = 1.6
#: 이보다 글줄이 적으면 판단 근거가 부족하다고 본다 (사진·도판만 있는 페이지 등).
MIN_LINES = 4


@dataclass
class Orientation:
    #: 시계방향 회전각. 0 또는 90만 나온다 (180/270은 판별하지 않는다).
    rotate: int
    confidence: float
    reason: str
    lines_h: int = 0
    lines_v: int = 0


def estimate_axis(bgr: np.ndarray) -> Orientation:
    """글줄이 가로로 눕는 축을 고른다.

    현재 방향과 90도 돌린 방향에서 각각 글줄 수를 세어 많은 쪽을 택한다.
    두 방향이 비슷하면 손대지 않는다 — 애매할 때 돌리는 건 손해다.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = WORK_LONG_EDGE / max(gray.shape[:2])
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    n_h = len(find_text_lines(gray))
    n_v = len(find_text_lines(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)))

    if max(n_h, n_v) < MIN_LINES:
        return Orientation(0, 0.0, f"본문 줄이 거의 없어 방향 판단 보류 (가로 {n_h}, 세로 {n_v})",
                           n_h, n_v)

    lo, hi = min(n_h, n_v), max(n_h, n_v)
    ratio = hi / max(lo, 1)
    if ratio < DOMINANCE:
        return Orientation(0, 0.0,
                           f"가로·세로 글줄 수가 비슷해 회전 안 함 (가로 {n_h}, 세로 {n_v})",
                           n_h, n_v)

    # 우세비 1.6배에서 0, 4배 이상에서 1이 되도록
    conf = float(np.clip((ratio - DOMINANCE) / (4.0 - DOMINANCE), 0.0, 1.0))
    if n_v > n_h:
        return Orientation(90, conf, f"세로 글줄이 우세해 90도 회전 (가로 {n_h}, 세로 {n_v})",
                           n_h, n_v)
    return Orientation(0, conf, f"이미 가로 방향 (가로 {n_h}, 세로 {n_v})", n_h, n_v)
