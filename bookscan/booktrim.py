"""책 영역만 남기고 바깥을 잘라낸다 — 곡선 경계 기준으로.

`bookmask`가 준 실루엣의 위/아래/왼쪽/오른쪽 경계를 각각 하나의 곡선으로 보고,
그 네 곡선이 출력 직사각형의 네 변이 되도록 다시 매핑한다(Coons 패치). 결과:

  * 책 바깥(책상·바닥)이 사라진다
  * 바깥 테두리의 휘어짐이 곧게 펴진다 — 사각형 재단으로는 못 하는 부분이다

사각형으로 잘라내면 휜 테두리와 직선 사이에 배경 쐐기가 남는다. 그래서 자르지
않고 **편다**.

경계가 매끄럽지 않거나 변위가 과하면 기하 변형을 포기하고, 마스크 바깥만 종이색
으로 지운 뒤 여백을 잘라내는 안전한 동작으로 내려간다. 잘못 편 페이지는 안 편
페이지보다 나쁘다는 원칙은 여기서도 같다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: 경계 곡선을 다항식으로 맞출 때의 차수. 책 테두리는 완만해서 3차면 충분하다.
POLY_DEG = 3
#: 곡선 피팅 잔차가 해당 변 길이의 이 비율을 넘으면 경계를 믿지 않는다.
MAX_FIT_RESIDUAL = 0.030
#: 한 경계가 이 비율 넘게 출렁이면 마스크가 잘못됐다고 본다.
MAX_CURVE_SWING = 0.25
#: 곡선을 바깥으로 미는 거리의 상한. 이만큼 밀어야 한다면 경계가 곡선으로
#: 설명되지 않는다는 뜻이라 기하 변형을 포기한다.
MAX_ENVELOPE_PUSH = 0.06
#: 한 변에서 잘라낼 수 있는 최대 비율. 마스크가 아무리 안쪽을 가리켜도 이 이상은
#: 안 자른다. 배경과 종이 색이 비슷하면 GrabCut이 페이지 안쪽을 배경으로 오인한다.
MAX_TRIM_FRACTION = 0.10
#: 글자가 있는 행/열로 인정할 최소 잉크 비율
INK_ROW_RATIO = 0.005
#: 맵은 성기게 만들어 늘린다. 매끄러운 맵이라 손실이 없고 메모리를 크게 아낀다.
MAP_GRID = 512


@dataclass
class TrimResult:
    image: np.ndarray
    applied: bool
    reason: str


def _boundaries(mask: np.ndarray) -> tuple[np.ndarray, ...] | None:
    """열마다 위/아래, 행마다 왼쪽/오른쪽 경계 위치. 빈 줄은 보간한다."""
    m = mask > 0
    col_has, row_has = m.any(axis=0), m.any(axis=1)
    if col_has.sum() < 8 or row_has.sum() < 8:
        return None
    h, w = mask.shape[:2]

    top = np.argmax(m, axis=0).astype(np.float32)
    bottom = (h - 1 - np.argmax(m[::-1], axis=0)).astype(np.float32)
    left = np.argmax(m, axis=1).astype(np.float32)
    right = (w - 1 - np.argmax(m[:, ::-1], axis=1)).astype(np.float32)

    xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    for arr, has, grid in ((top, col_has, xs), (bottom, col_has, xs),
                           (left, row_has, ys), (right, row_has, ys)):
        if not has.all():
            arr[~has] = np.interp(grid[~has], grid[has], arr[has])
    return top, bottom, left, right


def _fit(values: np.ndarray, lo: int, hi: int, span: float,
         outward: int) -> tuple[np.ndarray, float, float] | None:
    """[lo, hi] 구간을 0~1로 매개변수화해 다항식으로 평활화한다.

    그냥 최소제곱으로 맞추면 실제 경계의 **절반이 곡선 바깥**에 놓인다. 그 상태로
    재단하면 페이지 가장자리가 잘려나간다(실제로 "Part 3" 제목이 날아갔다).
    그래서 두 가지를 더 한다.

      1) 이상치를 한 번 걸러내고 다시 맞춘다. 마스크 경계의 톱니에 곡선이
         끌려가는 걸 막는다.
      2) 곡선을 바깥으로 밀어, 실제 경계의 97%가 곡선 안쪽에 오게 한다.
         배경이 조금 남는 게 본문이 잘리는 것보다 낫다.

    `outward`는 바깥 방향(-1이면 값이 작아지는 쪽, +1이면 커지는 쪽).
    (평활화된 곡선, 잔차, 바깥으로 민 거리) 반환.
    """
    if hi - lo < 8:
        return None
    seg = values[lo:hi + 1].astype(np.float64)
    t = np.linspace(0.0, 1.0, len(seg))
    try:
        coef = np.polyfit(t, seg, POLY_DEG)
        resid = seg - np.polyval(coef, t)
        rms = float(np.sqrt(np.mean(resid ** 2)))
        keep = np.abs(resid) < 2.5 * rms if rms > 1e-6 else np.ones(len(seg), bool)
        if keep.sum() > POLY_DEG * 3:
            coef = np.polyfit(t[keep], seg[keep], POLY_DEG)
    except Exception:
        return None

    pred = np.polyval(coef, t)
    residual = float(np.sqrt(np.mean((pred - seg) ** 2))) / max(span, 1.0)

    # 실제 경계의 97%를 품도록 바깥으로 민다
    diff = seg - pred
    shift = float(np.percentile(diff, 3.0 if outward < 0 else 97.0))
    shift = min(shift, 0.0) if outward < 0 else max(shift, 0.0)

    smooth = np.polyval(coef, np.linspace(0.0, 1.0, MAP_GRID)) + shift
    return smooth.astype(np.float32), residual, abs(shift) / max(span, 1.0)


def _content_bounds(bgr: np.ndarray) -> tuple[int, int, int, int]:
    """글자가 실제로 존재하는 범위 (위, 아래, 왼쪽, 오른쪽).

    재단이 여기를 넘어 들어오지 못하게 막는 안전선이다. 배경과 종이 색이 비슷하면
    GrabCut이 페이지 안쪽을 배경으로 오인하는데(사진 01에서 제목이 잘렸다),
    글자는 국소 대비로 잡히므로 이 선은 그런 실패에 휘둘리지 않는다.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = 800 / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = gray.shape[:2]
    # 어두운 글자만 찾으면 색지 위 흰 글자를 놓친다 (실제로 그 때문에 컬러 간지의
    # 큰 제목이 잘렸다). 국소 대비의 절댓값을 보면 두 극성이 다 잡힌다.
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    binary = np.abs(gray.astype(np.float32) - blur.astype(np.float32)) > 12.0
    rows = binary.mean(axis=1) > INK_ROW_RATIO
    cols = binary.mean(axis=0) > INK_ROW_RATIO
    if not rows.any() or not cols.any():
        return 0, h - 1, 0, w - 1
    r = np.where(rows)[0]
    c = np.where(cols)[0]
    return (int(r[0] / sh * h), int(r[-1] / sh * h),
            int(c[0] / sw * w), int(c[-1] / sw * w))


def _corners(mask: np.ndarray) -> np.ndarray | None:
    """마스크 윤곽에서 네 모서리를 고른다 (좌표합·좌표차 최소/최대)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(pts) < 8:
        return None
    s, d = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    return np.stack([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]])  # TL, TR, BR, BL


def _coons_maps(top, bottom, left, right, out_w: int, out_h: int):
    """네 경계 곡선을 출력 사각형의 네 변으로 보내는 좌표맵."""
    g = MAP_GRID
    u = np.linspace(0.0, 1.0, g, dtype=np.float32)
    U, V = np.meshgrid(u, u)

    (tx, ty), (bx, by), (lx, ly), (rx, ry) = top, bottom, left, right
    p00 = np.array([tx[0], ty[0]]);   p10 = np.array([tx[-1], ty[-1]])
    p01 = np.array([bx[0], by[0]]);   p11 = np.array([bx[-1], by[-1]])

    def blend(t0, t1, l0, l1, c00, c10, c01, c11):
        return ((1 - V) * t0[None, :] + V * t1[None, :]
                + (1 - U) * l0[:, None] + U * l1[:, None]
                - ((1 - U) * (1 - V) * c00 + U * (1 - V) * c10
                   + (1 - U) * V * c01 + U * V * c11))

    map_x = blend(tx, bx, lx, rx, p00[0], p10[0], p01[0], p11[0])
    map_y = blend(ty, by, ly, ry, p00[1], p10[1], p01[1], p11[1])
    map_x = cv2.resize(map_x.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y


def _paper_color(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """책 안쪽 밝은 부분의 색 = 종이색. 지운 자리를 이걸로 채운다."""
    inside = bgr[mask > 0]
    if len(inside) < 100:
        return np.array([255, 255, 255], np.uint8)
    lum = inside.mean(axis=1)
    bright = inside[lum >= np.percentile(lum, 85)]
    return np.clip(bright.mean(axis=0), 0, 255).astype(np.uint8)


def _erase_background(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크 바깥을 종이색으로 지우고 남은 여백을 잘라낸다 (안전한 폴백)."""
    out = bgr.copy()
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    out[inner == 0] = _paper_color(bgr, mask)
    x, y, w, h = cv2.boundingRect(inner if inner.any() else mask)
    if w <= 16 or h <= 16:
        return out
    # 여기서도 글자를 자르면 안 된다
    ink_t, ink_b, ink_l, ink_r = _content_bounds(bgr)
    x0, y0 = min(x, ink_l), min(y, ink_t)
    x1, y1 = max(x + w, ink_r + 1), max(y + h, ink_b + 1)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(bgr.shape[1], x1), min(bgr.shape[0], y1)
    return out[y0:y1, x0:x1]


def trim_to_book(bgr: np.ndarray, mask: np.ndarray | None,
                 enabled: bool = True, geometry: bool = True) -> TrimResult:
    """책 영역만 남긴다. 경계가 믿을 만하면 곡선을 펴서, 아니면 지우고 잘라서.

    `geometry=False`면 아래 쿤스 패치 단계를 통째로 건너뛰고 배경만 지운다.
    앞단의 곡선 워프(`pagecurve` + `warp.curved_maps`)가 이미 테두리를 폈을 때
    쓴다. 편 걸 또 펴면 두 번 편 셈이 되고, 3500픽셀짜리 리샘플링만 한 번 더
    낭비된다.
    """
    if not enabled:
        return TrimResult(bgr, False, "사용자가 끔")
    if mask is None or not (mask > 0).any():
        return TrimResult(bgr, False, "책 영역을 못 따냄 — 그대로 둠")
    if not geometry:
        return TrimResult(_erase_background(bgr, mask), True,
                          f"배경 지움 (곡선 보정이 이미 폄, {(1 - float((mask > 0).mean())) * 100:.0f}% 제거)")

    h, w = bgr.shape[:2]
    covered = float((mask > 0).mean())
    if covered > 0.985:
        return TrimResult(bgr, False, f"책이 화면을 거의 채움 ({covered:.3f}) — 재단 불필요")

    bnd = _boundaries(mask)
    corners = _corners(mask)
    if bnd is None or corners is None:
        return TrimResult(_erase_background(bgr, mask), True, "경계 추출 실패 — 배경만 지움")
    top, bottom, left, right = bnd
    tl, tr, br, bl = corners

    x_lo_t, x_hi_t = int(round(min(tl[0], tr[0]))), int(round(max(tl[0], tr[0])))
    x_lo_b, x_hi_b = int(round(min(bl[0], br[0]))), int(round(max(bl[0], br[0])))
    y_lo_l, y_hi_l = int(round(min(tl[1], bl[1]))), int(round(max(tl[1], bl[1])))
    y_lo_r, y_hi_r = int(round(min(tr[1], br[1]))), int(round(max(tr[1], br[1])))

    fits = [
        _fit(top, x_lo_t, x_hi_t, h, -1), _fit(bottom, x_lo_b, x_hi_b, h, +1),
        _fit(left, y_lo_l, y_hi_l, w, -1), _fit(right, y_lo_r, y_hi_r, w, +1),
    ]
    if any(f is None for f in fits):
        return TrimResult(_erase_background(bgr, mask), True, "경계 피팅 실패 — 배경만 지움")

    worst = max(f[1] for f in fits)
    if worst > MAX_FIT_RESIDUAL:
        return TrimResult(_erase_background(bgr, mask), True,
                          f"경계가 매끄럽지 않음 (잔차 {worst:.3f}) — 배경만 지움")
    push = max(f[2] for f in fits)
    if push > MAX_ENVELOPE_PUSH:
        return TrimResult(_erase_background(bgr, mask), True,
                          f"경계가 들쭉날쭉함 (보정 {push:.3f}) — 배경만 지움")

    swing = max(float(np.ptp(fits[0][0])) / h, float(np.ptp(fits[1][0])) / h,
                float(np.ptp(fits[2][0])) / w, float(np.ptp(fits[3][0])) / w)
    if swing > MAX_CURVE_SWING:
        return TrimResult(_erase_background(bgr, mask), True,
                          f"경계가 과하게 출렁임 ({swing:.3f}) — 배경만 지움")

    # 두 겹의 안전선으로 곡선이 안쪽으로 파고드는 걸 막는다.
    #   1) 글자가 있는 곳은 절대 못 넘어온다
    #   2) 그와 별개로 한 변에서 잘라낼 수 있는 양 자체를 제한한다
    ink_t, ink_b, ink_l, ink_r = _content_bounds(bgr)
    lim_v, lim_h = h * MAX_TRIM_FRACTION, w * MAX_TRIM_FRACTION
    top_v = np.minimum(fits[0][0], min(ink_t, lim_v))
    bot_v = np.maximum(fits[1][0], max(ink_b, h - 1 - lim_v))
    left_v = np.minimum(fits[2][0], min(ink_l, lim_h))
    right_v = np.maximum(fits[3][0], max(ink_r, w - 1 - lim_h))

    # 곡선을 (x, y) 쌍으로. top/bottom은 x가 매개변수, left/right는 y가 매개변수.
    lin = np.linspace
    top_c = (lin(x_lo_t, x_hi_t, MAP_GRID, dtype=np.float32), top_v)
    bot_c = (lin(x_lo_b, x_hi_b, MAP_GRID, dtype=np.float32), bot_v)
    left_c = (left_v, lin(y_lo_l, y_hi_l, MAP_GRID, dtype=np.float32))
    right_c = (right_v, lin(y_lo_r, y_hi_r, MAP_GRID, dtype=np.float32))

    out_w = int(round(max(np.mean(right_c[0] - left_c[0]), 16.0)))
    out_h = int(round(max(np.mean(bot_c[1] - top_c[1]), 16.0)))
    if out_w < 32 or out_h < 32:
        return TrimResult(_erase_background(bgr, mask), True, "재단 결과가 너무 작음 — 배경만 지움")
    out_w, out_h = min(out_w, w * 2), min(out_h, h * 2)

    map_x, map_y = _coons_maps(top_c, bot_c, left_c, right_c, out_w, out_h)
    out = cv2.remap(bgr, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 가장자리에 남은 배경 얼룩을 종이색으로 덮는다
    warped_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.erode(warped_mask, np.ones((3, 3), np.uint8), iterations=1)
    if (warped_mask == 0).any():
        out[warped_mask == 0] = _paper_color(bgr, mask)

    return TrimResult(out, True,
                      f"책 경계로 재단 (잔차 {worst:.3f}, 배경 {(1 - covered) * 100:.0f}% 제거)")
