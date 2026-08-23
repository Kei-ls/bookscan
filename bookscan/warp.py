"""원근 보정. 검출된 사각형을 정면에서 본 직사각형으로 편다.

사각형에는 `warp_page`(호모그래피)를, 굽은 테두리(`pagecurve.PageCurve`)에는
`warp_curved`(쿤스 패치 + 곡률 기반 재매개화)를 쓴다.
"""

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

#: 곡선 워프 좌표맵을 만들 격자. 맵이 매끄러워서 성기게 만들어 늘려도 손실이 없다.
MAP_GRID = 512
#: 국소 확대율 보정의 상한. 책등 바로 앞은 종이가 거의 옆을 보고 있어서 이론상
#: 확대율이 0으로 가는데, 그 근처는 테두리 측정 자체가 못 믿을 곳이다. 실제 책의
#: 눌림은 1.2~1.35배 수준이라 1.5면 넉넉하면서도 폭주를 막는다.
MAX_STRETCH = 1.5


def _fit_size(w: float, h: float, aspect: str, max_long_edge: int) -> tuple[int, int]:
    """비율 프리셋과 최대 길이를 반영한 출력 크기."""
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


def target_size(quad: np.ndarray, aspect: str = "auto", max_long_edge: int = 3500) -> tuple[int, int]:
    """워프 결과의 (폭, 높이).

    마주보는 변 중 긴 쪽을 취한다. 원근 때문에 가까운 쪽 변이 길게 찍히는데,
    그 쪽에 맞춰야 해상도 손실이 없다.
    """
    tl, tr, br, bl = quad
    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    return _fit_size(float(w), float(h), aspect, max_long_edge)


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


def _flat_axis(mid: np.ndarray, ruler: np.ndarray,
               max_stretch: float = MAX_STRETCH) -> tuple[np.ndarray, float]:
    """중앙선을 따라간 사진 위 거리를, 종이를 폈을 때의 거리로 바꾼다.

    원근에서 확대율은 방향을 가리지 않는다(등방). 그러니 **세로 자가 짧게 찍힌
    곳은 가로도 같은 비율로 눌려 있다.** 책등 근처가 왜 가로로 뭉개지는지가
    이 한 문장으로 설명되고, 되돌리는 법도 같이 나온다 — 그 자리의 자 길이가
    가장 긴 데의 몇 분의 몇인지 재서, 그 역수만큼 늘려주면 된다.

    글자를 보고 재는 방법(`dewarp.py`)과 달리 **글자가 없어도 된다.** 사진 속
    그림이 한 면을 다 채워도, 테두리만 보이면 잴 수 있다.

    (누적 길이 (G,), 실제로 쓴 최대 확대율) 반환.
    """
    step = np.linalg.norm(np.diff(mid, axis=0), axis=1)   # 구간별 사진 위 길이
    seg = 0.5 * (ruler[:-1] + ruler[1:])                  # 구간 대표 자 길이
    ref = float(np.percentile(ruler, 90.0))               # 가장 덜 눌린 곳 = 기준
    gain = np.clip(ref / np.maximum(seg, 1e-6), 1.0, max_stretch)
    flat = np.concatenate([[0.0], np.cumsum(step * gain)])
    return flat, float(gain.max())


def curved_maps(curve, aspect: str = "auto", max_long_edge: int = 3500):
    """굽은 테두리를 직사각형으로 펴는 좌표맵. (map_x, map_y, w, h, 최대확대율).

    두 가지가 동시에 일어난다.

      * **모양**: 네 곡선을 직사각형의 네 변으로 보내는 쿤스(Coons) 패치. 위/아래가
        휜 만큼 안쪽 행들도 같이 따라 움직여서, 활처럼 휜 페이지가 곧게 펴진다.
      * **간격**: 격자를 균등하게 찍지 않고 `_flat_axis`가 준 '펼친 길이'에 맞춰
        찍는다. 눌린 곳에 출력 픽셀을 더 많이 배정하는 셈이라 글자 간격이 돌아온다.

    쿤스 패치만 쓰면 테두리는 곧아지지만 책등 근처 글자는 여전히 눌린 채로
    남는다. 두 번째가 사용자가 말한 '굽은 정도에 따른 후보정'에 해당한다.

    두 가지를 **서로 다른 좌표계에서** 잰다. 헷갈리기 쉬운 부분이라 이유를 적어 둔다.

      * **모양은 직사각형 좌표계에서.** 호모그래피를 바탕에 깔고 곡선은 거기서
        벗어난 잔차만 얹는다. 겹선형 쿤스 패치만 쓰면 원근을 풀지 못해서, 곧은
        사각형을 넣어도 호모그래피와 평균 48픽셀이나 어긋났다. 바탕을 깐 뒤에는
        0.8픽셀이다.
      * **자 길이는 원본 좌표계에서.** 이건 반대로 해 봤다가 확실히 나빠졌다
        (비율 오차 0.026 → 0.109). 호모그래피는 **네 모서리를 못박기** 때문에
        좌우 변의 자 길이를 강제로 같게 만드는데, 단면 사진에서 종이가 말려
        들어가는 곳이 바로 그 좌우 변이다. 재야 할 눌림을 재기도 전에 지워버린다.
        펼침면은 눌리는 곳이 한가운데라 영향이 없어서, 처음엔 개선인 줄 알았다.

    남는 한계도 적어 둔다. 테두리만 보고서는 "책이 접혀서 왼쪽이 멀어진 것"과
    "평평한 종이를 왼쪽에서 비스듬히 찍은 것"을 구별할 수 없다. 둘 다 왼쪽 자가
    짧게 나오기 때문이다. 이 함수는 전부 접힘으로 본다. 위에서 내려찍는 보통의
    책 사진에서는 그 가정이 맞고 실측도 좋지만, 많이 비스듬한 사진에서는 결과가
    실제보다 조금 갸름해진다 (`tools/eval_curve.py`의 bent_single_tilted).
    """
    T, B, L, R = curve.top, curve.bottom, curve.left, curve.right
    g_in = len(T)
    if g_in < 8 or len(B) != g_in or len(L) != g_in or len(R) != g_in:
        return None

    ruler_v = np.linalg.norm(B - T, axis=1)  # u마다의 세로 자
    ruler_u = np.linalg.norm(R - L, axis=1)  # v마다의 가로 자
    if ruler_v.min() < 1e-3 or ruler_u.min() < 1e-3:
        return None

    flat_x, gain_x = _flat_axis(0.5 * (T + B), ruler_v)
    flat_y, gain_y = _flat_axis(0.5 * (L + R), ruler_u)
    if flat_x[-1] < 32 or flat_y[-1] < 32:
        return None
    out_w, out_h = _fit_size(flat_x[-1], flat_y[-1], aspect, max_long_edge)

    # 출력 격자를 매개변수 공간으로 되돌린다 (펼친 길이의 역함수)
    par = np.linspace(0.0, 1.0, g_in)
    u = np.interp(np.linspace(0.0, flat_x[-1], MAP_GRID), flat_x, par)
    v = np.interp(np.linspace(0.0, flat_y[-1], MAP_GRID), flat_y, par)

    # 바탕은 호모그래피로 깐다. 쿤스 패치의 겹선형 보간은 원근을 풀지 못해서,
    # 비스듬히 찍은 사진에서 곧은 격자를 조금씩 부채꼴로 벌린다. 사각형 부분은
    # 호모그래피에 맡기고, **곡선은 거기서 벗어난 잔차만** 얹는다. 잔차는 작아서
    # 겹선형으로 섞어도 탈이 없다.
    H, rw, rh = warp_transform(curve.corners, "auto", max_long_edge=1 << 24)
    try:
        inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    def to_rect(p: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(
            np.ascontiguousarray(p, np.float32).reshape(1, -1, 2), H).reshape(-1, 2)

    # **각 변을 자기 좌표로 매개화해야 한다.** 곡선 위 점을 매개변수 순서대로
    # 늘어놓고 "i번째 점은 직사각형 변의 i/N 지점" 이라고 두면 안 된다. 호모그래피는
    # 직선 위 점들의 간격 비를 보존하지 않기 때문에, 곧은 변조차 큰 잔차를 갖게
    # 된다(실측: 곧은 사각형인데 평균 48px). 위/아래 변은 x로, 좌/우 변은 y로
    # 매개화하면 곧은 변의 잔차가 정확히 0이 되고, 남는 건 진짜 휘어짐뿐이다.
    Tr, Br, Lr, Rr = (to_rect(c) for c in (T, B, L, R))
    xs = u * (rw - 1)
    ys = v * (rh - 1)
    dy_t = np.interp(xs, Tr[:, 0], Tr[:, 1])
    dy_b = np.interp(xs, Br[:, 0], Br[:, 1]) - (rh - 1)
    dx_l = np.interp(ys, Lr[:, 1], Lr[:, 0])
    dx_r = np.interp(ys, Rr[:, 1], Rr[:, 0]) - (rw - 1)

    # 네 모서리는 호모그래피가 직사각형 꼭짓점에 정확히 보내므로 잔차가 0이다.
    # 덕분에 쿤스 패치의 모서리 보정항이 통째로 사라지고 이 두 줄만 남는다.
    U, V = u[None, :], v[:, None]  # 행 = v, 열 = u
    rect_x = xs[None, :] + (1 - U) * dx_l[:, None] + U * dx_r[:, None]
    rect_y = ys[:, None] + (1 - V) * dy_t[None, :] + V * dy_b[None, :]
    src = cv2.perspectiveTransform(
        np.stack([rect_x, rect_y], axis=-1).reshape(1, -1, 2).astype(np.float32),
        inv).reshape(MAP_GRID, MAP_GRID, 2)

    map_x = cv2.resize(src[:, :, 0], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(src[:, :, 1], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y, out_w, out_h, max(gain_x, gain_y)


def warp_curved(bgr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """`curved_maps`가 만든 맵으로 이미지를 편다."""
    return cv2.remap(bgr, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def remap_mask(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """마스크를 같은 맵으로 옮긴다. 바깥은 배경(0)."""
    out = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (out > 127).astype(np.uint8) * 255


def clamp_corners(corners: np.ndarray | list, shape: tuple[int, int]) -> np.ndarray:
    """코너를 이미지 범위 안으로 (약간의 여유를 두고) 밀어넣는다."""
    h, w = shape[:2]
    q = np.asarray(corners, dtype=np.float32).reshape(4, 2).copy()
    q[:, 0] = np.clip(q[:, 0], -w * 0.02, w * 1.02)
    q[:, 1] = np.clip(q[:, 1], -h * 0.02, h * 1.02)
    return q
