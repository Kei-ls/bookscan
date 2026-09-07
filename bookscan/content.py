"""본문 블록 — 재단 경계가 안쪽으로 넘어오지 못하게 막는 안전선.

이 파이프라인이 페이지를 망치는 방식에는 뚜렷한 비대칭이 있다. 경계가 종이
바깥으로 조금 나가면 배경이 조금 남을 뿐이지만, 안쪽으로 조금 들어오면
**글자가 잘린다.** 잘린 글자는 뒤 단계에서 되돌릴 방법이 없다. 두 실수의 값이
다르니 경계를 한 방향으로만 밀어주는 장치를 따로 둔다.

`booktrim`도 같은 생각으로 `content_bounds`를 쓰고 있었지만, 그건 이미 펴진
이미지 안에서 자르는 상자를 넓히는 것뿐이다. **곡선 워프가 안 떠온 픽셀은
거기에 아예 없다.** 그래서 이 모듈의 안전선은 워프보다 **먼저**, 원본 좌표계에서
걸린다.

고치려는 건 오직 하나, **경계가 글자를 관통한 자리**다. 그걸 알아보는 조건도
둘뿐이다 (`outward_reach`).

  1) **안쪽에도 잉크가 있어야 한다.** 경계 바로 안쪽이 빈 여백이면, 바깥에 뭐가
     있든 우리가 자른 글자가 아니다.
  2) **바깥으로 이어져야 한다.** 잉크를 따라 걸어나가되 빈틈이 일정 폭 이상
     벌어지면 멈춘다. 덕분에 경계 너머가 **옆 페이지**일 때는 책등 여백에서
     걸음이 멈춘다 — 넓게 잡힌 사각형을 이 장치가 **더 넓히지는 않는다**.

두 조건 다 "잉크"가 정말 글자여야 성립한다. 국소 대비만 보면 **종이의 끝**이
글자와 똑같이 생겼고(실측: 멀쩡한 사진에서도 네 변이 죄다 20~40px씩 밀렸다),
그 띠는 두께가 있어서 안쪽 창까지 물들여 1)번 조건마저 무력화한다.

그래서 세 번째 조건이 붙는다. **한 변을 따라 훑은 띠 안에서, 그 변과 나란하게
쭉 이어지는 줄은 글자가 아니다.** 종이의 끝(그리고 거기 이어지는 책등 그림자와
옆 페이지)은 변을 따라 거의 모든 표본에 걸쳐 나타나고, 글자는 몇 %에만 걸린다.
실측하면 그 줄은 채움비율 0.9, 본문 한 열은 0.15로 사이가 넓다.

연결 성분의 뻗은 길이로 걸러 봤다가 물러섰다. 왼쪽 여백의 글자가 바로 옆
종이 끝선과 한 덩어리로 이어져 있어서, 선을 지우면 **지키려던 글자가 같이
지워졌다.** 채움비율은 표본 프레임 안에서 재기 때문에 그런 일이 없다 — 글자
하나가 변 전체를 채울 수는 없으니까.

밀어내는 양은 한 변마다 하나의 값이다. 변의 모양은 그대로 두고 통째로 평행이동
시킨다. 점마다 다르게 밀면 곡선이 울퉁불퉁해지고, 그걸 매끄럽다고 가정하는
`warp.curved_maps`가 뒤에서 탈이 난다.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detect import order_corners

#: 잉크 마스크를 만들 작업 크기. 글자는 국소 대비로 잡으므로 이 정도면 충분하고,
#: 원본 해상도가 달라져도 같은 굵기의 획을 보게 된다.
INK_LONG_EDGE = 800
#: 국소 대비를 볼 때의 흐림 반경과 문턱. 어두운 글자만 찾으면 색지 위 흰 글자를
#: 놓치므로 차이의 **절댓값**을 본다 (`booktrim`에서 옮겨온 규칙).
INK_SIGMA = 3.0
INK_THRESHOLD = 12.0
#: 글자가 있는 행/열로 인정할 최소 잉크 비율
INK_ROW_RATIO = 0.005
#: 한 변을 따라 이 비율 이상의 표본에서 잉크가 나오는 깊이(offset)는 글자가
#: 아니라 그 변과 나란한 줄로 본다. 실측: 종이 끝선 0.9, 본문 한 열 0.15.
EDGE_FILL = 0.35

#: 한 변을 바깥으로 밀 수 있는 최대 거리 (그 변에 수직인 쪽 치수 대비).
#: 잘린 글자를 되찾는 데는 1%면 넉넉하다. 이보다 크게 밀어야 한다면 경계가
#: 조금 어긋난 게 아니라 검출 자체가 틀린 것이라, 여기서 손대봐야 소용없다.
MAX_PUSH = 0.030
#: 잉크 사이 빈틈이 이보다 넓게 벌어지면 다른 덩어리로 보고 멈춘다. 같은 폭이
#: 경계 안쪽을 들여다보는 창의 깊이로도 쓰인다. 책등이나 페이지 여백은 이보다
#: 훨씬 넓고, 글자와 글자 사이 틈은 이보다 훨씬 좁다.
PUSH_GAP = 0.008
#: 경계 안쪽을 들여다보는 창의 깊이 (변에 수직인 쪽 치수 대비). 잘린 글자는
#: 이 안에 나머지 획이 있다. 페이지 여백은 이보다 훨씬 넓어 빈 채로 남는다.
INNER_LOOK = 0.020
#: 잉크를 걷어낸 뒤 더 두는 여백. 글자가 가장자리에 딱 붙는 걸 막는다.
PUSH_MARGIN = 0.004
#: 이보다 작게 밀어야 한다면 잡음을 본 것으로 보고 그냥 둔다.
MIN_PUSH = 0.0015
#: 한 변에서 뽑을 표본 수의 상한. 아래쪽은 변 길이(작업 좌표 1픽셀당 하나)로
#: 정해진다 — 글자 하나가 표본 여러 개에 걸쳐야 중앙값 필터를 견딘다.
MAX_SAMPLES = 2048
#: 양 끝(모서리) 근처는 이웃 변과 겹쳐 지저분하다. 이만큼 잘라내고 본다.
END_SKIP = 0.03
#: 밀어낼 거리 프로파일에 거는 중앙값 필터의 폭 (표본 수). 표본 한둘이 튀는
#: 잡음은 지우고, 글자 하나가 만드는 봉우리는 남긴다.
MEDIAN_WIDTH = 5


def ink_mask(bgr: np.ndarray, long_edge: int | None = INK_LONG_EDGE) -> tuple[np.ndarray, float]:
    """글자가 있는 곳을 255로 만든 마스크와, 원본 대비 축소 배율.

    이진화가 아니라 **국소 대비**를 본다. 종이색과 배경색이 비슷해도, 색지 위
    흰 글자여도 잡힌다. 이 두 경우가 실제로 페이지를 잘라먹은 적이 있다.

    `long_edge=None`이면 줄이지 않는다. 경계를 미는 판단은 그렇게 해야 한다 —
    800픽셀로 줄이면 획 하나가 서너 픽셀이 되어, 경계가 가로지른 글자가 표본
    몇 개에만 걸리고 그대로 묻힌다(실측: 눈으로 보이는 잘림을 못 잡아냈다).
    `content_bounds`는 상자 하나만 얻으면 되니 줄여도 된다.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    scale = 1.0 if long_edge is None else long_edge / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    blur = cv2.GaussianBlur(gray, (0, 0), INK_SIGMA)
    diff = np.abs(gray.astype(np.float32) - blur.astype(np.float32))
    return ((diff > INK_THRESHOLD).astype(np.uint8) * 255), scale


def content_bounds(bgr: np.ndarray) -> tuple[int, int, int, int]:
    """글자가 실제로 존재하는 범위 (위, 아래, 왼쪽, 오른쪽). 원본 좌표계.

    축에 나란한 상자라 기울어진 페이지에서는 실제 본문보다 넉넉해진다. 안쪽으로
    못 들어오게 막는 용도이므로 넉넉한 쪽이 안전한 방향이다.
    """
    h, w = bgr.shape[:2]
    ink, scale = ink_mask(bgr)
    sh, sw = ink.shape[:2]
    binary = ink > 0
    rows = binary.mean(axis=1) > INK_ROW_RATIO
    cols = binary.mean(axis=0) > INK_ROW_RATIO
    if not rows.any() or not cols.any():
        return 0, h - 1, 0, w - 1
    r, c = np.where(rows)[0], np.where(cols)[0]
    return (int(r[0] / sh * h), int(r[-1] / sh * h),
            int(c[0] / sw * w), int(c[-1] / sw * w))


def _sample(img: np.ndarray, base: np.ndarray, normal: np.ndarray,
            offs: np.ndarray) -> np.ndarray:
    """base의 각 점에서 normal 방향으로 offs만큼 떨어진 곳의 값. (S, K).

    화면 밖은 0(빈 곳)으로 읽는다. 덕분에 밀어내는 거리가 사진 경계를 넘지
    못한다 — 사진 밖에는 되찾을 글자가 없기 때문이다.
    """
    mx = (base[:, None, 0] + offs[None, :] * normal[0]).astype(np.float32)
    my = (base[:, None, 1] + offs[None, :] * normal[1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _median_smooth(a: np.ndarray, width: int = MEDIAN_WIDTH) -> np.ndarray:
    if len(a) < width or width < 3:
        return a
    pad = width // 2
    win = np.lib.stride_tricks.sliding_window_view(np.pad(a, pad, mode="edge"), width)
    return np.median(win, axis=1)


def outward_reach(ink: np.ndarray, base: np.ndarray, normal: np.ndarray,
                  max_push: float, gap: float, inner: float) -> np.ndarray:
    """각 점에서 **가로질린 글자**가 바깥으로 이어지는 거리. 아니면 0.

    모듈 설명의 세 조건이 그대로 여기 있다. 변과 나란한 줄은 잉크로 세지 않고(3),
    안쪽 창이 비어 있으면 시작조차 하지 않으며(1), 바깥으로 걷다 빈틈이 `gap`
    보다 벌어지면 멈춘다(2).
    """
    gap_px = max(1, int(round(gap)))
    in_px = max(1, int(round(inner)))
    k = int(max(2, round(max_push))) + 1

    offs = np.arange(-in_px, k, dtype=np.float32)
    hit = _sample(ink, base, normal, offs) > 127

    # 3) 변과 나란히 쭉 이어지는 깊이는 종이의 끝이지 글자가 아니다.
    hit[:, hit.mean(axis=0) > EDGE_FILL] = False

    # 1) 경계 바로 안쪽이 비어 있으면 우리가 자른 글자가 아니다.
    alive = hit[:, :in_px].any(axis=1)

    # 2) 바깥으로 걸어나가며 잉크를 따라간다.
    out = hit[:, in_px:]
    last = np.full(len(base), -1, np.int32)
    blank = np.zeros(len(base), np.int32)
    for i in range(k):
        col = out[:, i]
        last = np.where(col & alive, i, last)
        blank = np.where(col, 0, blank + 1)
        alive &= blank < gap_px
    return np.where(last >= 0, last + 1.0, 0.0).astype(np.float32)


def side_push(ink: np.ndarray, a: np.ndarray, b: np.ndarray, normal: np.ndarray,
              perp: float) -> float:
    """한 변을 바깥으로 얼마나 밀어야 글자를 안 자르는지. 작업 좌표계 거리.

    변을 따라 촘촘히 훑어 가장 크게 삐져나온 곳에 맞춘다. 잘린 글자는 변의 한
    군데에만 있을 수 있으니 평균이나 백분위로는 잡히지 않는다. 대신 표본 한둘이
    튀는 잡음에 끌려가지 않도록 중앙값 필터를 먼저 걸고 최댓값을 본다.
    """
    e = b - a
    length = float(np.linalg.norm(e))
    if length < 16.0:
        return 0.0
    n = int(np.clip(round(length), 64, MAX_SAMPLES))
    ts = np.linspace(END_SKIP, 1.0 - END_SKIP, n, dtype=np.float32)
    base = a[None, :] + ts[:, None] * e[None, :]

    reach = outward_reach(ink, base, normal, perp * MAX_PUSH,
                          perp * PUSH_GAP, perp * INNER_LOOK)
    push = float(_median_smooth(reach).max())
    if push < perp * MIN_PUSH:
        return 0.0
    return float(min(push + perp * PUSH_MARGIN, perp * MAX_PUSH))


def outward_normal(a: np.ndarray, b: np.ndarray, center: np.ndarray) -> np.ndarray:
    """변(또는 곡선의 현) a→b의 법선 중 `center`에서 멀어지는 쪽 (단위 벡터)."""
    e = b - a
    length = float(np.linalg.norm(e))
    if length < 1e-6:
        return np.array([0.0, 0.0], np.float32)
    u = e / length
    n = np.array([-u[1], u[0]], np.float32)
    if float(np.dot(n, (a + b) * 0.5 - center)) < 0:
        n = -n
    return n


def _intersect(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray,
               fallback: np.ndarray) -> np.ndarray:
    """두 직선의 교점. 거의 나란하면 fallback."""
    den = float(d0[0] * d1[1] - d0[1] * d1[0])
    if abs(den) < 1e-9:
        return fallback
    w = p1 - p0
    t = float(w[0] * d1[1] - w[1] * d1[0]) / den
    hit = p0 + t * d0
    return hit if np.isfinite(hit).all() else fallback


def push_sides(quad: np.ndarray, pushes: tuple[float, float, float, float],
               center: np.ndarray | None = None) -> np.ndarray:
    """네 변을 각자의 바깥 방향으로 밀고, 다시 교차시켜 모서리를 잡는다.

    변마다 평행이동만 하므로 방향은 그대로고, 모서리는 원래 자리에서 자연스럽게
    따라 움직인다. 변 순서는 `order_corners` 기준으로 0=위, 1=오른쪽, 2=아래,
    3=왼쪽이다.
    """
    q = np.asarray(quad, np.float32).reshape(4, 2)
    cen = q.mean(axis=0) if center is None else center
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        lines.append((a + outward_normal(a, b, cen) * pushes[i], b - a))
    return np.stack([
        _intersect(lines[(i - 1) % 4][0], lines[(i - 1) % 4][1],
                   lines[i][0], lines[i][1], q[i])
        for i in range(4)
    ]).astype(np.float32)


def guard_quad(bgr: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, str]:
    """사각형이 글자를 자르지 않도록 네 변을 필요한 만큼 바깥으로 민다.

    (사각형, 사유) 반환. 밀 필요가 없으면 사유는 빈 문자열이다.
    """
    q = order_corners(np.asarray(quad, np.float32))
    ink, scale = ink_mask(bgr, long_edge=None)
    qs = q * scale
    center = qs.mean(axis=0)

    quad_w = float(max(np.linalg.norm(qs[1] - qs[0]), np.linalg.norm(qs[2] - qs[3])))
    quad_h = float(max(np.linalg.norm(qs[3] - qs[0]), np.linalg.norm(qs[2] - qs[1])))
    # 변 i는 qs[i] → qs[i+1]. 0=위, 1=오른쪽, 2=아래, 3=왼쪽.
    perps = (quad_h, quad_w, quad_h, quad_w)

    pushes = []
    for i in range(4):
        a, b = qs[i], qs[(i + 1) % 4]
        pushes.append(side_push(ink, a, b, outward_normal(a, b, center),
                                max(24.0, perps[i])))

    if not any(p > 0 for p in pushes):
        return q, ""

    out = push_sides(qs, tuple(pushes), center)

    names = ("위", "오른쪽", "아래", "왼쪽")
    moved = ", ".join(f"{names[i]} {pushes[i] / scale:.0f}px"
                      for i in range(4) if pushes[i] > 0)
    return (out / scale).astype(np.float32), f"글자를 자르지 않도록 넓힘 ({moved})"
