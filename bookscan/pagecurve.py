"""굽은 페이지 윤곽선 검출 — 책 테두리를 직선 네 개가 아니라 곡선 네 개로 본다.

`detect.py`가 내놓는 건 사각형이다. 펼친 책에는 맞지 않는다. 종이가 책등 쪽으로
말려 들어가면서 위/아래 테두리가 활처럼 휘기 때문이다. 사각형의 변은 그 활의
현(chord)이라, 가운데로 갈수록 실제 종이에서 멀어진다.

이 모듈은 그 사각형을 **씨앗**으로만 쓰고, 각 변을 따라가며 진짜 테두리가
어디인지 다시 찾는다.

  1) **어림잡기** — 책 실루엣 마스크가 있으면 변에 수직으로 깊게 훑어 경계가
     대충 어디쯤인지 알아낸다. 마스크는 속이 꽉 찬 도형이라 가짜 봉우리라는 게
     없어서, 얼마나 깊이 파고든 테두리든 한 번에 찾아낸다.
  2) **다듬기** — 그 어림 곡선 위에 띠를 다시 얹고, 이번엔 **사진의 밝기
     기울기**로 정확한 위치를 잡는다. 띠를 좁혀가며 두 번 반복하되, 나온
     후보들 중 **경계 안팎의 밝기 단차가 가장 큰 것**을 고른다.
  3) 뽑힌 점들에 3차 다항식을 강건하게 맞춘다 (이상치 걸러내고 재적합)
  4) **확인** — 고른 자리의 단차가 충분한지 본다. 아니면 그 변을 통째로 버린다.

마스크에 위치만 묻고 최종 판정은 사진에 맡기는 게 이 순서의 핵심이다. 마스크는
틀린다 — 흰 책상 위 흰 종이에서는 GrabCut이 종이 한가운데를 가로지르는 울퉁불퉁한
덩어리를 내놓는다. 그걸 그대로 믿으면 평평한 종이에 5%짜리 가짜 휘어짐이 생기고,
그 상태로 펴면 멀쩡한 페이지가 뒤틀린다. 4)번 관문이 그런 마스크를 걸러낸다:
종이 한가운데에는 넘을 단차가 없기 때문이다.

반대로 마스크 없이 기울기만으로 시작할 수도 있다(재단을 꺼도 곡선 보정은 되어야
한다). 그때는 현 근처만 좁게 보고 시작해서 걸어 들어간다 — 처음부터 넓게 보면
본문 첫 줄이나 책상 무늬가 딸려 들어오기 때문이다.

**휘어짐을 재는 것 자체가 목적이다.** 곡선에서 직선 성분을 빼고 남은 양이
`bow`이고, 이 값이 `warp.warp_curved`가 얼마나 세게 펼지를 결정한다. 거의 0이면
곡선으로 볼 이유가 없으니 curved=False를 돌려주고, 호출부는 기존 원근 보정을
그대로 쓴다. 평평한 사진에서는 아무것도 달라지지 않아야 한다는 뜻이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detect import order_corners

#: 곡선 하나를 몇 점으로 표현할 것인가. 완만한 곡선이라 이 정도면 차고 넘친다.
CURVE_POINTS = 256
#: 한 변에서 뽑을 표본 수
SAMPLES = 128
#: 양 끝(모서리) 근처는 두 변이 겹쳐 신호가 지저분하다. 이만큼 잘라내고 본다.
END_SKIP = 0.05
#: 탐색 띠의 폭. 변에 수직인 쪽 치수 대비 비율로, (바깥쪽, 안쪽)이다.
#:
#: **안팎을 다르게 잡는 게 중요하다.** 사각형의 변은 활의 현이라, 진짜 테두리는
#: 언제나 그 현의 **안쪽**에 있다. 바깥은 조금만 열어두면 된다.
#:
#: 마스크는 속이 꽉 찬 실루엣이라 바깥에서 안으로 훑다가 처음 닿는 곳이 곧
#: 경계다. 가짜 봉우리라는 게 존재할 수 없으니 마음껏 깊이 봐도 된다. 펼침면의
#: 책등 부근은 실제로 높이의 12%까지 파고들어서, 이 여유가 없으면 놓친다.
MASK_OUT, MASK_IN = 0.08, 0.45
#: 기울기 쪽은 다르다. 깊이 볼수록 본문 첫 줄이나 책등 그림자가 딸려 들어온다.
#: 좁게 시작해서 아래 정밀화 단계로 걸어 들어간다.
EDGE_IN = 0.18
#: 정밀화를 돌 때마다 띠 반폭을 EDGE_IN의 이 비율로 줄인다
REFINE_SHRINK = (0.50, 0.20)
#: 경계 곡선 차수. 책 테두리는 완만해서 3차면 충분하고, 더 올리면 출렁인다.
POLY_DEG = 3

#: 다항식 잔차(변 길이 대비)가 이보다 크면 그 변을 믿지 않는다
MAX_RESIDUAL = 0.012
#: 이상치를 걸러내고 남은 표본 비율이 이보다 낮으면 측정 실패
MIN_INLIER_RATIO = 0.55
#: 경계 안팎의 밝기 차가 이만큼(0~255)은 나야 진짜 테두리로 인정한다.
#: 흰 책상 위 흰 종이도 실측 15 정도는 나온다. 6이면 그런 최악의 경우는
#: 통과시키면서, 종이 한가운데를 찍은 엉터리 마스크는 걸러낸다.
MIN_EDGE_STEP = 6.0
#: 그 단차의 부호가 표본들 사이에서 이만큼은 일치해야 한다. 진짜 테두리는
#: 한쪽이 일관되게 밝다. 부호가 반반이면 잡음을 본 것이다.
MIN_STEP_AGREE = 0.70
#: 이보다 덜 휘면 곡선 취급할 이유가 없다 (현 길이 대비)
MIN_BOW = 0.008
#: 이보다 크면 테두리가 아니라 엉뚱한 걸 따라간 것으로 본다
MAX_BOW = 0.25


@dataclass
class PageCurve:
    """네 개의 곡선으로 표현한 페이지 테두리. 좌표계는 입력 이미지와 같다.

    매개변수 방향을 맞춰 놓은 게 중요하다. top과 bottom은 둘 다 왼→오(u 방향),
    left와 right는 둘 다 위→아래(v 방향)다. 그래서 같은 인덱스 i끼리가 서로
    마주보는 짝이 되고, top[i]와 bottom[i]를 이으면 그게 곡면의 자(ruling)다.
    """

    top: np.ndarray     # (N,2) TL -> TR
    bottom: np.ndarray  # (N,2) BL -> BR
    left: np.ndarray    # (N,2) TL -> BL
    right: np.ndarray   # (N,2) TR -> BR
    bows: tuple[float, float, float, float]  # top, bottom, left, right (부호 있음)
    source: str  # "mask" | "edge"
    curved: bool
    reason: str

    @property
    def corners(self) -> np.ndarray:
        """TL, TR, BR, BL."""
        return np.stack([self.top[0], self.top[-1],
                         self.bottom[-1], self.bottom[0]]).astype(np.float32)

    @property
    def bow(self) -> float:
        """가장 크게 휜 변의 휘어짐. UI와 로그에 쓰는 대표값."""
        return max(abs(b) for b in self.bows)

    def polygon(self) -> np.ndarray:
        """시계방향 폐곡선 (N,2). 오버레이 그리기와 마스크 채우기에 쓴다."""
        return np.concatenate([
            self.top, self.right[1:], self.bottom[::-1][1:], self.left[::-1][1:],
        ]).astype(np.float32)

    def scaled(self, factor: float) -> "PageCurve":
        """다른 해상도의 좌표계로 옮긴다."""
        return PageCurve(
            self.top * factor, self.bottom * factor,
            self.left * factor, self.right * factor,
            self.bows, self.source, self.curved, self.reason,
        )


def _sample_along(img: np.ndarray, base: np.ndarray, normal: np.ndarray,
                  offs: np.ndarray, nearest: bool = False,
                  replicate: bool = False) -> np.ndarray:
    """base의 각 점에서 normal 방향으로 offs만큼 떨어진 곳의 값. (S, K) 반환.

    마스크는 화면 밖을 배경(0)으로 봐야 맞고, 밝기는 가장자리 값을 늘려 읽어야
    맞다. 0으로 읽으면 화면 밖이 새까만 걸로 보여서 없는 단차가 생긴다.
    """
    mx = (base[:, None, 0] + offs[None, :] * normal[0]).astype(np.float32)
    my = (base[:, None, 1] + offs[None, :] * normal[1]).astype(np.float32)
    flag = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    border = cv2.BORDER_REPLICATE if replicate else cv2.BORDER_CONSTANT
    return cv2.remap(img, mx, my, flag, borderMode=border, borderValue=0)


def _edge_step(gray: np.ndarray, base: np.ndarray, normal: np.ndarray,
               off: np.ndarray, gap: float = 3.0, width: float = 7.0) -> float:
    """경계 안팎의 밝기 차. '여기 정말 테두리가 있나'를 묻는 마지막 관문.

    경계선 자체는 흐릿하게 번져 있으니 `gap`만큼 띄우고 양쪽을 각각 평균 낸다.
    부호가 표본들 사이에서 일치하지 않으면 잡음을 본 것이므로 0을 돌려준다.
    """
    pts = base + off[:, None] * normal[None, :]
    inn = np.linspace(-gap, -gap - width, 6, dtype=np.float32)
    out = np.linspace(gap, gap + width, 6, dtype=np.float32)
    d = (_sample_along(gray, pts, normal, inn, replicate=True).mean(axis=1)
         - _sample_along(gray, pts, normal, out, replicate=True).mean(axis=1))
    med = float(np.median(d))
    if abs(med) < 1e-6:
        return 0.0
    agree = float((np.sign(d) == np.sign(med)).mean())
    return abs(med) if agree >= MIN_STEP_AGREE else 0.0


def _offsets_from_mask(mask: np.ndarray, base: np.ndarray, normal: np.ndarray,
                       offs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """마스크 경계를 가로지르는 지점. offs는 바깥→안쪽 순서여야 한다."""
    vals = _sample_along(mask, base, normal, offs, nearest=True) > 127
    # 띠의 바깥 끝이 이미 마스크 안이면 경계가 띠 밖에 있다는 뜻 — 못 믿는다
    ok = vals.any(axis=1) & ~vals[:, 0]
    idx = np.argmax(vals, axis=1)
    return offs[idx], ok


def _offsets_from_gradient(grad: np.ndarray, base: np.ndarray, normal: np.ndarray,
                           offs: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """밝기 기울기가 가장 센 지점. 현 근처를 선호하는 가중치를 곱해서 고른다."""
    vals = _sample_along(grad, base, normal, offs).astype(np.float32)
    prior = np.exp(-(offs.astype(np.float32) ** 2) / (2.0 * max(sigma, 1e-3) ** 2))
    w = vals * prior[None, :]

    k = np.argmax(w, axis=1)
    rows = np.arange(len(k))
    ok = w[rows, k] > 1e-6

    # 봉우리 좌우 값으로 포물선을 맞춰 소수점 아래를 건진다. 띠 간격이 1픽셀보다
    # 넓을 수 있어서 이 보정이 실제로 의미가 있다.
    step = float(offs[1] - offs[0]) if len(offs) > 1 else 1.0
    out = offs[k].astype(np.float32)
    inner = (k > 0) & (k < w.shape[1] - 1)
    if inner.any():
        i = rows[inner]
        y0, y1, y2 = w[i, k[i] - 1], w[i, k[i]], w[i, k[i] + 1]
        den = y0 - 2 * y1 + y2
        safe = np.where(np.abs(den) > 1e-9, den, 1.0)
        shift = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / safe, 0.0)
        out[i] += np.clip(shift, -1.0, 1.0) * step
    return out, ok


def _robust_poly(t: np.ndarray, off: np.ndarray,
                 deg: int = POLY_DEG) -> tuple[np.ndarray, float, float] | None:
    """이상치를 걸러가며 다항식을 맞춘다. (계수, RMS 잔차, 인라이어 비율)."""
    if len(t) < deg * 3:
        return None
    keep = np.ones(len(t), bool)
    coef = None
    for _ in range(3):
        if keep.sum() < deg * 3:
            return None
        try:
            coef = np.polyfit(t[keep], off[keep], deg)
        except Exception:
            return None
        r = off - np.polyval(coef, t)
        # 표준편차 대신 MAD를 쓴다. 표본 몇 개가 크게 튀면 표준편차 자체가
        # 부풀어 올라 이상치를 이상치로 못 알아본다.
        mad = float(np.median(np.abs(r - np.median(r))))
        s = max(1.4826 * mad, 1e-3)
        nxt = np.abs(r) <= 3.0 * s
        if (nxt == keep).all():
            break
        keep = nxt
    if coef is None or keep.sum() < deg * 3:
        return None
    rms = float(np.sqrt(np.mean((off[keep] - np.polyval(coef, t[keep])) ** 2)))
    return coef, rms, float(keep.mean())


def _bow(coef: np.ndarray, t: np.ndarray) -> float:
    """직선 성분을 뺀 나머지의 최대 크기. 이게 '얼마나 휘었나'다.

    변 전체가 안쪽으로 밀려 있거나(상수항) 한쪽으로 기울어 있는 것(1차항)은
    사각형이 조금 어긋났다는 뜻이지 휘었다는 뜻이 아니다. 그래서 뺀다.
    """
    y = np.polyval(coef, t)
    line = np.polyval(np.polyfit(t, y, 1), t)
    d = y - line
    return float(d[np.argmax(np.abs(d))])


def _fit_side(gray: np.ndarray, grad: np.ndarray, mask: np.ndarray | None,
              a: np.ndarray, b: np.ndarray, center: np.ndarray,
              perp: float) -> tuple[np.ndarray, float, str] | None:
    """한 변을 곡선으로 만든다. (곡선 위 점 (N,2), bow, 출처) 반환.

    `perp`는 이 변에 수직인 쪽의 치수(위/아래 변이면 사각형 높이)다. 탐색 깊이를
    전부 여기에 비례시켜서, 사진 해상도가 달라져도 같게 동작한다.
    """
    e = b - a
    length = float(np.linalg.norm(e))
    if length < 24.0:
        return None
    u = e / length
    n = np.array([-u[1], u[0]], np.float32)
    # 사각형 중심에서 멀어지는 쪽을 +로 잡는다. 그래야 "바깥 → 안쪽" 순서로
    # 띠를 훑는다고 말할 수 있고, 마스크 경계 찾기가 단순해진다.
    if float(np.dot(n, (a + b) * 0.5 - center)) < 0:
        n = -n

    ts = np.linspace(END_SKIP, 1.0 - END_SKIP, SAMPLES, dtype=np.float32)
    base = a[None, :] + ts[:, None] * e[None, :]

    def scan_grad(pts: np.ndarray, half: float, sigma_frac: float = 0.8):
        k = max(11, int(half * 2) | 1)
        offs = np.linspace(half, -half, k, dtype=np.float32)
        return _offsets_from_gradient(grad, pts, n, offs, half * sigma_frac)

    # 1) 어림잡기 — 마스크가 있으면 깊게 훑는다. 없으면 현 근처만 좁게 본다.
    fit, src = None, "edge"
    if mask is not None:
        k = max(21, int(perp * (MASK_OUT + MASK_IN)) | 1)
        offs = np.linspace(perp * MASK_OUT, -perp * MASK_IN, k, dtype=np.float32)
        off, ok = _offsets_from_mask(mask, base, n, offs)
        if ok.sum() >= POLY_DEG * 3:
            fit = _robust_poly(ts[ok], off[ok])
            if fit is not None:
                src = "mask"
    if fit is None:
        # 마스크 없이 시작할 때는 가중치를 좁게 건다. 띠가 넓으니 본문 첫 줄이
        # 함께 들어오는데, 그 봉우리가 테두리보다 셀 때가 있기 때문이다.
        off, ok = scan_grad(base, perp * EDGE_IN, sigma_frac=0.30)
        if ok.sum() < POLY_DEG * 3:
            return None
        fit = _robust_poly(ts[ok], off[ok])
        if fit is None:
            return None

    # 2) 다듬기 — 여기서부터는 마스크가 아니라 사진을 본다. 어림 곡선 위로 띠를
    #    옮겨 얹으므로, 깊이 파고든 변에서도 경계가 띠 한가운데에 온다.
    #
    #    후보들 중 **밝기 단차가 가장 큰 것**을 고른다. 처음엔 다항식 잔차로
    #    골라 봤는데 그러면 안 된다: 마스크는 한 픽셀 격자라 톱니가 섞이고 기울기는
    #    봉우리를 따라가서, 두 방식의 잔차는 애초에 같은 자로 잰 값이 아니다.
    #    실제로 그 비교 때문에 옳은 정밀화가 거부되고 부호가 뒤집힌 곡선이 남았다.
    #    "테두리인가"를 묻는 데는 단차가 곧바른 잣대다.
    def edge_strength(coef: np.ndarray) -> float:
        return _edge_step(gray, base, n, np.polyval(coef, ts).astype(np.float32))

    best, best_q = fit, edge_strength(fit[0])
    for shrink in REFINE_SHRINK:
        shift = np.polyval(fit[0], ts).astype(np.float32)
        off_i, ok_i = scan_grad(base + shift[:, None] * n[None, :],
                                max(5.0, perp * EDGE_IN * shrink))
        if ok_i.sum() < POLY_DEG * 3:
            break
        nxt = _robust_poly(ts[ok_i], (off_i + shift)[ok_i])  # 현 기준으로 되돌려 적합
        if nxt is None:
            break
        q = edge_strength(nxt[0])
        if q < best_q * 0.9:
            break  # 더 흐릿한 데로 걸어갔다 = 테두리를 놓쳤다
        fit = nxt
        if q > best_q:
            best, best_q = nxt, q

    coef, rms, inlier = best
    if rms / length > MAX_RESIDUAL or inlier < MIN_INLIER_RATIO:
        return None

    # 3) 확인 — 찾은 자리에 정말 밝기 단차가 있나. 마스크가 종이 한가운데를
    #    가리켰다면 여기서 걸린다.
    if best_q < MIN_EDGE_STEP:
        return None

    tt = np.linspace(0.0, 1.0, CURVE_POINTS, dtype=np.float32)
    d = np.polyval(coef, tt).astype(np.float32)
    pts = a[None, :] + tt[:, None] * e[None, :] + d[:, None] * n[None, :]
    return pts.astype(np.float32), _bow(coef, tt) / length, src


def _stitch(curves: list[np.ndarray]) -> list[np.ndarray]:
    """네 곡선의 끝점을 서로 맞춰 닫힌 테두리로 만든다.

    변마다 따로 맞춘 곡선이라 모서리에서 몇 픽셀씩 어긋난다. 이웃한 두 곡선의
    끝점 중간을 새 모서리로 삼고, 각 곡선에 선형 램프를 더해 거기에 닿게 한다.
    램프는 양 끝에서만 크고 가운데서는 거의 0이라 곡선의 모양은 그대로 남는다.
    """
    top, bottom, left, right = curves
    tl = (top[0] + left[0]) * 0.5
    tr = (top[-1] + right[0]) * 0.5
    bl = (bottom[0] + left[-1]) * 0.5
    br = (bottom[-1] + right[-1]) * 0.5

    def ramp(c: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        t = np.linspace(0.0, 1.0, len(c), dtype=np.float32)[:, None]
        return (c + (1 - t) * (p0 - c[0])[None, :] + t * (p1 - c[-1])[None, :]).astype(np.float32)

    return [ramp(top, tl, tr), ramp(bottom, bl, br),
            ramp(left, tl, bl), ramp(right, tr, br)]


def detect_curve(bgr: np.ndarray, quad: np.ndarray,
                 mask: np.ndarray | None = None) -> PageCurve | None:
    """사각형을 씨앗으로 굽은 테두리를 찾는다. 못 찾으면 None.

    mask(책 실루엣, bgr과 같은 크기)가 있으면 그 경계를 먼저 쓴다. 없으면 밝기
    기울기만으로 찾는다 — 재단을 꺼도 곡선 보정은 되어야 하기 때문이다.
    """
    h, w = bgr.shape[:2]
    q = order_corners(np.asarray(quad, np.float32))
    center = q.mean(axis=0)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), max(1.2, min(h, w) * 0.0015))
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)

    # 띠 폭은 그 변에 수직인 쪽 치수에 맞춘다. 위/아래 변은 높이, 좌/우 변은 폭.
    quad_w = float(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))
    quad_h = float(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))

    sides = (
        (q[0], q[1], quad_h),  # top:    TL -> TR
        (q[3], q[2], quad_h),  # bottom: BL -> BR
        (q[0], q[3], quad_w),  # left:   TL -> BL
        (q[1], q[2], quad_w),  # right:  TR -> BR
    )
    curves: list[np.ndarray] = []
    bows: list[float] = []
    srcs: list[str] = []
    for a, b, perp in sides:
        got = _fit_side(gray, grad, mask, a, b, center, max(24.0, perp))
        if got is None:
            return None
        pts, bow, src = got
        curves.append(pts)
        bows.append(bow)
        srcs.append(src)

    top, bottom, left, right = _stitch(curves)
    bows_t = (bows[0], bows[1], bows[2], bows[3])
    source = "mask" if srcs.count("mask") >= 3 else "edge"
    worst = max(abs(x) for x in bows_t)

    if worst > MAX_BOW:
        return PageCurve(top, bottom, left, right, bows_t, source, False,
                         f"휘어짐이 비정상적으로 큼 ({worst:.3f}) — 측정 실패로 보고 생략")
    if worst < MIN_BOW:
        return PageCurve(top, bottom, left, right, bows_t, source, False,
                         f"테두리가 거의 곧음 (휘어짐 {worst:.3f}) — 원근 보정만")
    return PageCurve(top, bottom, left, right, bows_t, source, True,
                     f"굽은 테두리 검출 ({source}, 위 {bows_t[0]:+.3f} 아래 {bows_t[1]:+.3f})")
