"""페이지 사각형 검출.

한 가지 방법으로는 모든 촬영 조건을 못 덮는다. 성격이 다른 세 가지 방법을
순서대로 시도하고, 공통 채점 함수로 가장 좋은 후보를 고른다. 전부 실패하면
프레임 전체를 낮은 confidence로 돌려주고 UI가 수동 보정을 유도한다.

  1) contour  - 배경과 밝기 대비가 있을 때 (가장 흔한 경우)
  2) lab      - 종이는 "밝고 채도 낮다"는 성질. 어두운/알록달록한 배경에 강함
  3) lines    - 흰 책상 위 흰 종이처럼 대비는 없지만 모서리 직선은 살아있을 때
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_AREA_RATIO = 0.12  # 프레임의 12% 미만이면 페이지로 안 봄
MAX_AREA_RATIO = 0.999
#: 이 점수 이상이면 뒤쪽 방법을 시도하지 않고 조기 종료.
#: 낮게 잡으면 애매한 사진에서 더 나은 방법을 시도해보지도 않고 멈춰버린다.
GOOD_ENOUGH = 0.93
#: 큰 후보가 작은 후보를 감싸고 있을 때, 큰 쪽 점수가 이 비율 이상이면 큰 쪽을 택한다
CONTAINMENT_TOLERANCE = 0.88

# --- 펼침면 병합 ---------------------------------------------------------
#: 짝이 되려면 두 후보의 면적 차가 이 배 안이어야 한다 (양쪽 페이지는 크기가 비슷하다)
PAIR_AREA_RATIO = 2.2
#: 합집합 껍질이 두 면적의 합보다 이 배 넘게 크면 둘 사이가 벌어진 것 = 짝이 아니다
PAIR_HULL_SLACK = 1.35
#: 서로 이만큼 넘게 겹치면 같은 페이지의 변형일 뿐이다
PAIR_OVERLAP_MAX = 0.15
#: 병합 후보를 승격시키는 문턱. 일반 승격(CONTAINMENT_TOLERANCE)보다 관대한 이유는
#: "비슷한 크기의 페이지 둘이 나란히 붙어 있다"는 구조적 근거가 이미 있기 때문이다.
SPREAD_TOLERANCE = 0.78
#: 병합 결과가 원래 후보의 몇 배 면적이어야 하는가 (두 페이지니까 2배 근처)
SPREAD_AREA_MIN, SPREAD_AREA_MAX = 1.55, 2.9


@dataclass
class Detection:
    corners: np.ndarray  # (4,2) float32, TL/TR/BR/BL 순서
    confidence: float
    method: str

    def to_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.corners]


def order_corners(pts: np.ndarray) -> np.ndarray:
    """네 점을 TL, TR, BR, BL 순서로 정렬한다.

    좌표합이 최소인 점이 TL, 최대가 BR. 좌표차(y-x)가 최소면 TR, 최대면 BL.
    회전이 45도를 크게 넘지 않는 한 안정적으로 동작한다.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _polygon_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _corner_angle_score(quad: np.ndarray) -> float:
    """네 내각이 90도에 얼마나 가까운지. 원근 때문에 어긋나는 건 어느 정도 허용."""
    scores = []
    for i in range(4):
        p0, p1, p2 = quad[(i - 1) % 4], quad[i], quad[(i + 1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        deg = np.degrees(np.arccos(cos))
        # 90도에서 40도까지 벗어나도 0점은 아니게
        scores.append(max(0.0, 1.0 - abs(deg - 90.0) / 45.0))
    return float(np.mean(scores))


def _edge_support(grad: np.ndarray, quad: np.ndarray, samples: int = 240) -> float:
    """사각형 둘레가 실제 이미지 경계 위에 놓여 있는 비율.

    검출된 사각형이 진짜 종이 테두리인지, 아니면 아무데나 그은 선인지를
    가르는 가장 결정적인 신호다.
    """
    h, w = grad.shape[:2]
    per_side = max(8, samples // 4)
    hits = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        t = np.linspace(0.02, 0.98, per_side)[:, None]  # 코너 근처는 제외
        pts = p0[None, :] * (1 - t) + p1[None, :] * t
        xs = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
        # 선에 정확히 안 걸쳐도 인정되도록 3x3 이웃의 최대값을 본다
        vals = np.zeros(len(xs), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy = np.clip(ys + dy, 0, h - 1)
                xx = np.clip(xs + dx, 0, w - 1)
                vals = np.maximum(vals, grad[yy, xx])
        hits.append(vals)
    vals = np.concatenate(hits)
    return float(np.mean(vals > 0.18))


def score_quad(quad: np.ndarray, grad: np.ndarray, shape: tuple[int, int]) -> float:
    """0~1 confidence. UI의 '확인 필요' 배지를 결정한다."""
    h, w = shape[:2]
    area = _polygon_area(quad)
    ratio = area / float(h * w)
    if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
        return 0.0
    if not _is_convex(quad):
        return 0.0

    # 너무 작은 것보다 큰 쪽을 선호하되, 프레임 전체와 같아지는 건 의심스럽다
    area_score = float(np.clip((ratio - MIN_AREA_RATIO) / 0.45, 0.0, 1.0))
    if ratio > 0.985:
        area_score *= 0.35

    angle = _corner_angle_score(quad)
    support = _edge_support(grad, quad)

    # 변 길이가 극단적으로 찌그러진 건 배제
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    if min(sides) < 1e-3:
        return 0.0
    aspect = max(sides) / min(sides)
    aspect_score = float(np.clip(1.0 - (aspect - 1.0) / 6.0, 0.1, 1.0))

    return float(0.45 * support + 0.25 * angle + 0.18 * area_score + 0.12 * aspect_score)


def _is_convex(quad: np.ndarray) -> bool:
    return bool(cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)))


def _gradient_map(gray: np.ndarray) -> np.ndarray:
    """0~1로 정규화한 gradient 크기.

    정규화 기준을 99퍼센타일로 잡으면 본문 글자의 진한 경계가 기준을 독점해서,
    흰 책상 위 흰 종이처럼 옅은 페이지 경계가 0에 가깝게 눌린다. 95퍼센타일이
    글자에 덜 휘둘리면서 옅은 경계를 살린다.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    hi = float(np.percentile(mag, 95.0))
    if hi < 1e-3:
        hi = float(mag.max()) or 1.0
    return np.clip(mag / hi, 0, 1).astype(np.float32)


def _quad_from_contour(cnt: np.ndarray) -> np.ndarray | None:
    """윤곽선을 사각형으로 근사. 여러 epsilon을 시도하고 안 되면 minAreaRect."""
    peri = cv2.arcLength(cnt, True)
    if peri < 1e-3:
        return None
    for eps in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx.reshape(4, 2).astype(np.float32))
    hull = cv2.convexHull(cnt)
    hp = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * hp, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx.reshape(4, 2).astype(np.float32))
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    return order_corners(np.asarray(box, dtype=np.float32))


def _candidates_from_mask(mask: np.ndarray, top: int = 6) -> list[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:top]
    out = []
    for c in cnts:
        q = _quad_from_contour(c)
        if q is not None:
            out.append(q)
    return out


# --------------------------------------------------------------------------
# 방법 1: 윤곽선
# --------------------------------------------------------------------------
def _detect_contour(bgr: np.ndarray, gray: np.ndarray) -> list[np.ndarray]:
    smooth = cv2.bilateralFilter(gray, 9, 60, 60)
    med = float(np.median(smooth))
    quads: list[np.ndarray] = []
    for mult in (0.66, 0.40):  # 대비가 약한 사진을 위해 낮은 임계도 시도
        lo = int(max(0, mult * med))
        hi = int(min(255, (mult + 0.67) * med))
        edges = cv2.Canny(smooth, lo, max(lo + 10, hi))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        quads.extend(_candidates_from_mask(edges))

    # 흰 책상 위 흰 종이: 국소 대비 평활화(CLAHE)로 몇 단계 안 되는 밝기 차를
    # 끌어올린 뒤 아주 낮은 임계로 Canny를 걸면 경계가 살아난다.
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(smooth)
    for lo, hi in ((10, 40), (20, 60)):
        edges = cv2.Canny(eq, lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        quads.extend(_candidates_from_mask(edges))
    return quads


# --------------------------------------------------------------------------
# 방법 2: 밝기/채도 분할
# --------------------------------------------------------------------------
def _detect_lab(bgr: np.ndarray, gray: np.ndarray) -> list[np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = cv2.magnitude(a, b)
    # 종이다움 = 밝고(L 높음) 색이 옅다(chroma 낮음)
    paperness = L - 2.2 * chroma
    paperness = cv2.normalize(paperness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    paperness = cv2.GaussianBlur(paperness, (7, 7), 0)

    quads: list[np.ndarray] = []
    _, otsu = cv2.threshold(paperness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (otsu,):
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        quads.extend(_candidates_from_mask(m))
    return quads


# --------------------------------------------------------------------------
# 방법 3: 직선 교점
# --------------------------------------------------------------------------
def _line_normal_form(x1, y1, x2, y2) -> tuple[float, float, float] | None:
    """직선을 ax+by=c (a²+b²=1) 형태로."""
    dx, dy = x2 - x1, y2 - y1
    n = float(np.hypot(dx, dy))
    if n < 1e-6:
        return None
    a, b = -dy / n, dx / n
    return a, b, a * x1 + b * y1


def _intersect(l1, l2) -> np.ndarray | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:  # 거의 평행
        return None
    return np.array([(c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det], dtype=np.float32)


def _detect_lines(bgr: np.ndarray, gray: np.ndarray) -> list[np.ndarray]:
    h, w = gray.shape[:2]
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    # 옅은 경계는 기본 Canny 임계(30/70)에서 통째로 사라진다. 낮은 임계부터
    # 시도하고, 선분이 충분히 모이면 거기서 멈춘다.
    segs = None
    for length_frac, (c1, c2) in ((0.05, (20.0, 50.0)), (0.05, (10.0, 30.0)),
                                  (0.12, (30.0, 70.0))):
        try:
            fld = cv2.ximgproc.createFastLineDetector(
                length_threshold=max(12, int(min(h, w) * length_frac)),
                distance_threshold=1.41,
                canny_th1=c1,
                canny_th2=c2,
                canny_aperture_size=3,
                do_merge=True,
            )
            found = fld.detect(smooth)
        except Exception:
            return []
        if found is not None and len(found) >= 4:
            segs = found
            break
    if segs is None or len(segs) == 0:
        return []

    horiz: list[tuple[float, tuple, float]] = []  # (위치, 직선, 길이)
    vert: list[tuple[float, tuple, float]] = []
    for s in segs.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, s)
        line = _line_normal_form(x1, y1, x2, y2)
        if line is None:
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        ang = min(ang, 180 - ang)  # 0=수평, 90=수직
        if ang < 35:
            horiz.append(((y1 + y2) / 2, line, length))
        elif ang > 55:
            vert.append(((x1 + x2) / 2, line, length))

    if len(horiz) < 2 or len(vert) < 2:
        return []

    # 프레임 가장자리에 딱 붙은 선은 사진 테두리일 가능성이 커서 제외
    margin_h, margin_w = h * 0.015, w * 0.015
    horiz = [t for t in horiz if margin_h < t[0] < h - margin_h] or horiz
    vert = [t for t in vert if margin_w < t[0] < w - margin_w] or vert
    if len(horiz) < 2 or len(vert) < 2:
        return []

    horiz.sort(key=lambda t: t[0])
    vert.sort(key=lambda t: t[0])
    # 가장 바깥쪽 두 개 + 가장 긴 두 개를 후보로 조합
    h_cands = {horiz[0][1]: 1, horiz[-1][1]: 1}
    v_cands = {vert[0][1]: 1, vert[-1][1]: 1}
    top = sorted(horiz, key=lambda t: -t[2])[:3]
    for _, line, _ in top:
        h_cands[line] = 1
    for _, line, _ in sorted(vert, key=lambda t: -t[2])[:3]:
        v_cands[line] = 1

    hl, vl = list(h_cands), list(v_cands)
    quads: list[np.ndarray] = []
    for i in range(len(hl)):
        for j in range(i + 1, len(hl)):
            for k in range(len(vl)):
                for m in range(k + 1, len(vl)):
                    pts = [
                        _intersect(hl[i], vl[k]), _intersect(hl[i], vl[m]),
                        _intersect(hl[j], vl[k]), _intersect(hl[j], vl[m]),
                    ]
                    if any(p is None for p in pts):
                        continue
                    q = np.array(pts, dtype=np.float32)
                    # 프레임에서 크게 벗어난 교점은 버린다
                    if (q[:, 0] < -w * 0.15).any() or (q[:, 0] > w * 1.15).any():
                        continue
                    if (q[:, 1] < -h * 0.15).any() or (q[:, 1] > h * 1.15).any():
                        continue
                    quads.append(order_corners(q))
    return quads


# --------------------------------------------------------------------------
def _contains(outer: np.ndarray, inner: np.ndarray, shape: tuple[int, int] | None = None) -> bool:
    """inner가 outer 안에 들어가는가. 약간의 여유를 둔다.

    여유가 필요한 이유가 둘 있다. 볼록 껍질을 사각형으로 근사하면 모서리가 조금씩
    깎여 들어가고, 직선 교점법은 프레임 밖 좌표를 내놓기도 한다. 엄격하게 보면
    실제로는 감싸고 있는데도 몇 픽셀 때문에 승격이 막힌다.
    """
    poly = outer.astype(np.float32).reshape(-1, 1, 2)
    pts = np.asarray(inner, dtype=np.float32).reshape(-1, 2)
    tol = 0.0
    if shape is not None:
        h, w = shape[:2]
        # 프레임 밖은 어차피 페이지가 아니므로 안으로 당겨서 비교한다
        pts = np.stack([np.clip(pts[:, 0], 0, w - 1), np.clip(pts[:, 1], 0, h - 1)], axis=1)
        tol = 0.012 * float(np.hypot(h, w))
    return all(cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -tol for x, y in pts)


def _overlap_area(a: np.ndarray, b: np.ndarray) -> float:
    area, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return float(area)


def _distinct(cands: list[Detection], limit: int) -> list[Detection]:
    """서로 충분히 다른 후보만 남긴다.

    직선 교점법은 같은 페이지의 미세한 변형을 수십 개씩 쏟아낸다. 그대로 짝짓기에
    넘기면 상위 후보가 전부 같은 페이지라서 반대쪽 페이지를 만날 기회가 없다.
    """
    kept: list[Detection] = []
    for c in sorted(cands, key=lambda d: -d.confidence):
        ca = _polygon_area(c.corners)
        if ca <= 0:
            continue
        dup = False
        for k in kept:
            ka = _polygon_area(k.corners)
            inter = _overlap_area(c.corners, k.corners)
            union = ca + ka - inter
            if union > 0 and inter / union > 0.6:
                dup = True
                break
        if not dup:
            kept.append(c)
        if len(kept) >= limit:
            break
    return kept


def _spread_candidates(cands: list[Detection], grad: np.ndarray,
                       shape: tuple[int, int]) -> list[Detection]:
    """나란히 붙은 페이지 둘을 하나의 펼침면 사각형으로 합친다.

    펼침면을 찍으면 한쪽 페이지만 딱 잡은 사각형이 양면 전체보다 높은 점수를 받는다.
    가운데 책등이 바깥 테두리만큼, 때로는 더 또렷하기 때문이다. 실제로 이 사진들에서
    '한쪽 면' 후보가 0.806점, '전체 책' 후보가 0.630점이었다. 점수만으로는 못 이긴다.

    그래서 점수 대신 구조를 본다. **비슷한 크기의 사각형 둘이 겹치지 않고 딱 붙어
    있으면** 그건 펼친 책이다. 두 후보의 볼록 껍질을 사각형으로 근사해 후보에 넣는다.
    """
    out: list[Detection] = []
    seen: set[tuple] = set()
    top = _distinct([c for c in cands if c.confidence >= 0.35], limit=10)
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i].corners, top[j].corners
            aa, ab = _polygon_area(a), _polygon_area(b)
            if min(aa, ab) <= 0 or max(aa, ab) / min(aa, ab) > PAIR_AREA_RATIO:
                continue
            if _overlap_area(a, b) > min(aa, ab) * PAIR_OVERLAP_MAX:
                continue
            hull = cv2.convexHull(np.vstack([a, b]).astype(np.float32))
            q = _quad_from_contour(hull)
            if q is None or not _is_convex(q):
                continue
            if _polygon_area(q) > (aa + ab) * PAIR_HULL_SLACK:
                continue  # 껍질이 헐렁하다 = 둘이 붙어 있지 않다
            key = tuple(np.round(q.ravel(), 1))
            if key in seen:
                continue
            seen.add(key)
            sc = score_quad(q, grad, shape)
            if sc > 0:
                out.append(Detection(corners=q, confidence=sc, method="spread"))
    return out


def _prefer_spread(best: Detection, merged: list[Detection],
                   shape: tuple[int, int]) -> Detection:
    """병합 후보가 최고점 후보를 감싸고 크기가 두 배쯤이면 그쪽을 택한다."""
    best_area = _polygon_area(best.corners)
    if best_area <= 0:
        return best
    pick = None
    for c in merged:
        ratio = _polygon_area(c.corners) / best_area
        if not (SPREAD_AREA_MIN <= ratio <= SPREAD_AREA_MAX):
            continue
        if c.confidence < best.confidence * SPREAD_TOLERANCE:
            continue
        if not _contains(c.corners, best.corners, shape):
            continue
        if pick is None or c.confidence > pick.confidence:
            pick = c
    return pick or best


def _prefer_enclosing(best: Detection, cands: list[Detection],
                      shape: tuple[int, int]) -> Detection:
    """더 큰 후보가 최고점 후보를 감싸고 있으면 그쪽을 택한다.

    펼침면을 찍으면 가운데 책등 그림자가 바깥 테두리보다 진해서, 한쪽 페이지만
    딱 잡은 사각형이 양면 전체보다 높은 점수를 받는 일이 흔하다. 이 보정이
    "책 펼쳐 찍었는데 한 면만 잘리는" 실패를 막는다.
    """
    improved = True
    while improved:
        improved = False
        best_area = _polygon_area(best.corners)
        for c in cands:
            if c is best or c.confidence < 0.5:
                continue
            if _polygon_area(c.corners) < best_area * 1.3:
                continue
            if c.confidence < best.confidence * CONTAINMENT_TOLERANCE:
                continue
            if _contains(c.corners, best.corners, shape):
                best, improved = c, True
                break
    return best


def detect_page(bgr: np.ndarray) -> Detection:
    """페이지 사각형을 찾는다. 입력 이미지 좌표계로 결과를 돌려준다."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grad = _gradient_map(gray)

    best: Detection | None = None
    cands: list[Detection] = []
    for name, fn in (("contour", _detect_contour), ("lab", _detect_lab), ("lines", _detect_lines)):
        try:
            quads = fn(bgr, gray)
        except Exception:
            quads = []
        for q in quads:
            sc = score_quad(q, grad, (h, w))
            if sc <= 0:
                continue
            det = Detection(corners=q, confidence=sc, method=name)
            cands.append(det)
            if best is None or sc > best.confidence:
                best = det
        if best is not None and best.confidence >= GOOD_ENOUGH:
            break  # 충분히 좋으면 나머지 방법은 건너뛴다

    if best is not None:
        # 펼침면 병합을 먼저 본다. 한쪽 면만 잡힌 상태에서 감싸기 승격을 돌리면
        # 그 페이지를 포함하는 엉뚱한 큰 사각형으로 먼저 끌려갈 수 있다.
        merged = _spread_candidates(cands, grad, (h, w))
        cands.extend(merged)
        best = _prefer_spread(best, merged, (h, w))
        best = _prefer_enclosing(best, cands, (h, w))

    if best is None or best.confidence < 0.25:
        # 전부 실패: 프레임 전체를 돌려주고 수동 보정하게 한다
        margin = 0.0
        full = np.array(
            [[margin, margin], [w - margin, margin], [w - margin, h - margin], [margin, h - margin]],
            dtype=np.float32,
        )
        keep = best.confidence if best else 0.0
        return Detection(corners=full, confidence=min(keep, 0.2), method="fallback")
    return best


def scale_corners(corners: np.ndarray | list, factor: float) -> np.ndarray:
    """작업본 좌표를 원본 좌표로 (또는 그 반대로) 환산."""
    return (np.asarray(corners, dtype=np.float32).reshape(4, 2) * float(factor)).astype(np.float32)
