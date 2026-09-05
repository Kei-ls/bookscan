"""합성 테스트 이미지 생성기.

실제 사진 없이 파이프라인을 검증하기 위해, 깨끗한 페이지에 알려진 왜곡을
인위적으로 입혀 '가짜 사진'을 만든다. 정답 코너 좌표를 알고 있으므로 검출
정확도를 픽셀 단위로 측정할 수 있고, dewarp가 평평한 페이지를 건드리지
않는지(오작동 방지)도 확인할 수 있다.

사용: python tools/make_synthetic.py [출력폴더]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bookscan.imgio import imwrite  # noqa: E402

RNG = np.random.default_rng(20260811)

LOREM = [
    "The quick brown fox jumps over the lazy dog",
    "Sphinx of black quartz judge my vow and rest",
    "Pack my box with five dozen liquor jugs today",
    "How vexingly quick daft zebras jump around us",
    "Bright vixens jump dozy fowl quack merrily on",
    "Five wine experts jokingly quizzed sample bags",
]


def make_page(w: int = 900, h: int = 1250, title: str = "CHAPTER ONE") -> np.ndarray:
    """글자가 들어간 깨끗한 페이지 (정답 이미지)."""
    page = np.full((h, w, 3), 252, np.uint8)
    cv2.putText(page, title, (int(w * 0.12), int(h * 0.12)),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    y = int(h * 0.20)
    line_h = int(h * 0.038)
    i = 0
    while y < h * 0.93:
        text = LOREM[i % len(LOREM)]
        indent = int(w * 0.10) + (int(w * 0.04) if i % 7 == 0 else 0)
        cv2.putText(page, text, (indent, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (30, 30, 30), 1, cv2.LINE_AA)
        y += line_h
        i += 1
    cv2.putText(page, "12", (w // 2 - 12, h - int(h * 0.03)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    return page


def curl(page: np.ndarray, strength: float = 0.055, side: str = "left") -> np.ndarray:
    """책등 쪽이 휘어 들어간 것처럼 원통형으로 구부린다.

    행을 아래로 밀고(수직 변위) 책등 근처를 가로로 압축한다 — 실제로 펼친 책을
    찍었을 때 나타나는 두 가지 왜곡이다.
    """
    h, w = page.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    # 책등 쪽에서 최대가 되는 정규화 프로파일
    t = xs / (w - 1)
    prof = (1.0 - t) ** 2 if side == "left" else t**2

    map_x = np.tile(xs, (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    map_y = map_y - (prof[None, :] * strength * h)
    # 책등 근처 수평 압축: 그쪽 픽셀을 더 촘촘히 샘플링
    squeeze = 1.0 - prof * 0.18
    center = 0.0 if side == "left" else float(w - 1)
    map_x = center + (map_x - center) / squeeze[None, :]

    return cv2.remap(page, map_x, map_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(252, 252, 252))


def bend(page: np.ndarray, strength: float = 0.30,
         side: str = "left") -> tuple[np.ndarray, np.ndarray]:
    """책을 원통형으로 세운 것처럼 만든다. **테두리까지 같이 휜다.**

    위의 `curl`은 페이지 **안쪽 글자만** 밀기 때문에 실루엣이 반듯한 사각형으로
    남는다. 실제 사진은 그렇지 않다. 책등 쪽 종이는 카메라에서 멀어지면서 그
    부분이 통째로 작게 찍히고, 그래서 위/아래 테두리가 안쪽으로 오므라든다.
    곡선 윤곽선 검출을 시험하려면 그 오므라듦이 반드시 있어야 한다.

    모델은 분리 가능(separable)하게 잡았다. 열 u마다 확대율 m(u) 하나가 있고,
    가로 위치와 세로 크기가 둘 다 그 m(u)로 정해진다. 원근에서 확대율이 방향을
    가리지 않는다는 성질을 그대로 옮긴 것이고, 덕분에 역함수가 닫힌 형태로
    나와서 remap 좌표맵을 정확히 만들 수 있다.

    (휘어진 사진, 종이 마스크) 반환.
    """
    h, w = page.shape[:2]
    u = np.linspace(0.0, 1.0, w)
    if side == "left":
        f = (1.0 - u) ** 2
    elif side == "right":
        f = u**2
    else:  # "center" — 펼침면은 가운데(책등)가 가장 멀다
        g = np.exp(-((u - 0.5) ** 2) / (2 * 0.22**2))
        f = (g - g[0]) / (1.0 - g[0])

    m = 1.0 / (1.0 + strength * f)  # 국소 확대율 (멀수록 작다)
    x_img = np.concatenate([[0.0], np.cumsum(0.5 * (m[:-1] + m[1:]))])
    out_w = int(np.ceil(x_img[-1])) + 1
    out_h = int(np.ceil((h - 1) * m.max())) + 1
    yc = (out_h - 1) * 0.5

    cols = np.arange(out_w, dtype=np.float64)
    src_x = np.interp(cols, x_img, np.arange(w, dtype=np.float64))
    m_at = np.interp(src_x, np.arange(w, dtype=np.float64), m)

    rows = np.arange(out_h, dtype=np.float64)[:, None]
    map_y = (rows - yc) / m_at[None, :] + (h - 1) * 0.5
    map_x = np.tile(src_x[None, :], (out_h, 1))

    inside = ((map_y >= 0) & (map_y <= h - 1)).astype(np.uint8) * 255
    warped = cv2.remap(page, map_x.astype(np.float32), map_y.astype(np.float32),
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped, inside


def spread(left: np.ndarray, right: np.ndarray, gutter_px: int = 26) -> np.ndarray:
    """두 페이지를 붙이고 가운데에 책등 그림자를 넣는다."""
    h = min(left.shape[0], right.shape[0])
    left = cv2.resize(left, (left.shape[1], h))
    right = cv2.resize(right, (right.shape[1], h))
    out = np.hstack([left, np.full((h, gutter_px, 3), 235, np.uint8), right])
    gx = left.shape[1] + gutter_px // 2
    xs = np.arange(out.shape[1], dtype=np.float32)
    shadow = 1.0 - 0.55 * np.exp(-((xs - gx) ** 2) / (2 * (gutter_px * 1.6) ** 2))
    return np.clip(out.astype(np.float32) * shadow[None, :, None], 0, 255).astype(np.uint8)


def photograph(
    page: np.ndarray,
    canvas: tuple[int, int] = (1600, 1200),
    bg_color: tuple[int, int, int] = (120, 105, 95),
    tilt: float = 0.18,
    margin: float = 0.12,
    lighting: float = 0.45,
    page_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """페이지를 배경 위에 원근 왜곡 + 조명 얼룩을 넣어 배치.

    (사진, 정답 코너, 정답 실루엣) 반환. 실루엣은 **사진 좌표계의 진짜 테두리**라,
    검출한 곡선을 픽셀 단위로 채점할 수 있게 해 준다. 이게 없던 동안 곡선이 통째로
    25픽셀 밀려 있는 걸 어떤 테스트도 잡지 못했다.
    """
    CW, CH = canvas
    ph, pw = page.shape[:2]

    scene = np.full((CH, CW, 3), bg_color, np.uint8)
    noise = RNG.normal(0, 7, (CH, CW, 3))
    scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 목표 사각형: 페이지의 원래 가로세로비를 유지한 채 캔버스 안에 최대 크기로
    # 넣는다. 비율을 무시하면 세로 페이지가 가로로 늘어나 실제 사진과 달라진다.
    mx, my = CW * margin, CH * margin
    box_w, box_h = CW - 2 * mx, CH - 2 * my
    pa = pw / ph
    fit_w = min(box_w, box_h * pa)
    fit_h = fit_w / pa
    cx, cy = CW / 2, CH / 2
    x0, x1 = cx - fit_w / 2, cx + fit_w / 2
    y0, y1 = cy - fit_h / 2, cy + fit_h / 2
    base = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    jitter = RNG.uniform(-1, 1, (4, 2)) * np.array([CW * tilt * 0.5, CH * tilt * 0.5])
    dst = (base + jitter).astype(np.float32)

    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, M, (CW, CH), flags=cv2.INTER_CUBIC)
    # 종이 모양이 사각형이 아닐 수 있다 (bend가 만든 오므라든 실루엣)
    src_mask = page_mask if page_mask is not None else np.full((ph, pw), 255, np.uint8)
    mask = cv2.warpPerspective(src_mask, M, (CW, CH), flags=cv2.INTER_NEAREST)

    m3 = (mask[:, :, None] > 127)
    scene = np.where(m3, warped, scene)

    # 조명 얼룩: 한쪽이 밝고 반대쪽이 어두운 부드러운 기울기 + 비네팅
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    ang = RNG.uniform(0, 2 * np.pi)
    ramp = (np.cos(ang) * xx / CW + np.sin(ang) * yy / CH)
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)  # numpy 2.x는 ndarray.ptp() 없음
    field = 1.0 - lighting * ramp
    r = np.sqrt(((xx - CW / 2) / (CW / 2)) ** 2 + ((yy - CH / 2) / (CH / 2)) ** 2)
    field *= 1.0 - 0.28 * np.clip(r, 0, 1.4) ** 2
    # 살짝 누런 조명
    tint = np.array([0.94, 0.99, 1.04], np.float32)
    lit = scene.astype(np.float32) * field[:, :, None] * tint[None, None, :]

    out = np.clip(lit + RNG.normal(0, 3.0, lit.shape), 0, 255).astype(np.uint8)
    return out, dst, mask


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    truth: dict[str, dict] = {}

    page = make_page()

    # 세로 페이지는 세로 캔버스에 담아야 실제 촬영과 비슷해진다
    portrait = (1200, 1600)
    cases = [
        ("flat_dark_bg", dict(page=page, canvas=portrait, bg_color=(110, 95, 85), tilt=0.16)),
        ("flat_frontal", dict(page=page, canvas=portrait, bg_color=(90, 80, 75), tilt=0.03,
                              lighting=0.30)),
        ("white_on_white", dict(page=page, canvas=portrait, bg_color=(232, 233, 236), tilt=0.14,
                                lighting=0.35)),
        ("strong_light", dict(page=page, canvas=portrait, bg_color=(70, 65, 60), tilt=0.20,
                              lighting=0.70)),
        ("curled", dict(page=curl(page, 0.06, "left"), canvas=portrait, bg_color=(100, 90, 82),
                        tilt=0.10)),
    ]
    for name, kw in cases:
        img, corners, _gt = photograph(**kw)
        imwrite(out_dir / f"{name}.jpg", img, quality=94)
        truth[name] = {"corners": corners.tolist(), "kind": "single"}

    sp = spread(curl(make_page(title="CHAPTER TWO"), 0.055, "left"),
                curl(make_page(title="CHAPTER TWO"), 0.055, "right"))
    img, corners, _gt = photograph(sp, canvas=(1800, 1150), bg_color=(95, 88, 80),
                              tilt=0.10, margin=0.08, lighting=0.40)
    imwrite(out_dir / "spread_curled.jpg", img, quality=94)
    truth["spread_curled"] = {"corners": corners.tolist(), "kind": "spread"}

    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    # --- 테두리까지 휘는 사진 (곡선 검출용) ---------------------------------
    # truth.json에 섞지 않고 따로 둔다. 여기 있는 사진들은 애초에 사각형으로
    # 표현할 수 없어서, 코너 오차라는 잣대 자체가 성립하지 않기 때문이다.
    bent: dict[str, dict] = {}
    flat_single = make_page(title="CHAPTER THREE")
    flat_spread = spread(make_page(title="CHAPTER FOUR"), make_page(title="CHAPTER FOUR"))
    bent_cases = [
        ("bent_single", flat_single, dict(strength=0.30, side="left"),
         dict(canvas=portrait, bg_color=(105, 92, 84), tilt=0.06, lighting=0.35)),
        ("bent_single_strong", flat_single, dict(strength=0.55, side="left"),
         dict(canvas=portrait, bg_color=(80, 74, 70), tilt=0.10, lighting=0.45)),
        # 비스듬히 찍은 데다 휘기까지 한 경우. 곡선 워프는 쿤스 패치라 원근을
        # 호모그래피만큼 정확히 풀지 못하는데, 그 한계가 어디쯤인지 재는 표본이다.
        ("bent_single_tilted", flat_single, dict(strength=0.30, side="left"),
         dict(canvas=portrait, bg_color=(100, 90, 84), tilt=0.24, lighting=0.40),
         0.06),
        # 펼침면은 책등에 **국소 주름**이 있는데 3차식은 그걸 담지 못한다. 차수를
        # 올리면 여기는 0.11%까지 좋아지지만 단면이 0.17% → 2.27%로 망가진다
        # (양 끝을 빼고 맞춘 고차식이 모서리에서 날뛴다). 전역 다항식이 아니라
        # 구간별 표현이 필요하다는 신호라, 지금 값을 알려진 한계로 고정해 둔다.
        ("bent_spread", flat_spread, dict(strength=0.34, side="center"),
         dict(canvas=(1800, 1150), bg_color=(95, 88, 80), tilt=0.06, margin=0.08,
              lighting=0.40), 0.01, 0.012),
    ]
    for case in bent_cases:
        name, flat, bkw, pkw = case[:4]
        warped, pmask = bend(flat, **bkw)
        img, corners, gt = photograph(warped, page_mask=pmask, **pkw)
        imwrite(out_dir / f"{name}.jpg", img, quality=94)
        # 정답 실루엣은 PNG로 (JPEG로 저장하면 테두리가 번져서 정답이 아니게 된다)
        cv2.imwrite(str(out_dir / f"{name}_mask.png"), gt)
        bent[name] = {
            "corners": corners.tolist(),
            "flat_aspect": flat.shape[1] / flat.shape[0],
            "kind": "spread" if bkw["side"] == "center" else "single",
            # 곡선 보정이 이만큼까지 나빠지는 건 알려진 한계로 봐준다
            # (tools/eval_curve.py, bookscan/warp.curved_maps 주석 참고)
            "aspect_tolerance": case[4] if len(case) > 4 else 0.01,
            #: 검출한 테두리가 정답 실루엣에서 벗어나도 되는 한계 (짧은 변 대비)
            "boundary_tolerance": case[5] if len(case) > 5 else 0.005,
            "mask": f"{name}_mask.png",
            **bkw,
        }
    (out_dir / "curve_truth.json").write_text(json.dumps(bent, indent=2), encoding="utf-8")

    print(f"{len(truth)}개 + 휜 테두리 {len(bent)}개 생성 → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
