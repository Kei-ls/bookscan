"""dewarp 검증.

두 가지를 본다.
  1) 휘어진 페이지에서 본문 줄의 직선성이 실제로 좋아지는가
  2) **평평한 페이지를 건드리지 않는가** — 이쪽이 더 중요하다.
     잘못 편 페이지는 안 편 페이지보다 나쁘기 때문.

사용: python tools/eval_dewarp.py [합성폴더]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bookscan.config import WORK_LONG_EDGE  # noqa: E402
from bookscan.detect import detect_page  # noqa: E402
from bookscan.dewarp import dewarp, find_text_lines  # noqa: E402
from bookscan.imgio import fit_long_edge, imread  # noqa: E402
from bookscan.spread import detect_gutter, split_spread  # noqa: E402
from bookscan.warp import warp_page  # noqa: E402


def straightness(bgr: np.ndarray) -> float:
    """본문 줄이 직선에서 벗어난 정도(페이지 높이 대비). 작을수록 곧다."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 1000 / max(gray.shape[1], 1)
    if s < 1.0:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h = gray.shape[0]
    devs = []
    for pts in find_text_lines(gray):
        x, y = pts[:, 0], pts[:, 1]
        if x.max() - x.min() < gray.shape[1] * 0.3:
            continue
        fit = np.polyval(np.polyfit(x, y, 1), x)  # 직선에 맞췄을 때의 잔차
        devs.append(float(np.sqrt(np.mean((y - fit) ** 2))))
    return float(np.mean(devs)) / h if devs else float("nan")


def page_from(path: Path) -> list[tuple[str, np.ndarray]]:
    """사진 → 워프된 페이지들 (펼침면이면 2장)."""
    img = imread(path)
    work, scale = fit_long_edge(img, WORK_LONG_EDGE)
    det = detect_page(work)
    warped = warp_page(img, det.corners / scale, max_long_edge=2000)
    sp = detect_gutter(warped)
    if sp.is_spread and sp.gutter is not None:
        left, right = split_spread(warped, sp.gutter)
        return [("L", left), ("R", right)]
    return [("", warped)]


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/synthetic")
    cases = ["flat_frontal", "flat_dark_bg", "curled", "spread_curled"]

    print(f"{'case':<20} {'적용':<5} {'줄':>3} {'전':>8} {'후':>8} {'개선':>8}   사유")
    print("-" * 96)
    ok = True
    for name in cases:
        for tag, page in page_from(root / f"{name}.jpg"):
            before = straightness(page)
            res = dewarp(page)
            after = straightness(res.image) if res.applied else before
            imp = (before - after) / before * 100 if before and not np.isnan(before) else 0.0
            label = f"{name}{('/' + tag) if tag else ''}"
            print(f"{label:<20} {'예' if res.applied else '아니오':<5} {res.n_lines:>3} "
                  f"{before:>8.4f} {after:>8.4f} {imp:>7.1f}%   {res.reason}")

            if name.startswith("flat") and res.applied:
                print("      ^^ 평평한 페이지를 건드렸다 — 안전장치 실패")
                ok = False
            if "curled" in name and res.applied and imp < 15:
                print("      ^^ 적용했는데 개선이 미미하다")
                ok = False
    print("-" * 96)
    print("판정:", "통과" if ok else "실패")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
