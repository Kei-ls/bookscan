"""굽은 테두리 검출과 곡선 보정이 실제로 효과가 있는지 잰다.

`make_synthetic.py`의 `bend`로 만든 사진은 **정답을 알고 있다.** 평평한 페이지를
가로세로비를 아는 상태에서 원통형으로 접었으니, 잘 폈다면 결과의 가로세로비가
원래대로 돌아와야 한다. 그래서 두 가지를 잰다.

  * **가로세로비 오차** — 책등 근처가 눌린 채 남으면 결과가 가로로 짧다.
    가로 보정(`warp._flat_axis`)이 제 몫을 하는지 여기서 드러난다.
  * **글줄 휘어짐** — 테두리를 편 만큼 안쪽 글줄도 같이 펴져야 한다.
    세로 보정(쿤스 패치)이 제 몫을 하는지 여기서 드러난다.

곡면 펴기(`dewarp.py`)는 꺼 놓고 잰다. 켜 두면 두 단계 중 어느 쪽 공인지
구별할 수 없기 때문이다. 실제 처리에서는 둘 다 켜져서, 테두리로 큰 왜곡을
잡고 글줄로 남은 잔재를 다듬는다.

사용: python tools/eval_curve.py [합성폴더]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bookscan.config import PageSettings  # noqa: E402
from bookscan.dewarp import MIN_LINES, find_text_lines  # noqa: E402
from bookscan.imgio import imread  # noqa: E402
from bookscan.pipeline import process  # noqa: E402


def line_bow(bgr: np.ndarray) -> float | None:
    """본문 줄 하나하나가 얼마나 휘어 있나의 평균 (높이 대비). 못 재면 None.

    `dewarp._fit_curves`를 그대로 쓰면 안 된다. 그건 줄들의 휘어짐을 **평균**
    내는데, 이 왜곡에서는 가운데 줄을 기준으로 위쪽 줄과 아래쪽 줄이 서로 반대로
    휘기 때문에 평균이 0에 가깝게 상쇄된다. 실제로 상쇄돼서 잰 값이 오히려
    좋아 보이는 착시를 한 번 겪었다.

    (덧붙이면, 이 상쇄가 곧 `dewarp.py`가 이 왜곡을 못 잡는 이유이기도 하다.
    글줄만 봐서는 안 되고 테두리를 봐야 한다는 근거가 여기 있다.)

    그래서 줄마다 2차 곡선과 1차 직선을 각각 맞추고 그 차이의 최댓값을 재서,
    부호를 없앤 뒤 평균 낸다.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    bows = []
    for pts in find_text_lines(gray):
        x, y = pts[:, 0], pts[:, 1]
        if x.max() - x.min() < w * 0.15:
            continue
        try:
            d = np.polyval(np.polyfit(x, y, 2), x) - np.polyval(np.polyfit(x, y, 1), x)
        except Exception:
            continue
        bows.append(float(np.abs(d).max()))
    if len(bows) < MIN_LINES:
        return None
    return float(np.mean(bows)) / max(h, 1)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/synthetic")
    truth_path = root / "curve_truth.json"
    if not truth_path.exists():
        print(f"{truth_path} 가 없습니다. tools/make_synthetic.py 를 먼저 돌리세요.")
        return 1
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    print(f"{'case':<20} {'곡선':<5} {'가로세로비':>11} {'비오차':>8} "
          f"{'글줄휘어짐':>11} {'ms':>6}")
    print("-" * 70)

    worse: list[str] = []
    for name, meta in truth.items():
        img = imread(root / f"{name}.jpg")
        want = float(meta["flat_aspect"])
        got: dict[bool, tuple[float, float | None]] = {}
        for use in (False, True):
            # 분할을 막아 통짜로 재고, 곡면 펴기는 꺼서 기하 단계만 본다
            st = PageSettings(curve=use, dewarp=False, spread="single", auto_rotate=False)
            t0 = time.perf_counter()
            res = process(img, st)
            ms = (time.perf_counter() - t0) * 1000
            page = res.pages[0].image
            h, w = page.shape[:2]
            aspect = w / max(h, 1)
            got[use] = (aspect, line_bow(page))
            bow = got[use][1]
            print(f"{name if not use else '':<20} {'켬' if use else '끔':<5} "
                  f"{aspect:>11.3f} {abs(aspect / want - 1):>8.3f} "
                  f"{('%11.4f' % bow) if bow is not None else '          -'} {ms:>6.0f}")

        off_err = abs(got[False][0] / want - 1)
        on_err = abs(got[True][0] / want - 1)
        off_bow, on_bow = got[False][1], got[True][1]
        tol = float(meta.get("aspect_tolerance", 0.01))
        if on_err > off_err + tol:
            worse.append(f"{name}: 비율 오차가 커짐 {off_err:.3f} → {on_err:.3f}")
        elif on_err > off_err:
            print(f"{'':<20} 알려진 한계: 비율 오차 {off_err:.3f} → {on_err:.3f} "
                  f"(허용 +{tol:.2f}) — 많이 비스듬한 사진은 접힘과 기울기를 구별 못 함")
        if off_bow is not None and on_bow is not None and on_bow > off_bow + 0.002:
            worse.append(f"{name}: 글줄이 더 휘었음 {off_bow:.4f} → {on_bow:.4f}")
        print(f"{'':<20} {'정답':<5} {want:>11.3f}")
        print("-" * 70)

    if worse:
        print("확인 필요:")
        for w in worse:
            print("  " + w)
        return 1
    print("곡선 보정이 모든 사진에서 비율과 글줄을 개선했거나 유지했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
