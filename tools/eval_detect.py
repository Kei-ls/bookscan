"""합성 이미지로 검출 정확도를 측정한다.

정답 코너를 알고 있으므로 픽셀 오차를 직접 잴 수 있다. 검출 로직을 고칠 때마다
이걸 돌려서 회귀가 없는지 확인한다.

사용: python tools/eval_detect.py [합성폴더]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bookscan.config import WORK_LONG_EDGE  # noqa: E402
from bookscan.detect import detect_page, order_corners  # noqa: E402
from bookscan.imgio import fit_long_edge, imread  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/synthetic")
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))

    print(f"{'case':<18} {'method':<10} {'conf':>6} {'평균오차':>9} {'최대오차':>9} {'ms':>6}")
    print("-" * 66)
    errors = []
    for name, meta in truth.items():
        path = root / f"{name}.jpg"
        img = imread(path)
        work, scale = fit_long_edge(img, WORK_LONG_EDGE)

        t0 = time.perf_counter()
        det = detect_page(work)
        ms = (time.perf_counter() - t0) * 1000

        # 검출 결과를 원본 좌표계로 되돌려 정답과 비교
        got = order_corners(det.corners / scale)
        want = order_corners(np.array(meta["corners"], dtype=np.float32))
        d = np.linalg.norm(got - want, axis=1)
        errors.append(d.mean())
        flag = "" if d.mean() < 25 else "   <-- 확인"
        print(f"{name:<18} {det.method:<10} {det.confidence:>6.2f} "
              f"{d.mean():>9.1f} {d.max():>9.1f} {ms:>6.0f}{flag}")

    print("-" * 66)
    print(f"전체 평균 코너 오차: {np.mean(errors):.1f} px")
    return 0 if np.mean(errors) < 25 else 1


if __name__ == "__main__":
    raise SystemExit(main())
