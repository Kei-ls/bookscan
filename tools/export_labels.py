"""편집 창에서 확인한 경계를 학습 자료로 뽑아낸다.

이 프로젝트의 편집 UI는 이미 라벨링 도구다. 코너를 끌어다 놓거나 `이 경계가
정확함`을 누르면 그 결과가 `manifest.json`에 남는다. 이 스크립트는 그걸
(사진, 마스크) 쌍으로 바꿔 `tools/finetune.py`가 먹을 수 있게 한다.

경계는 두 가지 중 있는 쪽을 쓴다.
  * `outline` — 곡선 보정이 적용됐을 때의 굽은 테두리 폐곡선 (103점).
    실제 종이 모양이라 사각형보다 정확하다.
  * `corners` — 그것이 없으면 사각형 네 점.

**확인 표시가 있는 사진만 뽑는다.** 자동 검출 결과를 그대로 학습에 넣으면
모델이 지금의 오차를 정답이라고 배운다. 지금 고치려는 게 바로 그 오차다.

한쪽으로 치우친 자료에 대한 경고
--------------------------------
사람은 결과가 **틀렸을 때** 손을 댄다. 그래서 손댄 것만 모으면 실패 사례만
쌓인다. 그런 자료로 학습하면 모델은 "책등 쪽을 넉넉히 잡아라"만 배우고, 원래
잘 되던 사진까지 넉넉히 잡아 배경을 물고 온다. 잘 나온 사진에도 확인 표시를
눌러 달라고 UI에 적어 둔 이유다. 이 스크립트는 뽑을 때 그 비율을 세어 알려준다.

사용:
    python tools/export_labels.py [--out labels] [--long-edge 1024] [프로젝트...]
    python tools/export_labels.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookscan.config import OUTPUT_DIR, ROOT  # noqa: E402
from bookscan.imgio import imread  # noqa: E402

#: 뽑아낸 사진의 긴 변. 학습 입력은 이보다 훨씬 작지만, 자를 때마다 원본을
#: 다시 읽지 않아도 되도록 넉넉히 남긴다. 증강에도 여유가 필요하다.
DEFAULT_LONG_EDGE = 1024
#: 마스크가 사진에서 차지하는 면적이 이 범위를 벗어나면 라벨이 깨진 것으로 본다
MIN_AREA, MAX_AREA = 0.05, 0.995


def _boundary(src: dict) -> np.ndarray | None:
    """원본 좌표계의 경계 폐곡선. 곡선이 있으면 그걸, 없으면 사각형."""
    for key in ("outline", "corners"):
        pts = src.get(key)
        if pts and len(pts) >= 4:
            return np.asarray(pts, dtype=np.float32)
    return None


def export_project(project_dir: Path, out_dir: Path, long_edge: int) -> list[dict]:
    manifest = project_dir / "manifest.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows: list[dict] = []

    for sid, src in sorted(data.get("sources", {}).items()):
        if not src.get("verified"):
            continue
        poly = _boundary(src)
        if poly is None:
            print(f"  건너뜀 {sid}: 경계가 없음")
            continue
        image_path = project_dir / src["original"]
        if not image_path.exists():
            print(f"  건너뜀 {sid}: 원본 파일이 없음 ({image_path.name})")
            continue

        img = imread(image_path)
        h, w = img.shape[:2]
        scale = min(1.0, long_edge / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(poly * scale).astype(np.int32)], 255)

        area = float((mask > 0).mean())
        if not (MIN_AREA <= area <= MAX_AREA):
            print(f"  건너뜀 {sid}: 마스크 면적이 이상함 ({area:.3f})")
            continue

        stem = f"{project_dir.name}_{sid}"
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
        rows.append({
            "id": stem,
            "project": project_dir.name,
            "source": sid,
            "filename": src.get("filename", ""),
            # 손으로 잡은 것인지, 자동 결과를 확인만 한 것인지. 자료가 한쪽으로
            # 치우쳤는지 나중에 이 값으로 셀 수 있다.
            "corrected": src.get("method") == "manual",
            "boundary": "outline" if src.get("outline") else "corners",
            "spine": (src.get("settings") or {}).get("spine", "auto"),
            "area": round(area, 4),
        })
        print(f"  뽑음 {stem}  ({'손으로 잡음' if rows[-1]['corrected'] else '확인만'}, "
              f"{rows[-1]['boundary']}, 면적 {area:.3f})")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("projects", nargs="*", help="비우면 전부")
    ap.add_argument("--out", default="labels", help="내보낼 폴더 (기본 labels)")
    ap.add_argument("--long-edge", type=int, default=DEFAULT_LONG_EDGE)
    ap.add_argument("--list", action="store_true", help="프로젝트 목록만 보여준다")
    args = ap.parse_args()

    projects = [p for p in sorted(OUTPUT_DIR.iterdir()) if (p / "manifest.json").exists()] \
        if OUTPUT_DIR.exists() else []
    if args.list:
        for p in projects:
            data = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
            srcs = data.get("sources", {})
            done = sum(1 for s in srcs.values() if s.get("verified"))
            print(f"{p.name}: 사진 {len(srcs)}장, 확인됨 {done}장")
        return 0

    if args.projects:
        wanted = set(args.projects)
        projects = [p for p in projects if p.name in wanted]
        if not projects:
            print("그런 이름의 프로젝트가 없습니다.")
            return 1

    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for p in projects:
        print(f"{p.name}:")
        rows += export_project(p, out_dir, args.long_edge)

    (out_dir / "index.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    fixed = sum(1 for r in rows if r["corrected"])
    print(f"\n{len(rows)}장 → {out_dir}")
    if not rows:
        print("확인 표시된 사진이 없습니다. 편집 창에서 "
              "`이 경계가 정확함`을 눌러 주세요.")
        return 1
    print(f"  손으로 고친 것 {fixed}장, 자동 결과를 확인만 한 것 {len(rows) - fixed}장")
    if fixed > len(rows) * 0.7:
        print("  ⚠ 고친 사진에 치우쳐 있습니다. 잘 나온 사진도 확인 표시를 해 주세요 —"
              "\n    실패 사례만 배우면 되던 것까지 넉넉히 잡게 됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
