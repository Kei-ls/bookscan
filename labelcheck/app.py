"""라벨 확인 프로그램 — 학습 자료를 눈으로 훑고 나쁜 것을 걸러낸다.

**bookscan과는 별개의 프로그램이다.** 이 패키지는 `bookscan`을 하나도 import하지
않는다. 하는 일이 다르기 때문이다 — bookscan은 사진을 고치고, 여기는 이미 뽑아낸
자료를 판정만 한다. 섞으면 둘 다 무거워진다.

왜 따로 훑어야 하나
-------------------
`tools/export_labels.py`가 뽑아낸 경계는 사람이 편집 창에서 확인한 것이지만,
거기서 확인한 건 **사각형 네 점**이고 실제로 라벨이 되는 건 **곡선 테두리**다.
그 곡선은 사람이 잡은 코너를 씨앗으로 다시 계산되므로, 코너를 정확히 놓아도
곡선이 빗나갈 수 있다 (실제로 책등 쪽에서 글자를 자른 적이 있다).

그러니 학습에 넣기 전에 **라벨 그 자체**를 봐야 한다. 잘못된 라벨 한 장은 없는
것보다 나쁘다 — 모델이 그걸 정답이라고 배운다.

쓰는 법
-------
    python -m labelcheck labels

좌우 화살표로 넘기고, K(유지) / X(버림) / U(보류)로 판정한다. 판정은 자료
폴더의 `review.json`에 쌓인다. 다 훑고 `유지한 것만 모으기`를 누르면 `kept/`
폴더가 생기고, 그걸 압축해 Colab에 올리면 된다.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
REVIEW_FILE = "review.json"
INDEX_FILE = "index.json"
KEPT_DIR = "kept"

VERDICTS = ("keep", "drop", "undecided")

app = FastAPI(title="labelcheck")
_lock = threading.Lock()


@dataclass
class Store:
    """자료 폴더 하나. 파일을 진실로 삼고 메모리에는 판정만 들고 있는다."""

    root: Path
    meta: dict[str, dict] = None       # id -> export_labels 가 남긴 정보
    verdicts: dict[str, str] = None
    order: list[str] = None

    def load(self) -> None:
        pairs: dict[str, dict] = {}
        for mask in sorted(self.root.glob("*_mask.png")):
            stem = mask.name[: -len("_mask.png")]
            for ext in (".jpg", ".jpeg", ".png"):
                img = mask.with_name(stem + ext)
                if img.exists():
                    pairs[stem] = {"id": stem, "image": img.name, "mask": mask.name}
                    break

        # export_labels 가 남긴 정보가 있으면 얹는다. 없어도 (tests/synthetic 처럼)
        # 사진과 마스크만으로 돌아가야 한다.
        idx = self.root / INDEX_FILE
        if idx.exists():
            try:
                for row in json.loads(idx.read_text(encoding="utf-8")):
                    if row.get("id") in pairs:
                        pairs[row["id"]].update(row)
            except (json.JSONDecodeError, TypeError):
                pass

        self.meta = pairs
        self.order = sorted(pairs)
        self.verdicts = {}
        rv = self.root / REVIEW_FILE
        if rv.exists():
            try:
                saved = json.loads(rv.read_text(encoding="utf-8"))
                self.verdicts = {k: v for k, v in saved.items()
                                 if k in pairs and v in VERDICTS}
            except (json.JSONDecodeError, AttributeError):
                pass

    def save(self) -> None:
        (self.root / REVIEW_FILE).write_text(
            json.dumps(self.verdicts, ensure_ascii=False, indent=1), encoding="utf-8")

    def verdict(self, item_id: str) -> str:
        return self.verdicts.get(item_id, "undecided")

    def rows(self) -> list[dict]:
        return [dict(self.meta[i], verdict=self.verdict(i)) for i in self.order]

    def stats(self) -> dict:
        rows = self.rows()
        keep = [r for r in rows if r["verdict"] == "keep"]
        counted = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
        corrected = sum(1 for r in keep if r.get("corrected"))
        spine: dict[str, int] = {}
        for r in keep:
            spine[r.get("spine") or "auto"] = spine.get(r.get("spine") or "auto", 0) + 1

        warnings: list[str] = []
        if keep and corrected > len(keep) * 0.7:
            warnings.append(
                f"유지한 {len(keep)}장 중 {corrected}장이 손으로 고친 사진입니다. "
                "실패 사례에 치우치면 모델이 '책등 쪽을 넉넉히'만 배워서 "
                "원래 잘 되던 사진까지 배경을 물고 옵니다.")
        if 0 < len(keep) < 20:
            warnings.append(
                f"유지한 자료가 {len(keep)}장뿐입니다. 이 정도면 모델이 이 몇 장을 "
                "외울 뿐이라 다른 사진에서 오히려 나빠집니다. 200~500장을 목표로 하세요.")
        if keep and len(spine) == 1 and "auto" not in spine:
            warnings.append(
                f"책등이 전부 {list(spine)[0]} 쪽입니다. 반대쪽 페이지도 찍어 주세요.")
        return {
            "total": len(rows), "counts": counted,
            "corrected": corrected, "spine": spine, "warnings": warnings,
        }


store = Store(Path("."))


# ------------------------------------------------------------------ 정적 파일
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((HERE / "index.html").read_text(encoding="utf-8"))


def _send(name: str) -> FileResponse:
    path = store.root / name
    if not path.is_file():
        raise HTTPException(404, "파일이 없습니다.")
    # 자료 폴더 밖으로 나가는 경로는 거절한다
    try:
        path.resolve().relative_to(store.root.resolve())
    except ValueError:
        raise HTTPException(403, "폴더 밖입니다.") from None
    return FileResponse(path)


@app.get("/api/items")
async def items() -> dict:
    with _lock:
        return {"root": str(store.root), "items": store.rows(), "stats": store.stats()}


@app.get("/api/image/{item_id}")
async def image(item_id: str):
    if item_id not in store.meta:
        raise HTTPException(404, "그런 항목이 없습니다.")
    return _send(store.meta[item_id]["image"])


@app.get("/api/mask/{item_id}")
async def mask(item_id: str):
    if item_id not in store.meta:
        raise HTTPException(404, "그런 항목이 없습니다.")
    return _send(store.meta[item_id]["mask"])


class VerdictBody(BaseModel):
    id: str
    verdict: str


@app.post("/api/verdict")
async def set_verdict(body: VerdictBody) -> dict:
    if body.id not in store.meta:
        raise HTTPException(404, "그런 항목이 없습니다.")
    if body.verdict not in VERDICTS:
        raise HTTPException(400, f"알 수 없는 판정: {body.verdict}")
    with _lock:
        if body.verdict == "undecided":
            store.verdicts.pop(body.id, None)
        else:
            store.verdicts[body.id] = body.verdict
        store.save()
        return {"id": body.id, "verdict": store.verdict(body.id), "stats": store.stats()}


class BulkBody(BaseModel):
    ids: list[str]
    verdict: str


@app.post("/api/verdict/bulk")
async def set_bulk(body: BulkBody) -> dict:
    if body.verdict not in VERDICTS:
        raise HTTPException(400, f"알 수 없는 판정: {body.verdict}")
    with _lock:
        for i in body.ids:
            if i not in store.meta:
                continue
            if body.verdict == "undecided":
                store.verdicts.pop(i, None)
            else:
                store.verdicts[i] = body.verdict
        store.save()
        return {"items": store.rows(), "stats": store.stats()}


@app.post("/api/export")
async def export_kept() -> dict:
    """유지한 것만 `kept/`로 모은다. 그 폴더를 압축해 Colab에 올리면 된다."""
    import shutil

    with _lock:
        keep = [i for i in store.order if store.verdict(i) == "keep"]
        if not keep:
            raise HTTPException(400, "유지하기로 한 자료가 없습니다.")
        out = store.root / KEPT_DIR
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        rows: list[dict[str, Any]] = []
        for i in keep:
            m = store.meta[i]
            shutil.copy2(store.root / m["image"], out / m["image"])
            shutil.copy2(store.root / m["mask"], out / m["mask"])
            rows.append({k: v for k, v in m.items() if k not in ("image", "mask")})
        (out / INDEX_FILE).write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"count": len(keep), "dir": str(out)}
