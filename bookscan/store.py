"""프로젝트 상태 관리.

출력 폴더 구조:
    output/<프로젝트>/
        originals/   원본 사본 (설정을 바꿔 다시 처리할 때 필요)
        pages/       최종 결과물. 읽는 순서대로 0001.jpg, 0002.jpg ...
        manifest.json

pages/의 파일 이름은 항상 현재 순서와 일치하도록 다시 매긴다. 사용자가
폴더를 열었을 때 그냥 순서대로 보이는 게 이 프로그램의 목적 중 하나다.
내부 id는 따로 유지해서 순서가 바뀌어도 UI 참조가 깨지지 않는다.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import IMAGE_EXTS, JPEG_QUALITY, OUTPUT_DIR, PageSettings, THUMB_LONG_EDGE
from .imgio import fit_long_edge, imread, imwrite
from .pipeline import ProcessResult, process

MANIFEST = "manifest.json"


@dataclass
class Source:
    """입력 사진 한 장."""

    id: str
    filename: str
    original: str  # 프로젝트 폴더 기준 상대경로
    settings: PageSettings = field(default_factory=PageSettings)
    corners: list[list[float]] | None = None  # 마지막 검출 결과
    #: 곡선 보정을 실제로 적용했을 때의 굽은 테두리 폐곡선 (원본 좌표계).
    #: 편집 화면에 겹쳐 그려서 무엇을 폈는지 눈으로 확인시킨다.
    outline: list[list[float]] | None = None
    confidence: float = 0.0
    method: str = ""
    is_spread: bool = False
    gutter: float | None = None
    notes: list[str] = field(default_factory=list)
    #: 이 경계가 정확하다고 사람이 확인해 준 사진. `tools/export_labels.py`가
    #: 학습 자료로 뽑아가는 기준이다. 코너를 손으로 잡으면 자동으로 켜진다 —
    #: 일부러 끌어다 놓은 자리라 그 자체가 확인이다.
    verified: bool = False
    added: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["settings"] = self.settings.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Source":
        d = dict(d)
        d["settings"] = PageSettings.from_dict(d.get("settings"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Page:
    """출력 페이지 한 장. PDF에 들어가는 단위."""

    id: str
    source: str
    half: str | None
    file: str  # 프로젝트 폴더 기준 상대경로
    width: int = 0
    height: int = 0
    version: int = 0  # 브라우저 캐시 무효화용
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Page":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class Project:
    """한 권(한 묶음)에 해당하는 작업 공간. 스레드 안전."""

    def __init__(self, name: str, root: Path | None = None) -> None:
        self.name = name
        self.dir = (root or OUTPUT_DIR) / name
        self.originals_dir = self.dir / "originals"
        self.pages_dir = self.dir / "pages"
        self.sources: dict[str, Source] = {}
        self.pages: dict[str, Page] = {}
        self.order: list[str] = []
        #: 새로 가져오는 사진에 적용할 기본 설정. 검출 모델 선택이 여기 산다.
        self.defaults = PageSettings()
        self._counter = 0
        self._lock = threading.RLock()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.originals_dir.mkdir(exist_ok=True)
        self.pages_dir.mkdir(exist_ok=True)
        self.load()

    # ---------------------------------------------------------------- 영속화
    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST

    def load(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.sources = {k: Source.from_dict(v) for k, v in data.get("sources", {}).items()}
        self.pages = {k: Page.from_dict(v) for k, v in data.get("pages", {}).items()}
        self.order = [p for p in data.get("order", []) if p in self.pages]
        self.defaults = PageSettings.from_dict(data.get("defaults"))
        self._counter = int(data.get("counter", 0))
        # 매니페스트에 없는데 남아있는 페이지도 순서 끝에 붙여준다
        for pid in self.pages:
            if pid not in self.order:
                self.order.append(pid)

    def save(self) -> None:
        with self._lock:
            data = {
                "version": 1,
                "name": self.name,
                "counter": self._counter,
                "defaults": self.defaults.to_dict(),
                "sources": {k: v.to_dict() for k, v in self.sources.items()},
                "pages": {k: v.to_dict() for k, v in self.pages.items()},
                "order": self.order,
            }
            tmp = self.manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.manifest_path)

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter:04d}"

    # ---------------------------------------------------------------- 가져오기
    def add_source(self, image: np.ndarray, filename: str) -> Source:
        """원본을 저장하고 Source를 만든다. 처리는 아직 안 한다."""
        with self._lock:
            sid = self._next_id("s")
            rel = f"originals/{sid}.jpg"
            imwrite(self.dir / rel, image, quality=95)
            # to_dict/from_dict를 거쳐 깊은 복사를 만든다. dataclasses.replace는
            # 리스트를 공유해서, 한 장의 코너를 고치면 다른 장까지 따라 바뀐다.
            src = Source(id=sid, filename=filename, original=rel,
                         settings=PageSettings.from_dict(self.defaults.to_dict()))
            self.sources[sid] = src
            return src

    def import_folder(self, folder: Path) -> list[Source]:
        """폴더의 이미지들을 이름순으로 가져온다."""
        files = sorted(
            (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: p.name.lower(),
        )
        out = []
        for f in files:
            try:
                out.append(self.add_source(imread(f), f.name))
            except Exception:
                continue  # 못 읽는 파일 하나 때문에 전체가 멈추면 안 된다
        return out

    # ---------------------------------------------------------------- 처리
    def process_source(self, sid: str) -> list[Page]:
        """원본을 처리해 페이지를 만든다. 이미 있으면 같은 자리에 교체한다."""
        src = self.sources[sid]
        original = imread(self.dir / src.original)
        result: ProcessResult = process(original, src.settings)

        with self._lock:
            old = [p for p in self.order if self.pages[p].source == sid]
            insert_at = self.order.index(old[0]) if old else len(self.order)
            for pid in old:
                self._remove_page_file(pid)
                self.order.remove(pid)
                self.pages.pop(pid, None)

            made: list[Page] = []
            for part in result.pages:
                pid = self._next_id("p")
                h, w = part.image.shape[:2]
                page = Page(id=pid, source=sid, half=part.half, file=f"pages/{pid}.jpg",
                            width=w, height=h, version=1, notes=list(part.notes))
                ext = ".png" if src.settings.mode == "bw" else ".jpg"
                page.file = f"pages/{pid}{ext}"
                imwrite(self.dir / page.file, part.image, quality=JPEG_QUALITY)
                self._write_thumb(page, part.image)
                self.pages[pid] = page
                made.append(page)

            for i, page in enumerate(made):
                self.order.insert(insert_at + i, page.id)

            src.corners = result.corners
            src.outline = result.outline
            if result.method == "manual":
                src.verified = True
            src.confidence = result.confidence
            src.method = result.method
            src.is_spread = result.is_spread
            src.gutter = result.gutter
            src.notes = result.notes
            self.renumber()
            self.save()
            return made

    def _write_thumb(self, page: Page, image: np.ndarray) -> None:
        thumb, _ = fit_long_edge(image, THUMB_LONG_EDGE)
        imwrite(self.dir / self._thumb_rel(page), thumb, quality=80)

    def _thumb_rel(self, page: Page) -> str:
        return f"pages/.thumbs/{page.id}.jpg"

    def _remove_page_file(self, pid: str) -> None:
        page = self.pages.get(pid)
        if not page:
            return
        for rel in (page.file, self._thumb_rel(page)):
            try:
                (self.dir / rel).unlink(missing_ok=True)
            except OSError:
                pass

    # ---------------------------------------------------------------- 순서/삭제
    def renumber(self) -> None:
        """pages/ 파일 이름을 현재 순서에 맞춰 다시 매긴다.

        같은 폴더 안에서 이름을 바꾸므로 충돌을 피하려고 2단계로 나눠 옮긴다.
        """
        with self._lock:
            planned: list[tuple[Page, str]] = []
            for i, pid in enumerate(self.order, start=1):
                page = self.pages[pid]
                ext = Path(page.file).suffix or ".jpg"
                planned.append((page, f"pages/{i:04d}{ext}"))
            if all(p.file == target for p, target in planned):
                return

            for page, target in planned:  # 1단계: 임시 이름으로
                if page.file == target:
                    continue
                src_path = self.dir / page.file
                if src_path.exists():
                    tmp = self.dir / f"pages/.tmp_{page.id}{Path(target).suffix}"
                    src_path.replace(tmp)
                    page.file = str(tmp.relative_to(self.dir)).replace("\\", "/")

            for page, target in planned:  # 2단계: 최종 이름으로
                src_path = self.dir / page.file
                if src_path.exists() and page.file != target:
                    src_path.replace(self.dir / target)
                page.file = target
                page.version += 1

    def reorder(self, new_order: list[str]) -> None:
        with self._lock:
            keep = [p for p in new_order if p in self.pages]
            keep += [p for p in self.order if p not in keep]
            self.order = keep
            self.renumber()
            self.save()

    def delete_page(self, pid: str) -> None:
        with self._lock:
            if pid not in self.pages:
                return
            self._remove_page_file(pid)
            self.pages.pop(pid, None)
            if pid in self.order:
                self.order.remove(pid)
            self.renumber()
            self.save()

    def delete_source(self, sid: str) -> None:
        with self._lock:
            for pid in [p for p in self.order if self.pages[p].source == sid]:
                self._remove_page_file(pid)
                self.pages.pop(pid, None)
                self.order.remove(pid)
            src = self.sources.pop(sid, None)
            if src:
                try:
                    (self.dir / src.original).unlink(missing_ok=True)
                except OSError:
                    pass
            self.renumber()
            self.save()

    # ---------------------------------------------------------------- 조회
    def ordered_pages(self) -> list[Page]:
        return [self.pages[p] for p in self.order if p in self.pages]

    def page_path(self, pid: str) -> Path:
        return self.dir / self.pages[pid].file

    def thumb_path(self, pid: str) -> Path:
        page = self.pages[pid]
        p = self.dir / self._thumb_rel(page)
        if not p.exists():  # 썸네일이 없으면 즉석에서 만든다
            try:
                self._write_thumb(page, imread(self.dir / page.file))
            except Exception:
                return self.dir / page.file
        return p

    def original_path(self, sid: str) -> Path:
        return self.dir / self.sources[sid].original


def list_projects(root: Path | None = None) -> list[str]:
    root = root or OUTPUT_DIR
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def delete_project(name: str, root: Path | None = None) -> None:
    target = (root or OUTPUT_DIR) / name
    if target.exists():
        shutil.rmtree(target)
