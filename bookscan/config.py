"""설정값과 페이지별 처리 옵션."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
WEB_DIR = ROOT / "web"

Mode = Literal["color", "gray", "bw"]
SpreadMode = Literal["auto", "single", "split"]
#: 결과 페이지의 가로세로비. "auto"는 사진에서 잰 값을 그대로 쓴다.
PaperMode = Literal["auto", "a4", "letter", "b5"]

#: 검출/미리보기 계산에 쓰는 작업본의 긴 변 길이. 원본은 최종 워프에만 쓴다.
WORK_LONG_EDGE = 1500
#: 썸네일 긴 변
THUMB_LONG_EDGE = 400
#: 편집 화면에 띄우는 원본 미리보기 긴 변
EDIT_LONG_EDGE = 1200
#: 결과물 긴 변 상한 (원본이 더 커도 이 이상으로는 안 키움)
MAX_OUTPUT_LONG_EDGE = 3500

JPEG_QUALITY = 92


@dataclass
class PageSettings:
    """사진 한 장을 어떻게 처리할지. UI에서 페이지별로 바꿀 수 있다."""

    #: 원본 이미지 좌표계의 페이지 네 모서리. None이면 자동 검출.
    corners: list[list[float]] | None = None
    mode: Mode = "color"
    #: 글줄이 가로로 눕도록 축을 자동으로 맞춘다. 상하(180도)는 판별하지 않는다.
    auto_rotate: bool = True
    #: 자동 회전에 **더해서** 적용할 회전 (0/90/180/270, 시계방향).
    #: 자동으로 축을 맞춘 뒤 상하가 뒤집혔으면 여기에 180을 준다.
    rotate: int = 0
    dewarp: bool = True
    #: 테두리를 직선 넷이 아니라 곡선 넷으로 잡고, 휜 만큼 펴서 원근 보정을 한다.
    #: 펼친 책처럼 종이가 말려 들어간 사진에서만 동작하고, 평평하면 저절로 꺼진다.
    curve: bool = True
    #: 책 실루엣을 따로 따서 그 경계로 재단한다 (책상·바닥 제거 + 테두리 펴기).
    trim: bool = True
    #: 페이지 검출에 쓸 모델 id (bookscan/models/catalog.py 참고).
    #: "classic"은 설치할 것이 없는 내장 알고리즘, 나머지는 선택 설치형 신경망이다.
    detector: str = "classic"
    spread: SpreadMode = "auto"
    #: 낱장의 가로세로비를 아는 판형으로 못박는다 (`warp.fit_paper`).
    #: 테두리만 보면 접힘과 기울기를 구별할 수 없어 비율이 조금 틀어지는데,
    #: 판형을 알면 그 자유도가 사라진다. "auto"면 잰 값을 그대로 쓴다.
    paper: PaperMode = "auto"
    #: 워프 결과 폭 기준 0~1로 정규화한 책등 위치. None이면 자동.
    gutter: float | None = None
    #: 대비/밝기 미세조정 (-100..100)
    brightness: int = 0
    contrast: int = 0
    sharpen: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PageSettings":
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AppConfig:
    output_dir: Path = OUTPUT_DIR
    input_dir: Path = INPUT_DIR
    defaults: PageSettings = field(default_factory=PageSettings)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
