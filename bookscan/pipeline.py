"""처리 파이프라인 — 사진 한 장을 최종 페이지 이미지로.

  원본 → 페이지 검출 → 원근 보정 → 회전 → 펼침면 분할 → 곡면 펴기 → 화질 보정

각 단계는 실패해도 예외를 던지지 않는다. 할 수 있는 데까지 하고 무슨 일이
있었는지 notes에 남긴다. 사진 한 장 때문에 배치 전체가 멈추면 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import MAX_OUTPUT_LONG_EDGE, WORK_LONG_EDGE, PageSettings
from .detect import Detection, detect_page
from .dewarp import dewarp
from .enhance import enhance
from .imgio import fit_long_edge, rotate90
from .spread import SpreadResult, detect_gutter, split_spread
from .warp import clamp_corners, warp_page


@dataclass
class PageImage:
    image: np.ndarray
    half: str | None = None  # None | "left" | "right"
    notes: list[str] = field(default_factory=list)


@dataclass
class ProcessResult:
    pages: list[PageImage]
    corners: list[list[float]]  # 원본 좌표계
    confidence: float
    method: str
    is_spread: bool
    gutter: float | None
    notes: list[str] = field(default_factory=list)


def process(original: np.ndarray, settings: PageSettings | None = None) -> ProcessResult:
    """사진 한 장을 처리해 1~2장의 페이지 이미지를 만든다."""
    st = settings or PageSettings()
    notes: list[str] = []

    # --- 1. 페이지 검출 (수동 지정이 있으면 그걸 쓴다) ---
    if st.corners:
        corners = clamp_corners(st.corners, original.shape)
        det = Detection(corners=corners, confidence=1.0, method="manual")
        notes.append("수동 지정 코너 사용")
    else:
        work, scale = fit_long_edge(original, WORK_LONG_EDGE)
        det = detect_page(work)
        det = Detection(det.corners / scale, det.confidence, det.method)
        notes.append(f"검출: {det.method} (confidence {det.confidence:.2f})")
        if det.method == "fallback":
            notes.append("페이지를 찾지 못해 사진 전체를 사용 — 코너를 직접 잡아주세요")

    # --- 2. 원근 보정 ---
    warped = warp_page(original, det.corners, max_long_edge=MAX_OUTPUT_LONG_EDGE)

    # --- 3. 회전 (분할보다 먼저 해야 책등이 세로로 선다) ---
    if st.rotate % 360:
        warped = rotate90(warped, st.rotate)
        notes.append(f"{st.rotate}도 회전")

    # --- 4. 펼침면 분할 ---
    sp: SpreadResult = detect_gutter(warped, st.spread)
    if st.gutter is not None and st.spread != "single":
        sp = SpreadResult(True, float(st.gutter), 1.0, "책등 위치 수동 지정")
    notes.append(sp.reason)

    if sp.is_spread and sp.gutter is not None:
        left, right = split_spread(warped, sp.gutter)
        parts = [PageImage(left, "left"), PageImage(right, "right")]
    else:
        parts = [PageImage(warped, None)]

    # --- 5~6. 각 페이지를 펴고 보정 ---
    for part in parts:
        dw = dewarp(part.image, enabled=st.dewarp)
        part.image = dw.image
        part.notes.append(f"곡면 펴기: {dw.reason}")
        part.image = enhance(
            part.image,
            mode=st.mode,
            brightness=st.brightness,
            contrast=st.contrast,
            sharpen=st.sharpen,
        )

    return ProcessResult(
        pages=parts,
        corners=[[float(x), float(y)] for x, y in det.corners],
        confidence=float(det.confidence),
        method=det.method,
        is_spread=bool(sp.is_spread),
        gutter=sp.gutter,
        notes=notes,
    )


def preview_detection(original: np.ndarray) -> Detection:
    """편집 UI가 초기 코너를 받아갈 때 쓰는 가벼운 검출."""
    work, scale = fit_long_edge(original, WORK_LONG_EDGE)
    det = detect_page(work)
    return Detection(det.corners / scale, det.confidence, det.method)
