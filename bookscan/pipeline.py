"""처리 파이프라인 — 사진 한 장을 최종 페이지 이미지로.

  원본 → (신경망 분할) → 페이지 검출 → 원근 보정 → 회전 → 책 경계 재단
       → 펼침면 분할 → 곡면 펴기 → 화질 보정

각 단계는 실패해도 예외를 던지지 않는다. 할 수 있는 데까지 하고 무슨 일이
있었는지 notes에 남긴다. 사진 한 장 때문에 배치 전체가 멈추면 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .bookmask import book_mask
from .booktrim import trim_to_book
from .config import MAX_OUTPUT_LONG_EDGE, WORK_LONG_EDGE, PageSettings
from .detect import Detection, detect_page
from .dewarp import dewarp
from .enhance import enhance
from .imgio import fit_long_edge, rotate90
from .models import get as get_model
from .models import ready as model_ready
from .models import segment as model_segment
from .orient import estimate_axis
from .spread import SpreadResult, detect_gutter, split_spread
from .warp import clamp_corners, warp_mask, warp_page, warp_transform


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

    work, scale = fit_long_edge(original, WORK_LONG_EDGE)

    # --- 1. 신경망 분할 (선택) ---
    # 마스크 한 장이 두 군데에 쓰인다: 검출 후보를 만드는 데, 그리고 재단할
    # 책 실루엣으로. 그래서 검출보다 먼저 한 번만 돌린다.
    info = get_model(st.detector)
    prior = None
    if info.neural:
        if not model_ready(info.id):
            notes.append(f"{info.name}: 설치되지 않아 내장 알고리즘을 씁니다")
        else:
            prior = model_segment(work, info.id)
            notes.append(f"신경망 분할: {info.name}"
                         + ("" if prior is not None else " — 실패, 내장 알고리즘으로 대체"))

    # --- 2. 페이지 검출 (수동 지정이 있으면 그걸 쓴다) ---
    if st.corners:
        corners = clamp_corners(st.corners, original.shape)
        det = Detection(corners=corners, confidence=1.0, method="manual")
        notes.append("수동 지정 코너 사용")
    else:
        det = detect_page(work, prior_mask=prior, prior_only=(info.id == "docscanner"))
        det = Detection(det.corners / scale, det.confidence, det.method)
        notes.append(f"검출: {det.method} (confidence {det.confidence:.2f})")
        if det.method == "fallback":
            notes.append("페이지를 찾지 못해 사진 전체를 사용 — 코너를 직접 잡아주세요")

    # --- 3. 책 영역 마스크 ---
    # 사각형은 펼친 책을 정확히 담지 못한다. 실제 실루엣을 따로 따서 워프에
    # 같이 태우고, 나중에 그 경계로 재단한다.
    mask = None
    if st.trim:
        mask = book_mask(work, det.corners * scale, prior=prior)
        if mask is None:
            notes.append("책 영역 분할 실패 — 사각형 그대로 사용")
        elif scale != 1.0:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # --- 4. 원근 보정 (마스크도 같은 변환을 태운다) ---
    warped = warp_page(original, det.corners, max_long_edge=MAX_OUTPUT_LONG_EDGE)
    wmask = None
    if mask is not None:
        M, ow, oh = warp_transform(det.corners, max_long_edge=MAX_OUTPUT_LONG_EDGE)
        wmask = warp_mask(mask, M, (ow, oh))

    # --- 5. 회전 (분할보다 먼저 해야 책등이 세로로 선다) ---
    # 책을 옆으로 눕혀 찍으면 이후 단계가 전부 헛돈다. 곡면 펴기는 글줄이
    # 가로로 누워 있다고 보고 곡률을 재고, 펼침면 분할은 책등이 세로로
    # 서 있다고 보고 찾기 때문이다.
    auto_rot = 0
    if st.auto_rotate:
        o = estimate_axis(warped)
        auto_rot = o.rotate
        notes.append(f"자동 회전: {o.reason}")
    rot = (auto_rot + st.rotate) % 360
    if rot:
        warped = rotate90(warped, rot)
        if wmask is not None:
            wmask = rotate90(wmask, rot)
    if st.rotate % 360:
        notes.append(f"수동 회전 {st.rotate % 360}도 (최종 {rot}도)")

    # --- 6. 책 경계로 재단 (분할보다 먼저 해야 배경이 한쪽 페이지로 몰리지 않는다) ---
    tr = trim_to_book(warped, wmask, enabled=st.trim)
    warped = tr.image
    if st.trim:
        notes.append(f"재단: {tr.reason}")

    # --- 7. 펼침면 분할 ---
    sp: SpreadResult = detect_gutter(warped, st.spread)
    if st.gutter is not None and st.spread != "single":
        sp = SpreadResult(True, float(st.gutter), 1.0, "책등 위치 수동 지정")
    notes.append(sp.reason)

    if sp.is_spread and sp.gutter is not None:
        left, right = split_spread(warped, sp.gutter)
        parts = [PageImage(left, "left"), PageImage(right, "right")]
    else:
        parts = [PageImage(warped, None)]

    # --- 8. 각 페이지를 펴고 보정 ---
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


def preview_detection(original: np.ndarray, detector: str = "classic") -> Detection:
    """편집 UI가 초기 코너를 받아갈 때 쓰는 가벼운 검출."""
    work, scale = fit_long_edge(original, WORK_LONG_EDGE)
    info = get_model(detector)
    prior = model_segment(work, info.id) if info.neural and model_ready(info.id) else None
    det = detect_page(work, prior_mask=prior, prior_only=(info.id == "docscanner"))
    return Detection(det.corners / scale, det.confidence, det.method)
