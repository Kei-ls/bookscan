"""처리 파이프라인 — 사진 한 장을 최종 페이지 이미지로.

  원본 → (신경망 분할) → 페이지 검출 → 굽은 테두리 검출 → 원근/곡선 보정
       → 회전 → 책 경계 재단 → 펼침면 분할 → 곡면 펴기 → 화질 보정

각 단계는 실패해도 예외를 던지지 않는다. 할 수 있는 데까지 하고 무슨 일이
있었는지 notes에 남긴다. 사진 한 장 때문에 배치 전체가 멈추면 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .bookmask import book_mask
from .booktrim import trim_to_book
from .config import (MAX_OUTPUT_LONG_EDGE, SINGLE_PAGE_ASPECT, WORK_LONG_EDGE,
                     PageSettings)
from .content import guard_quad, push_sides
from .detect import Detection, detect_page, order_corners
from .dewarp import dewarp
from .enhance import enhance
from .imgio import fit_long_edge, rotate90
from .models import get as get_model
from .models import ready as model_ready
from .models import segment as model_segment
from .orient import estimate_axis
from .pagecurve import CURVE_POINTS, detect_curve, guard_content, shift_sides
from . import spine as spine_mod
from .spread import SpreadResult, detect_gutter, split_spread
from .warp import (clamp_corners, curved_maps, fit_paper, remap_mask, warp_curved,
                   warp_mask, warp_page, warp_transform)

#: 테두리 폐곡선을 UI로 보낼 때 이만큼 건너뛰며 솎아낸다 (한 변당 약 24점).
OUTLINE_STEP = max(1, CURVE_POINTS // 24)


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
    #: 곡선 보정을 실제로 적용했을 때, 그때 쓴 테두리 폐곡선 (원본 좌표계).
    #: 편집 화면에 겹쳐 그려서 "무엇을 폈는지" 눈으로 확인할 수 있게 한다.
    outline: list[list[float]] | None = None


def _aspect_note(image: np.ndarray) -> str:
    """낱장 한 쪽치고 비율이 이상하면 경고 문구. 멀쩡하면 빈 문자열.

    한 쪽씩만 찍는다는 전제가 서면 결과 비율이 좁은 띠 안에 들어와야 한다.
    벗어났다는 건 크롭이 페이지와 어긋났다는 뜻이고, 그 원인은 대개 둘 중
    하나다 — 옆 페이지가 딸려 들어왔거나, 페이지 한쪽이 잘려 나갔거나.

    **고치지 않고 알리기만 한다.** 넓게 잡힌 크롭을 비율만 맞춰 눌러도 옆
    페이지는 그대로 남고 글자만 찌그러진다. 무엇이 이상한지 짚어주는 게 여기서
    할 수 있는 정직한 일이다.
    """
    h, w = image.shape[:2]
    if h < 16 or w < 16:
        return ""
    have = w / h
    lo, hi = SINGLE_PAGE_ASPECT
    if have > hi:
        return (f"가로세로비 {have:.2f} — 한 쪽치고 넓습니다 "
                f"(보통 {lo:.2f}~{hi:.2f}). 옆 페이지가 딸려 들어왔는지 확인하세요")
    if have < lo:
        return (f"가로세로비 {have:.2f} — 한 쪽치고 좁습니다 "
                f"(보통 {lo:.2f}~{hi:.2f}). 페이지가 잘려 나갔는지 확인하세요")
    return ""


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
        # 사각형이 글자를 물고 있으면 여기서 넓혀 둔다. 아래 책 마스크와 굽은
        # 테두리가 모두 이 사각형을 씨앗으로 쓰므로, 고치려면 여기가 먼저다.
        # 수동 지정 코너는 건드리지 않는다 — 사용자가 끌어다 놓은 자리가 답이다.
        guarded, gnote = guard_quad(work, det.corners * scale)
        if gnote:
            det = Detection(guarded / scale, det.confidence, det.method)
            notes.append("본문 보호: " + gnote)

    # --- 2b. 책등 쪽 변 (나머지 셋과 다르게 다룰 변 하나) ---
    # 여기는 종이가 끝나는 자리가 아니라 접혀 들어가는 자리다. 찾을 단차가 없고,
    # 그래서 검출도 마스크도 이 변에서만 안쪽으로 흘러내린다. 살짝 바깥으로 밀어
    # 둔다 — 넘어가 봐야 같은 책의 종이라 배경이 딸려오지 않는데, 모자라면
    # 글자가 잘리고 그건 되돌릴 수 없다.
    sp = spine_mod.detect(work, det.corners * scale, st.spine)
    notes.append(sp.reason)
    quad_w = 0.0
    if sp.index is not None:
        q = order_corners((det.corners * scale).astype(np.float32))
        qw = float(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))
        qh = float(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])))
        margin = qw * spine_mod.SAFETY_MARGIN
        # 이미 한 쪽치고 넓다면 밀지 않는다. 이 여백은 "책등 쪽 변이 안쪽으로
        # 흘러내린다"는 전제 위에 서 있는데, 넓게 잡힌 사각형은 그 반대로
        # 틀린 것이라(옆 페이지를 문 경우) 밀면 더 나빠지기만 한다. 어느 쪽으로
        # 틀렸는지는 책등을 직접 찾아야 알 수 있고, 그건 아직 못 한다.
        if qh > 1e-6 and (qw + margin) / qh > SINGLE_PAGE_ASPECT[1]:
            notes.append(f"책등 쪽 여백 생략 — 사각형이 이미 한 쪽보다 넓음 "
                         f"(가로세로비 {qw / qh:.2f})")
        else:
            quad_w = qw
            pushes = [0.0, 0.0, 0.0, 0.0]
            pushes[sp.index] = margin
            det = Detection(push_sides(q, tuple(pushes)) / scale,
                            det.confidence, det.method)
            notes.append(f"책등 쪽({spine_mod.SIDE_LABELS[sp.index]}) 변을 "
                         f"{margin / scale:.0f}px 바깥으로 — 글자가 잘리는 걸 막는다")

    # --- 3. 책 영역 마스크 ---
    # 사각형은 펼친 책을 정확히 담지 못한다. 실제 실루엣을 따로 따서 두 군데에
    # 쓴다: 굽은 테두리를 어디쯤에서 찾을지 알려주는 데, 그리고 재단할 경계로.
    mask_work = None
    if st.trim or st.curve:
        mask_work = book_mask(work, det.corners * scale, prior=prior)
        if mask_work is None and st.trim:
            notes.append("책 영역 분할 실패 — 사각형 그대로 사용")

    mask = None
    if mask_work is not None and st.trim:
        mask = mask_work if scale == 1.0 else cv2.resize(
            mask_work, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    # --- 4. 굽은 테두리 검출 ---
    # 여기가 사각형을 벗어나는 지점이다. 네 변을 곡선으로 다시 잡고, 휜 정도를 잰다.
    curve = None
    if st.curve:
        try:
            curve = detect_curve(work, det.corners * scale, mask=mask_work)
        except Exception:
            curve = None
        notes.append("테두리: " + (curve.reason if curve is not None else "곡선을 찾지 못함 — 원근 보정만"))
        # 여기가 글자를 지키는 마지막 자리다. `warp_curved`는 이 곡선 안쪽만
        # 떠오므로, 넘어가면 잘린 획을 되찾을 방법이 없다.
        if curve is not None and curve.curved:
            curve, cnote = guard_content(work, curve)
            if cnote:
                notes.append("본문 보호: " + cnote)
            # 곡선은 사각형을 씨앗으로 다시 찾은 것이라, 위에서 민 것과 무관하게
            # 또 흘러내려 있을 수 있다. 같은 여백을 여기서 한 번 더 준다.
            if sp.side is not None and quad_w > 0:
                d = [0.0, 0.0, 0.0, 0.0]  # 위, 아래, 왼쪽, 오른쪽
                d[2 if sp.side == "left" else 3] = quad_w * spine_mod.SAFETY_MARGIN
                curve = shift_sides(curve, tuple(d))
                notes.append("책등 쪽 테두리도 같은 만큼 바깥으로")

    # --- 5. 원근 보정 (마스크도 같은 변환을 태운다) ---
    # 곡선이 실제로 휘어 있을 때만 곡선 워프를 쓴다. 평평하면 지금까지처럼
    # 호모그래피 한 장으로 끝내서, 기존 결과가 한 픽셀도 달라지지 않게 한다.
    cmaps, full_curve = None, None
    if curve is not None and curve.curved:
        full_curve = curve.scaled(1.0 / scale)  # 작업본 → 원본 좌표계
        cmaps = curved_maps(full_curve, max_long_edge=MAX_OUTPUT_LONG_EDGE)
        if cmaps is None:
            notes.append("곡선 워프 실패 — 원근 보정으로 대체")

    outline: list[list[float]] | None = None
    if cmaps is not None:
        map_x, map_y, _, _, gain = cmaps
        # 매니페스트에 들어가므로 성기게 줄인다. 완만한 곡선이라 손실이 없다.
        outline = [[float(x), float(y)] for x, y in full_curve.polygon()[::OUTLINE_STEP]]
        warped = warp_curved(original, map_x, map_y)
        wmask = remap_mask(mask, map_x, map_y) if mask is not None else None
        notes.append(f"곡선 보정 적용 (눌린 곳을 최대 {gain:.2f}배 늘림)")
    else:
        warped = warp_page(original, det.corners, max_long_edge=MAX_OUTPUT_LONG_EDGE)
        wmask = None
        if mask is not None:
            M, ow, oh = warp_transform(det.corners, max_long_edge=MAX_OUTPUT_LONG_EDGE)
            wmask = warp_mask(mask, M, (ow, oh))

    # --- 6. 회전 (분할보다 먼저 해야 책등이 세로로 선다) ---
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

    # --- 7. 책 경계로 재단 (분할보다 먼저 해야 배경이 한쪽 페이지로 몰리지 않는다) ---
    # 곡선 워프가 이미 테두리를 폈다면 여기서 또 펴면 두 번 편 셈이 된다.
    # 배경을 지우고 여백을 잘라내는 일만 시킨다.
    tr = trim_to_book(warped, wmask, enabled=st.trim, geometry=cmaps is None)
    warped = tr.image
    if st.trim:
        notes.append(f"재단: {tr.reason}")

    # --- 8. 펼침면 분할 ---
    sp: SpreadResult = detect_gutter(warped, st.spread)
    if st.gutter is not None and st.spread != "single":
        sp = SpreadResult(True, float(st.gutter), 1.0, "책등 위치 수동 지정")
    notes.append(sp.reason)

    if sp.is_spread and sp.gutter is not None:
        left, right = split_spread(warped, sp.gutter)
        parts = [PageImage(left, "left"), PageImage(right, "right")]
    else:
        parts = [PageImage(warped, None)]

    # --- 9. 각 페이지를 펴고, 비율을 맞추고, 보정 ---
    # 비율 맞추기는 분할 뒤 낱장에 적용한다. 펼침면(1.4)에 낱장 비율(0.7)을
    # 씌우면 폭이 두 배가 된다. 화질 보정보다는 먼저 해서, 샤프닝한 결과를
    # 다시 리샘플해 뭉개는 일이 없게 한다.
    for part in parts:
        dw = dewarp(part.image, enabled=st.dewarp)
        part.image = dw.image
        part.notes.append(f"곡면 펴기: {dw.reason}")
        part.image, paper_note = fit_paper(
            part.image, st.paper, spread=(part.half is None and sp.is_spread),
            max_long_edge=MAX_OUTPUT_LONG_EDGE)
        if paper_note:
            part.notes.append(paper_note)
        # 쪼개지 않은 펼침면은 애초에 두 쪽이라 이 잣대가 맞지 않는다.
        if part.half is not None or not sp.is_spread:
            warn = _aspect_note(part.image)
            if warn:
                part.notes.append("확인 필요: " + warn)
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
        outline=outline,
    )


def preview_detection(original: np.ndarray, detector: str = "classic") -> Detection:
    """편집 UI가 초기 코너를 받아갈 때 쓰는 가벼운 검출."""
    work, scale = fit_long_edge(original, WORK_LONG_EDGE)
    info = get_model(detector)
    prior = model_segment(work, info.id) if info.neural and model_ready(info.id) else None
    det = detect_page(work, prior_mask=prior, prior_only=(info.id == "docscanner"))
    return Detection(det.corners / scale, det.confidence, det.method)
