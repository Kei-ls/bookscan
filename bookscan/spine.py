"""책등 쪽 변 — 네 변 중 하나만 다르게 다룬다.

펼친 책의 한 쪽을 찍으면 네 변 중 **셋만 진짜 경계**다. 위·아래·바깥쪽은
종이가 끝나고 배경이 시작되는 자리라 밝기가 뚝 떨어진다. 나머지 하나, 책등
쪽은 경계가 아니다. 종이가 접혀 들어갈 뿐 그 너머로도 같은 책의 같은 종이가
이어진다. `detect`도 `pagecurve`도 "종이와 배경의 단차"를 찾도록 만들어져
있으니, 그 변에서는 찾을 것이 애초에 없다.

지금까지 그 변을 나머지 셋과 똑같이 다뤄서 생긴 일이 둘이다. 경계가 본문 안쪽에
내려앉아 **글자를 잘랐고**(사진 4·5의 `13.` `15.` `16.`), 반대로 책등을 넘어
**옆 페이지를 물었다**(사진 1). 어느 쪽이 어느 변인지 알기만 하면 둘 다 다르게
대접할 수 있다.

자동 판정과 그 한계
-------------------
책등 쪽을 알아보는 신호는 **경계 너머가 여전히 종이인가**다. 실측(사진 5장,
경계 바깥/안쪽 평균 밝기의 비):

    사진      위     오른쪽    아래    왼쪽    실제 책등
    s0014    0.07    1.11    0.06    0.17    오른쪽
    s0015    0.26    0.01    0.13    0.95    왼쪽
    s0016    0.00    0.21    0.13    0.81    왼쪽
    s0017    0.00    0.23    0.12    0.87    왼쪽

책등 쪽만 0.8을 넘고 진짜 모서리는 0.0~0.3에 몰린다. 사이가 넓어 고르기 쉽다.

**그런데 이 신호는 경계 바깥이 사진 안에 있어야 잴 수 있다.** 페이지를 꽉 채워
찍으면 그 바깥이 프레임 밖이라 표본이 없다(실측: 사진 1의 오른쪽 변은 유효
표본이 2%였고, 화면 밖을 검게 읽어 판정이 뒤집혔다). 그래서 이 모듈은 근거가
얕으면 **기권한다.** 책등 쪽을 반대로 잡는 것은 안 잡느니만 못하다 — 지켜야 할
변을 밀어내고 밀어내야 할 변을 지키게 된다.

기권은 흠이 아니라 설계다. 어차피 촬영을 한 쪽씩으로 유도할 것이므로, 사용자가
`왼쪽`/`오른쪽`을 직접 고르는 쪽이 주(主)이고 자동은 거드는 정도면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .content import ink_mask, outward_normal
from .detect import order_corners

#: `order_corners` 기준 변 순서. 0=위, 1=오른쪽, 2=아래, 3=왼쪽.
SIDE_LABELS = ("위", "오른쪽", "아래", "왼쪽")
#: 책등 후보가 될 수 있는 변 (책등은 세로로 선다 — 회전은 이 단계 뒤에 한다)
CANDIDATES = {"left": 3, "right": 1}

#: 경계 바깥/안쪽을 볼 띠. 그 변에 수직인 쪽 치수 대비 (시작, 끝).
BAND_IN, BAND_OUT = 0.02, 0.10
#: 한 변을 따라 뽑는 표본 수
SAMPLES = 400
#: 표본 중 이 비율은 사진 안에 있어야 그 변을 판정에 쓴다
MIN_VALID = 0.35
#: 책등으로 인정할 최소 밝기비 (경계 너머도 종이여야 한다)
MIN_BRIGHT = 0.55
#: 반대쪽 변보다 이만큼은 앞서야 판정한다. 비슷하면 기권.
MIN_LEAD = 0.20
#: 책등 쪽 변을 바깥으로 미는 안전 여백 (페이지 폭 대비).
#: 여기는 배경이 아니라 같은 책의 종이라, 넉넉히 잡아도 배경이 딸려오지 않는다.
#: 글자가 잘리는 건 되돌릴 수 없고 책등 여백이 조금 남는 건 눈에 거슬리지도
#: 않으니, 한쪽으로 치우친 선택이 맞다.
SAFETY_MARGIN = 0.025


@dataclass
class Spine:
    #: "left" | "right" | None (모름/없음)
    side: str | None
    confidence: float
    reason: str

    @property
    def index(self) -> int | None:
        """`order_corners` 기준 변 번호."""
        return CANDIDATES.get(self.side) if self.side else None


def _side_evidence(work: np.ndarray, ink: np.ndarray, quad: np.ndarray,
                   i: int, perp: float) -> tuple[float, float, float]:
    """변 i의 (밝기비, 바깥 잉크비율, 유효 표본비율).

    화면 밖 표본은 **빼고** 센다. 0으로 읽으면 프레임 밖이 새까만 걸로 보여서
    거기가 배경인 양 판정된다 (실측: 그 때문에 사진 1의 책등을 반대로 잡았다).
    """
    h, w = work.shape[:2]
    a, b = quad[i], quad[(i + 1) % 4]
    n = outward_normal(a, b, quad.mean(axis=0))
    ts = np.linspace(0.08, 0.92, SAMPLES, dtype=np.float32)
    base = a[None, :] + ts[:, None] * (b - a)[None, :]
    span = np.arange(max(1, int(perp * BAND_IN)), max(2, int(perp * BAND_OUT)),
                     dtype=np.float32)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work

    def grid(offs):
        x = base[:, None, 0] + offs[None, :] * n[0]
        y = base[:, None, 1] + offs[None, :] * n[1]
        return x, y, cv2.remap(gray, x.astype(np.float32), y.astype(np.float32),
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    xo, yo, out = grid(span)
    xi, yi, inn = grid(-span)
    ok = ((xo >= 0) & (xo < w) & (yo >= 0) & (yo < h)
          & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h))
    valid = float(ok.mean())
    if ok.sum() < SAMPLES // 4:
        return 0.0, 0.0, valid

    ratio = float(out[ok].mean()) / max(float(inn[ok].mean()), 1.0)
    hit = cv2.remap(ink, xo.astype(np.float32), yo.astype(np.float32),
                    cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 127
    return ratio, float((hit & ok).sum() / max(ok.sum(), 1)), valid


def detect(work: np.ndarray, quad: np.ndarray, force: str = "auto") -> Spine:
    """책등이 어느 변인지. 근거가 얕으면 기권하고 None을 돌려준다.

    `force`가 "left"/"right"면 그대로 쓰고, "none"이면 책등이 없는 낱장으로 본다.
    """
    if force in ("left", "right"):
        return Spine(force, 1.0, f"사용자 지정: 책등 {'왼쪽' if force == 'left' else '오른쪽'}")
    if force == "none":
        return Spine(None, 1.0, "사용자 지정: 낱장 (책등 없음)")

    q = order_corners(np.asarray(quad, np.float32))
    ink, _ = ink_mask(work, long_edge=None)
    qw = float(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])))

    scores: dict[str, tuple[float, float, float]] = {}
    for name, i in CANDIDATES.items():
        scores[name] = _side_evidence(work, ink, q, i, max(24.0, qw))

    ranked = sorted(scores.items(), key=lambda kv: -kv[1][0])
    best, (ratio, inkr, valid) = ranked[0]
    second = ranked[1][1][0]

    if valid < MIN_VALID:
        return Spine(None, 0.0, f"책등 판정 기권 — 경계 바깥이 사진에 거의 없음 "
                                f"(유효 표본 {valid * 100:.0f}%)")
    if ratio < MIN_BRIGHT:
        return Spine(None, 0.0, f"책등 판정 기권 — 어느 변 너머에도 종이가 안 보임 "
                                f"(최대 밝기비 {ratio:.2f})")
    if ratio - second < MIN_LEAD:
        return Spine(None, 0.0, f"책등 판정 기권 — 좌우가 비슷함 "
                                f"({ratio:.2f} 대 {second:.2f})")
    label = "왼쪽" if best == "left" else "오른쪽"
    return Spine(best, min(1.0, ratio),
                 f"책등 {label} (밝기비 {ratio:.2f}, 바깥 잉크 {inkr:.2f})")
