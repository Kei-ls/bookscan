"""검출기(모델) 목록.

UI의 드롭다운은 이 목록을 그대로 그린다. 새 모델을 붙이려면 여기에 항목을
하나 더 만들고 runtime.py의 segment 분기만 늘리면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ROOT

#: 가중치를 받아 두는 곳. 저장소에 커밋하지 않는다(.gitignore).
WEIGHTS_DIR = ROOT / "models"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    #: 한 줄 설명 — 드롭다운 밑에 그대로 뜬다
    summary: str
    #: 신경망을 쓰는가 (torch + 가중치가 필요한가)
    neural: bool = False
    #: 이 가중치가 기대하는 입력 크기. 파인튜닝하며 올릴 수 있어서 모델마다 둔다.
    #: U-Net은 5번 풀링하므로 32의 배수여야 한다.
    input_size: tuple[int, int] = (256, 256)
    #: 내려받을 곳이 없고 이 컴퓨터에서 직접 만들어야 하는 가중치인가.
    #: 파일이 없으면 목록에 아예 띄우지 않는다 — 설치 버튼을 눌러도 받을 데가 없다.
    local_only: bool = False
    weights_name: str = ""
    weights_url: str = ""
    weights_bytes: int = 0
    #: 내려받은 파일을 검증할 SHA-256. 남이 만든 가중치를 그대로 믿지 않기 위함.
    weights_sha256: str = ""
    source: str = ""
    license: str = ""
    pip_packages: tuple[str, ...] = field(default_factory=tuple)
    pip_index: str = ""

    @property
    def weights_path(self):
        return WEIGHTS_DIR / self.weights_name if self.weights_name else None


DOCSCANNER = ModelInfo(
    id="docscanner",
    name="Document-Scanner (신경망만)",
    summary="신경망 분할 결과를 우선 믿습니다. 가장 빠르지만 모델이 틀린 사진에서는 "
            "내장 알고리즘보다 코너가 부정확합니다. 아래 '내장 + 신경망'을 권합니다.",
    neural=True,
    weights_name="Scanner-Detector.pth",
    weights_url="https://huggingface.co/ordaktaktak/Document-Scanner/resolve/main/Scanner-Detector.pth",
    weights_bytes=122337249,
    weights_sha256="f8c5809d06d49093581108bb71555c73e3f6044e3fcc5852682c1b78e9f8da10",
    source="https://github.com/Saeed-Biabani/Document-Scanner",
    license="MIT (c) 2024 Saeed-Biabani",
    pip_packages=("torch",),
    # 윈도우의 기본 PyPI torch 휠은 CUDA를 통째로 안고 있어 2GB가 넘는다.
    # 우리는 GPU를 안 쓰므로 CPU 전용 인덱스에서 받는다 (약 200MB).
    pip_index="https://download.pytorch.org/whl/cpu",
)

CLASSIC = ModelInfo(
    id="classic",
    name="내장 알고리즘 (기본)",
    summary="윤곽선·밝기분할·직선교점 3단 폴백. 설치할 것이 없습니다.",
)

HYBRID = ModelInfo(
    id="hybrid",
    name="내장 + 신경망 (권장)",
    summary="양쪽 후보를 모두 만들어 같은 채점 함수로 비교하고, 재단할 책 실루엣은 "
            "신경망 마스크를 씁니다. 코너 정확도는 내장과 같고 배경은 더 지웁니다.",
    neural=True,
    pip_packages=DOCSCANNER.pip_packages,
    pip_index=DOCSCANNER.pip_index,
)

FINETUNED = ModelInfo(
    id="finetuned",
    name="파인튜닝 (내 사진으로 학습)",
    summary="`tools/finetune.py`로 직접 학습한 가중치를 씁니다. 입력 해상도가 높아 "
            "책등 쪽 경계를 원본 기준 더 촘촘히 잡습니다. 가중치 파일이 있을 때만 "
            "목록에 뜹니다.",
    neural=True,
    input_size=(384, 384),
    local_only=True,
    weights_name="Scanner-Detector-ft.pth",
    source="tools/finetune.py",
    license="원본 가중치에서 파생 — MIT (c) 2024 Saeed-Biabani",
    pip_packages=DOCSCANNER.pip_packages,
    pip_index=DOCSCANNER.pip_index,
)

MODELS: dict[str, ModelInfo] = {
    m.id: m for m in (CLASSIC, DOCSCANNER, HYBRID, FINETUNED)}
DEFAULT_MODEL = CLASSIC.id


def get(model_id: str | None) -> ModelInfo:
    return MODELS.get(model_id or DEFAULT_MODEL, CLASSIC)


def weights_for(model_id: str) -> ModelInfo:
    """hybrid처럼 자기 가중치가 없는 항목을 실제 가중치 주인으로 바꿔준다.

    자기 가중치 파일을 가진 모델은 그대로 둔다 — 나중에 두 번째 신경망을 붙였을 때
    말없이 Document-Scanner의 가중치를 쓰게 되면 안 된다.
    """
    info = get(model_id)
    # url이 아니라 **파일 이름**으로 따진다 — 파인튜닝한 가중치는 내려받을 곳이
    # 없지만 엄연히 자기 것이다.
    if info.neural and not info.weights_name:
        return DOCSCANNER
    return info
