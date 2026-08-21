"""신경망 검출기 실행 환경 — 지연 로딩, 가중치 관리, 추론.

torch는 **선택 의존성**이다. 200MB짜리 패키지를 안 쓰는 사람에게까지 깔게 할
이유가 없다. 그래서 이 모듈은 torch를 최상단에서 import하지 않고, 실제로
신경망을 돌릴 때만 불러온다. torch가 없으면 조용히 None을 돌려주고 호출부는
고전 방식으로 되돌아간다.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import subprocess
import sys
import threading
import urllib.request
from typing import Any, Callable

import cv2
import numpy as np

from .catalog import MODELS, ModelInfo, WEIGHTS_DIR, get, weights_for
from .docscanner import INPUT_SIZE, build_model

#: 마스크가 프레임에서 차지하는 면적이 이 범위를 벗어나면 실패로 본다.
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.03, 0.98

_lock = threading.Lock()
_cache: dict[str, Any] = {}
_torch: Any = None


# ------------------------------------------------------------------ 설치 상태
def torch_installed() -> bool:
    """import하지 않고 설치 여부만 확인한다 (import는 몇 초씩 걸린다)."""
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def weights_installed(info: ModelInfo) -> bool:
    p = weights_for(info.id).weights_path
    return bool(p and p.exists() and p.stat().st_size > 1024)


def ready(model_id: str) -> bool:
    info = get(model_id)
    if not info.neural:
        return True
    return torch_installed() and weights_installed(info)


def status(model_id: str) -> dict:
    info = get(model_id)
    w = weights_for(model_id)
    have_torch = (not info.neural) or torch_installed()
    have_weights = (not info.neural) or weights_installed(info)
    missing = []
    if not have_torch:
        missing.append("torch")
    if not have_weights:
        missing.append("가중치")
    return {
        "id": info.id,
        "name": info.name,
        "summary": info.summary,
        "neural": info.neural,
        "ready": have_torch and have_weights,
        "torch": have_torch,
        "weights": have_weights,
        "missing": missing,
        "weights_bytes": w.weights_bytes if info.neural else 0,
        "source": info.source or w.source,
        "license": info.license or w.license,
    }


def status_all() -> list[dict]:
    return [status(mid) for mid in MODELS]


# ------------------------------------------------------------------ 설치 동작
def install_packages(model_id: str, log: Callable[[str], None] | None = None) -> None:
    """torch를 이 venv에 설치한다. 실패하면 예외를 던진다."""
    info = get(model_id)
    if not info.pip_packages:
        return
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    if info.pip_index:
        cmd += ["--index-url", info.pip_index]
    cmd += list(info.pip_packages)
    if log:
        log(" ".join(info.pip_packages) + " 설치 중… (수백 MB, 몇 분 걸립니다)")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError("pip 설치 실패:\n" + "\n".join(tail))
    # find_spec은 첫 조회 결과를 캐시한다. 방금 설치한 패키지가 안 보일 수 있다.
    importlib.invalidate_caches()
    if log:
        log("설치 완료")


def download_weights(model_id: str,
                     progress: Callable[[int, int], None] | None = None) -> None:
    """가중치를 내려받아 검증한다. 중간에 죽어도 반쪽짜리 파일이 남지 않게
    임시 파일에 받은 뒤 옮긴다."""
    info = weights_for(model_id)
    if not info.weights_url:
        return
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = info.weights_path
    tmp = dest.with_suffix(dest.suffix + ".part")

    digest = hashlib.sha256()
    req = urllib.request.Request(info.weights_url, headers={"User-Agent": "bookscan"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or info.weights_bytes or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)

    got = digest.hexdigest()
    if info.weights_sha256 and got != info.weights_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"가중치 해시가 다릅니다 (받은 값 {got[:16]}…). 내려받기를 중단했습니다.")
    tmp.replace(dest)
    with _lock:
        _cache.pop(info.id, None)


# ------------------------------------------------------------------ 모델 적재
def _load(info: ModelInfo) -> Any:
    global _torch
    with _lock:
        if info.id in _cache:
            return _cache[info.id]
        if _torch is None:
            _torch = importlib.import_module("torch")
        torch = _torch
        model = build_model(torch)
        # weights_only=True — 남이 올린 .pth는 결국 pickle이라 임의 코드를
        # 실행시킬 수 있다. 텐서만 읽고 나머지는 거부한다.
        state = torch.load(info.weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        torch.set_grad_enabled(False)
        _cache[info.id] = model
        return model


def _binarize(prob: np.ndarray) -> np.ndarray | None:
    """확률 맵을 0/255 마스크로.

    **고정 문턱 0.5는 쓰면 안 된다.** 이 체크포인트는 확률을 0~1로 벌리지 못하고
    실측 0.47~0.72 안에 뭉쳐 놓는다(모든 사진에서 최댓값이 0.724로 똑같았다).
    그래서 0.5로 자르면 프레임의 96~98%가 "문서"가 되어 마스크가 아무 일도
    하지 않는다. 확률의 절대값이 아니라 **분포 안에서의 상대 위치**를 봐야 한다.

    Otsu가 그 일을 한다(합성 6장 코너 오차 실측: Otsu 114px < 원 저장소의
    mean+std/2 방식 125px < 고정 0.5 244px). Otsu가 말이 안 되는 면적을 내놓으면
    원 저장소 방식으로 한 번 더 시도한다.
    """
    h, w = prob.shape[:2]
    u8 = cv2.normalize(prob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    author = float(prob.mean() + abs(prob.std() / 2))
    author_u8 = np.interp(author, (prob.min(), prob.max()), (0, 255))
    for thr in (float(otsu), float(author_u8)):
        mask = (u8 > thr).astype(np.uint8) * 255
        ratio = float((mask > 0).sum()) / (h * w)
        if MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO:
            return mask
    return None


def segment(bgr: np.ndarray, model_id: str = "docscanner") -> np.ndarray | None:
    """문서 영역 마스크(0/255, 입력과 같은 크기). 못 하면 None.

    모델은 256x256 회색조를 받는다. **이진 마스크를 키우지 않고 확률 맵을 키운
    다음 문턱을 건다** — 256픽셀 격자에서 자른 계단을 1500픽셀로 늘리면 경계가
    한 칸에 6픽셀씩 뭉개지기 때문이다.
    """
    info = weights_for(model_id)
    if not info.neural or not ready(info.id):
        return None
    try:
        model = _load(info)
        torch = _torch
    except Exception:
        return None

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    small = cv2.resize(gray, INPUT_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    std = float(small.std())
    if std < 1e-6:
        return None  # 완전 균일한 사진 — 정규화하면 0으로 나눈다
    x = torch.from_numpy((small - small.mean()) / std)[None, None]

    try:
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    except Exception:
        return None

    full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return _binarize(full)


def unload() -> None:
    """적재한 모델을 놓아준다 (메모리 회수용)."""
    with _lock:
        _cache.clear()
