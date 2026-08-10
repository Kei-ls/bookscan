"""이미지 입출력. EXIF 회전 반영, HEIC 지원, 리사이즈 헬퍼."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # 아이폰 사진은 대부분 HEIC라 이게 없으면 아예 못 읽는다.
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:  # pragma: no cover - 설치 실패 시에도 나머지는 동작해야 함
    HEIF_OK = False


def imread(path: str | Path) -> np.ndarray:
    """BGR uint8 배열로 읽는다. EXIF Orientation을 실제 픽셀에 반영한다.

    cv2.imread는 유니코드 경로와 HEIC를 못 다루므로 Pillow를 경유한다.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # 폰 사진의 회전 태그를 픽셀에 굽는다
        im = im.convert("RGB")
        rgb = np.asarray(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def imwrite(path: str | Path, bgr: np.ndarray, quality: int = 92) -> None:
    """확장자에 맞춰 저장. 유니코드 경로 안전."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if bgr.ndim == 2:
        im = Image.fromarray(bgr, mode="L")
    else:
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if ext in (".jpg", ".jpeg"):
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(path, "JPEG", quality=quality, subsampling=0, optimize=True)
    elif ext == ".png":
        im.save(path, "PNG", optimize=True)
    else:
        im.save(path)


def encode_jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """업로드된 바이트를 BGR 배열로. HEIC 포함."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        rgb = np.asarray(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def fit_long_edge(bgr: np.ndarray, long_edge: int) -> tuple[np.ndarray, float]:
    """긴 변이 long_edge가 되도록 축소. (이미지, 원본대비배율) 반환.

    확대는 하지 않는다. 배율은 좌표를 원본계로 되돌릴 때 쓴다.
    """
    h, w = bgr.shape[:2]
    cur = max(h, w)
    if cur <= long_edge:
        return bgr, 1.0
    scale = long_edge / cur
    out = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                     interpolation=cv2.INTER_AREA)
    return out, scale


def rotate90(bgr: np.ndarray, degrees: int) -> np.ndarray:
    """시계방향 회전. degrees는 0/90/180/270."""
    d = degrees % 360
    if d == 0:
        return bgr
    if d == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"90도 단위만 지원: {degrees}")
