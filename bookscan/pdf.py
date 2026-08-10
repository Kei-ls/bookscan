"""PDF 조립.

img2pdf를 쓰는 이유: JPEG 바이트를 다시 인코딩하지 않고 그대로 PDF 안에 넣는다.
Pillow로 저장하면 재인코딩이 일어나 화질이 한 번 더 깎이고 용량도 커진다.
"""

from __future__ import annotations

from pathlib import Path

import img2pdf
from PIL import Image

from .store import Project

#: mm 단위 용지 크기
PAGE_SIZES = {
    "auto": None,  # 이미지 비율 그대로
    "a4": (210.0, 297.0),
    "letter": (215.9, 279.4),
    "b5": (176.0, 250.0),
}


def _layout(size: str):
    """img2pdf 레이아웃 함수. auto면 이미지 크기를 그대로 쓴다."""
    dims = PAGE_SIZES.get(size)
    if dims is None:
        return None
    w, h = dims
    return img2pdf.get_layout_fun(
        pagesize=(img2pdf.mm_to_pt(w), img2pdf.mm_to_pt(h)),
        border=(0, 0),
        fit=img2pdf.FitMode.into,
        auto_orient=True,
    )


def _prepare(path: Path, tmp_dir: Path) -> Path:
    """img2pdf가 받아주는 형태로 맞춘다.

    img2pdf는 알파 채널이나 팔레트 PNG를 거부하므로 그런 경우에만 변환한다.
    JPEG는 손대지 않아야 무손실 임베드가 유지된다.
    """
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return path
    with Image.open(path) as im:
        if im.mode in ("RGB", "L", "1"):
            return path
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out = tmp_dir / f"{path.stem}_rgb.png"
        im.convert("RGB").save(out, "PNG")
        return out


#: Windows 파일명에 쓸 수 없는 문자
_BAD_CHARS = '\\/:*?"<>|'


def safe_filename(name: str, fallback: str) -> str:
    """사용자가 입력한 이름을 실제로 저장 가능한 파일명으로 만든다.

    이걸 안 하면 `2024/1권` 같은 이름에서 경로 조작이 일어나거나 OSError로 500이 난다.
    """
    name = (name or "").strip().strip(". ")
    for ch in _BAD_CHARS:
        name = name.replace(ch, "_")
    name = name.replace("\0", "")
    if not name:
        name = fallback
    if len(name) > 120:
        name = name[:120].rstrip()
    return name


def build_pdf(project: Project, filename: str | None = None, size: str = "auto") -> Path:
    """현재 순서대로 PDF를 만든다. 만들어진 파일 경로를 돌려준다."""
    pages = project.ordered_pages()
    if not pages:
        raise ValueError("페이지가 없습니다. 사진을 먼저 가져오세요.")

    name = safe_filename(filename or project.name, project.name)
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = safe_filename(name, project.name) + ".pdf"
    out_path = project.dir / name

    tmp_dir = project.dir / ".pdftmp"
    paths = []
    for page in pages:
        p = project.dir / page.file
        if p.exists():
            paths.append(str(_prepare(p, tmp_dir)))
    if not paths:
        raise ValueError("페이지 파일을 찾을 수 없습니다.")

    kwargs = {}
    layout = _layout(size)
    if layout is not None:
        kwargs["layout_fun"] = layout

    data = img2pdf.convert(paths, **kwargs)
    out_path.write_bytes(data)

    if tmp_dir.exists():
        for f in tmp_dir.iterdir():
            f.unlink(missing_ok=True)
        tmp_dir.rmdir()
    return out_path
