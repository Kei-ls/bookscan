"""FastAPI 서버.

이미지 처리는 CPU를 오래 쓰므로 전부 스레드풀로 넘긴다. OpenCV는 대부분의
연산에서 GIL을 놓아주기 때문에 이 방식이 실제로 효과가 있고, 처리 중에도
이벤트 루프가 막히지 않아 UI가 계속 반응한다.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .camera import Camera, OverlayStream, list_devices
from .config import INPUT_DIR, OUTPUT_DIR, WEB_DIR, PageSettings
from .imgio import decode_image
from .pdf import PAGE_SIZES, build_pdf
from .store import Project, delete_project, list_projects

app = FastAPI(title="Bookscan")

POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bookscan")
CAMERA = Camera()

_state: dict[str, Any] = {"project": None}
_state_lock = threading.Lock()

# 진행 상황을 브라우저로 흘려보내는 통로 (SSE)
_events: "queue.Queue[dict]" = queue.Queue()


def emit(kind: str, **data: Any) -> None:
    _events.put({"type": kind, **data})


def current_project() -> Project:
    with _state_lock:
        if _state["project"] is None:
            names = list_projects()
            _state["project"] = Project(names[0] if names else "내 스캔")
        return _state["project"]


async def in_pool(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(POOL, lambda: fn(*args, **kwargs))


# --------------------------------------------------------------------- 직렬화
def source_json(project: Project, sid: str) -> dict:
    s = project.sources[sid]
    return {
        "id": s.id,
        "filename": s.filename,
        "confidence": s.confidence,
        "method": s.method,
        "is_spread": s.is_spread,
        "gutter": s.gutter,
        "corners": s.corners,
        "notes": s.notes,
        "settings": s.settings.to_dict(),
    }


def page_json(project: Project, pid: str) -> dict:
    p = project.pages[pid]
    src = project.sources.get(p.source)
    return {
        "id": p.id,
        "source": p.source,
        "half": p.half,
        "file": p.file,
        "width": p.width,
        "height": p.height,
        "version": p.version,
        "notes": p.notes,
        "index": project.order.index(p.id) + 1 if p.id in project.order else 0,
        "confidence": src.confidence if src else 0.0,
        "filename": src.filename if src else "",
    }


def snapshot(project: Project) -> dict:
    return {
        "project": project.name,
        "dir": str(project.dir),
        "pages": [page_json(project, pid) for pid in project.order],
        "sources": {sid: source_json(project, sid) for sid in project.sources},
    }


# --------------------------------------------------------------------- 정적 파일
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# --------------------------------------------------------------------- 프로젝트
@app.get("/api/state")
async def get_state() -> dict:
    project = current_project()
    return {**snapshot(project), "projects": list_projects(),
            "default_input": str(INPUT_DIR), "output_root": str(OUTPUT_DIR)}


class ProjectBody(BaseModel):
    name: str


@app.post("/api/projects")
async def switch_project(body: ProjectBody) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "프로젝트 이름이 비어 있습니다.")
    if any(c in name for c in '\\/:*?"<>|'):
        raise HTTPException(400, '이름에 \\ / : * ? " < > | 는 쓸 수 없습니다.')
    with _state_lock:
        _state["project"] = Project(name)
    return await get_state()


@app.delete("/api/projects/{name}")
async def remove_project(name: str) -> dict:
    await in_pool(delete_project, name)
    with _state_lock:
        if _state["project"] is not None and _state["project"].name == name:
            _state["project"] = None
    return await get_state()


# --------------------------------------------------------------------- 가져오기
class ImportBody(BaseModel):
    folder: str


def _process_all(project: Project, sids: list[str]) -> None:
    total = len(sids)
    for i, sid in enumerate(sids, start=1):
        name = project.sources[sid].filename
        emit("progress", done=i - 1, total=total, current=name)
        try:
            project.process_source(sid)
        except Exception as exc:
            emit("error", filename=name, message=str(exc))
    emit("progress", done=total, total=total, current="")
    emit("done")


@app.post("/api/import/folder")
async def import_folder(body: ImportBody) -> dict:
    folder = Path(body.folder.strip('" ')).expanduser()
    if not folder.is_dir():
        raise HTTPException(400, f"폴더를 찾을 수 없습니다: {folder}")
    project = current_project()
    sources = await in_pool(project.import_folder, folder)
    if not sources:
        raise HTTPException(400, "폴더에 읽을 수 있는 이미지가 없습니다.")
    sids = [s.id for s in sources]
    asyncio.get_running_loop().run_in_executor(POOL, _process_all, project, sids)
    return {"imported": len(sids)}


@app.post("/api/import/upload")
async def import_upload(files: list[UploadFile]) -> dict:
    project = current_project()
    sids = []
    for f in files:
        data = await f.read()
        try:
            image = await in_pool(decode_image, data)
        except Exception:
            continue
        src = await in_pool(project.add_source, image, f.filename or "upload")
        sids.append(src.id)
    if not sids:
        raise HTTPException(400, "읽을 수 있는 이미지가 없습니다.")
    asyncio.get_running_loop().run_in_executor(POOL, _process_all, project, sids)
    return {"imported": len(sids)}


@app.get("/api/events")
async def events() -> StreamingResponse:
    """처리 진행률을 SSE로 흘려보낸다."""

    async def gen():
        while True:
            try:
                item = await asyncio.get_running_loop().run_in_executor(
                    POOL, _events.get, True, 25.0
                )
            except queue.Empty:
                yield b": keepalive\n\n"
                continue
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n".encode()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------- 이미지
def _send(path: Path, media: str = "image/jpeg") -> FileResponse:
    if not path.exists():
        raise HTTPException(404, "파일이 없습니다.")
    mt = "image/png" if path.suffix.lower() == ".png" else media
    return FileResponse(path, media_type=mt,
                        headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/api/pages/{pid}/thumb")
async def page_thumb(pid: str):
    project = current_project()
    if pid not in project.pages:
        raise HTTPException(404, "페이지가 없습니다.")
    return _send(await in_pool(project.thumb_path, pid))


@app.get("/api/pages/{pid}/full")
async def page_full(pid: str):
    project = current_project()
    if pid not in project.pages:
        raise HTTPException(404, "페이지가 없습니다.")
    return _send(project.page_path(pid))


@app.get("/api/sources/{sid}/original")
async def source_original(sid: str):
    project = current_project()
    if sid not in project.sources:
        raise HTTPException(404, "원본이 없습니다.")
    return _send(project.original_path(sid))


# --------------------------------------------------------------------- 재처리
class ReprocessBody(BaseModel):
    corners: list[list[float]] | None = None
    mode: str | None = None
    rotate: int | None = None
    auto_rotate: bool | None = None
    dewarp: bool | None = None
    trim: bool | None = None
    spread: str | None = None
    gutter: float | None = None
    brightness: int | None = None
    contrast: int | None = None
    reset_corners: bool = False
    reset_gutter: bool = False


@app.post("/api/sources/{sid}/reprocess")
async def reprocess(sid: str, body: ReprocessBody) -> dict:
    project = current_project()
    if sid not in project.sources:
        raise HTTPException(404, "원본이 없습니다.")
    st: PageSettings = project.sources[sid].settings

    if body.reset_corners:
        st.corners = None
    elif body.corners is not None:
        if len(body.corners) != 4:
            raise HTTPException(400, "코너는 4개여야 합니다.")
        st.corners = [[float(x), float(y)] for x, y in body.corners]
    if body.reset_gutter:
        st.gutter = None
    elif body.gutter is not None:
        st.gutter = float(body.gutter)
    if body.mode is not None:
        if body.mode not in ("color", "gray", "bw"):
            raise HTTPException(400, f"알 수 없는 모드: {body.mode}")
        st.mode = body.mode
    if body.rotate is not None:
        st.rotate = int(body.rotate) % 360
    if body.auto_rotate is not None:
        st.auto_rotate = bool(body.auto_rotate)
    if body.dewarp is not None:
        st.dewarp = bool(body.dewarp)
    if body.trim is not None:
        st.trim = bool(body.trim)
    if body.spread is not None:
        if body.spread not in ("auto", "single", "split"):
            raise HTTPException(400, f"알 수 없는 분할 설정: {body.spread}")
        st.spread = body.spread
    if body.brightness is not None:
        st.brightness = int(body.brightness)
    if body.contrast is not None:
        st.contrast = int(body.contrast)

    await in_pool(project.process_source, sid)
    return snapshot(project)


class RotateAllBody(BaseModel):
    #: 모든 원본의 수동 회전에 더할 각도 (보통 180)
    delta: int = 180


@app.post("/api/sources/rotate_all")
async def rotate_all(body: RotateAllBody) -> dict:
    """묶음 전체를 같은 각도만큼 더 돌린다.

    축(가로/세로)은 자동으로 맞지만 상하는 판별하지 않는다 — 시도했다가 반증했고
    자세한 근거는 orient.py에 적어 뒀다. 대신 한 번에 찍은 사진들은 상하가 모두
    같으므로, 페이지마다 누르는 대신 묶음을 통째로 뒤집게 한다.
    """
    if body.delta % 90:
        raise HTTPException(400, "90도 단위만 가능합니다.")
    project = current_project()
    sids = list(project.sources)
    if not sids:
        raise HTTPException(400, "처리할 원본이 없습니다.")
    for sid in sids:
        st = project.sources[sid].settings
        st.rotate = (st.rotate + int(body.delta)) % 360
    asyncio.get_running_loop().run_in_executor(POOL, _process_all, project, sids)
    return {"rotated": len(sids), "delta": int(body.delta) % 360}


class ReorderBody(BaseModel):
    order: list[str]


@app.post("/api/pages/reorder")
async def reorder(body: ReorderBody) -> dict:
    project = current_project()
    await in_pool(project.reorder, body.order)
    return snapshot(project)


@app.delete("/api/pages/{pid}")
async def delete_page(pid: str) -> dict:
    project = current_project()
    if pid not in project.pages:
        raise HTTPException(404, "페이지가 없습니다.")
    await in_pool(project.delete_page, pid)
    return snapshot(project)


@app.delete("/api/sources/{sid}")
async def delete_source(sid: str) -> dict:
    project = current_project()
    if sid not in project.sources:
        raise HTTPException(404, "원본이 없습니다.")
    await in_pool(project.delete_source, sid)
    return snapshot(project)


# --------------------------------------------------------------------- PDF
class PdfBody(BaseModel):
    filename: str | None = None
    size: str = "auto"


@app.post("/api/pdf")
async def make_pdf(body: PdfBody) -> dict:
    project = current_project()
    if body.size not in PAGE_SIZES:
        raise HTTPException(400, f"알 수 없는 용지 크기: {body.size}")
    try:
        path = await in_pool(build_pdf, project, body.filename, body.size)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"path": str(path), "name": path.name, "size": path.stat().st_size,
            "pages": len(project.order)}


@app.get("/api/pdf/download")
async def download_pdf(name: str):
    project = current_project()
    # `../..` 같은 이름으로 프로젝트 폴더 밖 파일을 읽어가지 못하게 막는다
    path = (project.dir / name).resolve()
    try:
        path.relative_to(project.dir.resolve())
    except ValueError:
        raise HTTPException(400, "잘못된 경로입니다.") from None
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(404, "PDF가 없습니다.")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


class RevealBody(BaseModel):
    path: str | None = None


@app.post("/api/reveal")
async def reveal(body: RevealBody) -> dict:
    """탐색기에서 출력 폴더 열기."""
    import subprocess

    project = current_project()
    target = Path(body.path) if body.path else project.dir
    if not target.exists():
        raise HTTPException(404, "경로가 없습니다.")
    try:
        if target.is_file():
            subprocess.Popen(["explorer", "/select,", str(target)])
        else:
            subprocess.Popen(["explorer", str(target)])
    except Exception as exc:
        raise HTTPException(500, f"폴더를 열 수 없습니다: {exc}") from exc
    return {"ok": True}


# --------------------------------------------------------------------- 카메라
@app.get("/api/cameras")
async def cameras() -> dict:
    devices = await in_pool(list_devices)
    return {
        "devices": [{"index": d.index, "name": d.name} for d in devices],
        "active": CAMERA.index,
        "hint": ("아이폰을 쓰려면 Camo / Iriun / EpocCam 같은 앱으로 Windows 가상 "
                 "카메라로 연결해야 목록에 나타납니다."),
    }


class CameraBody(BaseModel):
    index: int


@app.post("/api/camera/open")
async def camera_open(body: CameraBody) -> dict:
    try:
        await in_pool(CAMERA.open, body.index)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "active": CAMERA.index}


@app.post("/api/camera/close")
async def camera_close() -> dict:
    await in_pool(CAMERA.close)
    return {"ok": True}


@app.get("/api/camera/stream")
async def camera_stream():
    if not CAMERA.is_open:
        raise HTTPException(400, "카메라가 열려 있지 않습니다.")
    stream = OverlayStream(CAMERA)
    return StreamingResponse(stream.frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/camera/capture")
async def camera_capture() -> dict:
    if not CAMERA.is_open:
        raise HTTPException(400, "카메라가 열려 있지 않습니다.")
    frame = CAMERA.latest()
    if frame is None:
        raise HTTPException(400, "영상이 들어오지 않습니다.")
    project = current_project()
    n = len([s for s in project.sources.values() if s.filename.startswith("촬영")]) + 1
    src = await in_pool(project.add_source, frame, f"촬영 {n:03d}")
    await in_pool(project.process_source, src.id)
    return snapshot(project)


@app.on_event("shutdown")
def _shutdown() -> None:
    CAMERA.close()
    POOL.shutdown(wait=False)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()
