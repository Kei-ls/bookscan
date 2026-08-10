"""카메라 코드 경로 검증 (실제 장치 없이).

이 PC에는 웹캠이 없어서 실제 촬영은 테스트할 수 없다. 대신 가짜 프레임을
주입해 장치 I/O를 뺀 나머지 — 오버레이 그리기, MJPEG 스트림 생성,
촬영 → 파이프라인 → 페이지 추가 — 가 동작하는지 확인한다.

사용: python tools/test_camera_path.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

from bookscan import camera as cam_mod, server  # noqa: E402
from bookscan.imgio import imread  # noqa: E402
from bookscan.store import Project, delete_project  # noqa: E402

SYNTH = Path("tests/synthetic/flat_dark_bg.jpg")
ROOT = Path("output")


class FakeCamera(cam_mod.Camera):
    """실제 장치 대신 합성 사진 한 장을 계속 내보내는 카메라."""

    def __init__(self, frame: np.ndarray) -> None:
        super().__init__()
        self._fake = frame
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def index(self):
        return 0 if self._open else None

    def open(self, index: int, **kw) -> None:
        self._open = True

    def latest(self):
        return self._fake.copy() if self._open else None

    def close(self) -> None:
        self._open = False


def main() -> int:
    frame = imread(SYNTH)
    fake = FakeCamera(frame)
    ok = True

    # --- 1. 오버레이 그리기 ---
    from bookscan.detect import detect_page
    from bookscan.imgio import fit_long_edge

    small, scale = fit_long_edge(frame, cam_mod.OVERLAY_WORK)
    det = detect_page(small)
    drawn = cam_mod.draw_overlay(frame, det.corners / scale, det.confidence)
    assert drawn.shape == frame.shape, "오버레이가 프레임 크기를 바꿈"
    assert not np.array_equal(drawn, frame), "오버레이가 아무것도 그리지 않음"
    print(f"1) 오버레이       OK  (검출 {det.method}, conf {det.confidence:.2f})")

    # 아무것도 못 찾았을 때도 죽지 않아야 한다
    cam_mod.draw_overlay(frame, None, 0.0)
    print("2) 미검출 오버레이 OK  (quad=None에서 예외 없음)")

    # --- 3. MJPEG 스트림 ---
    fake.open(0)
    stream = cam_mod.OverlayStream(fake)
    gen = stream.frames()
    chunks = [next(gen) for _ in range(9)]  # 프레임 3장 분량
    blob = b"".join(chunks)
    n_frames = blob.count(b"--frame")
    has_jpeg = blob.count(b"\xff\xd8\xff") >= 3  # JPEG 시작 마커
    gen.close()
    assert n_frames == 3 and has_jpeg, f"MJPEG 형식 이상 (frames={n_frames}, jpeg={has_jpeg})"
    print(f"3) MJPEG 스트림   OK  ({n_frames}프레임, {len(blob)/1024:.0f} KB)")

    # --- 4. 촬영 → 파이프라인 → 페이지 추가 ---
    delete_project("_camtest", ROOT)
    project = Project("_camtest", ROOT)
    server._state["project"] = project
    server.CAMERA = fake
    cam_backup = server.CAMERA

    with TestClient(server.app) as client:
        r = client.get("/api/cameras")
        assert r.status_code == 200, r.text
        print(f"4) /api/cameras   OK  (devices={len(r.json()['devices'])})")

        before = len(project.order)
        r = client.post("/api/camera/capture")
        assert r.status_code == 200, f"촬영 실패: {r.status_code} {r.text}"
        data = r.json()
        after = len(data["pages"])
        assert after == before + 1, f"페이지가 안 늘었다: {before} -> {after}"
        page = data["pages"][-1]
        print(f"5) 촬영           OK  ({page['filename']}, {page['width']}x{page['height']}, "
              f"conf {page['confidence']:.2f})")
        assert (project.dir / page["file"]).exists(), "결과 파일이 없다"
        print(f"6) 결과 파일      OK  ({page['file']})")

        # 카메라가 닫힌 상태에서는 400을 돌려줘야 한다 (500으로 터지면 안 됨)
        fake.close()
        r = client.post("/api/camera/capture")
        assert r.status_code == 400, f"닫힌 카메라에서 {r.status_code} (400이어야 함)"
        r = client.get("/api/camera/stream")
        assert r.status_code == 400, f"닫힌 스트림에서 {r.status_code} (400이어야 함)"
        print("7) 닫힌 상태 처리 OK  (500이 아니라 400 반환)")

    delete_project("_camtest", ROOT)
    print("\n판정:", "통과" if ok else "실패")
    print("주의: 실제 장치 열기(cv2.VideoCapture)는 카메라가 없어 검증하지 못함.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
