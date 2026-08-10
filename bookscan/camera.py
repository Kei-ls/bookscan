"""카메라 캡처.

브라우저의 getUserMedia 대신 서버에서 cv2.VideoCapture로 프레임을 잡고 MJPEG로
내보낸다. 그래야 검출 로직(detect.py)을 그대로 재사용해 라이브 화면에 페이지
테두리를 겹쳐 그릴 수 있다.

전용 스레드가 계속 프레임을 읽어 '최신 한 장'만 들고 있는다. 카메라 내부 버퍼에
프레임이 쌓여 화면이 밀리는 걸 막기 위한 구조다.

아이폰을 쓰려면 Camo / Iriun / EpocCam 같은 앱으로 Windows 가상 카메라로
잡혀야 한다. Windows 11에는 macOS의 연속성 카메라 같은 기본 기능이 없다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .detect import detect_page
from .imgio import encode_jpeg, fit_long_edge

#: 오버레이용 검출은 이 해상도에서 (전체 해상도로 하면 프레임이 밀린다)
OVERLAY_WORK = 640
#: 몇 프레임마다 검출을 다시 할지
DETECT_EVERY = 4


@dataclass
class DeviceInfo:
    index: int
    name: str


def list_devices(max_probe: int = 6) -> list[DeviceInfo]:
    """연결된 카메라 목록. 이름을 못 얻으면 인덱스만이라도 돌려준다."""
    try:
        from pygrabber.dshow_graph import FilterGraph

        names = FilterGraph().get_input_devices()
        if names:
            return [DeviceInfo(i, n) for i, n in enumerate(names)]
    except Exception:
        pass

    # pygrabber를 못 쓰면 직접 열어보며 탐색 (느리므로 최후 수단)
    found = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                found.append(DeviceInfo(i, f"카메라 {i}"))
        finally:
            cap.release()
    return found


class Camera:
    """카메라 하나를 열고 최신 프레임을 계속 갱신한다."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._index: int | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: str | None = None

    @property
    def index(self) -> int | None:
        return self._index

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def error(self) -> str | None:
        return self._error

    def open(self, index: int, width: int = 1920, height: int = 1080) -> None:
        self.close()
        self._error = None
        # Windows에서는 DSHOW가 기본 백엔드보다 훨씬 빨리 열린다
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            self._error = (f"카메라 {index}번을 열 수 없습니다. 다른 앱이 쓰고 있거나 "
                           f"연결이 끊겼는지 확인하세요.")
            raise RuntimeError(self._error)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._cap = cap
        self._index = index
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        # 첫 프레임이 들어올 때까지 잠깐 기다린다 (웹캠 워밍업)
        deadline = time.time() + 4.0
        while time.time() < deadline and self.latest() is None:
            time.sleep(0.05)
        if self.latest() is None:
            self.close()
            self._error = "카메라를 열었지만 영상이 들어오지 않습니다."
            raise RuntimeError(self._error)

    def _loop(self) -> None:
        while not self._stop.is_set() and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            with self._lock:
                self._frame = frame
        # 루프 종료

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t.is_alive():
            t.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._index = None
        with self._lock:
            self._frame = None


def draw_overlay(frame: np.ndarray, quad: np.ndarray | None, confidence: float) -> np.ndarray:
    """검출된 페이지 테두리를 프레임 위에 그린다."""
    # cv2.putText는 한글을 못 그린다(Hershey 폰트에 없음). 라벨은 ASCII로만 쓴다.
    out = frame.copy()
    if quad is None:
        cv2.putText(out, "searching...", (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (60, 60, 240), 2, cv2.LINE_AA)
        return out

    pts = quad.astype(np.int32).reshape(-1, 1, 2)
    good = confidence >= 0.6
    color = (80, 220, 80) if good else (60, 170, 240)  # 초록 / 주황(BGR)

    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
    cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
    for x, y in quad.astype(int):
        cv2.circle(out, (int(x), int(y)), 9, color, -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 9, (255, 255, 255), 2, cv2.LINE_AA)

    label = f"{confidence * 100:.0f}%"
    cv2.putText(out, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out


class OverlayStream:
    """MJPEG 프레임 생성기. 몇 프레임마다 한 번씩만 검출을 돌린다."""

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        self._quad: np.ndarray | None = None
        self._conf = 0.0
        self._n = 0

    def _update_detection(self, frame: np.ndarray) -> None:
        small, scale = fit_long_edge(frame, OVERLAY_WORK)
        try:
            det = detect_page(small)
        except Exception:
            return
        if det.method == "fallback":
            self._quad, self._conf = None, 0.0
            return
        self._quad = det.corners / scale
        self._conf = det.confidence

    def frames(self, quality: int = 75):
        boundary = b"--frame\r\n"
        while self.camera.is_open:
            frame = self.camera.latest()
            if frame is None:
                time.sleep(0.05)
                continue
            if self._n % DETECT_EVERY == 0:
                self._update_detection(frame)
            self._n += 1

            shown = draw_overlay(frame, self._quad, self._conf)
            shown, _ = fit_long_edge(shown, 1000)  # 전송량 절약
            jpg = encode_jpeg(shown, quality)
            yield boundary + b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpg)}\r\n\r\n".encode()
            yield jpg + b"\r\n"
            time.sleep(0.03)
