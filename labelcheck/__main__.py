"""`python -m labelcheck` 진입점."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .app import app, store

# 윈도우 콘솔은 기본이 CP949라 em-dash(—) 같은 문자에서 그대로 죽는다. 한글은
# 멀쩡히 나가는데 문장부호 하나 때문에 프로그램이 안 뜨면 곤란하다.
# (bookscan은 콘솔에 아무것도 안 찍어서 이 문제를 피해 간다.)
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="labelcheck",
        description="학습 자료(사진 + 마스크)를 훑어보고 나쁜 라벨을 걸러낸다")
    ap.add_argument("labels", nargs="?", default="labels",
                    help="export_labels.py 가 만든 폴더 (기본 labels)")
    ap.add_argument("--host", default="127.0.0.1")
    # bookscan이 8765를 쓴다. 둘을 같이 띄워 놓고 오갈 수 있어야 한다.
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    root = Path(args.labels).resolve()
    if not root.is_dir():
        raise SystemExit(f"그런 폴더가 없습니다: {root}\n"
                         f"먼저 python tools/export_labels.py 를 돌리세요.")
    store.root = root
    store.load()
    if not store.order:
        raise SystemExit(f"{root} 에 (사진, 마스크) 쌍이 없습니다.")

    url = f"http://{args.host}:{args.port}/"
    print(f"라벨 확인: {root}  ({len(store.order)}장)")
    print(url)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
