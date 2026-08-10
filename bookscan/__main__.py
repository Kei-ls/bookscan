"""`python -m bookscan` 진입점."""

from __future__ import annotations

import argparse

from .server import serve


def main() -> None:
    ap = argparse.ArgumentParser(prog="bookscan", description="책/문서 사진 스캔 후처리 + PDF")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않음")
    args = ap.parse_args()
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
