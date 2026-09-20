"""Serve a review directory with byte-range support for audio seeking.

Run with the ktv environment (Flask is required)::

    python tools/serve_review.py --directory temp/nextfire-full-3346334398
"""
from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, send_from_directory


def create_app(directory: Path) -> Flask:
    app = Flask(__name__, static_folder=None)
    root = directory.resolve()

    @app.get("/")
    @app.get("/<path:filename>")
    def serve(filename: str = "review.html"):
        # Werkzeug implements Content-Range, 206 and 416 responses, including
        # seeking into the retained audio symlinks used by comparison runs.
        response = send_from_directory(root, filename, conditional=True)
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Cache-Control"] = "no-cache"
        return response

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    if not (args.directory / "review.html").is_file():
        parser.error("directory must contain review.html")
    create_app(args.directory).run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
