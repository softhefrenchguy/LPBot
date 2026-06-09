from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from lpbot.paper.paper_trade_v1.cli_paper_report import main as generate_report


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_log(event: str, **fields: object) -> None:
    payload = {"ts": _utc_now(), "event": event, **fields}
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve auto-updating paper report.")
    p.add_argument("--log-csv", default="artifacts/paper/paper_log.csv")
    p.add_argument("--out", default="artifacts/paper/report.html")
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--interval-seconds", type=int, default=3600)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--bind", default="0.0.0.0")
    return p.parse_args()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def _serve(directory: Path, bind: str, port: int) -> None:
    directory = directory.resolve()
    # Ensure the handler serves from the report directory
    import os
    os.chdir(directory)
    handler = _QuietHandler
    server = ThreadingHTTPServer((bind, port), handler)
    _json_log("report_server_start", bind=bind, port=port, dir=str(directory))
    server.serve_forever()


def _render_loop(args: argparse.Namespace) -> None:
    while True:
        try:
            _json_log("report_generate_start")
            # Run report generator as a subprocess to avoid blocking the server thread
            cmd = [
                "python",
                "-m",
                "lpbot.paper.paper_trade_v1.cli_paper_report",
                "--log-csv",
                args.log_csv,
                "--out",
                args.out,
                "--max-rows",
                str(args.max_rows),
            ]
            timeout_sec = max(60, min(900, args.interval_seconds - 5))
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/app",
                timeout=timeout_sec,
            )
            _json_log("report_generate_complete")
        except subprocess.TimeoutExpired:
            _json_log("report_generate_error", error="timeout")
        except subprocess.CalledProcessError as exc:
            _json_log("report_generate_error", error=exc.stderr.strip()[:1000])
        except Exception as exc:
            _json_log("report_generate_error", error=str(exc))
        time.sleep(args.interval_seconds)


def main() -> None:
    args = _parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = threading.Thread(target=_render_loop, args=(args,), daemon=True)
    t.start()

    serve_dir = out_path.parent
    _serve(serve_dir, args.bind, args.port)


if __name__ == "__main__":
    main()
