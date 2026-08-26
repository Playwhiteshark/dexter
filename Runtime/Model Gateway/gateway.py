from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 11435
UPSTREAM = "http://127.0.0.1:11434"

DEFAULT_REASONING_EFFORT = os.environ.get(
    "DEXTER_REASONING_EFFORT",
    "none",
).strip() or "none"
PRODUCTION_MODEL = os.environ.get(
    "DEXTER_MODEL",
    "gemma4-jarvis:12b",
).strip() or "gemma4-jarvis:12b"
PRELOAD_KEEP_ALIVE = "5m"
PRELOAD_TIMEOUT_SECONDS = 120

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def preload_model() -> tuple[bool, str]:
    payload = json.dumps(
        {
            "model": PRODUCTION_MODEL,
            "prompt": "",
            "stream": False,
            "keep_alive": PRELOAD_KEEP_ALIVE,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        UPSTREAM + "/api/generate",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=PRELOAD_TIMEOUT_SECONDS,
        ) as response:
            response.read()
        return True, f"{PRODUCTION_MODEL} is loaded"
    except Exception as exc:
        return False, f"Preload failed: {exc}"


def preload_model_background() -> None:
    def worker():
        ok, message = preload_model()
        prefix = "Preload OK" if ok else "Preload warning"
        print(f"{prefix}: {message}", flush=True)

    threading.Thread(
        target=worker,
        name="dexter-model-preload",
        daemon=True,
    ).start()


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _forward(self, body: bytes | None = None):
        target = UPSTREAM + self.path

        headers = {}
        for key, value in self.headers.items():
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            if key.lower() == "host":
                continue
            headers[key] = value

        if body is not None:
            headers["Content-Length"] = str(len(body))

        request = urllib.request.Request(
            target,
            data=body,
            method=self.command,
            headers=headers,
        )

        try:
            with urllib.request.urlopen(request) as upstream:
                self.send_response(upstream.status)

                for key, value in upstream.headers.items():
                    lower = key.lower()
                    if lower in HOP_BY_HOP_HEADERS:
                        continue
                    if lower == "content-length":
                        continue
                    self.send_header(key, value)

                self.send_header("Connection", "close")
                self.end_headers()

                content_type = upstream.headers.get("Content-Type", "")

                if "text/event-stream" in content_type.lower():
                    # Forward SSE lines immediately so Goose sees streamed
                    # reasoning/content tokens without waiting for a large buffer.
                    for line in upstream:
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    while True:
                        chunk = upstream.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()

        except urllib.error.HTTPError as exc:
            self.send_response(exc.code)

            for key, value in exc.headers.items():
                lower = key.lower()
                if lower in HOP_BY_HOP_HEADERS or lower == "content-length":
                    continue
                self.send_header(key, value)

            self.send_header("Connection", "close")
            self.end_headers()

            error_body = exc.read()
            if error_body:
                self.wfile.write(error_body)
                self.wfile.flush()

        except Exception as exc:
            self.send_error(502, f"Dexter model gateway error: {exc}")

        self.close_connection = True

    def do_GET(self):
        self._forward()

    def do_HEAD(self):
        self._forward()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)

        if self.path.endswith("/v1/chat/completions"):
            try:
                payload = json.loads(raw_body)
            except Exception:
                payload = None

            if isinstance(payload, dict):
                # Normal requests default to no reasoning.
                # If a caller explicitly supplies reasoning_effort later
                # (e.g. a future think_deeply path), preserve that value.
                payload.setdefault(
                    "reasoning_effort",
                    DEFAULT_REASONING_EFFORT,
                )
                raw_body = json.dumps(payload).encode("utf-8")

        self._forward(raw_body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(
        (LISTEN_HOST, LISTEN_PORT),
        GatewayHandler,
    )

    print(
        f"Dexter Model Gateway listening on "
        f"http://{LISTEN_HOST}:{LISTEN_PORT}"
    )
    print(f"Forwarding to {UPSTREAM}")
    print(
        f"Default reasoning_effort: "
        f"{DEFAULT_REASONING_EFFORT}"
    )
    print(
        f"Preload model: {PRODUCTION_MODEL} "
        f"(keep alive {PRELOAD_KEEP_ALIVE})"
    )
    print("Ctrl+C to stop.")

    preload_model_background()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
