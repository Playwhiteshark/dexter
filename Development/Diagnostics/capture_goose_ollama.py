from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 11436
UPSTREAM = "http://127.0.0.1:11434"
OUT_DIR = Path.cwd() / "goose_ollama_capture"
OUT_DIR.mkdir(exist_ok=True)

SKIP_REQUEST_HEADERS = {
    "host", "content-length", "connection", "accept-encoding"
}
SKIP_RESPONSE_HEADERS = {
    "transfer-encoding", "content-length", "connection", "content-encoding"
}


def stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def save_json(name: str, obj):
    path = OUT_DIR / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_HEAD(self):
        self._proxy()

    def _proxy(self):
        body = b""
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n:
            body = self.rfile.read(n)

        url = UPSTREAM + self.path
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in SKIP_REQUEST_HEADERS
        }

        request_obj = None
        if body:
            try:
                request_obj = json.loads(body.decode("utf-8"))
            except Exception:
                request_obj = None

        token = stamp()
        if request_obj is not None:
            save_json(f"{token}_request.json", request_obj)

            if self.path.startswith("/api/chat"):
                save_json("latest_goose_chat_request.json", request_obj)
                messages = request_obj.get("messages") or []
                tools = request_obj.get("tools") or []

                print("\n" + "=" * 78)
                print("CAPTURED /api/chat")
                print("=" * 78)
                print(f"model: {request_obj.get('model')}")
                print(f"messages: {len(messages)} | tools: {len(tools)}")
                for i, msg in enumerate(messages):
                    role = msg.get("role")
                    content = msg.get("content")
                    if isinstance(content, str):
                        print(f"  message[{i}] role={role!r} chars={len(content)}")
                    else:
                        print(f"  message[{i}] role={role!r} content_type={type(content).__name__}")

                system_parts = [
                    m.get("content", "")
                    for m in messages
                    if m.get("role") in ("system", "developer")
                    and isinstance(m.get("content"), str)
                ]
                system_text = "\n\n".join(system_parts)
                (OUT_DIR / "latest_goose_system_prompt.txt").write_text(
                    system_text, encoding="utf-8"
                )
                print(f"system/developer chars: {len(system_text)}")
                print(f"saved: {OUT_DIR / 'latest_goose_chat_request.json'}")
                print(f"saved: {OUT_DIR / 'latest_goose_system_prompt.txt'}")
                print("=" * 78)

        req = urllib.request.Request(
            url,
            data=body if self.command in ("POST", "PUT", "PATCH") else None,
            headers=headers,
            method=self.command,
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=600) as upstream:
                response_body = upstream.read()
                status = upstream.status
                response_headers = upstream.headers
        except urllib.error.HTTPError as exc:
            response_body = exc.read()
            status = exc.code
            response_headers = exc.headers
        except Exception as exc:
            payload = json.dumps({"error": f"capture proxy upstream failure: {exc}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            print(f"UPSTREAM ERROR {self.path}: {exc}", file=sys.stderr)
            return

        elapsed = time.perf_counter() - started

        response_obj = None
        try:
            response_obj = json.loads(response_body.decode("utf-8"))
        except Exception:
            pass

        if response_obj is not None:
            save_json(f"{token}_response.json", response_obj)
            if self.path.startswith("/api/chat"):
                save_json("latest_goose_chat_response.json", response_obj)
                print(f"response: HTTP {status}, {len(response_body)} bytes, {elapsed:.2f}s")
                print(f"saved: {OUT_DIR / 'latest_goose_chat_response.json'}")

        self.send_response(status)
        for k, v in response_headers.items():
            if k.lower() not in SKIP_RESPONSE_HEADERS:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(response_body)


def main():
    print("=" * 78)
    print("GOOSE -> OLLAMA CAPTURE PROXY")
    print("=" * 78)
    print(f"Listening : http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"Forwarding: {UPSTREAM}")
    print(f"Capture dir: {OUT_DIR}")
    print()
    print("This does not modify Goose or Ollama. It simply logs and forwards requests.")
    print("Press Ctrl+C after the Goose test.")
    print()

    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
