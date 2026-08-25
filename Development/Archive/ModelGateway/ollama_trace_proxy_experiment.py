from __future__ import annotations

import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 11435
UPSTREAM = "http://127.0.0.1:11434"

request_number = 0


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        global request_number
        request_number += 1
        current = request_number

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)

        try:
            payload = json.loads(raw_body)
        except Exception:
            payload = {}

        # Diagnostic experiment:
        # 1) disable reasoning
        # 2) remove tool schemas
        # Everything else remains unchanged.
        payload["reasoning_effort"] = "none"
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

        messages = payload.get("messages", [])
        tools = payload.get("tools", [])
        stream = payload.get("stream", False)

        message_json = json.dumps(messages, ensure_ascii=False)
        tool_json = json.dumps(tools, ensure_ascii=False)

        # Ask Ollama to include token usage in the stream.
        # This changes response metadata only, not the model prompt.
        if stream and self.path.endswith("/v1/chat/completions"):
            payload["stream_options"] = {
                **payload.get("stream_options", {}),
                "include_usage": True,
            }

            # Diagnostic experiment: disable model reasoning while leaving
            # the prompt, messages, tools, and all other request data unchanged.
            raw_body = json.dumps(payload).encode("utf-8")

        raw_body = json.dumps(payload).encode("utf-8")

        print()
        print("=" * 72)
        print(f"REQUEST {current}")
        print(f"Path:               {self.path}")
        print(f"Model:              {payload.get('model')}")
        print(f"Messages:           {len(messages)}")
        print(f"Message JSON chars: {len(message_json)}")
        print(f"Tools supplied:     {len(tools)}")
        print(f"Tool JSON chars:    {len(tool_json)}")
        print(f"Whole body chars:   {len(raw_body)}")
        print(f"Stream:             {stream}")
        print(f"reasoning_effort:   {payload.get('reasoning_effort', '<not supplied>')}")
        print(f"reasoning:          {payload.get('reasoning', '<not supplied>')}")
        print("=" * 72, flush=True)

        request = urllib.request.Request(
            UPSTREAM + self.path,
            data=raw_body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()

        first_chunk_time = None
        first_reasoning_time = None
        first_content_time = None

        reasoning_chars = 0
        content_chars = 0
        tool_call_chars = 0

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        try:
            with urllib.request.urlopen(request) as upstream:
                self.send_response(upstream.status)
                self.send_header(
                    "Content-Type",
                    upstream.headers.get(
                        "Content-Type",
                        "application/json",
                    ),
                )
                self.send_header("Connection", "close")
                self.end_headers()

                if stream:
                    for raw_line in upstream:
                        self.wfile.write(raw_line)
                        self.wfile.flush()

                        line = raw_line.decode(
                            "utf-8",
                            errors="replace",
                        ).strip()

                        if not line.startswith("data:"):
                            continue

                        data = line[5:].strip()

                        if data == "[DONE]":
                            continue

                        try:
                            event = json.loads(data)
                        except Exception:
                            continue

                        now = time.perf_counter()

                        if first_chunk_time is None:
                            first_chunk_time = now - started

                        usage = event.get("usage")
                        if usage:
                            prompt_tokens = usage.get(
                                "prompt_tokens",
                                prompt_tokens,
                            )
                            completion_tokens = usage.get(
                                "completion_tokens",
                                completion_tokens,
                            )
                            total_tokens = usage.get(
                                "total_tokens",
                                total_tokens,
                            )

                        for choice in event.get("choices", []):
                            delta = choice.get("delta") or {}

                            content = delta.get("content") or ""
                            if content:
                                content_chars += len(content)
                                if first_content_time is None:
                                    first_content_time = (
                                        now - started
                                    )

                            # Ollama/OpenAI-compatible versions have
                            # used more than one reasoning field name.
                            reasoning_parts = []

                            for key, value in delta.items():
                                if (
                                    "reason" in key.lower()
                                    or "think" in key.lower()
                                ):
                                    if isinstance(value, str):
                                        reasoning_parts.append(value)

                            reasoning_text = "".join(reasoning_parts)

                            if reasoning_text:
                                reasoning_chars += len(
                                    reasoning_text
                                )
                                if first_reasoning_time is None:
                                    first_reasoning_time = (
                                        now - started
                                    )

                            calls = delta.get("tool_calls") or []
                            if calls:
                                tool_call_chars += len(
                                    json.dumps(calls)
                                )

                else:
                    data = upstream.read()
                    self.wfile.write(data)
                    self.wfile.flush()

                wall = time.perf_counter() - started

                print()
                print(f"RESPONSE {current}")
                print(f"First chunk:        {first_chunk_time}")
                print(f"First reasoning:    {first_reasoning_time}")
                print(f"First content:      {first_content_time}")
                print(f"Reasoning chars:    {reasoning_chars}")
                print(f"Content chars:      {content_chars}")
                print(f"Tool-call chars:    {tool_call_chars}")
                print(f"Prompt tokens:      {prompt_tokens}")
                print(f"Completion tokens:  {completion_tokens}")
                print(f"Total tokens:       {total_tokens}")
                print(f"Proxy wall time:    {wall:.3f}")
                print("=" * 72, flush=True)

        except Exception as exc:
            print(f"PROXY ERROR: {exc}", flush=True)
            self.send_error(502, str(exc))

        self.close_connection = True


if __name__ == "__main__":
    server = ThreadingHTTPServer(
        (LISTEN_HOST, LISTEN_PORT),
        ProxyHandler,
    )

    print(
        f"Passive Ollama trace proxy: "
        f"http://{LISTEN_HOST}:{LISTEN_PORT}"
    )
    print(f"Forwarding unchanged to: {UPSTREAM}")
    print("Ctrl+C to stop.")

    server.serve_forever()