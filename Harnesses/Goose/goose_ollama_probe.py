#!/usr/bin/env python3
"""
goose_ollama_probe.py

Transparent local reverse proxy for observing Goose -> Ollama traffic.

What it records:
- Every HTTP request/response, including full /v1/chat/completions JSON/SSE.
- Per-request timing: request start, upstream headers, first response bytes, end.
- Model-request fields: model, tools, messages, sampling/reasoning fields.
- Streaming content/reasoning/tool-call deltas where present.
- 1-second system/GPU samples (CPU, RAM, NVIDIA GPU util/VRAM/power/temp).
- Periodic Ollama /api/ps snapshots to detect model load/unload.
- The matching slice of Ollama's Windows server.log for llama.cpp timing details.
- A digestible summary.md + summary.json when the probe exits.

The proxy is observational: it does not intentionally alter JSON bodies or model settings.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import ctypes
import datetime as dt
import http.client
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


# ------------------------------- utilities -------------------------------

def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def safe_json_loads(data: bytes | str) -> Any | None:
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        return json.loads(data)
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def parse_duration_seconds(text: str) -> float | None:
    """Parse Go-ish durations such as 19.65s, 523ms, 512.7µs."""
    text = text.strip()
    m = re.fullmatch(r"([\d.]+)\s*(ns|us|µs|ms|s|m)", text)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2)
    factors = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0}
    return value * factors[unit]


def flatten_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


# ------------------------------- Windows system stats -------------------------------

class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _filetime_to_int(ft: FILETIME) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


class WindowsSystemSampler:
    def __init__(self) -> None:
        self.prev = None
        self.kernel32 = ctypes.windll.kernel32 if os.name == "nt" else None

    def sample(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if os.name != "nt":
            return out

        idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
        if self.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            vals = (_filetime_to_int(idle), _filetime_to_int(kernel), _filetime_to_int(user))
            if self.prev is not None:
                idle_d = vals[0] - self.prev[0]
                kernel_d = vals[1] - self.prev[1]
                user_d = vals[2] - self.prev[2]
                total = kernel_d + user_d
                if total > 0:
                    out["cpu_percent"] = round(100.0 * (total - idle_d) / total, 2)
            self.prev = vals

        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if self.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            out["ram_used_percent"] = int(mem.dwMemoryLoad)
            out["ram_total_mb"] = round(mem.ullTotalPhys / (1024**2), 1)
            out["ram_used_mb"] = round((mem.ullTotalPhys - mem.ullAvailPhys) / (1024**2), 1)
            out["pagefile_used_mb"] = round((mem.ullTotalPageFile - mem.ullAvailPageFile) / (1024**2), 1)
        return out


def sample_nvidia() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if p.returncode != 0 or not p.stdout.strip():
            return {}
        vals = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
        if len(vals) < 5:
            return {}
        def num(s: str) -> float | None:
            try:
                return float(s)
            except Exception:
                return None
        return {
            "gpu_util_percent": num(vals[0]),
            "gpu_mem_used_mb": num(vals[1]),
            "gpu_mem_total_mb": num(vals[2]),
            "gpu_power_w": num(vals[3]),
            "gpu_temp_c": num(vals[4]),
        }
    except Exception:
        return {}


def query_ollama_ps(host: str, port: int) -> list[dict[str, Any]] | None:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=2)
        conn.request("GET", "/api/ps")
        r = conn.getresponse()
        data = r.read()
        conn.close()
        obj = safe_json_loads(data)
        if isinstance(obj, dict) and isinstance(obj.get("models"), list):
            return obj["models"]
    except Exception:
        pass
    return None


# ------------------------------- response parsing -------------------------------

def parse_openai_response(body: bytes, content_type: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "format": "unknown",
        "content_chars": 0,
        "reasoning_chars": 0,
        "tool_call_count": 0,
        "finish_reasons": [],
        "usage": None,
        "content_preview": "",
        "reasoning_preview": "",
    }
    text = body.decode("utf-8", errors="replace")

    if "text/event-stream" in content_type.lower() or text.lstrip().startswith("data:"):
        result["format"] = "sse"
        contents: list[str] = []
        reasonings: list[str] = []
        tool_indices: set[Any] = set()
        usages = []
        finish_reasons = []

        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            obj = safe_json_loads(payload)
            if not isinstance(obj, dict):
                continue
            if obj.get("usage") is not None:
                usages.append(obj.get("usage"))
            for choice in obj.get("choices", []) or []:
                if choice.get("finish_reason"):
                    finish_reasons.append(choice["finish_reason"])
                delta = choice.get("delta") or {}
                c = delta.get("content")
                if isinstance(c, str):
                    contents.append(c)
                for key in ("reasoning", "reasoning_content", "thinking"):
                    rr = delta.get(key)
                    if isinstance(rr, str):
                        reasonings.append(rr)
                for tc in delta.get("tool_calls", []) or []:
                    if isinstance(tc, dict):
                        tool_indices.add(tc.get("index", len(tool_indices)))

        content = "".join(contents)
        reasoning = "".join(reasonings)
        result.update({
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
            "tool_call_count": len(tool_indices),
            "finish_reasons": finish_reasons,
            "usage": usages[-1] if usages else None,
            "content_preview": content[:500],
            "reasoning_preview": reasoning[:500],
        })
        return result

    obj = safe_json_loads(body)
    if isinstance(obj, dict):
        result["format"] = "json"
        choices = obj.get("choices") or []
        contents, reasonings = [], []
        tool_count = 0
        finishes = []
        for choice in choices:
            if choice.get("finish_reason"):
                finishes.append(choice["finish_reason"])
            msg = choice.get("message") or {}
            c = msg.get("content")
            if isinstance(c, str):
                contents.append(c)
            for key in ("reasoning", "reasoning_content", "thinking"):
                rr = msg.get(key)
                if isinstance(rr, str):
                    reasonings.append(rr)
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                tool_count += len(tcs)
        content = "".join(contents)
        reasoning = "".join(reasonings)
        result.update({
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
            "tool_call_count": tool_count,
            "finish_reasons": finishes,
            "usage": obj.get("usage"),
            "content_preview": content[:500],
            "reasoning_preview": reasoning[:500],
        })
    return result


def summarize_request_json(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    messages = obj.get("messages")
    tools = obj.get("tools")
    roles = []
    message_chars = 0
    system_chars = 0
    tool_result_chars = 0
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            roles.append(role)
            txt = flatten_text_content(m.get("content"))
            message_chars += len(txt)
            if role == "system":
                system_chars += len(txt)
            if role == "tool":
                tool_result_chars += len(txt)

    interesting = {}
    for k in (
        "model", "stream", "temperature", "top_p", "top_k", "min_p",
        "presence_penalty", "frequency_penalty", "repeat_penalty",
        "max_tokens", "max_completion_tokens", "reasoning_effort",
        "reasoning", "think", "stream_options", "response_format",
        "tool_choice", "parallel_tool_calls", "seed",
    ):
        if k in obj:
            interesting[k] = obj[k]

    return {
        **interesting,
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "message_roles": roles,
        "message_text_chars": message_chars,
        "system_text_chars": system_chars,
        "tool_result_text_chars": tool_result_chars,
        "tool_schema_count": len(tools) if isinstance(tools, list) else 0,
        "request_keys": sorted(obj.keys()),
    }


# ------------------------------- probe state -------------------------------

@dataclass
class RequestMetric:
    request_id: int
    method: str
    path: str
    started_at: str
    start_perf: float
    upstream_headers_ms: float | None = None
    first_body_byte_ms: float | None = None
    total_ms: float | None = None
    status: int | None = None
    request_bytes: int = 0
    response_bytes: int = 0
    content_type: str = ""
    request_summary: dict[str, Any] | None = None
    response_summary: dict[str, Any] | None = None
    error: str | None = None


class ProbeState:
    def __init__(self, run_dir: Path, upstream_host: str, upstream_port: int, sample_interval: float, ps_interval: float):
        self.run_dir = run_dir
        self.requests_dir = run_dir / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.sample_interval = sample_interval
        self.ps_interval = ps_interval
        self.started_wall = now_iso()
        self.started_perf = time.perf_counter()
        self.request_counter = 0
        self.lock = threading.Lock()
        self.metrics: list[RequestMetric] = []
        self.stop_event = threading.Event()
        self.finalized = False
        self.sys_sampler = WindowsSystemSampler()
        self.server_log_path = self._find_ollama_server_log()
        self.server_log_start = self.server_log_path.stat().st_size if self.server_log_path and self.server_log_path.exists() else 0

        self.events_path = run_dir / "events.jsonl"
        self.req_metrics_path = run_dir / "request_metrics.jsonl"
        self.perf_path = run_dir / "performance.csv"
        self.ollama_ps_path = run_dir / "ollama_ps.jsonl"

        with self.perf_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "elapsed_s", "cpu_percent", "ram_used_percent", "ram_used_mb", "ram_total_mb",
                "pagefile_used_mb", "gpu_util_percent", "gpu_mem_used_mb", "gpu_mem_total_mb",
                "gpu_power_w", "gpu_temp_c",
            ])
            writer.writeheader()

        self.event("probe_started", {
            "run_dir": str(run_dir),
            "upstream": f"http://{upstream_host}:{upstream_port}",
            "server_log": str(self.server_log_path) if self.server_log_path else None,
        })

    def _find_ollama_server_log(self) -> Path | None:
        localappdata = os.environ.get("LOCALAPPDATA")
        if not localappdata:
            return None
        p = Path(localappdata) / "Ollama" / "server.log"
        return p

    def next_request_id(self) -> int:
        with self.lock:
            self.request_counter += 1
            return self.request_counter

    def event(self, event_type: str, data: dict[str, Any]) -> None:
        rec = {"timestamp": now_iso(), "elapsed_s": round(time.perf_counter() - self.started_perf, 6), "event": event_type, **data}
        with self.lock:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def add_metric(self, metric: RequestMetric) -> None:
        with self.lock:
            self.metrics.append(metric)
            with self.req_metrics_path.open("a", encoding="utf-8") as f:
                d = asdict(metric)
                d.pop("start_perf", None)
                f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")

    def start_sampler(self) -> None:
        threading.Thread(target=self._sampler_loop, name="probe-sampler", daemon=True).start()

    def _sampler_loop(self) -> None:
        next_ps = 0.0
        while not self.stop_event.is_set():
            start = time.perf_counter()
            rec = {
                "timestamp": now_iso(),
                "elapsed_s": round(start - self.started_perf, 3),
                **self.sys_sampler.sample(),
                **sample_nvidia(),
            }
            fields = [
                "timestamp", "elapsed_s", "cpu_percent", "ram_used_percent", "ram_used_mb", "ram_total_mb",
                "pagefile_used_mb", "gpu_util_percent", "gpu_mem_used_mb", "gpu_mem_total_mb",
                "gpu_power_w", "gpu_temp_c",
            ]
            with self.lock:
                with self.perf_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writerow({k: rec.get(k, "") for k in fields})

            if start >= next_ps:
                models = query_ollama_ps(self.upstream_host, self.upstream_port)
                psrec = {
                    "timestamp": now_iso(),
                    "elapsed_s": round(start - self.started_perf, 3),
                    "models": models,
                }
                with self.lock:
                    with self.ollama_ps_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(psrec, ensure_ascii=False, default=str) + "\n")
                next_ps = start + self.ps_interval

            elapsed = time.perf_counter() - start
            self.stop_event.wait(max(0.05, self.sample_interval - elapsed))

    def copy_server_log_excerpt(self) -> str:
        if not self.server_log_path or not self.server_log_path.exists():
            return ""
        try:
            with self.server_log_path.open("rb") as f:
                f.seek(self.server_log_start)
                data = f.read()
            text = data.decode("utf-8", errors="replace")
            (self.run_dir / "ollama_server_excerpt.log").write_text(text, encoding="utf-8")
            return text
        except Exception as e:
            self.event("server_log_copy_error", {"error": repr(e)})
            return ""

    def finalize(self) -> None:
        with self.lock:
            if self.finalized:
                return
            self.finalized = True
        self.stop_event.set()
        time.sleep(min(self.sample_interval + 0.1, 1.2))
        excerpt = self.copy_server_log_excerpt()
        try:
            summary = build_summary(self, excerpt)
            write_json(self.run_dir / "summary.json", summary)
            (self.run_dir / "summary.md").write_text(render_summary_md(summary), encoding="utf-8")
            self.event("probe_finalized", {"summary": str(self.run_dir / "summary.md")})
            print(f"\n[probe] Summary written to: {self.run_dir / 'summary.md'}")
            print(f"[probe] Full run directory: {self.run_dir}")
        except Exception:
            traceback.print_exc()


# ------------------------------- summary generation -------------------------------

def read_perf_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out: dict[str, Any] = {}
                for k, v in row.items():
                    if k == "timestamp":
                        out[k] = v
                    elif v == "":
                        out[k] = None
                    else:
                        try:
                            out[k] = float(v)
                        except Exception:
                            out[k] = v
                rows.append(out)
    except Exception:
        pass
    return rows


def stats_for(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return {
        "min": round(min(vals), 3),
        "mean": round(statistics.fmean(vals), 3),
        "max": round(max(vals), 3),
    }


def parse_ollama_server_excerpt(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "thinking_modes": [],
        "runner_start_seconds": [],
        "contexts": [],
        "offload": [],
        "samplers": [],
        "chat_completion_http_seconds": [],
        "llama_timings": [],
    }
    lines = text.splitlines()
    pending: dict[str, Any] = {}

    for line in lines:
        m = re.search(r"thinking\s*=\s*(\d+)", line)
        if m:
            out["thinking_modes"].append(int(m.group(1)))

        m = re.search(r'llama-server started in ([\d.]+) seconds', line)
        if m:
            out["runner_start_seconds"].append(float(m.group(1)))

        m = re.search(r"\bn_ctx\s*=\s*(\d+)", line)
        if m:
            out["contexts"].append(int(m.group(1)))

        m = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers", line)
        if m:
            out["offload"].append({"offloaded": int(m.group(1)), "total": int(m.group(2))})

        if "sampler params:" in line:
            pending.setdefault("sampler_block", [])
        elif "sampler_block" in pending:
            if line.strip() and ("top_k" in line or "repeat_last_n" in line or "mirostat" in line):
                pending["sampler_block"].append(line.strip())
            elif pending["sampler_block"]:
                joined = " ".join(pending.pop("sampler_block"))
                sm = {}
                for key in ("top_k", "top_p", "min_p", "temp", "repeat_penalty", "presence_penalty", "frequency_penalty"):
                    mm = re.search(rf"\b{re.escape(key)}\s*=\s*([-\d.]+)", joined)
                    if mm:
                        sm[key] = float(mm.group(1))
                if sm:
                    out["samplers"].append(sm)

        m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
        if m:
            pending["prompt_eval_ms"] = float(m.group(1))
            pending["prompt_eval_tokens"] = int(m.group(2))

        m = re.search(r"\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
        if m and "prompt eval time" not in line:
            pending["eval_ms"] = float(m.group(1))
            pending["eval_tokens"] = int(m.group(2))

        m = re.search(r"\btotal time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
        if m:
            pending["total_ms"] = float(m.group(1))
            pending["total_tokens"] = int(m.group(2))
            timing = {k: pending.pop(k) for k in list(pending.keys()) if k in {
                "prompt_eval_ms", "prompt_eval_tokens", "eval_ms", "eval_tokens", "total_ms", "total_tokens"
            }}
            if timing:
                out["llama_timings"].append(timing)

        if 'POST     "/v1/chat/completions"' in line:
            # GIN: "| 200 |    9.1573277s |"
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                secs = parse_duration_seconds(parts[2])
                if secs is not None:
                    out["chat_completion_http_seconds"].append(round(secs, 6))

    # Deduplicate stable repeated values while preserving order
    for key in ("thinking_modes", "contexts"):
        out[key] = list(dict.fromkeys(out[key]))
    return out


def build_summary(state: ProbeState, excerpt: str) -> dict[str, Any]:
    rows = read_perf_csv(state.perf_path)
    metrics = list(state.metrics)
    model_metrics = [m for m in metrics if "/v1/chat/completions" in m.path]

    gaps = []
    sorted_mm = sorted(model_metrics, key=lambda m: m.start_perf)
    for prev, cur in zip(sorted_mm, sorted_mm[1:]):
        if prev.total_ms is not None:
            prev_end = prev.start_perf + prev.total_ms / 1000.0
            gaps.append(max(0.0, cur.start_perf - prev_end))

    perf = {
        "samples": len(rows),
        "cpu_percent": stats_for(rows, "cpu_percent"),
        "ram_used_mb": stats_for(rows, "ram_used_mb"),
        "ram_used_percent": stats_for(rows, "ram_used_percent"),
        "gpu_util_percent": stats_for(rows, "gpu_util_percent"),
        "gpu_mem_used_mb": stats_for(rows, "gpu_mem_used_mb"),
        "gpu_power_w": stats_for(rows, "gpu_power_w"),
        "gpu_temp_c": stats_for(rows, "gpu_temp_c"),
    }

    model_req = []
    for m in sorted_mm:
        d = asdict(m)
        d.pop("start_perf", None)
        model_req.append(d)

    total_proxy_model_s = sum((m.total_ms or 0) / 1000.0 for m in model_metrics)

    loaded_transitions = []
    try:
        prev_loaded = None
        for line in state.ollama_ps_path.read_text(encoding="utf-8").splitlines():
            o = json.loads(line)
            models = o.get("models")
            loaded = bool(models)
            if loaded != prev_loaded:
                loaded_transitions.append({
                    "timestamp": o.get("timestamp"),
                    "elapsed_s": o.get("elapsed_s"),
                    "loaded": loaded,
                    "models": [x.get("name") or x.get("model") for x in (models or [])],
                })
                prev_loaded = loaded
    except Exception:
        pass

    return {
        "run": {
            "started_at": state.started_wall,
            "finished_at": now_iso(),
            "duration_s": round(time.perf_counter() - state.started_perf, 3),
            "run_dir": str(state.run_dir),
            "upstream": f"http://{state.upstream_host}:{state.upstream_port}",
        },
        "http": {
            "all_request_count": len(metrics),
            "model_request_count": len(model_metrics),
            "model_request_total_proxy_s": round(total_proxy_model_s, 3),
            "inter_model_request_gaps_s": [round(x, 3) for x in gaps],
            "inter_model_request_gap_total_s": round(sum(gaps), 3),
            "model_requests": model_req,
        },
        "performance": perf,
        "ollama_loaded_transitions": loaded_transitions,
        "ollama_server": parse_ollama_server_excerpt(excerpt),
    }


def render_summary_md(s: dict[str, Any]) -> str:
    run = s["run"]
    http = s["http"]
    perf = s["performance"]
    oll = s["ollama_server"]

    lines = [
        "# Goose ↔ Ollama Probe Summary",
        "",
        f"- Started: `{run['started_at']}`",
        f"- Finished: `{run['finished_at']}`",
        f"- Probe lifetime: **{run['duration_s']:.2f} s**",
        f"- Upstream: `{run['upstream']}`",
        f"- Model calls (`/v1/chat/completions`): **{http['model_request_count']}**",
        f"- Sum of model-call wall time seen by proxy: **{http['model_request_total_proxy_s']:.2f} s**",
        f"- Sum of gaps between model calls: **{http['inter_model_request_gap_total_s']:.2f} s**",
        "",
        "## Model request timeline",
        "",
        "| # | status | total s | headers ms | first body ms | req bytes | resp bytes | model | tools | msgs | reasoning field sent? | response reasoning chars | tool calls |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|",
    ]

    for m in http["model_requests"]:
        rq = m.get("request_summary") or {}
        rs = m.get("response_summary") or {}
        reasoning_sent = any(k in rq for k in ("reasoning_effort", "reasoning", "think"))
        lines.append(
            f"| {m['request_id']} | {m.get('status','')} | "
            f"{(m.get('total_ms') or 0)/1000:.3f} | "
            f"{m.get('upstream_headers_ms') if m.get('upstream_headers_ms') is not None else ''} | "
            f"{m.get('first_body_byte_ms') if m.get('first_body_byte_ms') is not None else ''} | "
            f"{m.get('request_bytes',0)} | {m.get('response_bytes',0)} | "
            f"`{rq.get('model','')}` | {rq.get('tool_schema_count','')} | {rq.get('message_count','')} | "
            f"{'yes' if reasoning_sent else 'no'} | {rs.get('reasoning_chars',0)} | {rs.get('tool_call_count',0)} |"
        )

    lines += ["", "## Request parameters observed", ""]
    for m in http["model_requests"]:
        rq = m.get("request_summary") or {}
        interesting = {k: rq.get(k) for k in (
            "model", "stream", "temperature", "top_p", "top_k", "min_p",
            "presence_penalty", "frequency_penalty", "repeat_penalty",
            "max_tokens", "max_completion_tokens", "reasoning_effort",
            "reasoning", "think", "stream_options", "tool_choice",
            "parallel_tool_calls"
        ) if k in rq}
        lines.append(f"**Request {m['request_id']}**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(interesting, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    lines += ["## Ollama / llama.cpp observations", ""]
    lines.append(f"- Thinking modes observed in server log: `{oll.get('thinking_modes')}`")
    lines.append(f"- Context sizes observed: `{oll.get('contexts')}`")
    lines.append(f"- Runner cold-start times: `{oll.get('runner_start_seconds')}` seconds")
    lines.append(f"- Layer offload observations: `{oll.get('offload')}`")
    if oll.get("samplers"):
        lines.append(f"- Actual sampler blocks observed: `{json.dumps(oll['samplers'], ensure_ascii=False)}`")
    lines.append(f"- `/v1/chat/completions` durations from Ollama log: `{oll.get('chat_completion_http_seconds')}` seconds")
    lines.append("")

    if oll.get("llama_timings"):
        lines += [
            "### llama.cpp generation timings",
            "",
            "| turn | prompt eval ms | prompt tokens | eval ms | output tokens | total ms | tok/s |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, t in enumerate(oll["llama_timings"], 1):
            eval_ms = t.get("eval_ms")
            eval_tokens = t.get("eval_tokens")
            tps = (eval_tokens / (eval_ms / 1000.0)) if eval_ms and eval_tokens else None
            lines.append(
                f"| {i} | {t.get('prompt_eval_ms','')} | {t.get('prompt_eval_tokens','')} | "
                f"{t.get('eval_ms','')} | {t.get('eval_tokens','')} | {t.get('total_ms','')} | "
                f"{f'{tps:.2f}' if tps is not None else ''} |"
            )
        lines.append("")

    lines += ["## Hardware telemetry", ""]
    def fmtstat(label: str, st: dict[str, float] | None, unit: str = ""):
        if st:
            lines.append(f"- {label}: mean **{st['mean']}{unit}**, max **{st['max']}{unit}**, min **{st['min']}{unit}**")
    fmtstat("CPU", perf.get("cpu_percent"), "%")
    fmtstat("RAM used", perf.get("ram_used_mb"), " MB")
    fmtstat("RAM load", perf.get("ram_used_percent"), "%")
    fmtstat("GPU utilization", perf.get("gpu_util_percent"), "%")
    fmtstat("GPU VRAM used", perf.get("gpu_mem_used_mb"), " MB")
    fmtstat("GPU power", perf.get("gpu_power_w"), " W")
    fmtstat("GPU temperature", perf.get("gpu_temp_c"), " °C")
    lines.append("")

    lines += ["## Model residency transitions", ""]
    transitions = s.get("ollama_loaded_transitions") or []
    if transitions:
        for t in transitions:
            lines.append(f"- `{t.get('timestamp')}` — loaded={t.get('loaded')} — models={t.get('models')}")
    else:
        lines.append("- No `/api/ps` residency transition data captured.")
    lines.append("")

    lines += [
        "## Files in this run",
        "",
        "- `summary.md` — this digest",
        "- `summary.json` — machine-readable digest",
        "- `request_metrics.jsonl` — one record per HTTP request",
        "- `events.jsonl` — proxy event timeline",
        "- `performance.csv` — 1-second hardware telemetry",
        "- `ollama_ps.jsonl` — periodic Ollama model residency snapshots",
        "- `ollama_server_excerpt.log` — Ollama server log bytes generated during this probe run",
        "- `requests/` — full raw/pretty request and response bodies",
        "",
        "> The probe is observational. Raw request files may contain prompts, tool schemas, tool results, paths, and other data Goose sent to Ollama.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------- HTTP proxy -------------------------------

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


class ProbeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "GooseOllamaProbe/1.0"

    @property
    def state(self) -> ProbeState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep console readable; canonical logs go to files.
        sys.stdout.write("[proxy] " + (fmt % args) + "\n")

    def do_HEAD(self): self._proxy()
    def do_GET(self): self._proxy()
    def do_POST(self): self._proxy()
    def do_PUT(self): self._proxy()
    def do_DELETE(self): self._proxy()
    def do_PATCH(self): self._proxy()

    def _proxy(self) -> None:
        req_id = self.state.next_request_id()
        start_perf = time.perf_counter()
        started_at = now_iso()

        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        req_json = safe_json_loads(body)
        req_summary = summarize_request_json(req_json)
        metric = RequestMetric(
            request_id=req_id,
            method=self.command,
            path=self.path,
            started_at=started_at,
            start_perf=start_perf,
            request_bytes=len(body),
            request_summary=req_summary,
        )

        prefix = f"req_{req_id:04d}"
        if body:
            if req_json is not None:
                write_json(self.state.requests_dir / f"{prefix}_request.json", req_json)
            else:
                (self.state.requests_dir / f"{prefix}_request.bin").write_bytes(body)

        self.state.event("request_started", {
            "request_id": req_id,
            "method": self.command,
            "path": self.path,
            "request_bytes": len(body),
            "request_summary": req_summary,
        })

        conn = None
        try:
            conn = http.client.HTTPConnection(self.state.upstream_host, self.state.upstream_port, timeout=660)

            headers = {}
            for k, v in self.headers.items():
                lk = k.lower()
                if lk in HOP_BY_HOP or lk in ("host", "content-length"):
                    continue
                headers[k] = v
            headers["Host"] = f"{self.state.upstream_host}:{self.state.upstream_port}"
            if body:
                headers["Content-Length"] = str(len(body))

            conn.request(self.command, self.path, body=body if body else None, headers=headers)
            upstream = conn.getresponse()
            metric.upstream_headers_ms = round((time.perf_counter() - start_perf) * 1000, 3)
            metric.status = upstream.status
            metric.content_type = upstream.getheader("Content-Type", "")

            self.send_response(upstream.status, upstream.reason)
            for k, v in upstream.getheaders():
                lk = k.lower()
                if lk in HOP_BY_HOP or lk in ("content-length", "connection"):
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()

            chunks: list[bytes] = []
            first_body = True
            while True:
                if hasattr(upstream, "read1"):
                    chunk = upstream.read1(4096)
                else:
                    chunk = upstream.read(4096)
                if not chunk:
                    break
                if first_body:
                    metric.first_body_byte_ms = round((time.perf_counter() - start_perf) * 1000, 3)
                    first_body = False
                chunks.append(chunk)
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    self.state.event("client_disconnected", {"request_id": req_id})
                    break

            resp_body = b"".join(chunks)
            metric.response_bytes = len(resp_body)
            metric.total_ms = round((time.perf_counter() - start_perf) * 1000, 3)

            # Full response capture.
            ctype = metric.content_type or ""
            obj = safe_json_loads(resp_body)
            if obj is not None and "text/event-stream" not in ctype.lower():
                write_json(self.state.requests_dir / f"{prefix}_response.json", obj)
            else:
                (self.state.requests_dir / f"{prefix}_response.txt").write_bytes(resp_body)

            if "/v1/chat/completions" in self.path:
                metric.response_summary = parse_openai_response(resp_body, ctype)

            self.state.event("request_finished", {
                "request_id": req_id,
                "status": metric.status,
                "upstream_headers_ms": metric.upstream_headers_ms,
                "first_body_byte_ms": metric.first_body_byte_ms,
                "total_ms": metric.total_ms,
                "response_bytes": metric.response_bytes,
                "response_summary": metric.response_summary,
            })

        except Exception as e:
            metric.error = repr(e)
            metric.total_ms = round((time.perf_counter() - start_perf) * 1000, 3)
            self.state.event("request_error", {
                "request_id": req_id,
                "error": metric.error,
                "traceback": traceback.format_exc(),
            })
            try:
                self.send_error(502, f"Probe upstream error: {e}")
            except Exception:
                pass
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            self.state.add_metric(metric)
            self.close_connection = True


# ------------------------------- main -------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Transparent Goose -> Ollama diagnostic proxy")
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=11436)
    ap.add_argument("--upstream", default="http://127.0.0.1:11434")
    ap.add_argument("--log-root", default=str(Path.cwd() / "diagnostic-runs"))
    ap.add_argument("--sample-interval", type=float, default=1.0, help="Hardware sampling interval in seconds")
    ap.add_argument("--ps-interval", type=float, default=3.0, help="Ollama /api/ps sampling interval in seconds")
    args = ap.parse_args()

    u = urlsplit(args.upstream if "://" in args.upstream else "http://" + args.upstream)
    if u.scheme != "http":
        print("Only local HTTP upstreams are supported by this diagnostic probe.", file=sys.stderr)
        return 2
    upstream_host = u.hostname or "127.0.0.1"
    upstream_port = u.port or 11434

    root = Path(args.log_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{stamp}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"run_{stamp}_{suffix}"
    run_dir.mkdir(parents=True)

    state = ProbeState(run_dir, upstream_host, upstream_port, args.sample_interval, args.ps_interval)
    state.start_sampler()

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProbeHandler)
    server.state = state  # type: ignore[attr-defined]

    stopped = threading.Event()

    def shutdown_handler(signum=None, frame=None):
        if stopped.is_set():
            return
        stopped.set()
        state.event("shutdown_requested", {"signal": signum})
        # shutdown must be called from another thread if serve_forever is active.
        threading.Thread(target=server.shutdown, daemon=True).start()

    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown_handler)

    atexit.register(state.finalize)

    print("=" * 72)
    print("Goose ↔ Ollama diagnostic probe")
    print(f"Listening : http://{args.listen_host}:{args.listen_port}")
    print(f"Forwarding: http://{upstream_host}:{upstream_port}")
    print(f"Run logs  : {run_dir}")
    print("Ctrl+C after the Goose test to generate summary.md / summary.json")
    print("=" * 72)

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
