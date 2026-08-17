from __future__ import annotations

import psutil


def list_processes_info(
    query: str | None = None,
    max_results: int = 100,
) -> dict:
    if max_results < 1 or max_results > 500:
        raise ValueError("max_results must be between 1 and 500.")

    query_lower = query.lower().strip() if query else None
    results: list[dict] = []

    for process in psutil.process_iter(
        attrs=["pid", "name", "exe", "status", "username"]
    ):
        try:
            info = process.info
            haystack = f"{info.get('name') or ''} {info.get('exe') or ''}".lower()
            if query_lower and query_lower not in haystack:
                continue

            results.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "exe": info.get("exe"),
                    "status": info.get("status"),
                    "username": info.get("username"),
                }
            )
            if len(results) >= max_results:
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {
        "status": "ok",
        "operation": "list_processes",
        "query": query,
        "processes": results,
        "count": len(results),
        "truncated": len(results) >= max_results,
    }


def process_running(
    pid: int | None = None,
    name: str | None = None,
) -> dict:
    if pid is None and not name:
        raise ValueError("Provide pid, name, or both.")
    if pid is not None and pid <= 0:
        raise ValueError("pid must be a positive integer.")

    def normalize_process_name(value: str) -> str:
        normalized = value.strip().lower()
        if normalized.endswith(".exe"):
            normalized = normalized[:-4]
        return normalized

    name_normalized = normalize_process_name(name) if name else None
    matches: list[dict] = []

    if pid is not None:
        if not psutil.pid_exists(pid):
            return {
                "status": "ok",
                "operation": "is_process_running",
                "running": False,
                "matches": [],
            }

        try:
            process = psutil.Process(pid)
            process_name = process.name()

            if (
                name_normalized
                and normalize_process_name(process_name) != name_normalized
            ):
                return {
                    "status": "ok",
                    "operation": "is_process_running",
                    "running": False,
                    "matches": [],
                }

            matches.append({"pid": pid, "name": process_name})

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    else:
        for process in psutil.process_iter(attrs=["pid", "name"]):
            try:
                process_name = process.info.get("name") or ""

                if normalize_process_name(process_name) == name_normalized:
                    matches.append(
                        {
                            "pid": process.info["pid"],
                            "name": process_name,
                        }
                    )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    return {
        "status": "ok",
        "operation": "is_process_running",
        "running": bool(matches),
        "matches": matches[:50],
        "truncated": len(matches) > 50,
    }