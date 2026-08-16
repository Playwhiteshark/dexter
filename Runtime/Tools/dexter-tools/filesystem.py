from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from send2trash import send2trash

from config import (
    ALLOWED_ROOTS,
    ALLOW_PERMANENT_DELETE,
    DEFAULT_EXCLUDED_DIR_NAMES,
    MAX_CONTENT_MATCHES,
    MAX_CONTENT_SEARCH_FILE_BYTES,
    MAX_LIST_ENTRIES,
    MAX_READ_CHARS,
    MAX_READ_FILE_BYTES,
    MAX_SEARCH_RESULTS,
)


def _resolved_roots() -> list[Path]:
    return [Path(root).resolve(strict=False) for root in ALLOWED_ROOTS]


def _is_within_allowed_roots(path: Path) -> bool:
    return any(path == root or path.is_relative_to(root) for root in _resolved_roots())


def resolve_allowed_path(path: str) -> Path:
    """Resolve a path and enforce Dexter's configured filesystem boundary."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Path must be a non-empty string.")

    resolved = Path(path).expanduser().resolve(strict=False)

    if not _is_within_allowed_roots(resolved):
        raise ValueError(
            f"Path is outside Dexter's allowed filesystem roots: {resolved}"
        )

    return resolved


def list_allowed_roots() -> dict:
    roots = [str(root) for root in _resolved_roots()]
    return {
        "status": "ok",
        "operation": "list_allowed_directories",
        "roots": roots,
        "count": len(roots),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _iso_timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _require_exists(path: Path, label: str = "Path") -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _require_file(path: Path, label: str = "Path") -> None:
    _require_exists(path, label)
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")


def _require_directory(path: Path, label: str = "Path") -> None:
    _require_exists(path, label)
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")


def _is_hidden_name(name: str) -> bool:
    return name.startswith(".")


def _normalize_extensions(extensions: list[str] | None) -> set[str] | None:
    if not extensions:
        return None

    normalized: set[str] = set()
    for extension in extensions:
        value = extension.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = "." + value
        normalized.add(value)

    return normalized or None


def _should_skip_dir(name: str, include_hidden: bool) -> bool:
    if name in DEFAULT_EXCLUDED_DIR_NAMES:
        return True
    if not include_hidden and _is_hidden_name(name):
        return True
    return False


def _walk(root: Path, include_hidden: bool) -> Iterable[tuple[Path, list[str], list[str]]]:
    """Walk without descending through links/junctions that escape allowed roots."""
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        safe_dirs: list[str] = []
        for name in dirs:
            if _should_skip_dir(name, include_hidden):
                continue
            try:
                resolved = (current_path / name).resolve(strict=False)
            except OSError:
                continue
            if _is_within_allowed_roots(resolved):
                safe_dirs.append(name)
        dirs[:] = safe_dirs
        yield current_path, dirs, files


def get_file_sha256(path: str) -> dict:
    file_path = resolve_allowed_path(path)
    _require_file(file_path, "File")

    return {
        "status": "ok",
        "operation": "sha256",
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def get_path_info(path: str) -> dict:
    target = resolve_allowed_path(path)
    _require_exists(target)

    stat = target.stat()
    kind = "directory" if target.is_dir() else "file" if target.is_file() else "other"

    return {
        "status": "ok",
        "operation": "file_info",
        "path": str(target),
        "name": target.name,
        "type": kind,
        "size_bytes": stat.st_size,
        "created_utc": _iso_timestamp(stat.st_ctime),
        "modified_utc": _iso_timestamp(stat.st_mtime),
        "accessed_utc": _iso_timestamp(stat.st_atime),
        "readable": os.access(target, os.R_OK),
        "writable": os.access(target, os.W_OK),
    }


def list_directory_entries(
    path: str,
    max_entries: int = 200,
    include_hidden: bool = False,
) -> dict:
    directory = resolve_allowed_path(path)
    _require_directory(directory, "Directory")

    if max_entries < 1 or max_entries > MAX_LIST_ENTRIES:
        raise ValueError(
            f"max_entries must be between 1 and {MAX_LIST_ENTRIES}."
        )

    entries: list[dict] = []
    total_seen = 0

    try:
        # Sort by name only before validation. Calling is_dir() in the sort key
        # can follow a link/junction before we have checked its resolved target.
        children = sorted(directory.iterdir(), key=lambda p: p.name.lower())
    except PermissionError as exc:
        raise PermissionError(f"Cannot list directory: {directory}") from exc

    for child in children:
        if not include_hidden and _is_hidden_name(child.name):
            continue

        total_seen += 1
        if len(entries) >= max_entries:
            continue

        try:
            resolved = child.resolve(strict=False)
            if not any(
                resolved == root or resolved.is_relative_to(root)
                for root in _resolved_roots()
            ):
                # A symlink/junction can point outside an allowed root. Do not
                # expose or traverse its target.
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "type": "blocked_link",
                    }
                )
                continue

            stat = child.stat()
            entries.append(
                {
                    "name": child.name,
                    "path": str(resolved),
                    "type": (
                        "directory"
                        if child.is_dir()
                        else "file"
                        if child.is_file()
                        else "other"
                    ),
                    "size_bytes": stat.st_size if child.is_file() else None,
                    "modified_utc": _iso_timestamp(stat.st_mtime),
                }
            )
        except (OSError, PermissionError) as exc:
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "type": "unavailable",
                    "error": str(exc),
                }
            )

    return {
        "status": "ok",
        "operation": "list_directory",
        "directory": str(directory),
        "entries": entries,
        "returned": len(entries),
        "total_seen": total_seen,
        "truncated": total_seen > len(entries),
    }


def _name_matches(name: str, pattern: str) -> bool:
    pattern = pattern.strip()
    if not pattern:
        raise ValueError("Search pattern must not be empty.")

    lower_name = name.lower()
    lower_pattern = pattern.lower()

    if any(char in pattern for char in "*?[]"):
        return fnmatch.fnmatch(lower_name, lower_pattern)

    return lower_pattern in lower_name


def search_paths(
    root: str,
    pattern: str,
    extensions: list[str] | None = None,
    include_directories: bool = False,
    include_hidden: bool = False,
    max_results: int = 100,
) -> dict:
    search_root = resolve_allowed_path(root)
    _require_directory(search_root, "Search root")

    if max_results < 1 or max_results > MAX_SEARCH_RESULTS:
        raise ValueError(
            f"max_results must be between 1 and {MAX_SEARCH_RESULTS}."
        )

    allowed_extensions = _normalize_extensions(extensions)
    matches: list[dict] = []
    scanned = 0
    truncated = False

    for current, dirs, files in _walk(search_root, include_hidden):
        names = files + (dirs if include_directories else [])

        for name in names:
            scanned += 1
            candidate = current / name
            try:
                resolved = candidate.resolve(strict=False)
            except OSError:
                continue
            if not _is_within_allowed_roots(resolved):
                continue

            is_file = resolved.is_file()
            if is_file and allowed_extensions is not None:
                if resolved.suffix.lower() not in allowed_extensions:
                    continue

            if not _name_matches(name, pattern):
                continue

            matches.append(
                {
                    "path": str(resolved),
                    "name": candidate.name,
                    "type": "file" if is_file else "directory" if resolved.is_dir() else "other",
                }
            )

            if len(matches) >= max_results:
                truncated = True
                break

        if truncated:
            break

    return {
        "status": "ok",
        "operation": "search_files",
        "root": str(search_root),
        "pattern": pattern,
        "matches": matches,
        "count": len(matches),
        "scanned_entries": scanned,
        "truncated": truncated,
    }



def _looks_binary(data: bytes) -> bool:
    if not data:
        return False

    sample = data[:4096]
    if b"\x00" in sample:
        return True

    # Treat a high proportion of non-whitespace control bytes as binary while
    # allowing ordinary tabs/newlines/carriage returns.
    controls = sum(byte < 32 and byte not in {9, 10, 13} for byte in sample)
    return controls / len(sample) > 0.20


def _decode_text(data: bytes) -> tuple[str, str]:
    last_error: UnicodeDecodeError | None = None

    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        "File could not be decoded as UTF-8 or Windows-1252 text."
    ) from last_error


def read_text_file(
    path: str,
    start_line: int = 1,
    max_lines: int = 400,
    max_chars: int = 50_000,
) -> dict:
    file_path = resolve_allowed_path(path)
    _require_file(file_path, "File")

    if start_line < 1:
        raise ValueError("start_line must be at least 1.")
    if max_lines < 1 or max_lines > 5000:
        raise ValueError("max_lines must be between 1 and 5000.")
    if max_chars < 1 or max_chars > MAX_READ_CHARS:
        raise ValueError(f"max_chars must be between 1 and {MAX_READ_CHARS}.")

    size = file_path.stat().st_size
    if size > MAX_READ_FILE_BYTES:
        raise ValueError(
            f"File is too large for read_file ({size} bytes; "
            f"limit {MAX_READ_FILE_BYTES})."
        )

    data = file_path.read_bytes()
    if _looks_binary(data):
        raise ValueError(f"File appears to be binary, not text: {file_path}")

    text, encoding = _decode_text(data)
    lines = text.splitlines()
    start_index = min(start_line - 1, len(lines))
    selected = lines[start_index:start_index + max_lines]
    content = "\n".join(selected)

    char_truncated = len(content) > max_chars
    if char_truncated:
        content = content[:max_chars]

    end_line = start_index + len(selected)

    return {
        "status": "ok",
        "operation": "read_file",
        "path": str(file_path),
        "encoding": encoding,
        "start_line": start_index + 1 if lines else 0,
        "end_line": end_line,
        "total_lines": len(lines),
        "content": content,
        "truncated": (
            char_truncated
            or end_line < len(lines)
            or start_line - 1 > len(lines)
        ),
    }


def write_file_verified(
    path: str,
    content: str,
    overwrite: bool = False,
) -> dict:
    """Create or overwrite a UTF-8 text file and verify the exact bytes written.

    ``path`` is the complete target path, including the filename and extension
    when one is desired. Extensionless files such as ``Dockerfile`` or
    ``.gitignore`` are valid.

    Existing files are rejected unless ``overwrite=True``. Parent directories
    are never created implicitly.
    """
    target = resolve_allowed_path(path)

    if not isinstance(content, str):
        raise TypeError("content must be a string.")

    existed_before = target.exists()

    if existed_before:
        if target.is_dir():
            raise IsADirectoryError(
                "Cannot write file because a directory exists at the target "
                f"path: {target}"
            )
        if not target.is_file():
            raise ValueError(
                "Cannot write file because the target is not a regular file: "
                f"{target}"
            )
        if not overwrite:
            raise FileExistsError(
                f"File already exists and overwrite is false: {target}"
            )

    _require_directory(target.parent, "Parent directory")

    encoded = content.encode("utf-8")
    expected_hash = hashlib.sha256(encoded).hexdigest()

    # Binary mode avoids Windows newline translation, so verification is for
    # the exact UTF-8 bytes corresponding to ``content``.
    mode = "wb" if overwrite else "xb"
    try:
        with target.open(mode) as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        # Preserve overwrite=False if another process creates the target after
        # the initial existence check.
        raise FileExistsError(
            f"File already exists and overwrite is false: {target}"
        ) from None

    if not target.is_file():
        raise RuntimeError("Write returned but the target file does not exist.")

    written_bytes = target.read_bytes()
    actual_hash = hashlib.sha256(written_bytes).hexdigest()
    size_matches = len(written_bytes) == len(encoded)
    bytes_match = written_bytes == encoded
    hashes_match = actual_hash == expected_hash

    if not size_matches or not bytes_match or not hashes_match:
        raise RuntimeError(
            "Write completed but exact-content verification failed."
        )

    return {
        "status": "ok",
        "operation": "write_file",
        "path": str(target),
        "encoding": "utf-8",
        "created": not existed_before,
        "overwritten": existed_before,
        "size_bytes": len(written_bytes),
        "sha256": actual_hash,
        "verified": True,
    }


def search_text_contents(
    root: str,
    query: str,
    extensions: list[str] | None = None,
    case_sensitive: bool = False,
    include_hidden: bool = False,
    max_results: int = 100,
    max_matches_per_file: int = 5,
) -> dict:
    search_root = resolve_allowed_path(root)
    _require_directory(search_root, "Search root")

    if not query:
        raise ValueError("Content search query must not be empty.")
    if max_results < 1 or max_results > MAX_CONTENT_MATCHES:
        raise ValueError(
            f"max_results must be between 1 and {MAX_CONTENT_MATCHES}."
        )
    if max_matches_per_file < 1 or max_matches_per_file > 50:
        raise ValueError("max_matches_per_file must be between 1 and 50.")

    allowed_extensions = _normalize_extensions(extensions)
    needle = query if case_sensitive else query.lower()
    matches: list[dict] = []
    scanned_files = 0
    skipped_large = 0
    skipped_binary = 0
    skipped_unreadable = 0
    truncated = False

    for current, _, files in _walk(search_root, include_hidden):
        for name in files:
            path = current / name

            if allowed_extensions is not None and path.suffix.lower() not in allowed_extensions:
                continue

            try:
                resolved = resolve_allowed_path(str(path))
                size = resolved.stat().st_size
                if size > MAX_CONTENT_SEARCH_FILE_BYTES:
                    skipped_large += 1
                    continue

                data = resolved.read_bytes()
                scanned_files += 1

                if _looks_binary(data):
                    skipped_binary += 1
                    continue

                text, encoding = _decode_text(data)
            except (OSError, PermissionError, ValueError):
                skipped_unreadable += 1
                continue

            per_file = 0
            for line_number, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.lower()
                if needle not in haystack:
                    continue

                matches.append(
                    {
                        "path": str(resolved),
                        "line_number": line_number,
                        "line": line[:1000],
                        "encoding": encoding,
                    }
                )
                per_file += 1

                if len(matches) >= max_results:
                    truncated = True
                    break
                if per_file >= max_matches_per_file:
                    break

            if truncated:
                break
        if truncated:
            break

    return {
        "status": "ok",
        "operation": "search_file_contents",
        "root": str(search_root),
        "query": query,
        "matches": matches,
        "count": len(matches),
        "scanned_files": scanned_files,
        "skipped_large": skipped_large,
        "skipped_binary": skipped_binary,
        "skipped_unreadable": skipped_unreadable,
        "truncated": truncated,
    }


def create_directory_verified(path: str) -> dict:
    directory = resolve_allowed_path(path)

    existed_before = directory.exists()
    if existed_before and not directory.is_dir():
        raise FileExistsError(
            f"Cannot create directory because a non-directory already exists: {directory}"
        )

    directory.mkdir(parents=True, exist_ok=True)

    verified = directory.is_dir()
    if not verified:
        raise RuntimeError("Directory creation returned but verification failed.")

    return {
        "status": "ok",
        "operation": "create_directory",
        "path": str(directory),
        "created": not existed_before,
        "already_existed": existed_before,
        "verified": True,
    }


def copy_file_verified(source: str, destination: str) -> dict:
    src = resolve_allowed_path(source)
    dst = resolve_allowed_path(destination)

    _require_file(src, "Source")

    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    _require_directory(dst.parent, "Destination directory")

    source_hash = sha256_file(src)
    source_size = src.stat().st_size

    shutil.copy2(src, dst)

    destination_exists = dst.is_file()
    destination_hash = sha256_file(dst) if destination_exists else None
    destination_size = dst.stat().st_size if destination_exists else None
    hashes_match = source_hash == destination_hash
    sizes_match = source_size == destination_size

    if not destination_exists or not hashes_match or not sizes_match:
        raise RuntimeError("Copy completed but postcondition verification failed.")

    return {
        "status": "ok",
        "operation": "copy",
        "source": str(src),
        "destination": str(dst),
        "source_exists": src.is_file(),
        "destination_exists": destination_exists,
        "source_sha256": source_hash,
        "destination_sha256": destination_hash,
        "hashes_match": hashes_match,
        "sizes_match": sizes_match,
        "verified": True,
    }


def move_file_verified(source: str, destination: str) -> dict:
    src = resolve_allowed_path(source)
    dst = resolve_allowed_path(destination)

    _require_file(src, "Source")

    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    _require_directory(dst.parent, "Destination directory")

    source_hash = sha256_file(src)
    source_size = src.stat().st_size

    shutil.move(str(src), str(dst))

    source_gone = not src.exists()
    destination_exists = dst.is_file()
    destination_hash = sha256_file(dst) if destination_exists else None
    destination_size = dst.stat().st_size if destination_exists else None
    hashes_match = source_hash == destination_hash
    sizes_match = source_size == destination_size

    if not source_gone or not destination_exists or not hashes_match or not sizes_match:
        raise RuntimeError("Move completed but postcondition verification failed.")

    return {
        "status": "ok",
        "operation": "move",
        "source": str(src),
        "destination": str(dst),
        "source_exists": src.exists(),
        "destination_exists": destination_exists,
        "source_sha256_before": source_hash,
        "destination_sha256_after": destination_hash,
        "hashes_match": hashes_match,
        "sizes_match": sizes_match,
        "verified": True,
    }


def rename_file_verified(path: str, new_name: str) -> dict:
    src = resolve_allowed_path(path)
    _require_file(src, "File")

    if not new_name or new_name in {".", ".."}:
        raise ValueError("new_name must be a non-empty file name.")
    if Path(new_name).name != new_name or "/" in new_name or "\\" in new_name:
        raise ValueError(
            "new_name must contain only a file name, not a directory path."
        )

    dst = resolve_allowed_path(str(src.parent / new_name))
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    source_hash = sha256_file(src)
    source_size = src.stat().st_size
    src.rename(dst)

    source_gone = not src.exists()
    destination_exists = dst.is_file()
    destination_hash = sha256_file(dst) if destination_exists else None
    sizes_match = destination_exists and dst.stat().st_size == source_size
    hashes_match = source_hash == destination_hash

    if not source_gone or not destination_exists or not hashes_match or not sizes_match:
        raise RuntimeError("Rename completed but postcondition verification failed.")

    return {
        "status": "ok",
        "operation": "rename",
        "source": str(src),
        "destination": str(dst),
        "source_exists": False,
        "destination_exists": True,
        "hashes_match": hashes_match,
        "sizes_match": sizes_match,
        "verified": True,
    }


def delete_file_verified(path: str, permanent: bool = False) -> dict:
    target = resolve_allowed_path(path)
    _require_file(target, "File")

    size_before = target.stat().st_size
    hash_before = sha256_file(target)

    if permanent:
        if not ALLOW_PERMANENT_DELETE:
            raise PermissionError(
                "Permanent deletion is disabled by Dexter policy. "
                "Use the default Recycle Bin deletion instead."
            )
        target.unlink()
        method = "permanent"
    else:
        send2trash(str(target))
        method = "recycle_bin"

    source_gone = not target.exists()
    if not source_gone:
        raise RuntimeError("Delete returned but the source path still exists.")

    return {
        "status": "ok",
        "operation": "delete_file",
        "path": str(target),
        "method": method,
        "size_bytes_before": size_before,
        "sha256_before": hash_before,
        "source_exists": False,
        "verified": True,
    }
