"""
Local archiving service for tbd items.

Items live under `.tbd/<status>/` as Markdown files. Only items in the
`done` or `closed` states may transition to `archived`; archived files are
stored under `.tbd/archive/<status>/`. All moves are guarded by a shared
lock file (`.tbd/archive.lock`) with timeout and stale-lock reclamation,
and every file is written via a same-directory temporary name followed by
an atomic replace, so a failure never leaves a half-written file behind.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .strif import hash_file, iso_timestamp, new_uid

ARCHIVABLE_STATUSES = ("done", "closed")
"""Only these statuses may transition to `archived`."""

ALL_STATUSES = ("open", "in_progress", "blocked", "done", "closed")

ARCHIVE_DIRNAME = "archive"
LOCK_FILENAME = "archive.lock"

EXIT_NOTHING = 0
"""Exit code: nothing to archive."""
EXIT_ERROR = 1
"""Exit code: an error occurred (including lock contention)."""
EXIT_APPLIED = 2
"""Exit code: an archive plan was applied (or is pending in dry-run mode)."""

DEFAULT_LOCK_TIMEOUT = 10.0
DEFAULT_LOCK_STALE_AFTER = 300.0
DEFAULT_PORT = 8765


class ArchiveError(RuntimeError):
    """Base error for archive failures."""


class InvalidTransitionError(ArchiveError):
    """Raised when the state machine rejects a status transition."""


class LockTimeoutError(ArchiveError):
    """Raised when the archive lock cannot be acquired in time."""


class ArchiveStateMachine:
    """
    State machine for tbd item statuses. Only `done` and `closed` may
    transition to `archived`. Any other transition raises
    `InvalidTransitionError` and must not touch the filesystem.
    """

    ALLOWED_TRANSITIONS = frozenset({(status, "archived") for status in ARCHIVABLE_STATUSES})

    def validate(self, from_status: str, to_status: str = "archived") -> str:
        if (from_status, to_status) not in self.ALLOWED_TRANSITIONS:
            raise InvalidTransitionError(
                f"Illegal transition: {from_status!r} -> {to_status!r}. "
                f"Only {sorted(ARCHIVABLE_STATUSES)} can be archived."
            )
        return to_status


@dataclass
class ArchiveLock:
    """
    A cross-process lock file at `.tbd/archive.lock`.

    Acquisition uses O_CREAT|O_EXCL. A lock whose recorded age exceeds
    `stale_after` seconds (or whose contents are unreadable) is considered
    stale and reclaimed. Failing to acquire within `timeout` seconds raises
    `LockTimeoutError` rather than silently skipping.
    """

    path: Path
    timeout: float = DEFAULT_LOCK_TIMEOUT
    stale_after: float = DEFAULT_LOCK_STALE_AFTER
    _fd: int | None = field(default=None, init=False, repr=False)

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._reclaim_if_stale():
                    continue
                if time.monotonic() >= deadline:
                    raise LockTimeoutError(
                        f"Could not acquire archive lock within {self.timeout:.1f}s: "
                        f"{self.path}. Another process holds it; exiting."
                    ) from None
                time.sleep(0.05)
                continue
            payload = json.dumps(
                {"pid": os.getpid(), "created": time.time(), "at": iso_timestamp()}
            )
            os.write(fd, payload.encode("utf-8"))
            self._fd = fd
            return

    def _reclaim_if_stale(self) -> bool:
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            created = float(info.get("created", 0))
        except (OSError, ValueError, AttributeError):
            created = 0.0
        if created <= 0:
            try:
                created = self.path.stat().st_mtime
            except OSError:
                return False
        if time.time() - created <= self.stale_after:
            return False
        try:
            self.path.unlink()
            return True
        except OSError:
            return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            self.path.unlink()
        except OSError:
            pass

    def __enter__(self) -> ArchiveLock:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


@dataclass
class PlannedMove:
    """One planned action: move `source` to `target`, or drop a duplicate."""

    source: Path
    target: Path
    status: str
    action: str  # "move" or "dedupe" (identical file already archived)

    def describe(self) -> str:
        if self.action == "dedupe":
            return f"dedupe {self.source} (already archived as {self.target})"
        return f"move {self.source} -> {self.target}"


def _status_dir(tbd_dir: Path, status: str) -> Path:
    return tbd_dir / status


def _archive_status_dir(tbd_dir: Path, status: str) -> Path:
    return tbd_dir / ARCHIVE_DIRNAME / status


def _hash8(path: Path) -> str:
    return hash_file(path, algorithm="sha1").hex[:8]


def _same_contents(a: Path, b: Path) -> bool:
    return (
        a.is_file()
        and b.is_file()
        and a.stat().st_size == b.stat().st_size
        and _hash8(a) == _hash8(b)
    )


def plan_archive(tbd_dir: str | Path) -> list[PlannedMove]:
    """
    Compute the archive plan without touching any files. Idempotent: items
    already archived with identical content become `dedupe` actions, and
    name conflicts are resolved with a content-hash suffix, never by
    overwriting.
    """
    tbd_dir = Path(tbd_dir)
    machine = ArchiveStateMachine()
    plan: list[PlannedMove] = []
    for status in ARCHIVABLE_STATUSES:
        machine.validate(status, "archived")
        src_dir = _status_dir(tbd_dir, status)
        if not src_dir.is_dir():
            continue
        for source in sorted(p for p in src_dir.iterdir() if p.is_file()):
            dest_dir = _archive_status_dir(tbd_dir, status)
            target = dest_dir / source.name
            if _same_contents(source, target):
                plan.append(PlannedMove(source, target, status, "dedupe"))
                continue
            if target.exists():
                target = dest_dir / f"{source.stem}.{_hash8(source)}{source.suffix}"
                while target.exists() and not _same_contents(source, target):
                    target = (
                        dest_dir / f"{source.stem}.{_hash8(source)}.{new_uid(24)}{source.suffix}"
                    )
                if _same_contents(source, target):
                    plan.append(PlannedMove(source, target, status, "dedupe"))
                    continue
            plan.append(PlannedMove(source, target, status, "move"))
    return plan


def _atomic_move(source: Path, target: Path) -> None:
    """
    Move `source` to `target` by copying to a temporary name in the same
    directory and atomically replacing. On any exception the temporary file
    is removed so no half-written file is left behind.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{new_uid()}.partial")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    source.unlink()


def apply_plan(plan: list[PlannedMove]) -> list[PlannedMove]:
    """
    Apply a plan, returning the actions actually performed. If any action
    fails, all previously applied moves are rolled back so the state
    machine never observes a partial migration.
    """
    applied: list[PlannedMove] = []
    try:
        for move in plan:
            if move.action == "dedupe":
                move.source.unlink()
            else:
                _atomic_move(move.source, move.target)
            applied.append(move)
    except BaseException as err:
        for move in reversed(applied):
            try:
                if move.action == "move" and move.target.exists() and not move.source.exists():
                    os.replace(move.target, move.source)
            except OSError:
                pass
        raise ArchiveError(f"Archive apply failed and was rolled back: {err}") from err
    return applied


def status_counts(tbd_dir: str | Path) -> dict[str, int]:
    """Item counts per status, including archived items per source status."""
    tbd_dir = Path(tbd_dir)
    counts: dict[str, int] = {}
    for status in ALL_STATUSES:
        src_dir = _status_dir(tbd_dir, status)
        counts[status] = (
            len([p for p in src_dir.iterdir() if p.is_file()]) if src_dir.is_dir() else 0
        )
    archived = 0
    for status in ARCHIVABLE_STATUSES:
        dest_dir = _archive_status_dir(tbd_dir, status)
        if dest_dir.is_dir():
            archived += len([p for p in dest_dir.iterdir() if p.is_file()])
    counts["archived"] = archived
    return counts


def run_archive(
    tbd_dir: str | Path,
    apply: bool = False,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    out=sys.stdout,
    err=sys.stderr,
) -> int:
    """
    Plan (and optionally apply) archiving. Returns EXIT_NOTHING when there
    is nothing to archive, EXIT_APPLIED when a plan was applied (or is
    pending in dry-run mode), and EXIT_ERROR on any failure.
    """
    tbd_dir = Path(tbd_dir)
    lock = ArchiveLock(tbd_dir / LOCK_FILENAME, timeout=lock_timeout)
    try:
        with lock:
            plan = plan_archive(tbd_dir)
            if not plan:
                print("tbd archive: nothing to archive.", file=out)
                return EXIT_NOTHING
            print("tbd archive plan:", file=out)
            for move in plan:
                print(f"  {move.describe()}", file=out)
            if not apply:
                print("Dry run only; pass --apply to move files.", file=out)
                return EXIT_APPLIED
            applied = apply_plan(plan)
            print(f"tbd archive: applied {len(applied)} action(s).", file=out)
            return EXIT_APPLIED
    except ArchiveError as exc:
        print(f"tbd archive: error: {exc}", file=err)
        return EXIT_ERROR


_PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>tbd archive</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ccc; padding: 4px 10px; }}
button {{ font-size: 1rem; padding: 6px 16px; }}
.err {{ color: #b00; }}
</style></head><body>
<h1>tbd archive</h1>
<h2>Status counts</h2>
<table><tr><th>status</th><th>items</th></tr>{counts_rows}</table>
<form method="post" action="/archive/run">
<button type="submit">Archive now</button>
</form>
{error_block}
<h2>Last moves</h2>
<ul>{moves_items}</ul>
<h2>Last plan</h2>
<ul>{plan_items}</ul>
<p><small>Last run: {last_run}</small></p>
</body></html>"""


class ArchiveServerState:
    """In-memory state shown on the /archive page."""

    def __init__(self, tbd_dir: Path, lock_timeout: float = DEFAULT_LOCK_TIMEOUT):
        self.tbd_dir = tbd_dir
        self.lock_timeout = lock_timeout
        self.last_plan: list[str] = []
        self.last_moves: list[str] = []
        self.last_error: str | None = None
        self.last_run: str = "never"

    def run_once(self) -> None:
        lock = ArchiveLock(self.tbd_dir / LOCK_FILENAME, timeout=self.lock_timeout)
        try:
            with lock:
                plan = plan_archive(self.tbd_dir)
                self.last_plan = [m.describe() for m in plan]
                applied = apply_plan(plan)
                self.last_moves = [m.describe() for m in applied]
                self.last_error = None
        except ArchiveError as exc:
            self.last_error = str(exc)
        self.last_run = iso_timestamp()

    def render_page(self) -> str:
        counts = status_counts(self.tbd_dir)
        counts_rows = "".join(
            f"<tr><td>{status}</td><td>{count}</td></tr>" for status, count in counts.items()
        )
        moves_items = "".join(f"<li>{m}</li>" for m in self.last_moves) or "<li>(none)</li>"
        plan_items = "".join(f"<li>{m}</li>" for m in self.last_plan) or "<li>(none)</li>"
        error_block = f'<p class="err">Error: {self.last_error}</p>' if self.last_error else ""
        return _PAGE_TEMPLATE.format(
            counts_rows=counts_rows,
            moves_items=moves_items,
            plan_items=plan_items,
            error_block=error_block,
            last_run=self.last_run,
        )


def _make_handler(state: ArchiveServerState) -> type[BaseHTTPRequestHandler]:
    class ArchiveHandler(BaseHTTPRequestHandler):
        def _send_html(self, body: str, status: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path in ("/", "/archive"):
                self._send_html(state.render_page())
            else:
                self._send_html("<h1>404</h1>", status=404)

        def do_POST(self) -> None:
            if self.path == "/archive/run":
                state.run_once()
                self.send_response(303)
                self.send_header("Location", "/archive")
                self.end_headers()
            else:
                self._send_html("<h1>404</h1>", status=404)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ArchiveHandler


def serve(
    tbd_dir: str | Path,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    """Run the local archive web service, opening /archive in a browser."""
    tbd_dir = Path(tbd_dir)
    state = ArchiveServerState(tbd_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(state))
    url = f"http://127.0.0.1:{server.server_address[1]}/archive"
    print(f"tbd archive service at {url} (serving {tbd_dir})")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
