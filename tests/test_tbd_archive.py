"""
Tests for the tbd archive service: state machine, file lock, atomic moves,
idempotency, and the local web page.
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import pytest

from strif.strif import main
from strif.tbd_archive import (
    EXIT_APPLIED,
    EXIT_ERROR,
    EXIT_NOTHING,
    ArchiveError,
    ArchiveLock,
    ArchiveServerState,
    ArchiveStateMachine,
    InvalidTransitionError,
    LockTimeoutError,
    apply_plan,
    plan_archive,
    run_archive,
    status_counts,
)


def write_item(tbd_dir: Path, status: str, name: str, content: str) -> Path:
    status_dir = tbd_dir / status
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def run_quiet(tbd_dir: Path, apply: bool, lock_timeout: float = 1.0):
    out, err = io.StringIO(), io.StringIO()
    code = run_archive(tbd_dir, apply=apply, lock_timeout=lock_timeout, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def all_files(tbd_dir: Path) -> list[Path]:
    return sorted(p for p in tbd_dir.rglob("*") if p.is_file())


#
# ---- 1. Empty directory ----


def test_empty_directory_nothing_to_archive(tmp_path: Path):
    tbd_dir = tmp_path / ".tbd"
    tbd_dir.mkdir()

    code, out, _ = run_quiet(tbd_dir, apply=True)

    assert code == EXIT_NOTHING
    assert "nothing to archive" in out
    # No archive directory or stray files are created for an empty run.
    assert not (tbd_dir / "archive").exists()
    assert all_files(tbd_dir) == []


def test_empty_directory_exit_code_via_cli(tmp_path: Path):
    tbd_dir = tmp_path / ".tbd"
    tbd_dir.mkdir()
    assert main(["tbd", "archive", "run", "--apply", "--tbd-dir", str(tbd_dir)]) == EXIT_NOTHING


#
# ---- 2. Name conflicts get a content-hash suffix, never an overwrite ----


def test_name_conflict_uses_hash_suffix(tmp_path: Path):
    tbd_dir = tmp_path / ".tbd"
    existing = write_item(tbd_dir, "archive/done", "task.md", "old content")
    source = write_item(tbd_dir, "done", "task.md", "new content")

    code, _, _ = run_quiet(tbd_dir, apply=True)

    assert code == EXIT_APPLIED
    # The pre-existing archived file is untouched.
    assert existing.read_text(encoding="utf-8") == "old content"
    # The conflicting item lands next to it with a content-hash suffix.
    digest = hashlib.sha1(b"new content").hexdigest()[:8]
    hashed = tbd_dir / "archive" / "done" / f"task.{digest}.md"
    assert hashed.is_file()
    assert hashed.read_text(encoding="utf-8") == "new content"
    assert not source.exists()


#
# ---- 3. Repeated runs are idempotent ----


def test_repeated_apply_is_idempotent(tmp_path: Path):
    tbd_dir = tmp_path / ".tbd"
    write_item(tbd_dir, "done", "a.md", "alpha")
    write_item(tbd_dir, "closed", "b.md", "beta")

    # Dry run first: reports a plan but moves nothing.
    code, out, _ = run_quiet(tbd_dir, apply=False)
    assert code == EXIT_APPLIED
    assert "Dry run" in out
    assert (tbd_dir / "done" / "a.md").is_file()
    assert not (tbd_dir / "archive").exists()

    code, _, _ = run_quiet(tbd_dir, apply=True)
    assert code == EXIT_APPLIED
    assert (tbd_dir / "archive" / "done" / "a.md").read_text(encoding="utf-8") == "alpha"
    assert (tbd_dir / "archive" / "closed" / "b.md").read_text(encoding="utf-8") == "beta"
    assert not (tbd_dir / "done" / "a.md").exists()
    assert not (tbd_dir / "closed" / "b.md").exists()

    # Second apply: nothing left to do, and no duplicates appear.
    code, out, _ = run_quiet(tbd_dir, apply=True)
    assert code == EXIT_NOTHING
    assert "nothing to archive" in out
    archived = [p.relative_to(tbd_dir) for p in all_files(tbd_dir)]
    assert archived == [Path("archive/closed/b.md"), Path("archive/done/a.md")]


def test_identical_readded_item_is_deduped(tmp_path: Path):
    tbd_dir = tmp_path / ".tbd"
    write_item(tbd_dir, "done", "a.md", "alpha")
    assert run_quiet(tbd_dir, apply=True)[0] == EXIT_APPLIED

    # The same item reappears in done/ (e.g. re-synced): it is dropped, not duplicated.
    write_item(tbd_dir, "done", "a.md", "alpha")
    plan = plan_archive(tbd_dir)
    assert [m.action for m in plan] == ["dedupe"]
    code, _, _ = run_quiet(tbd_dir, apply=True)
    assert code == EXIT_APPLIED
    archived = [p.relative_to(tbd_dir) for p in all_files(tbd_dir)]
    assert archived == [Path("archive/done/a.md")]


#
# ---- 4. Lock contention exits explicitly; stale locks are reclaimed ----


def test_lock_contention_fails_explicitly(tmp_path: Path):
    tbd_dir = tmp_path / ".tbd"
    tbd_dir.mkdir()
    write_item(tbd_dir, "done", "a.md", "alpha")
    lock_path = tbd_dir / "archive.lock"

    with ArchiveLock(lock_path, timeout=1.0):
        # A second locker times out instead of silently skipping.
        with pytest.raises(LockTimeoutError):
            ArchiveLock(lock_path, timeout=0.2).acquire()
        # The CLI-level runner reports the error and exits with EXIT_ERROR.
        code, _, err = run_quiet(tbd_dir, apply=True, lock_timeout=0.2)
        assert code == EXIT_ERROR
        assert "lock" in err.lower()
        # Nothing was moved while the lock was held.
        assert (tbd_dir / "done" / "a.md").is_file()

    # Lock file is released afterwards and archiving works again.
    assert not lock_path.exists()
    assert run_quiet(tbd_dir, apply=True)[0] == EXIT_APPLIED


def test_stale_lock_is_reclaimed(tmp_path: Path):
    tbd_dir = tmp_path / ".tbd"
    tbd_dir.mkdir()
    lock_path = tbd_dir / "archive.lock"
    stale = {"pid": 999999, "created": time.time() - 3600}
    lock_path.write_text(json.dumps(stale), encoding="utf-8")

    lock = ArchiveLock(lock_path, timeout=0.5, stale_after=300.0)
    lock.acquire()  # Reclaims the stale lock instead of timing out.
    lock.release()
    assert not lock_path.exists()


#
# ---- 5. A mid-apply exception rolls back and leaves no partial files ----


def test_midway_exception_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tbd_dir = tmp_path / ".tbd"
    first = write_item(tbd_dir, "done", "a.md", "alpha")
    second = write_item(tbd_dir, "done", "b.md", "beta")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-apply")
        return real_replace(src, dst)

    monkeypatch.setattr("strif.tbd_archive.os.replace", flaky_replace)

    plan = plan_archive(tbd_dir)
    assert len(plan) == 2
    with pytest.raises(ArchiveError, match="rolled back"):
        apply_plan(plan)

    # Both source files are restored byte-for-byte.
    assert first.read_text(encoding="utf-8") == "alpha"
    assert second.read_text(encoding="utf-8") == "beta"
    # No half-written file and no leftover temporary file anywhere.
    leftovers = [p for p in all_files(tbd_dir) if ".partial" in p.name]
    assert leftovers == []
    assert not (tbd_dir / "archive").exists() or all_files(tbd_dir / "archive") == []

    # And the archive can be retried cleanly afterwards.
    monkeypatch.undo()
    assert run_quiet(tbd_dir, apply=True)[0] == EXIT_APPLIED
    assert (tbd_dir / "archive" / "done" / "a.md").is_file()
    assert (tbd_dir / "archive" / "done" / "b.md").is_file()


#
# ---- 6. Illegal state transitions are rejected without touching files ----


def test_invalid_transition_rejected(tmp_path: Path):
    machine = ArchiveStateMachine()

    for status in ("open", "in_progress", "blocked", "archived"):
        with pytest.raises(InvalidTransitionError):
            machine.validate(status, "archived")
    for status in ("done", "closed"):
        assert machine.validate(status, "archived") == "archived"

    # Items in non-archivable states are never planned or moved.
    tbd_dir = tmp_path / ".tbd"
    open_item = write_item(tbd_dir, "open", "todo.md", "still open")
    write_item(tbd_dir, "in_progress", "wip.md", "half done")

    assert plan_archive(tbd_dir) == []
    code, _, _ = run_quiet(tbd_dir, apply=True)
    assert code == EXIT_NOTHING
    assert open_item.read_text(encoding="utf-8") == "still open"
    assert (tbd_dir / "in_progress" / "wip.md").is_file()
    assert not (tbd_dir / "archive").exists()


#
# ---- Web page: counts, last moves, last plan, one click per archive run ----


def test_server_state_page_and_run_once(tmp_path: Path):
    tbd_dir = tmp_path / ".tbd"
    write_item(tbd_dir, "done", "a.md", "alpha")
    write_item(tbd_dir, "open", "todo.md", "still open")

    state = ArchiveServerState(tbd_dir)
    page = state.render_page()
    assert "done" in page and "open" in page  # per-status counts table

    state.run_once()  # what the page button triggers

    assert state.last_error is None
    assert any("a.md" in move for move in state.last_moves)
    assert any("a.md" in step for step in state.last_plan)
    assert (tbd_dir / "archive" / "done" / "a.md").is_file()
    assert (tbd_dir / "open" / "todo.md").is_file()

    page = state.render_page()
    assert "a.md" in page  # last moves and last plan are shown
    counts = status_counts(tbd_dir)
    assert counts["done"] == 0
    assert counts["open"] == 1
    assert counts["archived"] == 1
