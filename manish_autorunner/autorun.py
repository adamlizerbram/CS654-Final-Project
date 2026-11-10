"""Personal JupyterHub <-> GitHub autorunner for Manish (Track 1).

Watches ``manish_autorunner/job.sh`` on the current git branch. Whenever
``job.sh`` is updated and pushed from a local machine, this runner pulls it on
JupyterHub, runs it inside ``./.conda/cs654`` (if present), streams output, and
periodically commits/pushes ``manish_autorunner/status/`` back to git so progress
and ``results/summary*.csv`` can be inspected from anywhere.

Usage (from repo root on JupyterHub):
  python manish_autorunner/autorun.py --watch
  python manish_autorunner/autorun.py --cmd "python -m eval.evaluate --agent random --episodes 10"
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_DIR = Path(__file__).resolve().parent
JOB_FILE = RUNNER_DIR / "job.sh"
STATUS_DIR = RUNNER_DIR / "status"
STATE_FILE = STATUS_DIR / "state.json"
LOG_TAIL_FILE = STATUS_DIR / "latest.log"
MAX_GIT_LOG_LINES = 400


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_idle_job(path: Path) -> bool:
    if not path.exists():
        return True
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.startswith("# IDLE")
    return True


def _detect_gpu() -> str:
    if not shutil.which("nvidia-smi"):
        return "none (cpu)"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        return out.splitlines()[0] if out else "unknown gpu"
    except Exception:
        return "unknown gpu"


def _build_job_env() -> dict[str, str]:
    env = os.environ.copy()
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in env:
                env[key] = val

    conda_bin = REPO_ROOT / ".conda" / "cs654" / "bin"
    if conda_bin.is_dir():
        env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
        env["CONDA_PREFIX"] = str(conda_bin.parent)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def _current_branch() -> str:
    res = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    return res.stdout.strip()


def _sync_from_remote(branch: str) -> None:
    _git(["fetch", "origin", branch], check=False)
    _git(["pull", "--rebase", "--autostash", "origin", branch], check=False)


def _copy_summary_csvs() -> None:
    results_dir = REPO_ROOT / "results"
    if not results_dir.is_dir():
        return
    for src in sorted(results_dir.glob("summary*.csv")):
        if src.is_file():
            shutil.copy2(src, STATUS_DIR / src.name)


def _write_tail_log(full_log: Path) -> None:
    if not full_log.exists():
        LOG_TAIL_FILE.write_text("", encoding="utf-8")
        return
    lines = full_log.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-MAX_GIT_LOG_LINES:]
    LOG_TAIL_FILE.write_text("\n".join(tail) + ("\n" if tail else ""), encoding="utf-8")


def _write_state(payload: dict[str, Any]) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _git_push_status(branch: str, message: str, *, use_git: bool) -> None:
    if not use_git:
        return
    _git(["add", "--", "manish_autorunner/status/"], check=False)
    diff = _git(["diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        return
    commit_args: list[str] = []
    if _git(["config", "user.name"], check=False).returncode != 0:
        commit_args.extend(["-c", "user.name=Manish Thapa"])
    if _git(["config", "user.email"], check=False).returncode != 0:
        commit_args.extend(["-c", "user.email=mthapa@sdsu.edu"])
    commit = _git([*commit_args, "commit", "-m", message], check=False)
    if commit.returncode != 0:
        return
    push = _git(["push", "origin", branch], check=False)
    if push.returncode != 0:
        _git(["pull", "--rebase", "--autostash", "origin", branch], check=False)
        _git(["push", "origin", branch], check=False)


def run_shell_command(
    command: str,
    *,
    job_hash: str,
    branch: str,
    sync_seconds: int = 30,
    use_git: bool = True,
) -> int:
    """Execute a shell command in REPO_ROOT and sync status/logs periodically."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    full_log_dir = REPO_ROOT / "logs"
    full_log_dir.mkdir(parents=True, exist_ok=True)
    full_log = full_log_dir / "autorun_full.log"

    started_at = _utc_now()
    gpu = _detect_gpu()
    state: dict[str, Any] = {
        "status": "running",
        "job_hash": job_hash,
        "started_at": started_at,
        "updated_at": started_at,
        "exit_code": None,
        "gpu": gpu,
        "host": os.uname().nodename,
        "command_preview": command.strip().splitlines()[0][:120] if command.strip() else "",
    }
    _write_state(state)
    LOG_TAIL_FILE.write_text(f"[{started_at}] Starting job...\n", encoding="utf-8")
    _git_push_status(branch, f"autorun: started job {job_hash[:8]}", use_git=use_git)

    env = _build_job_env()
    proc = subprocess.Popen(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pump_output() -> None:
        assert proc.stdout is not None
        with full_log.open("w", encoding="utf-8") as lf:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
                lf.flush()

    pump = threading.Thread(target=_pump_output, daemon=True)
    pump.start()

    last_sync = time.monotonic()
    while proc.poll() is None:
        time.sleep(2)
        if time.monotonic() - last_sync >= sync_seconds:
            _write_tail_log(full_log)
            _copy_summary_csvs()
            state["updated_at"] = _utc_now()
            _write_state(state)
            _git_push_status(branch, f"autorun: progress {job_hash[:8]}", use_git=use_git)
            last_sync = time.monotonic()

    pump.join(timeout=5)
    rc = int(proc.returncode or 0)
    _write_tail_log(full_log)
    _copy_summary_csvs()
    state["status"] = "done" if rc == 0 else "failed"
    state["exit_code"] = rc
    state["updated_at"] = _utc_now()
    _write_state(state)
    _git_push_status(
        branch,
        f"autorun: {state['status']} (rc={rc}) {job_hash[:8]}",
        use_git=use_git,
    )
    return rc


def watch_loop(
    *,
    poll_seconds: int = 20,
    sync_seconds: int = 30,
    use_git: bool = True,
) -> None:
    """Poll remote git branch for updates to manish_autorunner/job.sh and run them."""
    branch = _current_branch() if use_git else "local"
    print(f"[autorun] Watching {JOB_FILE} on branch {branch!r} (poll={poll_seconds}s)")

    prev_state = _load_state()
    last_hash = str(prev_state.get("job_hash", ""))

    # Announce watcher online if idle or no prior state.
    if not prev_state:
        _write_state(
            {
                "status": "idle",
                "job_hash": _sha256(JOB_FILE),
                "started_at": _utc_now(),
                "updated_at": _utc_now(),
                "exit_code": 0,
                "gpu": _detect_gpu(),
                "host": os.uname().nodename,
                "command_preview": "# IDLE",
            }
        )
        _git_push_status(branch, "autorun: watcher online", use_git=use_git)

    while True:
        try:
            if use_git:
                _sync_from_remote(branch)

            current_hash = _sha256(JOB_FILE)
            if current_hash and current_hash != last_hash and not _is_idle_job(JOB_FILE):
                cmd = JOB_FILE.read_text(encoding="utf-8")
                print(f"[autorun] Detected new job {current_hash[:8]}; executing...")
                run_shell_command(
                    cmd,
                    job_hash=current_hash,
                    branch=branch,
                    sync_seconds=sync_seconds,
                    use_git=use_git,
                )
                last_hash = current_hash
        except KeyboardInterrupt:
            print("\n[autorun] Stopped by user.")
            break
        except Exception as exc:
            print(f"[autorun] Watcher error: {exc}", file=sys.stderr)
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Personal JupyterHub autorunner.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll git for changes to manish_autorunner/job.sh.",
    )
    mode.add_argument(
        "--run-job",
        action="store_true",
        help="Run manish_autorunner/job.sh once right now.",
    )
    mode.add_argument(
        "--cmd",
        help="Run an inline shell command once and sync logs/status.",
    )
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--sync-seconds", type=int, default=30)
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Disable git pull/commit/push (useful for local testing).",
    )
    args = parser.parse_args(argv)

    use_git = not args.no_git
    branch = _current_branch() if use_git else "local"

    if args.watch:
        watch_loop(
            poll_seconds=args.poll_seconds,
            sync_seconds=args.sync_seconds,
            use_git=use_git,
        )
        return 0
    if args.run_job:
        cmd = JOB_FILE.read_text(encoding="utf-8")
        return run_shell_command(
            cmd,
            job_hash=_sha256(JOB_FILE),
            branch=branch,
            sync_seconds=args.sync_seconds,
            use_git=use_git,
        )
    cmd_hash = hashlib.sha256(args.cmd.encode("utf-8")).hexdigest()
    return run_shell_command(
        args.cmd,
        job_hash=cmd_hash,
        branch=branch,
        sync_seconds=args.sync_seconds,
        use_git=use_git,
    )


if __name__ == "__main__":
    raise SystemExit(main())
