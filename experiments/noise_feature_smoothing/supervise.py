"""Serial stage runner with a wall-clock deadline and bounded repair retries."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATE = HERE / "supervisor.json"
INITIAL_ATTEMPT = HERE / "process.stderr.log"
# Count the earlier failed launch toward the total budget.
START = INITIAL_ATTEMPT.stat().st_mtime if INITIAL_ATTEMPT.exists() else time.time()
END = START + 7 * 3600
STATUS = {"started_at_epoch": START, "deadline_epoch": END,
          "completed": [], "retries": {}, "pid": os.getpid()}
CHILD = None


def save(status, **extra):
    STATUS.update(status=status, **extra)
    STATUS["updated_at"] = datetime.datetime.now().astimezone().isoformat()
    STATUS["remaining_seconds"] = max(0, END - time.time())
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(STATUS, indent=2), encoding="utf-8")
    temp.replace(STATE)
    print(json.dumps(STATUS, ensure_ascii=True), flush=True)


def stop_child():
    if CHILD is None or CHILD.poll() is not None:
        return
    try:
        CHILD.send_signal(signal.CTRL_BREAK_EVENT)
        CHILD.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        # Windows venv executables can launch a second Python process.
        subprocess.run(["taskkill", "/PID", str(CHILD.pid), "/T", "/F"],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            CHILD.wait(timeout=2)
        except subprocess.TimeoutExpired:
            CHILD.kill()
            CHILD.wait()


def run():
    global CHILD
    save("running", stage=0)
    env = dict(os.environ, GEOLIFE_DEADLINE_EPOCH=str(END), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(HERE / "matplotlib_cache"))
    for stage in range(6):
        for retry in range(4):
            if time.time() >= END:
                save("deadline")
                return
            STATUS["retries"][f"stage{stage}"] = retry
            save("running", stage=stage, retry=retry)
            script_hash = hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest()
            snapshots = HERE / "source_snapshots"
            snapshots.mkdir(exist_ok=True)
            (snapshots / f"stage{stage}_attempt{retry}_{script_hash[:12]}.py").write_bytes(
                (HERE / "run_experiment.py").read_bytes())
            with (HERE / f"stage{stage}_attempt{retry}.stdout.log").open("ab") as stdout, (
                    HERE / f"stage{stage}_attempt{retry}.stderr.log").open("ab") as stderr:
                CHILD = subprocess.Popen(
                    [sys.executable, "-B", "-u", str(HERE / "run_experiment.py"), "--stage", str(stage)],
                    cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
                save("running", child_pid=CHILD.pid)
                while CHILD.poll() is None:
                    if time.time() >= END:
                        save("deadline", note="Saving partial products; stopping active worker.")
                        stop_child()
                        return
                    time.sleep(min(1, max(0.01, END - time.time())))
            if CHILD.returncode == 0:
                acceptance = json.loads((HERE / f"stage{stage}_acceptance.json").read_text())
                assert acceptance["passed"] is True
                STATUS["completed"].append(f"stage{stage}")
                save("stage_passed", child_exitcode=0)
                break
            save("repair_required", child_exitcode=CHILD.returncode)
            if retry == 3:
                save("terminated", note="Initial attempt and three retries failed.")
                return
            # The supervising agent reviews logs, searches primary documentation, records
            # an implementation-only repair, then creates this per-attempt approval file.
            request = HERE / f"repair_stage{stage}_attempt{retry}.json"
            request.write_text(json.dumps({"stage": stage, "attempt": retry,
                "code_sha256": script_hash, "child_exitcode": CHILD.returncode,
                "required": "Diagnose, search, record diff, repair implementation only."},
                indent=2), encoding="utf-8")
            ready = HERE / f"retry_stage{stage}_after{retry}.json"
            while not ready.exists():
                if time.time() >= END:
                    save("deadline")
                    return
                time.sleep(1)
            repair = json.loads(ready.read_text(encoding="utf-8"))
            assert repair["protocol_unchanged"] is True
        else:
            return
    save("complete")


if __name__ == "__main__":
    try:
        run()
    except BaseException as exc:
        stop_child()
        save("terminated", error=repr(exc))
        raise
