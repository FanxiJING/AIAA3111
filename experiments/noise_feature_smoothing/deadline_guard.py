"""Independent wall-clock guard and Windows system/display execution lease."""
import ctypes
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def run():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    lease = kernel.SetThreadExecutionState(0x80000003)
    assert lease, ctypes.get_last_error()
    state = json.loads((HERE / "supervisor.json").read_text())
    end = state["deadline_epoch"]
    status = {"deadline_epoch": end, "supervisor_pid": state["pid"],
              "system_and_display_awake": True}
    write(HERE / "deadline_guard.json", dict(status, status="active", updated_at=time.time()))
    try:
        while True:
            state = json.loads((HERE / "supervisor.json").read_text())
            if state["status"] in ("complete", "terminated", "deadline"):
                write(HERE / "deadline_guard.json", dict(status, status="released",
                      observed_supervisor_status=state["status"], updated_at=time.time()))
                return
            if time.time() >= end:
                checkpoint_path = HERE / "checkpoint.json"
                checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
                checkpoint.update(status="deadline", remaining_seconds=0, updated_at=time.time())
                write(checkpoint_path, checkpoint)
                write(HERE / "deadline_guard.json", dict(status, status="stopping",
                      updated_at=time.time(), note="Partial artifacts and checkpoint saved."))
                # The stage worker's entire descendant tree includes Windows venv launchers.
                subprocess.run(["taskkill", "/PID", str(state["pid"]), "/T", "/F"],
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                return
            time.sleep(min(1, max(.01, end-time.time())))
    finally:
        kernel.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    run()
