"""Write a scratch split folder that lists all 64 users as "train", so build_windows.py cuts pure
(label-cut) windows for every user. Used only for the report-only all-users CV (step 6).
Part 1's data/splits is read, never written."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "data" / "splits"
DST = HERE / "scratch" / "splits_all_train"
DST.mkdir(parents=True, exist_ok=True)
users = json.loads((SRC / "train_users.json").read_text()) + json.loads((SRC / "test_users.json").read_text())
(DST / "train_users.json").write_text(json.dumps(sorted(users), indent=2))
(DST / "test_users.json").write_text("[]")
print(f"{len(users)} users -> {DST}")
