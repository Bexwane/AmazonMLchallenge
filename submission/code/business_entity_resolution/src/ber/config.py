import os
from pathlib import Path

# Folder holding train/ and test/ (the official `dataset/` directory). Override with BER_DATA.
DATA_DIR = Path(os.environ.get("BER_DATA", "D:/6ab10eb3b23ba_student_resource/student_resource/dataset"))
# Where cached intermediate artifacts go. Override with BER_WORK.
WORK_DIR = Path(os.environ.get("BER_WORK", Path(__file__).resolve().parents[2] / "artifacts"))

SEED = 42
