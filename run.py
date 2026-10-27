import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("Running the prediction pipeline...", flush=True)

subprocess.run(
    [sys.executable, str(ROOT / "src" / "predict.py")],
    cwd=ROOT,
    check=True,
)

print("Starting the Zenith dashboard...", flush=True)

subprocess.run(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app" / "dashboard.py"),
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--server.headless=true",
    ],
    cwd=ROOT,
    check=True,
)