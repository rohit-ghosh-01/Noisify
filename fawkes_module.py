"""
fawkes_module.py

Runs Fawkes on a folder using its default behavior (auto-detects and
cloaks faces). No file organizing, no checking — that's handled by
core_module via a bash script after this returns.
"""

import subprocess


def cloak_folder(folder_path: str, mode: str = "low") -> str:
    """Run fawkes on folder_path. Returns 'done' or 'failed'."""
    try:
        result = subprocess.run(
            ["fawkes", "-d", folder_path, "-m", mode],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "failed"

    return "done" if result.returncode == 0 else "failed"