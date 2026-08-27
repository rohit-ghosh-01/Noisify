"""
test_fawkes.py

Simple manual test for the full pipeline as core_module will run it:
    fawkes_module.cloak_folder()
    -> organize_cloaked.sh
    -> verification_module.verify_cloak()

Usage:
    python test_fawkes.py /path/to/folder/with/images
"""

import subprocess
import sys
from pathlib import Path

import fawkes_module
import verification_module

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_fawkes.py /path/to/folder")
        sys.exit(1)

    folder_path = sys.argv[1]
    folder = Path(folder_path)

    print(f"[1] Folder: {folder}")
    if not folder.is_dir():
        print("FAIL: folder does not exist.")
        sys.exit(1)

    original_images = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not original_images:
        print("FAIL: no images found in folder to test with.")
        sys.exit(1)

    print("[2] Running fawkes_module.cloak_folder() ...")
    status = fawkes_module.cloak_folder(folder_path)
    print(f"    Status: {status}")

    if status != "done":
        print("FAIL: fawkes_module did not return 'done'.")
        sys.exit(1)

    print("[3] Running organize_cloaked.sh ...")
    script_path = Path(__file__).parent / "organize_cloaked.sh"
    if not script_path.is_file():
        print(f"FAIL: organize_cloaked.sh not found at {script_path}")
        sys.exit(1)

    result = subprocess.run(
        ["bash", str(script_path), folder_path],
        capture_output=True,
        text=True,
    )
    organize_status = result.stdout.strip()
    print(f"    Status: {organize_status}")
    if result.stderr:
        print(f"    stderr: {result.stderr.strip()}")

    if organize_status != "done":
        print("FAIL: organize_cloaked.sh did not return 'done'.")
        sys.exit(1)

    print("[4] Checking cloaked_version folder ...")
    cloaked_dir = folder / "cloaked_version"
    cloaked_files = sorted(cloaked_dir.glob("*")) if cloaked_dir.is_dir() else []
    print(f"    Files found: {[f.name for f in cloaked_files]}")

    if not cloaked_files:
        print("FAIL: cloaked_version folder is empty or missing.")
        sys.exit(1)

    print("[5] Running verification_module.verify_cloak() on each matched pair ...")
    any_verified = False
    for original in original_images:
        match = next((c for c in cloaked_files if c.stem.startswith(original.stem)), None)
        if not match:
            print(f"    Skipping {original.name}: no matching cloaked file found.")
            continue

        print(f"    Comparing {original.name} <-> {match.name}")
        verdict = verification_module.verify_cloak(str(original), str(match))
        print(f"    Result: {verdict}")

        if verdict["status"] == "ok":
            any_verified = True

    if not any_verified:
        print("FAIL: no image pair could be verified.")
        sys.exit(1)

    print("\nPASS: fawkes_module, organize_cloaked.sh, and verification_module all ran successfully.")


if __name__ == "__main__":
    main()