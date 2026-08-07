"""Diagnostic: report template-match confidence on the current crafting screen.

Usage:
    - Put the game on the alchemy (or any) crafting screen.
    - Run:  uv run python scripts/debug_crafting_screen.py [device_serial]

Captures a live screenshot via adb and prints the best match confidence for
each crafting-screen template (colour, grey and process-upgrade), in both
colour and grayscale, so we can see why wait_for_any_template times out.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ADB = Path.home() / "AppData/Local/AdbAutoPlayer/binaries/adb.exe"
TEMPLATES_DIR = (
    Path(__file__).resolve().parent.parent
    / "adb_auto_player/games/afk_journey/templates/homestead"
)

TEMPLATES = [
    "cook_button.png",
    "alchem_button.png",
    "forge_button.png",
    "gray_make_button.png",
    "gray_alchem_button.png",
    "gray_forge_button.png",
    "process_upgrade_confirm.png",
    "process_upgrade_button.png",
    "multiplier_x10.png",
    "multiplier_state.png",
]


def capture(serial: str) -> np.ndarray:
    """Capture a screenshot from the given device via adb."""
    raw = subprocess.run(
        [str(ADB), "-s", serial, "exec-out", "screencap", "-p"],
        capture_output=True,
        check=True,
    ).stdout
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode screenshot.")
    return img


def best_conf(
    base: np.ndarray, template: np.ndarray, grayscale: bool
) -> tuple[float, tuple[int, int]]:
    """Return the best match confidence and location for a template."""
    b, t = base, template
    if grayscale:
        b = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(b, t, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


def main() -> None:
    """Capture the current screen and print per-template match confidence."""
    serial = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    base = capture(serial)
    print(f"Screenshot size: {base.shape[1]}x{base.shape[0]} (device {serial})")
    out = Path.home() / "AppData/Local/Temp/hs_crafting_screen.png"
    cv2.imwrite(str(out), base)
    print(f"Saved screenshot to {out}\n")
    print(f"{'template':<32}{'color':>10}{'gray':>10}  best-loc(color)")
    for name in TEMPLATES:
        path = TEMPLATES_DIR / name
        if not path.exists():
            print(f"{name:<32}  MISSING TEMPLATE FILE")
            continue
        tmpl = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if tmpl is None:
            print(f"{name:<32}  UNREADABLE")
            continue
        cv, loc = best_conf(base, tmpl, grayscale=False)
        gv, _ = best_conf(base, tmpl, grayscale=True)
        print(f"{name:<32}{cv:>10.3f}{gv:>10.3f}  {loc}")


if __name__ == "__main__":
    main()
