"""Check, on a real screen, that every patch is drawn exactly as the program builds it.

Opens a small PsychoPy window for a few seconds and reads back what was drawn. Two
things are checked, both of which went wrong before 2026-11-10:

1. **The right way up.** PsychoPy draws an array's first row at the bottom; ours is the
   top. Upside down, a 45 deg tilt shows as 135 deg.
2. **Pixel for pixel.** A patch whose edge falls between pixels is resampled: about 1
   column in 8 is doubled or dropped, breaking the stripes and the dithering.

Run:  .venv/bin/python tools/check_pixels.py      (prints PASS or FAIL)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from specvis.core.gabor import GaborParameters, dither_8bit, render_brightness  # noqa: E402
from specvis.procedure.perimetry import (PerimetrySettings, background_for,  # noqa: E402
                                         default_calibration)
from specvis.stimulus.backend import Texture  # noqa: E402
from specvis.stimulus.psychopy_backend import PsychoPyBackend  # noqa: E402


def drawn(backend, image, centre):
    win = backend._win
    stim = backend._to_stim(Texture(image, centre))
    win.clearBuffer()
    stim.draw()
    frame = np.asarray(win._getFrame(buffer="back"))[..., 0].astype(int)
    win.flip()
    return frame


def main() -> int:
    s = PerimetrySettings()
    cal = default_calibration(s)
    mean, bg = background_for(s, cal)
    b = PsychoPyBackend(fullscreen=False)
    b.open(900, 700, bg)
    w, h = b.size_px
    ok = True
    try:
        # 1. orientation: white band at the top of the array must be at the top on screen
        img = np.full((100, 100), 0.2)
        img[:20, :] = 1.0
        fr = drawn(b, img, (w / 2, h / 2))
        rows = np.where(fr == 255)[0]
        up = rows.size and rows.mean() < fr.shape[0] / 2
        print(("PASS" if up else "FAIL"), "patches are drawn the right way up")
        ok &= bool(up)

        # 2. exact pixels, at awkward (fractional) positions, for real dithered Gabors
        base = GaborParameters(s.diameter_degrees, s.spatial_frequency_cpd,
                               s.orientation_degrees, 0.0, s.diameter_degrees / 6.0, mean, 1.0)
        rng = np.random.default_rng(0)
        worst = 0
        for c in (1.0, 0.1, 0.01):
            for dx in (0.0, 0.3, 0.5, 1.7):
                img = dither_8bit(render_brightness(base.with_contrast(c), cal, 61.2, 61.2), rng)
                cx, cy = w / 2 + dx, h / 2 + dx
                fr = drawn(b, img, (cx, cy))
                ph, pw = img.shape
                x0, y0 = round(cx - pw / 2), round(cy - ph / 2)
                got = fr[y0:y0 + ph, x0:x0 + pw]
                worst = max(worst, int(np.abs(got - np.round(img * 255)).max()))
        exact = worst == 0
        print(("PASS" if exact else "FAIL"),
              f"patches are drawn pixel for pixel (largest difference {worst} grey levels)")
        ok &= exact
        back = np.unique(drawn(b, np.full((50, 50), bg), (w / 2, h / 2)))
        same = back.tolist() == [round(bg * 255)]
        print(("PASS" if same else "FAIL"),
              f"a blank patch is invisible (screen grey {back.tolist()}, background "
              f"{round(bg * 255)})")
        ok &= same
    finally:
        b.close()
    print("ALL PASS" if ok else "SOMETHING FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
