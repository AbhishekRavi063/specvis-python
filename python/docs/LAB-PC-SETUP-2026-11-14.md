# Lab PC setup and screen calibration (14 Nov 2026)

The stimulus screen of the lab PC was calibrated with the ColorCAL on 14 Nov 2026. This
report records what was set up, what was measured, and what is still open, for later
sharing. It contains **no participant data**.

## Result

The ColorCAL calibration **passed**. A luminance scale named **"Lab screen"** was saved
(14 Nov 2026, 20:30) with no warnings. It can be chosen in the app (Step 2).

| Quantity | Value |
|---|---|
| Black (grey 0) | 0.125 cd/m² |
| White (grey 255) | 117.1 cd/m² |
| Background (target 10 cd/m²) | grey 87 = 9.925 cd/m² |
| Strongest patch at this background | 0.1 dB |
| One grey step at the background | 32.9 dB (the faintest without dithering) |
| Middle of the range | **16.5 dB**: where the staircase seeds start (without dithering) |
| Uniformity | not measured (the meter is fixed at the centre) |

Contrast dB is −20 log10(Michelson contrast); larger = fainter.

## Equipment and conditions

| Item | Detail |
|---|---|
| Monitor | **BenQ, 24.5" 16:9**. The monitor reports a picture area of **544 × 303 mm**. 1920 × 1080, one screen (PsychoPy screen 0). |
| Windows display scaling | 100% (DPI 96, checked) |
| Photometer | Cambridge Research Systems **ColorCAL MKII**, serial 473, firmware 200, USB (VID 0861:1001), on COM9 |
| Meter position | fixed on a tripod (Manfrotto 495 head) against the screen centre; not moved |
| Zero calibration | done with the monitor switched off: `calibrateZero()` gave True, and the dark reading was 0.0 cd/m² |
| Software | branch `full-threshold`, commit `c60e54c` at the time of the measurement; Python 3.11.5 (64-bit); psychopy-crs 0.0.2; PyQt6 6.7.1 / Qt 6.7.3 |
| Power | screen-off and sleep set to never; no screen saver |
| Other apps | LightScan closed; no other Python running |
| Monitor brightness setting | **not yet recorded: write it down and never change it** |
| Dynamic contrast / eco / overdrive off; 30-minute warm-up; dark room | **to be confirmed in writing** |

## Readings (saved run: one reading per level, 42 levels)

The 17 levels from 79 to 95 bracket the 10 cd/m² background.

| Grey | cd/m² | | Grey | cd/m² | | Grey | cd/m² |
|---|---|---|---|---|---|---|---|
| 0 | 0.1251 | | 80 | 8.3545 | | 94 | 11.4461 |
| 5 | 0.1463 | | 81 | 8.5712 | | 95 | 11.6796 |
| 10 | 0.2027 | | 82 | 8.8222 | | 105 | 13.8960 |
| 15 | 0.3315 | | 83 | 9.0006 | | 120 | 17.6909 |
| 20 | 0.5192 | | 84 | 9.2287 | | 135 | 22.1775 |
| 25 | 0.7945 | | 85 | 9.4725 | | 150 | 27.2007 |
| 30 | 1.1062 | | 86 | 9.7006 | | 165 | 34.6069 |
| 35 | 1.5262 | | 87 | 9.9251 | | 180 | 43.9958 |
| 40 | 1.9800 | | 88 | 10.1461 | | 195 | 54.4707 |
| 45 | 2.5273 | | 89 | 10.3872 | | 210 | 65.9334 |
| 50 | 3.1262 | | 90 | 10.5842 | | 225 | 79.2765 |
| 55 | 3.8216 | | 91 | 10.7738 | | 240 | 95.1452 |
| 60 | 4.5696 | | 92 | 10.9749 | | 255 | 117.1297 |
| 75 | 7.2997 | | 93 | 11.1988 | | | |
| 79 | 8.1513 | | | | | | |

On the lab PC the files are `python\calibration\photometer\20261114_203016_readings.csv`
and `20261114_203016_summary.json`.

## Checks

| Check | Expected | Measured | Verdict |
|---|---|---|---|
| Dark end positive | grey 0 about 0.05–0.3, nothing negative | 0.125; no zero or negative readings | pass |
| Rising with grey level | each value above the one before | all 42 rise | pass |
| White | about 100–150 cd/m² | 117.1 | pass |
| Background | about 10 cd/m² | 9.925 at grey 87 | pass |
| Repeatability | two runs agree | within about 1% at every shared level (e.g. grey 60: 4.558 vs 4.570) | pass |
| Warnings | none | none | pass |
| Uniformity | under about 10% difference | not measured | not done |

An earlier run placed the background near grey 105. In that run, other windows came in
front of the grey screen and the meter measured them. **Grey 87 is the value to trust.**

## Problems found on the day, and their fixes

| Problem | Cause | Fix (commit) |
|---|---|---|
| PyQt6 6.11 failed: "DLL load failed … procedure could not be found" | Qt 6.11 does not load on this PC, even with the latest Visual C++ runtime | Windows installs PyQt6 6.7.1 / Qt 6.7.3 (`445c691`) |
| `test_safeguards` failed on Windows | `time.monotonic()` only ticks every 15.6 ms on Windows | gaze and patch times use `perf_counter` (`b0e1683`) |
| `tobii-research` would not install | the Tobii SDK is published for Python 3.10 only; the app runs on 3.11 | the SDK runs in `python\.venv-tobii` (3.10.11) through `specvis.bat tracker-bridge`; it finds the Pro Fusion (250 Hz) |
| The calibration tool froze while looking for the meter | it tried every COM port; Intel AMT and Bluetooth ports hang when opened | the ColorCAL is found by its USB ID only (`79743ec`); `--port COM9` also works |
| Dark greys read 0 or negative | the meter's zero was off | zero calibration with the monitor off (`--auto-zero`, `694b7f2`, or by hand) |
| "Python not responding"; other windows came in front of the grey | the window did not redraw while the meter read | readings run in the background and the window keeps redrawing (`c60e54c`) |
| Each reading took 22.4 s | psychopy-crs 0.0.2 waits out 5 s timeouts around the reply | the reply is read directly (about 2.6 s), checked once against the driver (`40a30d5`) |
| Closing the window took about 1 hour, with nothing saved yet | not yet known | results and the scale are saved before the window closes (`40a30d5`) |
| ESC lost all readings | nothing was saved when stopped | readings so far are saved as `*_readings_PARTIAL.csv` (`40a30d5`) |

The fixes in `40a30d5` were tested with a simulated meter and port only. **The next real
run on the lab PC should confirm them.**

## The eye tracker and the viewing distance

- The **Tobii Pro Fusion's specified operating distance is 50–80 cm** (Tobii product
  page). On this 24.5" screen, the full 24-2 needs about 33–37 cm, so the tracker cannot
  be used with it.
- Two grids fit the tracker's range. Both are in the app, with an eye-tracker preset each:

  | Grid | Spots | Distance |
  |---|---|---|
  | 24-2 without the top/bottom rows | 42 | 52 cm |
  | Aflah grid | 68 | 55 cm |

- The plan is to test both grids on the same people and compare: time, where the border
  is found, agreement with the clinical report, repeatability, and fixation losses.
- **The team has not decided yet.**

## Update 15 Nov: the grid for the eye tracker, and the working distances

**The working distances**, recalculated with the monitor's own size (picture area
544 × 303 mm, so half the height is 151.5 mm). Up/down is the tight direction, and 1° is
added for half the 2° patch:

| Grid | Needs up/down | Farthest distance (151.5 mm ÷ tan) | Tobii (50–80 cm)? |
|---|---|---|---|
| Full 24-2 (52 spots) | 22° | 37.5 cm | no |
| **24-2 within ±15° (34 spots)** | 16° | **52.8 cm** | **yes: 50–52.8 cm** |
| 24-2 without top/bottom rows (42 spots) | 16° | 52.8 cm | yes |
| Aflah grid (68 spots) | 14° | 60.7 cm | yes |
| 30-2 | 28° | 28.5 cm | no |

**About the screen height:** the reported size (544 × 303) is slightly off 16:9. With square
pixels, 544 mm wide gives 306 mm tall. The app uses the smaller height (303 mm) for the
"does it fit" check, which is the safe side, and places the spots with square pixels.

**New default grid, following Padmini's points** (equidistant, equal on all sides, test at
about 50 cm):
- the 24-2 positions within **±15° in every direction** (3°, 9°, 15° on both axes), 6° apart
  everywhere;
- 36 points, minus the 2 blind-spot points, so **34 tested**;
- chin rest at **51 cm**, with the eye-distance check at **±1 cm**: the Tobii measures the
  real distance;
- precedent: Elliott et al. 2015 mapped the central 30° diameter (±15°) with the Octopus G2.

Preset: **"Lab monitor – eye tracker, 24-2 within 15 deg (51 cm)"**.

The Aflah grid and the 24-2 without its top/bottom rows stay available for comparison.

## Open items

- [ ] Record the monitor brightness setting used for this calibration
- [ ] Confirm dynamic contrast, eco and overdrive were off, and the 30-minute warm-up
- [ ] Preflight with the measured width (544 mm), e.g.
      `python\specvis.bat preflight --width-mm 544 --distance-mm 520`
- [ ] Decide whether to measure uniformity (`--uniformity`; the meter must be moved)
- [ ] Confirm the photometer-tool fixes (`40a30d5`) on a real run
- [ ] Team: the grid with the eye tracker (24-2 without its top/bottom rows, or the Aflah grid)
- [ ] Team: whether `python/calibration/luminance_scales.json` (screen calibrations)
      should be shared on GitHub or kept local
