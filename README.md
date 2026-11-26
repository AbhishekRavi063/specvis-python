# ReFreM perimetry (Python)

A monocular visual-field test (perimetry) for the ReFreM study, built in Python
(PsychoPy) as a rebuild of Specvis.

- **How it works:** a striped patch (Gabor) flashes at the points of a grid while the
  participant looks at a central +. They press a key when they see the patch.
- **Result:** at every point the program finds the threshold contrast, the contrast seen
  about half the time. The result is a map of seen, weak and blind spots.
- **Why:** the map places the EEG stimuli (Block 2) in truly blind and truly seeing parts of
  the field.

**Private project.** Do not make this repository public. Participant data, results,
calibration measurements and documents are never stored in it (see section 9).

---

## Contents

1. [Install](#1-install-once-per-computer)
2. [Everyday commands](#2-everyday-commands)
3. [Set up the lab screen](#3-set-up-the-lab-screen-once-and-after-any-change)
4. [Eye tracker](#4-eye-tracker-tobii-pro-fusion)
5. [Run a test (Block 1: perimetry)](#5-run-a-test-block-1-perimetry)
6. [What happens during a session](#6-what-happens-during-a-session)
7. [Block 2](#7-block-2-orientation--flicker-rate)
8. [Presets](#8-presets)
9. [Data: what is saved and where](#9-data-what-is-saved-and-where)
10. [Folder structure](#10-folder-structure)
11. [Known limits](#11-known-limits)
12. [Licence](#12-licence)

---

## 1. Install (once per computer)

You need **Python 3.11** from [python.org](https://www.python.org/downloads/). On Windows,
tick **"Add Python to PATH"**.

| Mac | Windows |
|---|---|
| double-click `python/Start ReFreM (Mac).command` | double-click `python\Start ReFreM (Windows).bat` |

The first start installs everything. It takes 5–15 minutes, needs internet and uses about
1–2 GB. After that the app opens directly.

To install from a terminal instead (run in the repository folder):

```bash
python/specvis.sh setup
```

```bash
python/specvis.sh test
```

All tests must say **PASS**. On Windows, use `python\specvis.bat` in place of
`python/specvis.sh` in every command in this README.

## 2. Everyday commands

| Command | What it does |
|---|---|
| `app` | the ReFreM app (Specvis-style window). On Windows it starts and stops the eye-tracker bridge itself. |
| `test` | runs every check in `python/tests` |
| `preflight` | checks the screen, ColorCAL, luminance scale and eye tracker before a session |
| `calibrate` | measures the screen with the ColorCAL and makes a **luminance scale** |
| `uniformity` | measures **screen evenness**: one grey at 9 places, with on-screen target rings |
| `flatfield` | makes, lists and checks **flat-field maps** (evenness correction) |
| `display` | measures the refresh rate, reversal rates and dropped frames |
| `block2` | Block 2 (orientation × flicker rate) on a finished Block 1 map |
| `build-table` | builds the normal table from valid pilot sessions |
| `redraw-maps` | redraws the maps of saved sessions |
| `tracker-setup` | once: the Python 3.10 environment for the Tobii SDK |
| `tracker-bridge` | runs the eye-tracker bridge on its own (`--fake` to try without the device) |
| `tracker-display` | shows or sets the eye tracker's screen geometry |
| `perimetry`, `app-simple`, `run` | older one-window versions, kept for testing |

Every tool takes `--help`, for example `python/specvis.sh calibrate --help`.

## 3. Set up the lab screen (once, and after any change)

### 3.1 Prepare the monitor

1. Measure the monitor's **visible width and height** with a ruler, in mm.
2. Choose a brightness setting, write it down, and **never change it** after calibrating.
3. Turn OFF:
   - dynamic contrast;
   - eco or auto brightness;
   - overdrive / motion blur reduction (such as DyAc);
   - on a Mac: True Tone and Night Shift.
4. Windows: set display scaling to **100%**, and the screen to its highest refresh rate.
5. Let the screen warm up for **30 minutes** in the dark test room.

### 3.2 Luminance scale (ColorCAL)

The luminance scale tells the app how bright each grey level really is. Without one, the
screen is *assumed*, and every result says "ASSUMED screen".

1. Plug in the ColorCAL and put it on its tripod against the **centre** of the screen.
2. Run:

   ```bash
   python/specvis.sh calibrate --name "Lab screen <date>" --background-cd 10
   ```

   - **Zero calibration:** it may ask you to cover the sensor. With `--auto-zero` it
     switches the screens off for about 10 seconds instead, so the screen goes dark.
   - It then shows full-screen greys for a few minutes. Don't touch anything.
3. Write down what it prints:
   - black and white (cd/m²);
   - the background grey;
   - the strongest dB, the one-grey-step dB, and the middle of the range.
4. The scale appears in the app (Step 2).

If the ColorCAL isn't found, add `--port`:
- Mac: run `ls /dev/cu.usbmodem*` to find the port.
- Windows: look in Device Manager → Ports (COM), for example `--port COM9`.

**Contrast dB** = −20·log10(Michelson contrast). A **larger dB means a fainter patch**.
With the presets (no dithering):
- the strongest patch is about 0.1 dB;
- the faintest is one grey step at the background (about 33–34 dB on the lab screen);
- the staircase seeds start in the middle of that range.

### 3.3 Screen evenness (`uniformity`)

This measures how much dimmer the screen is away from its centre.

```bash
python/specvis.sh uniformity --label "BenQ" --note "Brightness 20"
```

1. The screen shows one grey (the 10 cd/m² background by default; `--grey` or `--cd`
   changes it).
2. A **ring** marks each of the 9 places in turn: centre, top, bottom, left, right and the 4
   corners, 30 mm in from the edges.
3. Put the meter in the ring and press **ENTER**.
   - **R** measures the previous place again.
   - **ESC** stops. What was measured is kept, with `_PARTIAL` in the file name.

It saves a CSV (cd/m² and % against the centre for each place) and a summary in
`python/output/uniformity/`. It prints whether every place is within ±10% of the centre.
- `--figure` adds a map of the screen.
- `--grid "24-2 within 9"` also estimates the error at the grid's spots.

`calibrate --uniformity` runs the same 9 places straight after a calibration.

### 3.4 Flat-field correction (`flatfield`), optional

A flat-field map makes the background equally bright everywhere. Where the screen is
dimmer, the app draws the background (and the patches) slightly brighter.

- It is **off unless a map is chosen**. With no map, the app draws exactly as before.
- **Strength:** 80% of the measured difference is corrected by default (`--strength`).
- **How:** a fixed (static) dither pattern keeps the average brightness right everywhere.
- **Patches keep their contrast:** each patch is scaled with the background under it.
- **Speed:** the background is drawn once; only the patch and the + are redrawn, so the
  frame rate is kept.

1. Make a map from one or more uniformity runs (the median of each place is used):

   ```bash
   python/specvis.sh flatfield --median run1.csv run2.csv run3.csv --name "BenQ median" --monitor BenQ_XL2546K
   ```

   - `--from-csv file.csv` uses one run.
   - `--list` shows the saved maps.
2. Check it with the meter (A/B: plain, corrected, plain at each place):

   ```bash
   python/specvis.sh flatfield --check "BenQ median" --scale "Lab screen <date>"
   ```

3. In the app, **Step 2 → Flat-field map → Choose**.

The chosen map's name and strength are recorded in every session and in its report
("flat-field correction: <name>, 80%", or "none").

Maps are saved in `python/calibration/flatfield_maps.json`. They stay on the lab computer.

### 3.5 Frame timing (`display`)

```bash
python/specvis.sh display --fullscreen --scale "Lab screen <date>" --flat-field "BenQ median"
```

It measures the real refresh rate and counts dropped frames, with or without the
flat-field map. It also prints the median frame time and warns **TOO SLOW** if a frame
takes too long.

## 4. Eye tracker (Tobii Pro Fusion)

The Tobii Python library works only on **Python 3.10**, and the app uses 3.11. So a small
**bridge** program runs on 3.10 and passes gaze to the app.

**Once:**
1. Install **Python 3.10** from python.org.
2. Run `python/specvis.sh tracker-setup`. This makes `python/.venv-tobii`.
3. Install **Tobii Pro Eye Tracker Manager**. Set the **display area** to the stimulus
   screen (`tracker-display` shows or sets it).

**Each session:**
- On Windows, `app` starts the bridge itself and closes it at the end.
- To run it on its own: `python/specvis.sh tracker-bridge`, and leave it open. It prints
  samples per second, which should be about 120 or 250.
- To try the app without the device: `python/specvis.sh tracker-bridge --fake`.

The Fusion tracks from about 50 to 80 cm. The presets use 51–60 cm.

**Check everything before a session:**

```bash
python/specvis.sh preflight --width-mm 542.3 --height-mm 305 --distance-mm 570
```

Every line should say **OK**. A **TODO** line says what to do.

## 5. Run a test (Block 1: perimetry)

Start the app with `python/specvis.sh app` (or the Start ReFreM file). It has four steps
and START, as in Specvis.

1. **Step 1 – Patient**
   - *New* participant (an ID only, for example `P01`) or *Existing*.
   - Choose the tested eye.
   - Record the age.
   - *Settings* loads a preset from `python/presets/` (section 8).
2. **Step 2 – Screen and luminance scale**
   - Check the screen width, height and viewing distance.
   - It must say **"The grid fits on this screen."** *Screen check* shows a box that must
     measure 10 cm.
   - Choose the **luminance scale** (section 3.2).
   - Choose the **flat-field map**, or leave it at *None* (section 3.4).
3. **Step 3 – Stimulus and background:** keep the preset's values unless the team decides.
4. **Step 4 – Fixation and other**
   - Fixation monitoring: eye tracker, or blind-spot checks.
   - The procedure: **Full Threshold 4-2** is the study method.
   - Pilot mode.
5. **START:** check the summary. It shows the dB range and must not say "ASSUMED screen".
   Then press **Start**.

**Keys:** **SPACE** = seen, **P** = pause, **ESC** = stop (what was done is kept).

At the end, the Results window shows the maps and the checks. Then use *Test the other
eye*.

**Before every test:**
- dark room;
- screen settings unchanged since calibration;
- chin rest at the preset's distance;
- the other eye covered with an **opaque** patch.

## 6. What happens during a session

### 6.1 Head position (with the eye tracker)

The positioning screen turns **green** when all three are right:
- the eye distance is right (within the preset's tolerance);
- both eyes are **centred left–right**, between the circle's sides;
- both eyes are inside the tracker's view, left–right.

If not, it says what to do, for example "Eyes too far right: move the head to the left".

- **Eye height is not judged.** It is recorded in `session.json` as
  `gaze_check.eyes_above_screen_centre_mm`.
- Eyes near the top or bottom of the tracker's view still give green, with a note that
  tracking may drop out.
- Setting `head_check_vertical` to true brings back the old height check.

Then the tracker is calibrated for the tested eye, and its accuracy is measured.

### 6.2 The person's own blind spot

Each person's blind spot is found and mapped at the start, and saved per eye. Later
sessions only re-check it.

1. **Search:** it finds the blind spot near where it usually is.
2. **Map:** from that point it walks outwards in each direction until the patch is seen
   twice.
   - It starts **2.5° out** and steps **1°**, then checks once half way back, so each edge
     is found to 0.5°.
   - If the first spot is already seen, it walks inwards instead.
   - It tries **right, up, left and down** first. The 4 diagonals are added only if one
     direction finds no edge; then 6 of the 8 must find one.
   - It goes out to **6°** at most.
3. **Oval:** an oval is fitted to the edges.
   - If the oval's centre is more than 1° from the point found, it maps once more.
   - If too few directions find an edge (for example in a hemianopic eye), it stops and
     says so.

A normal blind spot takes about 28 patches, about 1 minute (simulated).

With the eye tracker, the **probe block** then runs before the test (our design):
- Two flashes are marked 1 and 2. The patch is in one of them, at the blind spot's centre.
- The person says which flash had it (guessing if need be), then whether they saw
  anything.
- A few trials at the mirror point on the other side check that the task works.

### 6.3 The test

- **Grids** (Step 2 → field):
  - **24-2 central:** 42 points, 6° apart, 51 cm;
  - **24-2 within 9:** 30 points, rows at 3° and 9° only, 57 cm;
  - **5° grid:** 56 points, 5° apart, 60 cm.

  Points inside the blind spot are left out.
- **Trial:** the + stays on → a Gabor for 200 ms → up to 1.2 s to answer → a random gap of
  300–600 ms.
- **Background:** 10 cd/m², as on the Humphrey perimeter.
- **Method:** Full Threshold 4-2 staircase, as published in the Open Perimetry Interface
  (OPI).
  - **Steps:** 8 then 4 of our contrast dB.
  - **Seeds:** 4 seed points start in the middle of the measurable range.
  - **Other points:** they start from their finished neighbours.
  - **Other methods:** ZEST (full) and SITA Fast are still in Step 4, for comparison.
- **Fixation:**
  - With the eye tracker, a patch shown while the eye was off the + is thrown away and
    repeated later.
  - A patch missed during a blink is repeated.
  - Without the eye tracker, patches at the person's own blind spot check fixation.
- **Checks:**
  - blank trials count false presses;
  - strong patches count misses;
  - blind-spot patches count losses of fixation.

  A session is **VALID** only if all checks pass.
- **Results:** sensitivity in contrast dB, on a grid map and a contour map.

### 6.4 The normal table (pilot)

1. Test healthy people, both eyes, with **Pilot mode On**. Every session must be VALID.
2. Build the table: run `python/specvis.sh build-table`, or use Step 4 → *Settings* →
   **Build normal table from valid pilot sessions**.
3. A table is valid only on the screen, distance and background it was made with.

## 7. Block 2 (orientation × flicker rate)

Block 2 shows horizontal or vertical stripes at every spot of a finished Block 1 map.

- **Order:** static first, then reversing at 20 Hz and 38 Hz.
- **Contrast:** each spot gets its own threshold contrast; "not seen" spots and the blind
  spot get the highest.
- **Catch trials:** included.

To run it:

```bash
python/specvis.sh block2 --participant P01 --scale "Lab screen <date>" --tracker
```

| Option | What it does |
|---|---|
| `--participant` | finds that person's latest finished Block 1 run |
| `--eye L` or `--eye R` | picks that eye's run |
| `--block1 <folder>` | picks one run by hand |
| `--flat-field <name>` | uses a flat-field map |
| `--no-map --windowed` | a try-out without a map |

With `--tracker`, Block 1's setup (head position, calibration, accuracy check) runs first.

Results go to `python/output/block2/`. For figures and a results table from one run, use
`python/tools/block2_report.py <run folder>` (with the `.venv` Python).

## 8. Presets

`python/presets/` holds the shared settings, so every computer runs the same test. All
use the BenQ 24.5" (542.3 × 305 mm), Full Threshold 4-2 and no dithering.

| Preset | Grid | Distance | Fixation |
|---|---|---|---|
| `1_lab_eye_tracker_24-2` | 24-2 central, 42 points | 51 cm | eye tracker |
| `2_lab_no_eye_tracker_24-2` | 24-2 central, 42 points | 51 cm | blind-spot checks |
| `3_lab_eye_tracker_5deg_60cm` | 5° grid, 56 points | 60 cm | eye tracker |
| `4_lab_eye_tracker_24-2_30pts_57cm` | 24-2 within 9, 30 points | 57 cm | eye tracker |

To change a preset:
1. Load it in the app.
2. Change the settings.
3. Use *Save settings* over the preset.
4. Commit it.

## 9. Data: what is saved and where

Everything is saved on the computer that ran the test:

```
python/output/participants.json                     participant records (IDs)
python/output/perimetry/<ID>_<eye>_<date>_<time>/   one folder per eye (Block 1):
    map.png, contour_map.png, summary.csv (one row per point),
    trials.csv (every patch), gaze.csv (eye tracker), session.json
python/output/block2/<ID>_<eye>_<date>_<time>/      Block 2
python/output/uniformity/                           screen-evenness runs
python/calibration/                                 luminance scales, flat-field maps,
                                                    photometer readings, tracker backups
python/output/app_errors.log                        only if something went wrong
```

`session.json` records the settings used, including:
- the luminance scale;
- the flat-field map (or "none");
- the blind-spot map;
- the gaze check.

**None of this goes to GitHub.** These are all git-ignored:
- `python/output/`;
- the measurements in `python/calibration/`;
- documents (`*.md` other than this README, `*.pdf`, `*.docx`, …);
- the `.venv` folders.

To share data, zip the session folders and pass them on privately. Deleting a session in
the app moves it to `_deleted/`.

## 10. Folder structure

```
python/
  specvis/
    app/          the Specvis-style app (specvis_ui.py), results maps, stores, assets
    procedure/    perimetry.py (Block 1, blind-spot map, head check), block2.py,
                  fixation.py, prior_table.py (normal table), session.py
    threshold/    full_threshold.py (4-2), zest.py, staircase42.py, bisection.py
    core/         Gabor, luminance, photometry, flatfield, geometry, reversal timing
    stimulus/     the display seam and the PsychoPy backend
    data/         trial and Gabor logs
  presets/        shared settings
  tests/          run with `python/specvis.sh test`
  tools/          calibrate, uniformity, flatfield, display, preflight, tracker tools,
                  block2_report, check_pixels
  run_app.py, run_block2.py, run_test_process.py, run_perimetry.py, build_prior_table.py
  specvis.sh, specvis.bat, Start ReFreM (Mac).command, Start ReFreM (Windows).bat
```

## 11. Known limits

- **Without a luminance scale** the screen is assumed (gamma 2.2), and the results say so.
- **The faintest patch** without dithering is one grey step at the background.
- **Timing** is the software's own report. There is no photodiode check yet.
- **Settings marked "our suggestion"** in the presets are our own choices, still to tune on
  real participants. Examples: the fixation window, blink handling, and the blind-spot
  map's start and steps.
- **The flat-field correction** is new. Check it with `flatfield --check` before relying
  on it.

## 12. Licence

Built on Specvis (Dzwiniel et al., GPL-3.0-or-later). This Python version follows the same
licence.
