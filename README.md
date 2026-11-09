# ReFreM perimetry - Python

A monocular visual-field test (perimetry) for the ReFreM study, in a Python rebuild of
Specvis. A striped patch (Gabor) flashes at 68 points of one eye's visual field; the
participant presses SPACE when they see it. At every point the program finds the
faintest contrast still seen. The result is a map - seen / weak / blind - used to place
the EEG Gabors in truly blind and truly seeing parts of the field.

**Private project.** Do not make this repository public. Participant data is never stored
in it (see *Data* below).

---

## 1. Install (once per computer)

Needs **Python 3.11** from [python.org](https://www.python.org/downloads/)
(Windows: tick **"Add Python to PATH"**). Then:

| Mac | Windows |
|---|---|
| double-click `python/Start ReFreM (Mac).command` | double-click `python\Start ReFreM (Windows).bat` |

The first start installs everything (5-15 min, needs internet, ~1-2 GB). After that it
opens the app directly. From a terminal instead:

```bash
python/specvis.sh setup     # once  (Windows: python\specvis.bat setup)
python/specvis.sh app       # the app (Windows: python\specvis.bat app)
python/specvis.sh test      # check the install: all tests must pass
```

## 2. Run a test

The app follows Specvis: **Step 1** Patient, **Step 2** Screen and luminance scale,
**Step 3** Stimulus and background, **Step 4** Fixation and other, **START**.

1. **Step 1** - *New* participant (ID only, e.g. `PILOT02`), tested eye, *Settings* (load a
   preset from `python/presets/`).
2. **Step 2** - screen width and patient distance must be right: it must say **"The grid
   fits on this screen."** *Screen check* shows a box that must measure 10 cm.
3. **Step 3** - leave the defaults (2 deg, 4 c/deg, 200 ms) unless the team decides.
4. **Step 4** - **Pilot mode On** for the first 3-5 healthy people.
5. **START** - check the summary, press **Start**.

During the test: **SPACE** = seen, **P** = pause, **ESC** = stop (what was done is kept).
At the end the Results window shows the maps and the checks; then *Test the other eye*.

**Before every test:** dark room; screen brightness fixed (auto-brightness, True Tone,
Night Shift, eco modes off); chin rest at the set distance (57 cm lab monitor, 35 cm
MacBook); the other eye covered with an **opaque** patch.

Full operator guide: [`python/HANDOVER.md`](python/HANDOVER.md).

## 3. How the test works

- **Grid:** 72 points (+/-1 ... +/-21 deg across, +/-1 ... +/-13 deg up/down, none on a
  midline); the 4 inside the natural blind spot are skipped -> **68 tested**.
- **Trial:** fixation cross 500 ms -> Gabor 200 ms -> up to 1200 ms to answer -> random gap
  300-800 ms.
- **Method: ZEST** (Bayesian). Each point starts at 50% contrast (or at a starting guess),
  the estimate is updated after every answer, the next patch is shown at the estimate, and
  the point stops when the estimate's SD < 0.15 log10 (4-20 patches; ~6 on average).
- **Starting guesses, SITA-style** (Pragathi's proposal): from **our own normal table** and
  from finished **neighbours in the same quadrant** (never across a midline, so stroke
  borders are not blurred). No final smoothing.
- **Quick blind rule:** missed 3x at 100% contrast -> blind. **Floor rule:** seen even at the
  faintest contrast -> reported as such, not as a threshold.
- **Checks** (catch trials): 10% blank (false presses <= 15%), 5% strong patches (misses
  <= 10%), 5% blind-spot patches (seen <= 20%). A session is **VALID** only if all pass.
- **Time:** ~16-20 min per eye in pilot mode.

## 4. The normal table (pilot first)

1. Test **3-5 healthy people**, both eyes, **Pilot mode On** (no guesses). Each run VALID.
2. Step 4 -> *Settings* -> **Build normal table from valid pilot sessions** (invalid sessions
   are skipped; people are counted, not eyes; left and right eyes are aligned).
3. Everyone after that: **Pilot mode Off** -> the test starts each point near the normal value
   (faster) and marks points worse than normal as *weak*.

For patients, the table should come from 20-30 age-matched (about 40-75) healthy people,
tested on the **lab monitor** (a table is only valid on the screen it was made on).

## 5. Data

Everything is saved automatically, on the computer that ran the test:

```
python/output/participants.json                  participant records (IDs)
python/output/perimetry/<ID>_<eye>_<date>/        one folder per eye:
    map.png, summary.csv (one row per point), trials.csv (every patch), session.json
python/output/perimetry/prior_table.csv           the normal table
python/output/app_errors.log                      only if something went wrong
```

`python/output/` is **ignored by git** - it never goes to GitHub. To combine sessions from
several computers (for the table): use different IDs per person, zip the session folders,
share them privately, unzip into one computer's `python/output/perimetry/`, then build the
table. One eye is ~150-300 KB. Deleting a session in the app moves it to `_deleted/`.

## 6. Presets - same settings everywhere

`python/presets/` is shared through this repository:

- `1_macbook_practice` - 14" MacBook at 35 cm, practice only (not for the table)
- `2_lab_monitor_pilot` - lab monitor at 57 cm, pilot mode on
- `3_lab_monitor_study` - lab monitor at 57 cm, pilot mode off

The lab presets have **no screen width yet**: measure the monitor's visible width once,
enter it in Step 2, run *Screen check*, *Save settings* over the preset, and commit it.

## 7. Folder structure

```
python/
  specvis/
    app/          the Specvis-style app (specvis_ui.py), results maps, stores, assets
    procedure/    perimetry.py (the test), prior_table.py (normal table), session.py
    threshold/    zest.py (ZEST), staircase42.py (4-2), bisection.py (Specvis's own)
    core/         Gabor, luminance, ladder, geometry, reversal timing
    stimulus/     the display seam + PsychoPy backend (frame-counted, Retina-correct)
    data/         trial and Gabor logs
  presets/        shared settings
  tests/          run with python/specvis.sh test
  docs/           PARITY.md (Java comparison), REBUILD-PLAN.md
  run_app.py, run_test_process.py, run_perimetry.py, build_prior_table.py
  HANDOVER.md     the operator guide
  README.md       technical notes on the port
```

## 8. Known limits

- Contrast is **nominal** until the monitor is calibrated with a photometer (Step 2 ->
  luminance scale *New*).
- Timing is software self-report (no photodiode check yet).
- No eye tracker yet: fixation is checked only with blind-spot catch trials.
- Tested on a 14" MacBook Pro (Apple silicon). **Windows not yet tested** - set display
  scaling to 100% and run *Screen check* first.
- 4 c/deg is hard to see far from the centre even for healthy eyes (outer points often
  "weak"); a team decision whether to keep it.

## 9. Licence

Built on Specvis (Dzwiniel et al., GPL-3.0-or-later); this Python version follows the same
licence.
