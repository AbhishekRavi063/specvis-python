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
4. **Step 4** - for the healthy people who build the normal table: **Procedure ZEST (full)**
   and **Pilot mode On** (see section 4 for how many and which ages).
5. **START** - check the summary, press **Start**.

During the test: **SPACE** = seen, **P** = pause, **ESC** = stop (what was done is kept).
At the end the Results window shows the maps and the checks; then *Test the other eye*.

**Before every test:** dark room; screen brightness fixed (auto-brightness, True Tone,
Night Shift, eco modes off); chin rest at the set distance (57 cm lab monitor, 35 cm
MacBook); the other eye covered with an **opaque** patch.

A longer operator guide is kept with the team, outside the repository.

## 3. How the test works

- **Grid:** 72 points (+/-1 ... +/-21 deg across, +/-1 ... +/-13 deg up/down, none on a
  midline); the 4 inside the natural blind spot are skipped -> **68 tested**.
- **Trial:** the + stays on -> Gabor 200 ms -> time to answer (up to 1200 ms; after a few
  answers it follows the person's own reaction times) -> random gap 300-600 ms (+300 ms after
  a "not seen", as in SITA Fast).
- **Background:** **10 cd/m2**, the Humphrey/SITA background (Padmini). On a normal 8-bit
  screen so dark a grey cannot draw faint patches (one grey step is ~4% contrast), so the
  patches are **dithered** (noisy-bit method): tiny random rounding that shows the right
  contrast on average. Without a photometer scale the screen is *assumed* (gamma 2.2,
  300 cd/m2 peak) and every result says so.
- **Method: SITA Fast with a ZEST engine** (Step 4 -> Procedure; "ZEST (full)" is the
  older, slower search). Taken from the published SITA papers (Bengtsson et al. 1997,
  Bengtsson & Heijl 1998, Heijl et al. 2019) and Turpin et al. 2003:
  - every point is a Bayesian estimate (ZEST), updated after every answer;
  - the 4 seed points (one per quadrant) keep the strict search; their results start
    their neighbours;
  - **starting guess** = our normal table + finished neighbours in the same quadrant
    (never across a midline), shaped like SITA's two models: mostly "normal", 20%
    "abnormal" (anywhere worse);
  - **stopping** is looser where vision is good and stricter where it is poor (SITA's
    error-related factor). A point at the normal level can stop after **2 "seen"**
    without a miss; anything worse than normal needs at least 4 patches with both a
    "seen" and a "not seen"; a point near blind must be seen twice;
  - **second search** at a point that ends far (0.8 log units) from its guess;
  - **blind:** never seen, then missed twice at 100% contrast;
  - presses less than 180 ms after the patch appears are **false presses**, not answers;
  - at the end every point is **re-estimated** with the false-press and miss rates the
    person actually had (SITA's post-processing).
- **Checks** (catch trials, fewer than before): 5% blank (false presses <= 15%), 5% strong
  patches (misses <= 10%), 5% blind-spot patches (seen <= 20%). A check fails only with
  **at least 3** bad answers as well, so one or two chance slips do not fail a good
  session. A session is **VALID** only if all pass.
- **Time (simulated):** pilot mode (no table) ~10 min per eye; with a normal table ~3-4 min
  for a healthy eye and ~6-9 min for a hemianopic eye. Real times will be a little longer.
- **Results:** sensitivity in **contrast dB** (= -20 log10 contrast; higher = better) on
  both the grid map and the contour map.

## 4. The normal table (pilot first)

1. Test **healthy people**, both eyes, **Procedure ZEST (full)** and **Pilot mode On** (no guesses). Each run VALID.
2. Step 4 -> *Settings* -> **Build normal table from valid pilot sessions** (invalid sessions
   are skipped; people are counted, not eyes; left and right eyes are aligned). The table
   is a **hill of vision** fitted to everyone's thresholds, using the published model
   (Pointer & Hess 1989; Baldwin, Meese & Baker 2012): sensitivity falls linearly in dB
   with eccentricity, one slope each for temporal, nasal, up and down, combined
   elliptically. Each point's table value is the hill's height there.
3. Everyone after that: **Pilot mode Off** -> the test starts each point near the normal value
   (faster) and marks points worse than normal as *weak*.

### Who and how many (pilot participants for the table)

There is no published minimum for a table like ours; these are practical estimates.

| Part of the table | Needs |
|---|---|
| The hill (5 fitted numbers) | about **10** healthy people |
| The normal spread (SD), used to mark spots as *weak* | about **15-20** healthy people |
| An age correction (one number for the whole field) | at least about **5 older** healthy people (about **50-70**), besides the young group |

Expected group: mostly lab members aged **20-30**, plus a few older healthy people.

| Young (20-30) | Older (50-70) | What is possible |
|---|---|---|
| 15-20 | 5 or more | the hill + one age correction (planned, not yet in the software) |
| 15-20 | 1-4 | the hill only; the older people are used to check the table |
| about 10 | 0 | the hill only, as a feasibility pilot |

Rules:

- **Minimum: about 15-20 healthy people**, all sessions VALID, both eyes each.
- Record every participant's **age** in Step 1.
- The people who build the table are **not** reused as healthy controls in the study.
- Build the table on the **lab monitor**, at the **final background luminance**: a table is
  only valid on the screen and background it was made with.
- With young people only, the table serves as SITA Fast's **starting guess**; patients'
  *weak / intact* spots should be judged against their own seeing hemifield.

Details and sources: `documentation/ReFreM-SITA-Fast-Study-Plan.md` in the specvis repo.

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
  tools/          check_pixels.py (checks the patch is drawn exactly on a real screen)
  run_app.py, run_test_process.py, run_perimetry.py, build_prior_table.py
```

## 8. Known limits

- The background and contrast are **assumed** until the monitor is calibrated with a
  photometer (Step 2 -> luminance scale *New*); then the 10 cd/m2 background is exact.
- SITA Fast settings were tuned on simulated observers, not yet on people; compare a few
  real sessions with "ZEST (full)" before relying on them.
- Timing is software self-report (no photodiode check yet).
- No eye tracker yet: fixation is checked only with blind-spot catch trials.
- Tested on a 14" MacBook Pro (Apple silicon). **Windows not yet tested** - set display
  scaling to 100% and run *Screen check* first.
- 4 c/deg is hard to see far from the centre even for healthy eyes (outer points often
  "weak"); a team decision whether to keep it.

## 9. Licence

Built on Specvis (Dzwiniel et al., GPL-3.0-or-later); this Python version follows the same
licence.
