# ReFreM perimetry - handover

How to install, run and use the perimetry test. For Aflah and anyone running it in the lab.

## What it is

A monocular visual-field test. A striped patch (Gabor) flashes at 68 points, one eye at a
time; the person presses SPACE when they see it. At every point the program finds the
faintest contrast still seen (ZEST). The result is a map: seen / weak / blind.

The map is used to choose the EEG Gabor positions.

## 1. Install (once per computer)

You need **Python 3.11** (python.org). Then:

| Mac | Windows |
|---|---|
| double-click `python/Start ReFreM (Mac).command` | double-click `python\Start ReFreM (Windows).bat` |

The first start installs everything (a few minutes, needs internet). After that it opens
the app straight away.

From a terminal instead: `python/specvis.sh setup` then `python/specvis.sh app`
(Windows: `python\specvis.bat setup`, `python\specvis.bat app`).

Check the install: `python/specvis.sh test` - all tests must pass.

## 2. Set up the screen and room (every time)

- Dark room, same lighting for everyone.
- Screen: auto-brightness **off**, True Tone / Night Shift / eco / dynamic contrast **off**,
  brightness fixed. Laptop: plugged in.
- Chin rest, eye **57 cm** from the lab monitor (35 cm for the MacBook).
- Other eye covered with an **opaque patch** (not a hand).

## 3. Run a test (the app - Specvis-style window)

The app looks and works like Specvis (Step 1-4, START, dark bottom bar with Back/Next),
with our test inside. (`python/specvis.sh app-simple` opens the earlier one-window app.)

1. **Step 1 - Patient**: *New* (ID only, e.g. `PILOT02`; names are optional, leave them
   empty) or *Existing*. Choose the **tested eye** (the open eye). Choose a **Preset**.
2. **Step 2 - Screen and luminance scale**: check the size (mm) and the patient distance.
   It must say **"The grid fits on this screen."** First time on a monitor: *Screen check*
   shows a box that must measure **10 cm**. With a photometer: *New* scale, type the six
   readings, check *Fit*.
3. **Step 3 - Stimulus and background**: leave as is (team default: 2 deg, 4 c/deg, 200 ms).
4. **Step 4 - Fixation and other**: leave as is. *Settings* (procedure): **pilot mode on**
   for the first 3-5 healthy people.
5. **START** - read the summary, press **Start**.

The test runs full screen. SPACE = seen, P = pause, ESC = stop (what was done is kept).
At the end the **Results** window shows the Specvis-style contour map (isofactor,
interpolation, colour/monochrome, export), the grid, every point and the checks. Choose
**Test the other eye now** or **Repeat this eye**. *Results* in Step 1 lists every session
with Info, Map, Compare results and Delete (Delete moves it to `_deleted/`, recoverable).

**A session must say VALID.** If not, the window says why (most often: the blind spot was
seen, so the other eye was not fully covered). Repeat that eye after a break.

## 4. Presets - same settings for everyone

Presets are in `python/presets/` and travel with the code, so everyone gets the same ones.

- *MacBook 14" - practice*: practice only, not for the table.
- *Lab monitor - pilot (normal table)*: pilot mode on.
- *Lab monitor - controls and patients*: pilot mode off.

The lab presets have **no screen width yet**. Measure the lab monitor's visible width
once, enter it in step 2, run the *Screen check*, then *Save settings file...* over the
two lab presets. Share that change with the code so nobody re-types it.

## 5. The normal table

After 3-5 valid pilot sessions (both eyes each), go to step 4 and press **Build normal
table**. Then use the *controls and patients* preset (pilot mode off): the test starts each
point from the table and from finished neighbours in the same quadrant, so it is faster.

Only valid sessions are used. People are counted, not eyes. Fewer than 10 people is fine
for testing the method, not yet a reliable table. For patients the table should come from
age-matched (about 40-75) healthy people, on the lab monitor.

## 6. Where the data is

`python/output/` on the computer that ran the test:

- `participants.json` - the participant records
- `perimetry/<ID>_<eye>_<date>/` - one folder per eye: `map.png`, `summary.csv` (one row
  per point), `trials.csv` (every flash), `session.json` (settings and checks)
- `perimetry/prior_table.csv` - the normal table

**This folder is never put on GitHub** (git ignores it). To share data, zip
`python/output/` and send it privately. Nothing in the app deletes a session.

## 7. Known limits

- Contrast is **nominal** until the monitor is calibrated with a photometer.
- Timing is software self-report (no photodiode check yet).
- No eye tracker yet: fixation is checked with blind-spot catch trials only.
- Tested on a Mac (14" MacBook Pro). **Windows is not tested yet**: on Windows, set the
  display scaling to 100% for the stimulus monitor and run the Screen check before use.
- 4 c/deg is hard to see far from the centre even for healthy eyes; points at 17-21 deg
  often come out "weak". A team decision whether to keep it.

## 8. If something goes wrong

| Problem | Fix |
|---|---|
| "The grid does not fit" | Wrong width/distance, or the screen is too small at that distance. Check both; on a laptop use 35 cm. |
| Blind spot seen (NOT VALID) | Cover the other eye fully; check the right eye was chosen; keep looking at the +. |
| Many false presses | The person is guessing: explain again, press only when sure. |
| Screen check box is not 10 cm | The width is wrong (or Windows scaling is not 100%). |
| App does not start | Run `python/specvis.sh test` (Windows: `specvis.bat test`) and send the output. |
