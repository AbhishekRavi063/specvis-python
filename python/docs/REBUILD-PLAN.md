# Plan: Specvis rebuilt in Python, screen for screen

Draft for approval, 9 November 2026. Goal: everything Specvis (Java) does, in Python,
so the Java can be retired and the project can use Python libraries freely.

## What "the same" means

1. **Same screens, same fields, same order.** Specvis's flow is Step 1 Patient, Step 2
   Screen and luminance scale, Step 3 Stimulus and background, Step 4 Fixation and other,
   then START, the procedure, and the results window. All 26 Java screens are rebuilt
   (PyQt6), with the same labels.
2. **Same numbers.** Given the same settings and the same key presses, Python produces
   the same thresholds and the same files as the Java, byte for byte where the Java's
   format allows. Checked against output captured from the real Java (the way the port's
   11,500 parity cases already are), not by reading the code.
3. **Same files.** `.sset` settings files, `session_info.txt`, `session_data.txt`,
   `trials.csv`, `gabor_presentations.csv` and the behavioural logs, in the Java's
   formats. Old Specvis files load.
4. **Same flaws, kept on purpose** (as the port already does): the truncated ladder and the
   bisection's floor blindness. A port that quietly fixes things cannot be checked
   against the original. Our own improvements (ZEST perimetry, 4-2) stay as extra
   options, clearly named.

## Already done in the port

Scientific core, checked against the Java: bisection, 4-2 staircase, Gabor, luminance
fit, ladders, dB, reversal timing, screen geometry, `trials.csv`, `gabor_presentations.csv`.
Plus our ZEST perimetry app. About 6-10% of the Java by volume (see `PARITY.md`).

## The work, in phases

Each phase ends usable and tested; nothing waits for the end.

| Phase | What | Java it replaces | Check |
|---|---|---|---|
| **A. Data** | Patient records (all Specvis fields: ID, name, gender, date of birth, age, contact, acuity L/R, notes); luminance scales; every settings class; `.sset` read and write; **read-only import of the Java's `patients.s`, `screenLuminanceScales.s` and `Results/`** | `Patient`, `LuminanceScale`, 12 settings classes, `SpecvisData` (~3,700 lines) | Load every `.sset` in the repo; import the three Java data folders and compare field by field |
| **B. Procedures** | The Basic procedure with all four fixation-monitor variants (none, blind spot, fixation-point change, both); ellipse, polygon and Gabor stimuli; bisection, 4-2 and fixed-contrast methods; pause/cancel keys; the stimulus distribution with sphericity | `ProcedureBasicFixMonitor*`, `ProcedureBasicStimulus/Data`, `FixedContrastMethod`, `ThresholdFactory` (~7,000) | Scripted key presses through Java and Python, same thresholds and same files |
| **C. Behavioural** | R1/R2/R3 with PAS, reversal plan, orientation schedule, behavioural log, unseen-discrimination analysis | `ProcedureBehavioural`, `BehaviouralTrialLog`, `OrientationSchedule`, `UnseenDiscriminationAnalysis` (~1,700) | Same, against the Java |
| **D. Screens** | All 26 screens: start window; new / existing / edit patient; screen and luminance scale (new / existing / edit / fit); stimulus and background with preview; stimuli-distribution preview; fixation and other with its five sub-dialogs; procedure settings; procedure preview and START; procedure results (show / save results / save settings) | 26 FXML screens + controllers (~8,000) | Every field present and wired; offscreen tests of each screen; side-by-side screenshots against the Java |
| **E. Results** | Patient results list (info / map / delete); the contour map (bicubic interpolation, iso-bands, dB / isofactor, colour bar, axes, fixation and blind-spot overlay, export); compare maps; compare table | `Contour2DMap`, `IsoCell`, results and compare controllers (~4,300) | Same map data and colour bands as the Java for the same session |
| **F. Handover** | Launchers, `HANDOVER.md`, Windows test, a parity checklist | - | Everything green before the Java is retired |

**Retiring the Java** comes only after the checklist is complete, and means an archive
(a git tag on the last Java version, then removal from the main branch), not deletion of
the history or of any experiment data.

## Decisions needed before starting

1. **Patient names.** Specvis stores first and last name, contact details. Keep the
   fields (same as Specvis) but leave them optional, recommending IDs only?
2. **Specvis's flaws.** Reproduce them exactly (recommended, as above), with our fixes as
   separate options?
3. **Our ZEST perimetry app.** Keep it as one more procedure inside the rebuilt Specvis
   (recommended), so there is one program.
4. **Order.** The pilot and the 17 November meeting do not need any of this: the current
   app covers them. Start Phase A now, alongside the pilot?

## Honest size

The Java is about 32,000 lines. This is several days of work across the phases, not one
session. Phase A plus B gives a Python Specvis that runs the same tests as the Java;
D and E give the same windows and maps.

## Risks

- **Java data import**: `patients.s` and `screenLuminanceScales.s` are Java-serialised.
  Python can read them with a library (`javaobj-py3`); to be confirmed on the real files
  before Phase A is called done. No Java file is modified.
- **Parity fixtures** need the Java to run once more to capture its output (the existing
  Java workspace, not this repository). That reads the Java; it does not change it.
- **Windows**: not tested yet anywhere in the port.
