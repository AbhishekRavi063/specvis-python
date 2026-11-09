# Specvis / ReFreM — Python port

A fresh implementation in Python, living entirely inside this folder.

**Nothing outside `python/` is touched.** The Java tree in `specvis-source/`, the
`build/` output, `specvis.sh`, `experiments/` and every participant record stay exactly
as they are. The Java remains the reference implementation until this one is validated
against it.

---

## Why a port, and what "correct" means here

The Java is a working, published instrument. A rewrite is only worth anything if it
produces the *same numbers*, so every ported component is checked against output
captured from the Java rather than against my reading of it.

`tests/java_bisection_reference.csv` is the first of these fixtures: every answer
sequence the procedure can actually reach, at six vector lengths, with the next index,
the surviving index, the completion flag and the full candidate mask recorded at each
step. `tests/test_bisection_parity.py` replays all 148 states.

> If a parity test fails, **the port is wrong, not the fixture.** Regenerate a fixture
> only when the Java itself has deliberately changed.

---

## Running it

```bash
python/specvis.sh app       # the ReFreM perimetry app (see HANDOVER.md)
python/specvis.sh run       # settings dialog -> threshold session -> results map
python/specvis.sh perimetry # one eye of monocular perimetry -> visual-field map
python/specvis.sh build-table  # normal table from pilot perimetry sessions
python/specvis.sh test      # every check in python/tests
python/specvis.sh display   # refresh rate, presentable reversal rates, dropped frames
python/specvis.sh setup     # create python/.venv on a new machine (Python 3.11)
```

`run` saves each session to `python/output/<participant>_<date>_<time>/`: `trials.csv`
and `gabor_presentations.csv` in the Java's formats, `summary.csv` (one row per
location: valid / ceiling / floor, threshold, trials) and `session.json` (settings,
measured refresh, actual reversal rate, dropped frames, calibration status).

The session is a demonstration-grade runner: four quadrant locations, a reversing
Gabor, 4-2 staircase or Specvis bisection per location, interleaved at random, with
practice and blank catch trials. It does not yet have the Java's five setup screens,
photometer entry, fixation monitoring or the R1/R2/R3 behavioural mode.

## What is built and verified

```
specvis/
  core/
    luminance.py     brightness <-> cd/m^2, both directions
    ladder.py        brightness vector, contrast ladder, contrast dB
    gabor.py         Gabor generation, DC correction, reversal, display floor
    reversal.py      reversal rate -> whole frames, what a display can present
    geometry.py      degrees <-> pixels, sphericity, quadrant and grid locations
  threshold/
    base.py          LocationThreshold, Result, Termination
    bisection.py     Specvis's own algorithm -- bit-exact with the Java
    staircase42.py   the 4-2 dB staircase, both boundaries detected
  data/
    trial_log.py     trials.csv, byte-compatible with the Java
    gabor_log.py     gabor_presentations.csv, byte-compatible with the Java
  stimulus/
    backend.py       the display seam, and NullBackend for tests
    psychopy_backend.py   frame-counted presentation, Retina-correct pixels
  procedure/
    session.py       the session loop, results, output files
run_session.py       the `run` command
tests/
  test_bisection_parity.py   148 reachable states vs the Java
  test_java_parity.py        11,499 cases vs the Java's own output
  test_core.py               ladder, calibration, Gabor invariants
  test_staircase42.py        step sizes, stopping rule, boundaries, floor
  test_reversal_and_geometry.py
  test_backend_seam.py
  test_session.py            whole sessions against simulated observers
```

Facts the tests pin, each reproducing a measured property of the Java:

| property | value |
|---|---|
| bisection parity | 148/148 reachable states, 6 vector lengths |
| default ladder | `[10, 13, 17, 23, 31, 42, 56, 74, 100]` — truncated, as upstream |
| ladder irregularity | 0.36 dB, from upstream's discarded rounding |
| Gabor DC deviation | < 1e-12 of nominal mean, including 1 c/deg cosine phase |
| pattern reversal | identical mean luminance, 96% of pixels differ |
| 4-2 worked trace | `0.2530 → 0.1596 → 0.2010 → 0.1596`, converged in 4 |
| both boundaries | floor and ceiling presented, then reported **invalid** |

### Deliberately preserved, including the flaws

The port reproduces upstream behaviour rather than improving it, because a port that
quietly fixes things cannot be validated against the original:

- **The truncated ladder.** `Functions.logspace` discards its own rounding result and
  the caller truncates, biasing every rung dimmer. `brightness_vector(...)` does the
  same by default; pass `rounding="nearest"` for the ladder the settings describe.
- **The bisection's floor blindness.** A participant who sees the dimmest rung is
  reported at that rung as a valid threshold, with no failure signal, while the ceiling
  correctly returns "no threshold". Documented in `BisectionLocationThreshold` and
  analysed in `documentation/ANALYSIS-specvis-threshold-algorithm.md`.
- **Java's out-of-range behaviour.** `remaining_levels` raises on answer arrays the
  real procedure cannot produce, exactly where the Java throws, rather than absorbing
  them.

---

## Still to build

| area | status |
|---|---|
| session runner, results map, session output | **demo-grade** (`specvis.sh run`) |
| setup UI (the Java's five screens) | **perimetry app built** (`specvis.sh app`, see `HANDOVER.md`): participants, presets, screen check, stimulus and procedure pages, results / history / compare. The 4-location `run` session still uses one dialog |
| photometer calibration entry | not started; uncalibrated scale, labelled |
| fixation monitoring (blind-spot checks) | not started |
| behavioural R1/R2/R3 / PAS mode | not started; Elliott block design under discussion |
| constant stimuli method | not started |
| visual field pre-test inside the port | **built** (`specvis.sh perimetry`): 72-point ZEST grid, seed points, starting guesses from our own normal table and same-quadrant neighbours, quick blind rule, pilot mode, catch trials, `map.png`. Not yet run on real people |
| EEG markers (LSL), photodiode, eye tracker | later milestone |

---

## The framework decision — settled

**PsychoPy on Python 3.11, with PyQt6 for the setup screens.**

Decided against the proposal (TPN/109205, section 6a), not on general preference. Four
requirements in it each independently rule out a generic GUI toolkit:

| proposal requirement | what it needs | PsychoPy | plain Qt |
|---|---|---|---|
| reversal at **25–40 Hz** | vsync-locked frame counting, 12.5–20 ms phases | `waitBlanking`, frame-by-frame control, dropped-frame reporting | no frame-timing guarantee |
| **EEG 64/128 ch** | triggers synchronised to stimulus onset | parallel port, serial, LSL | nothing; would be hand-rolled |
| **eye tracker** for fixation | Tobii integration | `iohub` supports Tobii natively | nothing |
| luminance **matched with a photometer** | calibration pipeline | Monitor Center, photometer support | nothing |

The eye-tracker row is on its own decisive: iohub speaks to Tobii directly, and writing
that from scratch against a vendor SDK is a project, not a task.

Specvis is a static perimetry program, and the note in the project memory already said
the 25–40 Hz requirement was "the most likely reason the Specvis base turns out to be
the wrong foundation". That is exactly what the port is for — so the replacement should
be the tool built for this problem rather than another general-purpose one.

### The seam that makes it safe

`specvis/stimulus/backend.py` defines eight methods. Everything above it — staircases,
Gabor generation, geometry, trial log — is pure Python and numpy and never learns what
draws anything. `NullBackend` implements the seam with no display at all, which is how
the presentation logic is tested without a monitor.

If PsychoPy ever has to go, one file is rewritten.

`present_train` is specified in **frames**, never milliseconds. A wall-clock timer
cannot hold a 12.5 ms phase: the jitter is comparable to the interval being measured,
and the SSVEP spectrum smears.

### What the monitor has to be

Computed by `presentable_rates`, and it matters for procurement:

| refresh | exact reversal rates in the 25–40 Hz band |
|---|---|
| **120 Hz** | **40**, 30 |
| **100 Hz** | 33.3, **25** |
| 76 Hz | 38 |
| 118.9 Hz (this laptop, adaptive) | 39.63, 29.73 — neither is a round number |

**No common refresh gives both 25 and 40 exactly** — that needs a rate divisible by 200.
Pick the band edge that matters before buying the panel. The development MacBook's
ProMotion display is adaptive and measures 118.9 Hz, so it is for development only,
never for timing validation.

**Retina displays.** On a Retina Mac, PsychoPy's `pix` units are logical points
(two physical pixels each) while `win.size` reports physical pixels. The backend
converts in one place (`_to_units`), so everything above the seam works in physical
pixels and a texel lands on exactly one pixel. Without that conversion every position,
size and spatial frequency is wrong by a factor of two.

## Honest limits

- **No timing claim is validated.** As in the Java: until a photodiode measures the
  actual display, any frame-accuracy statement is software self-report.
- **The luminance scale is nominal** unless a photometer measured it.
  `LuminanceCalibration.calibrated` is false for the uncalibrated default so every
  consumer can label its output.
- **No clinical use.** Same constraint as the Java tree.
