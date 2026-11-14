#!/bin/sh
#
# Run, test and check the Python port. The Python counterpart of ../specvis.sh.
#
#     python/specvis.sh app       Specvis - ReFreM (Specvis-style window, our test)
#     python/specvis.sh app-simple  the earlier one-window perimetry app
#     python/specvis.sh run       settings dialog, a threshold session, the results map
#     python/specvis.sh perimetry one eye of monocular perimetry, the visual-field map
#     python/specvis.sh build-table  normal table from the pilot perimetry sessions
#     python/specvis.sh redraw-maps  redraw the grid and contour maps of saved sessions
#     python/specvis.sh test      run every check in python/tests
#     python/specvis.sh display   measure the screen: refresh rate, reversal rates, drops
#     python/specvis.sh calibrate measure the screen with the ColorCAL -> a luminance scale
#     python/specvis.sh preflight check the screen, ColorCAL, scale and eye tracker
#     python/specvis.sh tracker-setup   once: Python 3.10 env with the Tobii SDK
#     python/specvis.sh tracker-bridge  before a session with the eye tracker (leave open)
#     python/specvis.sh setup     create python/.venv from requirements.txt (Python 3.11)
#
# Sessions are saved in python/output/<participant>_<date>_<time>/.
#
# Uses python/.venv. On another machine run `setup` once first; PYTHON311 overrides
# which interpreter it is built from:
#
#     PYTHON311=/path/to/python3.11 python/specvis.sh setup

set -e

HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.venv/bin/python"

need_venv() {
    if [ ! -x "$PY" ]; then
        echo "python/.venv not found. Create it first:  python/specvis.sh setup" >&2
        exit 1
    fi
}

case "${1:-}" in
    run)
        need_venv
        exec "$PY" "$HERE/run_session.py"
        ;;
    app)
        need_venv
        exec "$PY" "$HERE/run_app.py"
        ;;
    app-simple)
        need_venv
        exec "$PY" "$HERE/run_app.py" --simple
        ;;
    perimetry)
        need_venv
        exec "$PY" "$HERE/run_perimetry.py"
        ;;
    build-table)
        need_venv
        shift
        exec "$PY" "$HERE/build_prior_table.py" "$@"
        ;;
    redraw-maps)
        need_venv
        shift
        exec "$PY" "$HERE/redraw_maps.py" "$@"
        ;;
    test)
        need_venv
        exec "$PY" "$HERE/run_tests.py"
        ;;
    display)
        need_venv
        shift
        exec "$PY" "$HERE/tools/measure_display.py" "$@"
        ;;
    calibrate)
        need_venv
        shift
        exec "$PY" "$HERE/tools/calibrate_photometer.py" "$@"
        ;;
    preflight)
        need_venv
        shift
        exec "$PY" "$HERE/tools/preflight.py" "$@"
        ;;
    tracker-setup)
        BASE=${PYTHON310:-python3.10}
        if ! command -v "$BASE" >/dev/null 2>&1; then
            echo "Python 3.10 not found. Install it from python.org (macOS 64-bit universal2" >&2
            echo "installer), then run this again, or set PYTHON310=/path/to/python3.10" >&2
            exit 1
        fi
        "$BASE" -m venv "$HERE/.venv-tobii"
        "$HERE/.venv-tobii/bin/python" -m pip install --upgrade pip
        "$HERE/.venv-tobii/bin/python" -m pip install tobii-research
        echo "Done. Before each session:  python/specvis.sh tracker-bridge"
        ;;
    tracker-bridge)
        shift
        if [ -x "$HERE/.venv-tobii/bin/python" ]; then
            exec "$HERE/.venv-tobii/bin/python" "$HERE/tools/tobii_bridge.py" "$@"
        fi
        echo "python/.venv-tobii not found. Run first:  python/specvis.sh tracker-setup" >&2
        exit 1
        ;;
    setup)
        BASE=${PYTHON311:-python3.11}
        "$BASE" -m venv "$HERE/.venv"
        "$HERE/.venv/bin/python" -m pip install --upgrade pip
        "$HERE/.venv/bin/python" -m pip install -r "$HERE/requirements.txt"
        echo "Done. Try:  python/specvis.sh test"
        ;;
    *)
        echo "usage: python/specvis.sh {app|app-simple|run|perimetry|build-table|redraw-maps|test|display|calibrate|preflight|tracker-setup|tracker-bridge|setup}" >&2
        exit 2
        ;;
esac
