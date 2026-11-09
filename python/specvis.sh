#!/bin/sh
#
# Run, test and check the Python port. The Python counterpart of ../specvis.sh.
#
#     python/specvis.sh app       Specvis - ReFreM (Specvis-style window, our test)
#     python/specvis.sh app-simple  the earlier one-window perimetry app
#     python/specvis.sh run       settings dialog, a threshold session, the results map
#     python/specvis.sh perimetry one eye of monocular perimetry, the visual-field map
#     python/specvis.sh build-table  normal table from the pilot perimetry sessions
#     python/specvis.sh test      run every check in python/tests
#     python/specvis.sh display   measure the screen: refresh rate, reversal rates, drops
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
    test)
        need_venv
        exec "$PY" "$HERE/run_tests.py"
        ;;
    display)
        need_venv
        shift
        exec "$PY" "$HERE/tools/measure_display.py" "$@"
        ;;
    setup)
        BASE=${PYTHON311:-python3.11}
        "$BASE" -m venv "$HERE/.venv"
        "$HERE/.venv/bin/python" -m pip install --upgrade pip
        "$HERE/.venv/bin/python" -m pip install -r "$HERE/requirements.txt"
        echo "Done. Try:  python/specvis.sh test"
        ;;
    *)
        echo "usage: python/specvis.sh {app|app-simple|run|perimetry|build-table|test|display|setup}" >&2
        exit 2
        ;;
esac
