#!/bin/sh
# Double-click to start the ReFreM perimetry app on a Mac.
# The first time, it sets everything up (needs Python 3.11; takes a few minutes).
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "First start: setting up (this takes a few minutes)..."
    ./specvis.sh setup || { echo "Setup failed. Is Python 3.11 installed?"; read -r _; exit 1; }
fi
exec ./specvis.sh app
