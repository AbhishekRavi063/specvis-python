@echo off
rem Windows counterpart of specvis.sh.
rem
rem     python\specvis.bat setup        create python\.venv (needs Python 3.11 installed)
rem     python\specvis.bat app          the ReFreM perimetry app
rem     python\specvis.bat perimetry    one eye, simple dialog instead of the app
rem     python\specvis.bat build-table  normal table from the pilot sessions
rem     python\specvis.bat redraw-maps  redraw the maps of saved sessions
rem     python\specvis.bat test         every check in python\tests
rem     python\specvis.bat calibrate    measure the screen with the ColorCAL
rem     python\specvis.bat preflight    check screen, ColorCAL, scale, eye tracker
rem     python\specvis.bat tracker-setup   once: python\.venv-tobii (Python 3.10 + tobii-research)
rem     python\specvis.bat tracker-bridge  before each eye-tracker session; leave it open
rem
rem Results are saved in python\output\ (never on GitHub).

setlocal
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"

if /I "%~1"=="setup" goto setup
if /I "%~1"=="tracker-setup" goto trackersetup
if /I "%~1"=="tracker-bridge" goto trackerbridge
if not exist "%PY%" (
    echo python\.venv not found. Create it first:  python\specvis.bat setup
    exit /b 1
)
if /I "%~1"=="app"         ( "%PY%" "%HERE%run_app.py" & exit /b )
if /I "%~1"=="app-simple"  ( "%PY%" "%HERE%run_app.py" --simple & exit /b )
if /I "%~1"=="perimetry"   ( "%PY%" "%HERE%run_perimetry.py" & exit /b )
if /I "%~1"=="build-table" ( "%PY%" "%HERE%build_prior_table.py" %2 %3 %4 %5 & exit /b )
if /I "%~1"=="redraw-maps" ( "%PY%" "%HERE%redraw_maps.py" %2 %3 %4 %5 & exit /b )
if /I "%~1"=="test"        ( "%PY%" "%HERE%run_tests.py" & exit /b )
if /I "%~1"=="calibrate"   ( "%PY%" "%HERE%tools\calibrate_photometer.py" %2 %3 %4 %5 %6 %7 %8 %9 & exit /b )
if /I "%~1"=="preflight"   ( "%PY%" "%HERE%tools\preflight.py" %2 %3 %4 %5 %6 %7 %8 %9 & exit /b )
if /I "%~1"=="display"     ( "%PY%" "%HERE%tools\measure_display.py" %2 %3 %4 %5 & exit /b )
echo usage: python\specvis.bat {setup^|app^|app-simple^|perimetry^|build-table^|redraw-maps^|test^|display^|calibrate^|preflight^|tracker-setup^|tracker-bridge}
exit /b 2

:trackersetup
rem The Tobii Pro SDK (tobii-research) is published for Python 3.10 only.
where py >nul 2>nul || ( echo The Python launcher "py" was not found. Install Python 3.10 from python.org. & exit /b 1 )
py -3.10 -m venv "%HERE%.venv-tobii"
if errorlevel 1 (
    echo Could not create python\.venv-tobii. Install Python 3.10 ^(64-bit^) from python.org first.
    exit /b 1
)
"%HERE%.venv-tobii\Scripts\python.exe" -m pip install --upgrade pip
"%HERE%.venv-tobii\Scripts\python.exe" -m pip install tobii-research
echo Done. Before each session:  python\specvis.bat tracker-bridge
exit /b 0

:trackerbridge
if not exist "%HERE%.venv-tobii\Scripts\python.exe" (
    echo python\.venv-tobii not found. Run first:  python\specvis.bat tracker-setup
    exit /b 1
)
"%HERE%.venv-tobii\Scripts\python.exe" "%HERE%tools\tobii_bridge.py" %2 %3 %4
exit /b

:setup
where py >nul 2>nul && (py -3.11 -m venv "%HERE%.venv") || (python -m venv "%HERE%.venv")
if errorlevel 1 (
    echo Could not create the environment. Install Python 3.11 from python.org first.
    exit /b 1
)
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r "%HERE%requirements.txt"
echo Done. Try:  python\specvis.bat app
exit /b 0
