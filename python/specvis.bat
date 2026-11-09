@echo off
rem Windows counterpart of specvis.sh.
rem
rem     python\specvis.bat setup        create python\.venv (needs Python 3.11 installed)
rem     python\specvis.bat app          the ReFreM perimetry app
rem     python\specvis.bat perimetry    one eye, simple dialog instead of the app
rem     python\specvis.bat build-table  normal table from the pilot sessions
rem     python\specvis.bat test         every check in python\tests
rem
rem Results are saved in python\output\ (never on GitHub).

setlocal
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"

if /I "%~1"=="setup" goto setup
if not exist "%PY%" (
    echo python\.venv not found. Create it first:  python\specvis.bat setup
    exit /b 1
)
if /I "%~1"=="app"         ( "%PY%" "%HERE%run_app.py" & exit /b )
if /I "%~1"=="app-simple"  ( "%PY%" "%HERE%run_app.py" --simple & exit /b )
if /I "%~1"=="perimetry"   ( "%PY%" "%HERE%run_perimetry.py" & exit /b )
if /I "%~1"=="build-table" ( "%PY%" "%HERE%build_prior_table.py" %2 %3 %4 %5 & exit /b )
if /I "%~1"=="test"        ( "%PY%" "%HERE%run_tests.py" & exit /b )
echo usage: python\specvis.bat {setup^|app^|app-simple^|perimetry^|build-table^|test}
exit /b 2

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
