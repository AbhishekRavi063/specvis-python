@echo off
rem Double-click to start the ReFreM perimetry app on Windows.
rem The first time, it sets everything up (needs Python 3.11; takes a few minutes).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First start: setting up ^(this takes a few minutes^)...
    call specvis.bat setup || ( echo Setup failed. Is Python 3.11 installed? & pause & exit /b 1 )
)
call specvis.bat app
