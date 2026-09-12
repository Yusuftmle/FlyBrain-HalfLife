@echo off
title FlyBrain-HalfLife - Live Game Connector (MaleCNS v1.0 / Half-Life)
echo ======================================================================
echo    FlyBrain-HalfLife: Live Half-Life DirectInput Controller
echo    Hardware DirectInput Scancodes Active: W, A, S, D, Space, Click
echo ======================================================================
echo.
echo NOTE: Start your Half-Life game first, then run this script.
echo If keypresses do not reach Half-Life, right-click this .bat and
echo select "Run as administrator" (Yonetici Olarak Calistir).
echo.
cd /d "%~dp0"
python main.py --mode live --no-dry-run
if %errorlevel% neq 0 (
    echo.
    echo An error occurred.
    pause
)
pause
