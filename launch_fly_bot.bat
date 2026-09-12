@echo off
title FlyBrain-HalfLife - Autonomous 3rd-Person Bot Launcher
echo ======================================================================
echo    FlyBrain-HalfLife: Autonomous 3rd-Person Bot Launcher
echo    Launches Half-Life in TPS Mode with MaleCNS v1.0 Fly Connectome!
echo ======================================================================
echo.

set HL_PATH=C:\Program Files (x86)\Steam\steamapps\common\Half-Life\hl.exe

tasklist /FI "IMAGENAME eq hl.exe" 2>NUL | find /I /N "hl.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [*] Half-Life is already running. Using existing game window...
) else (
    if exist "%HL_PATH%" (
        echo [*] Launching Half-Life in 3rd-Person Bot Mode...
        start "" "%HL_PATH%" -game valve -windowed -noborder -w 800 -h 600 +name "FlyBrain_Bot" +thirdperson +cam_idealdist 130 +map crossfire
        echo [*] Waiting 6 seconds for Half-Life to load map...
        timeout /t 6 /nobreak >nul
    ) else (
        echo [!] Half-Life not found at default Steam path. Please start Half-Life manually.
    )
)

echo [*] Starting MaleCNS v1.0 Fly Connectome Neural Pipeline...
cd /d "%~dp0"
python main.py --mode live --no-dry-run
if %errorlevel% neq 0 (
    echo.
    echo An error occurred.
    pause
)
pause
