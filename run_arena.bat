@echo off
title FlyBrain-HalfLife - 3D FPS Arena Launcher
echo ======================================================================
echo    FlyBrain-HalfLife: MaleCNS v1.0 & FlyWire Autonomous Connectome
echo            Built-in 3D FPS Arena and Telemetry Dashboard
echo ======================================================================
echo.
python main.py --mode arena
if %errorlevel% neq 0 (
    echo.
    echo An error occurred. Please check your Python environment and dependencies.
    pause
)
