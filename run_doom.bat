@echo off
title FlyBrain - ViZDoom 3D Biological Arena Launcher
echo ======================================================================
echo    FlyBrain-HalfLife: MaleCNS v1.0 & FlyWire Autonomous Connectome
echo           Genuine 3D ViZDoom Arena & Biological Telemetry
echo ======================================================================
echo.
python main.py --mode doom --doom-scenario deadly_corridor
if %errorlevel% neq 0 (
    echo.
    echo An error occurred. Please ensure ViZDoom is installed: pip install vizdoom
    pause
)
