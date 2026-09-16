#!/usr/bin/env bash
# FlyBrain-HalfLife - Autonomous 3rd-Person Bot Launcher for Linux
echo "======================================================================"
echo "   FlyBrain-HalfLife: Autonomous 3rd-Person Bot Launcher (Linux)"
echo "   Launches Half-Life in TPS Mode with MaleCNS v1.0 Fly Connectome!"
echo "======================================================================"
echo ""

# Check if Half-Life process is already active
if pgrep -x "hl_linux" > /dev/null || pgrep -x "hl.exe" > /dev/null; then
    echo "[*] Half-Life is already running. Using existing game window..."
else
    # Common Steam Linux paths
    HL_PATHS=(
        "$HOME/.local/share/Steam/steamapps/common/Half-Life/hl.sh"
        "$HOME/.steam/steam/steamapps/common/Half-Life/hl.sh"
        "$HOME/.local/share/Steam/steamapps/common/Half-Life/hl_linux"
    )
    FOUND=""
    for p in "${HL_PATHS[@]}"; do
        if [ -f "$p" ]; then
            FOUND="$p"
            break
        fi
    done

    if [ -n "$FOUND" ]; then
        echo "[*] Launching Half-Life from: $FOUND"
        "$FOUND" -game valve -windowed -noborder -w 800 -h 600 +name "FlyBrain_Bot" +thirdperson +cam_idealdist 130 +map crossfire &
        echo "[*] Waiting 6 seconds for Half-Life to load map..."
        sleep 6
    else
        echo "[!] Half-Life not found at default Steam path. Please start Half-Life manually or run Steam."
    fi
fi

echo "[*] Starting MaleCNS v1.0 Fly Connectome Neural Pipeline..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 main.py --mode live --no-dry-run
