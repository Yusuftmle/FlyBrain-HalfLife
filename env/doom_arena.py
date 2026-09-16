"""
env/doom_arena.py - ViZDoom Embedded 3D Arena for FlyBrain Autonomous Agent
Runs genuine 1993 id Software 3D engine inside the biological connectome loop.
Perceives real 3D Doom corridors, demons, and pickups through 60x60 compound eye ommatidia.
"""
import os
import math
from typing import Tuple, Dict, Any, List, Optional
import numpy as np

try:
    import vizdoom as vzd
    VIZDOOM_AVAILABLE = True
except ImportError:
    VIZDOOM_AVAILABLE = False


class DoomArena:
    """
    State-of-the-Art 3D Game Environment powered by ViZDoom.
    Provides sub-millisecond direct-memory frame buffers and biophysical
    action mapping for Drosophila connectome navigation and combat.
    """
    def __init__(
        self,
        scenario: str = "deadly_corridor",
        width: int = 640,
        height: int = 480,
        render_hud: bool = False
    ):
        if not VIZDOOM_AVAILABLE:
            raise RuntimeError(
                "ViZDoom is not installed. Install via: pip install vizdoom"
            )

        self.w = width
        self.h = height
        self.scenario = scenario
        self.game = vzd.DoomGame()

        # Find scenario config
        cfg_path = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
            self.scenario = "deadly_corridor"

        self.game.load_config(cfg_path)
        self.game.set_window_visible(False)
        self.game.set_screen_format(vzd.ScreenFormat.BGR24)

        # Resolution mapping
        res_map = {
            (320, 240): vzd.ScreenResolution.RES_320X240,
            (640, 480): vzd.ScreenResolution.RES_640X480,
            (800, 600): vzd.ScreenResolution.RES_800X600,
        }
        res = res_map.get((width, height), vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_resolution(res)

        # Ensure delta turning button is available for smooth continuous analog steering
        self.game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
        self.game.set_render_hud(render_hud)

        self.game.init()
        self.game.new_episode()

        self._buttons = self.game.get_available_buttons()
        self.last_health = 100.0
        self.last_kill_count = 0
        self.episode_count = 1

    def step(self, turn_val: float, forward_val: float, is_attack: bool) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Steps the 3D Doom environment using motor commands:
        turn_val: [-1.0, 1.0] continuous yaw delta (DNp20 steering)
        forward_val: [-1.0, 1.0] forward/backward propulsion (DNpe017 vs MDN)
        is_attack: bool firing weapon
        """
        events = []

        if self.game.is_episode_finished():
            self.game.new_episode()
            self.episode_count += 1
            self.last_health = 100.0
            self.last_kill_count = 0

        # Construct action vector matching scenario buttons
        action = []
        for btn in self._buttons:
            val = 0.0
            if btn == vzd.Button.MOVE_FORWARD:
                val = 1.0 if forward_val > 0.06 else 0.0
            elif btn == vzd.Button.MOVE_BACKWARD:
                val = 1.0 if forward_val < -0.06 else 0.0
            elif btn == vzd.Button.TURN_LEFT:
                val = 1.0 if turn_val < -0.10 else 0.0
            elif btn == vzd.Button.TURN_RIGHT:
                val = 1.0 if turn_val > 0.10 else 0.0
            elif btn == vzd.Button.TURN_LEFT_RIGHT_DELTA:
                # Continuous analog turn: convert normalized [-1, 1] to degrees/tick
                val = float(turn_val * 16.0)
            elif btn == vzd.Button.ATTACK:
                val = 1.0 if is_attack else 0.0
            elif btn == vzd.Button.MOVE_LEFT:
                val = 1.0 if turn_val < -0.40 else 0.0
            elif btn == vzd.Button.MOVE_RIGHT:
                val = 1.0 if turn_val > 0.40 else 0.0
            action.append(val)

        # Step 2 game ticks (smooth ~35 FPS at 70Hz engine tick rate)
        self.game.make_action(action, 2)

        # Read state buffer
        state = self.game.get_state()
        if state is None:
            self.game.new_episode()
            state = self.game.get_state()

        frame = state.screen_buffer if state else np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Health & Combat telemetry
        if state:
            try:
                curr_health = float(self.game.get_game_variable(vzd.GameVariable.HEALTH))
                if curr_health < self.last_health:
                    damage = self.last_health - curr_health
                    events.append({"type": "damage", "mag": max(0.5, damage / 15.0)})
                elif curr_health > self.last_health:
                    events.append({"type": "forward", "mag": 0.4})  # Medkit pickup reward
                self.last_health = curr_health
            except Exception:
                pass

            try:
                kill_count = int(self.game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                if kill_count > self.last_kill_count:
                    events.append({"type": "kill", "mag": 1.5})
                    self.last_kill_count = kill_count
            except Exception:
                pass

        if forward_val > 0.08:
            events.append({"type": "forward", "mag": 0.05})

        return frame, events

    def close(self):
        """Releases ViZDoom C++ game engine resources."""
        try:
            self.game.close()
        except Exception:
            pass
