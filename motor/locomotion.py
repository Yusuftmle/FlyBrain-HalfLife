"""
locomotion.py - Smooth Locomotion & Steering Controller
Translates normalized continuous actions into smooth OS input commands without strafe interference.
"""
import time
from typing import Dict, Any, Optional
import numpy as np
from core.platform.base import BaseInputDriver, ActionKey
from motor.haltere import VirtualHaltere


class LocomotionController:
    """
    Executes smooth locomotion and gaze control from decoded analog actions.
    Fixes pathfinding anomalies:
      - Pure Corridor Steering: Uses mouse yaw + directional arrow keys.
        NO 'A'/'D' strafe during normal turns to avoid diagonal wall friction.
      - State-Based Forward Hold: Holds 'W' continuously while moving forward
        instead of flooding the OS input queue every frame.
      - Closed-Loop Gaze Control: Employs biological VirtualHaltere (spring-damper)
        for auto-horizon stabilization, anti-windup, and transient saccadic glances.
    """
    def __init__(
        self,
        input_driver: BaseInputDriver,
        mouse_gain_x: float = 110.0,
        mouse_gain_y: float = 40.0,
        walk_threshold: float = 0.02,
        turn_key_threshold: float = 0.15,
        turn_deadzone: float = 0.010,
        fire_cooldown: float = 0.35,
        pitch_range_pixels: float = 140.0,
        enable_strafe: bool = False
    ):
        self.driver = input_driver
        self.mouse_gain_x = mouse_gain_x
        self.mouse_gain_y = mouse_gain_y
        self.walk_threshold = walk_threshold
        self.turn_key_threshold = turn_key_threshold
        self.turn_deadzone = turn_deadzone
        self.fire_cooldown = fire_cooldown
        self.enable_strafe = enable_strafe

        # Biological Virtual Haltere (Gaze Stabilization Reflex)
        self.haltere = VirtualHaltere(
            pitch_range_pixels=pitch_range_pixels,
            k_spring=4.0,
            k_damper=0.6,
            deadband=0.025,
            max_correction_dy=8
        )

        # Current hardware state
        self.is_walking: bool = False
        self.is_stepping_back: bool = False
        self.is_turning_left: bool = False
        self.is_turning_right: bool = False
        self.last_fire_time: float = 0.0
        self.last_action_desc: str = "IDLE"
        self.mouse_accum_x: float = 0.0
        self.mouse_accum_y: float = 0.0
        self.last_mouse_dx: int = 0
        self._last_apply_time: float = time.time()

    def request_glance(self, target_pitch: float, duration: float = 0.4) -> None:
        """Requests transient exploratory glance (e.g. looking up at an obstacle)."""
        self.haltere.request_glance(target_pitch=target_pitch, duration=duration)

    def update_visual_horizon(
        self,
        ceiling_score: float = 0.0,
        floor_score: float = 0.0,
        vertical_flow: float = 0.0
    ) -> None:
        """Updates virtual haltere with dorsal light / optic flow horizon cues."""
        self.haltere.update_visual_horizon(
            ceiling_score=ceiling_score,
            floor_score=floor_score,
            vertical_flow=vertical_flow
        )

    def apply(
        self,
        action: Dict[str, Any],
        now: Optional[float] = None,
        suppress_input: bool = False
    ) -> Dict[str, Any]:
        """
        Applies decoded action dictionary to hardware drivers.

        Args:
            action: Output from NeuralDecoder (turn, forward, pitch, is_firing)
            now: Current timestamp
            suppress_input: If True (e.g. paused / unfocused), releases all keys and returns IDLE.
        """
        if now is None:
            now = time.time()

        if suppress_input:
            self.release_all()
            if hasattr(self, "haltere"):
                self.haltere.subpixel_accum_y = 0.0
            return {"action": "PAUSED [ODAK YOK - MASAÜSTÜ]", "is_firing": False}

        dt = max(0.005, min(0.1, now - self._last_apply_time))
        self._last_apply_time = now

        # Integrate visual horizon cues if passed in action
        if "visual_horizon" in action:
            vh = action["visual_horizon"]
            self.update_visual_horizon(
                ceiling_score=float(vh.get("ceiling_score", 0.0)),
                floor_score=float(vh.get("floor_score", 0.0)),
                vertical_flow=float(vh.get("vertical_flow", 0.0))
            )

        # Trigger transient glance if requested in action
        if "glance_pitch" in action:
            self.request_glance(
                target_pitch=float(action["glance_pitch"]),
                duration=float(action.get("glance_duration", 0.4))
            )

        turn = float(action.get("turn", 0.0))
        forward = float(action.get("forward", 0.0))
        pitch = float(action.get("pitch", 0.0))
        is_firing_req = bool(action.get("is_firing", False))

        actions_taken = []

        # 1. Pure Analog Mouse Yaw Steering (MochisRice / DOOMFLY Standard - Zero Screen Shake)
        mouse_dx = 0
        mouse_dy = 0
        if abs(turn) > self.turn_deadzone:
            self.mouse_accum_x += turn * self.mouse_gain_x
            target_dx = int(self.mouse_accum_x)

            # Human-like Slew Rate Limiter: Max 16 pixels/frame acceleration (~1.1 deg/frame at 30 FPS)
            # Prevents abrupt violent snaps while permitting smooth, decisive corridor cornering up to 36 px/frame
            max_step = 16
            clamped_dx = int(np.clip(target_dx, self.last_mouse_dx - max_step, self.last_mouse_dx + max_step))
            clamped_dx = int(np.clip(clamped_dx, -36, 36))
            self.last_mouse_dx = clamped_dx
            self.mouse_accum_x -= clamped_dx
            mouse_dx = clamped_dx

            if turn < 0:
                actions_taken.append("STEER LEFT")
            else:
                actions_taken.append("STEER RIGHT")

            # In natural fly biology and DOOMFLY: never fight mouse with keyboard arrow keys!
            self._clear_turn_keys()
        else:
            self.last_mouse_dx = 0
            self.mouse_accum_x = 0.0
            self._clear_turn_keys()

        # 1.5 Closed-Loop Gaze: Level horizon (matching DOOMFLY reference: pitch = 0)
        if abs(pitch) > 0.05:
            # Explicit pitch commanded (direct manual override)
            self.mouse_accum_y += pitch * self.mouse_gain_y
            mouse_dy = int(self.mouse_accum_y)
            self.mouse_accum_y -= mouse_dy
            if mouse_dy != 0 and hasattr(self, "haltere"):
                self.haltere.on_mouse_moved(dy=mouse_dy, dt=dt)
            if pitch < -0.05:
                actions_taken.append("LOOK UP")
            elif pitch > 0.05:
                actions_taken.append("LOOK DOWN")
        else:
            # Fixed level eye-line horizon (zero pitch jitter)
            mouse_dy = 0
            self.mouse_accum_y = 0.0

        if mouse_dx != 0 or mouse_dy != 0:
            self.driver.mouse_move_relative(dx=mouse_dx, dy=mouse_dy)

        # 2. Continuous Locomotor CPG Forward Drive (DOOMFLY / Drosophila biology)
        # 2. Continuous Locomotor CPG Forward Drive (DOOMFLY / Drosophila biology)
        # Holds 'W' continuously during locomotion. Backward stepping ('S') requires strong MDN activation (<-0.25).
        if forward > self.walk_threshold or (self.is_walking and forward > 0.0):
            if self.is_stepping_back:
                self.driver.release_action(ActionKey.BACKWARD)
                self.is_stepping_back = False

            if not self.is_walking:
                self.driver.press_action(ActionKey.FORWARD)
                self.is_walking = True
            actions_taken.append("WALK FORWARD")
        elif forward < -0.25:
            # MDN Moonwalker backward stepping (damage / obstacle deadlock)
            if self.is_walking:
                self.driver.release_action(ActionKey.FORWARD)
                self.is_walking = False

            if not self.is_stepping_back:
                self.driver.press_action(ActionKey.BACKWARD)
                self.is_stepping_back = True
            actions_taken.append(f"STEP BACK (MDN {abs(forward):.2f})")
        else:
            # Neutral / coasting: release forward walk, but DO NOT press backward key!
            if self.is_walking:
                self.driver.release_action(ActionKey.FORWARD)
                self.is_walking = False
            if self.is_stepping_back:
                self.driver.release_action(ActionKey.BACKWARD)
                self.is_stepping_back = False

        # 3. Primary Weapon Attack
        did_fire = False
        if is_firing_req:
            if (now - self.last_fire_time) >= self.fire_cooldown:
                self.driver.mouse_click(duration=0.01)
                self.last_fire_time = now
                did_fire = True
                actions_taken.append("FIRE!")

        self.last_action_desc = " + ".join(actions_taken) if actions_taken else "IDLE"
        return {
            "action": self.last_action_desc,
            "is_firing": did_fire,
            "is_walking": self.is_walking,
            "is_stepping_back": self.is_stepping_back,
            "turn": turn,
            "forward": forward,
            "haltere": self.haltere.get_diagnostics() if hasattr(self, "haltere") else {}
        }

    def _clear_turn_keys(self):
        """Clears keyboard turning arrow keys when in deadzone."""
        if self.is_turning_left:
            self.driver.release_action(ActionKey.TURN_LEFT)
            if self.enable_strafe:
                self.driver.release_action(ActionKey.STRAFE_LEFT)
            self.is_turning_left = False
        if self.is_turning_right:
            self.driver.release_action(ActionKey.TURN_RIGHT)
            if self.enable_strafe:
                self.driver.release_action(ActionKey.STRAFE_RIGHT)
            self.is_turning_right = False

    def release_all(self):
        """Releases all held keys and mouse actions."""
        self._clear_turn_keys()
        if self.is_walking:
            self.driver.release_action(ActionKey.FORWARD)
            self.is_walking = False
        if self.is_stepping_back:
            self.driver.release_action(ActionKey.BACKWARD)
            self.is_stepping_back = False
        if self.enable_strafe:
            self.driver.release_action(ActionKey.STRAFE_LEFT)
            self.driver.release_action(ActionKey.STRAFE_RIGHT)
        self.mouse_accum_x = 0.0
        self.mouse_accum_y = 0.0
        self.last_mouse_dx = 0
        if hasattr(self, "haltere"):
            self.haltere.subpixel_accum_y = 0.0
        self.driver.release_all()
        self.last_action_desc = "IDLE"
