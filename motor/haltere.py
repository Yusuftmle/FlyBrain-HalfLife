"""
FlyBrain Half-Life - Virtual Haltere (Biyolojik Jiroskop & Gaze Stabilization Reflex)
Inspired by Drosophila melanogaster haltere biomechanics & Lobula Plate Vertical System (VS1-VS10).

Maintains closed-loop pitch tracking, auto-horizon restitution (spring-damper),
transient exploratory glances (saccades), anti-windup clamping, and visual horizon corrections.
"""

import time
import math
import numpy as np
from typing import Dict, Any, Tuple


class VirtualHaltere:
    """
    Simulates fly haltere equilibrium and gaze stabilization reflex for vertical camera pitch.
    
    Coordinate Convention (Standard FPS / Half-Life):
      - current_pitch = 0.0: Level horizon (ideal walking gaze)
      - current_pitch < 0.0: Tilted UP (looking towards ceiling; -1.0 ~ -85 deg)
      - current_pitch > 0.0: Tilted DOWN (looking towards floor; +1.0 ~ +85 deg)
      - mouse dy < 0: Moves camera UP
      - mouse dy > 0: Moves camera DOWN
    """

    def __init__(
        self,
        pitch_range_pixels: float = 140.0,
        k_spring: float = 4.0,
        k_damper: float = 0.6,
        deadband: float = 0.025,
        max_correction_dy: int = 8
    ):
        """
        Args:
            pitch_range_pixels: Number of mouse dy pixels representing full 0 -> 85 deg sweep.
            k_spring: Spring constant returning gaze to level horizon (0.0).
            k_damper: Damping coefficient preventing overshoot oscillations.
            deadband: Angle deadzone around target where no correction is dispatched.
            max_correction_dy: Maximum mouse dy dispatched per frame to prevent jarring snaps.
        """
        self.pitch_scale = 1.0 / max(10.0, float(pitch_range_pixels))
        self.k_spring = float(k_spring)
        self.k_damper = float(k_damper)
        self.deadband = float(deadband)
        self.max_correction_dy = int(max_correction_dy)

        # State Variables
        self.current_pitch: float = 0.0       # Normalized [-1.0, 1.0]
        self.target_pitch: float = 0.0        # Default to 0.0 (Horizon)
        self.pitch_velocity: float = 0.0      # Estimated d(pitch)/dt
        self.glance_remaining: float = 0.0    # Transient saccade countdown (seconds)
        self.subpixel_accum_y: float = 0.0    # Sub-pixel accumulator for smooth mouse dy
        self.last_update_time: float = time.perf_counter()

        # Visual Horizon Correction State
        self.visual_bias_y: float = 0.0

    def request_glance(self, target_pitch: float, duration: float = 0.4) -> None:
        """
        Requests a transient exploratory glance (e.g. looking up at an obstacle to find openings).
        Once duration expires, the spring automatically levels gaze back to horizon (0.0).
        
        Args:
            target_pitch: Desired pitch angle in [-1.0, 1.0] (e.g. -0.25 to look up).
            duration: Active duration in seconds before spring takes over.
        """
        self.target_pitch = float(np.clip(target_pitch, -0.85, 0.85))
        self.glance_remaining = max(self.glance_remaining, float(duration))

    def update_visual_horizon(
        self,
        ceiling_score: float = 0.0,
        floor_score: float = 0.0,
        vertical_flow: float = 0.0
    ) -> None:
        """
        Integrates visual cues (Dorsal Light & Optic Flow) to prevent desynchronization with the game.
        
        Args:
            ceiling_score: [0.0, 1.0] Confidence that viewport shows ceiling/lights.
            floor_score: [0.0, 1.0] Confidence that viewport shows ground only.
            vertical_flow: Lobula Plate VS net vertical optic flow (dy).
        """
        self.visual_bias_y = 0.0

        # Ceiling override: If retina sees ceiling texture/lights, force pitch downwards
        if ceiling_score > 0.55:
            # We are tilted UP, possibly hitting ceiling limit
            if self.current_pitch > -0.60:
                self.current_pitch = -0.75  # Calibrate estimated angle
            # Downward correction bias (positive pitch)
            self.visual_bias_y += (ceiling_score - 0.50) * 2.0 * 0.40

        # Floor override: If retina sees pure ground texture, force pitch upwards
        elif floor_score > 0.55:
            if self.current_pitch < 0.60:
                self.current_pitch = 0.75
            # Upward correction bias (negative pitch)
            self.visual_bias_y -= (floor_score - 0.50) * 2.0 * 0.40

        # VS Optic Flow stabilization: If moving forward and visual field drifts uniformly
        if abs(vertical_flow) > 0.15:
            # Uniform vertical flow indicates uncompensated head pitch rotation
            self.visual_bias_y += np.clip(-vertical_flow * 0.20, -0.25, 0.25)

    def compute_step(self, dt: float) -> int:
        """
        Computes closed-loop mouse dy for gaze stabilization and active glance tracking.
        
        Returns:
            mouse_dy: Integer mouse displacement (negative = look UP, positive = look DOWN).
        """
        dt = max(0.005, min(0.1, float(dt)))

        # 1. Update Transient Glance Timer
        if self.glance_remaining > 0.0:
            self.glance_remaining -= dt
            if self.glance_remaining <= 0.0:
                self.glance_remaining = 0.0
                self.target_pitch = 0.0  # Return target to Horizon!

        # 2. Closed-Loop Spring-Damper Error
        # Target is target_pitch; current is current_pitch
        # If current_pitch is -0.30 (looking UP) and target is 0.0:
        # error = 0.0 - (-0.30) = +0.30 (need to move DOWN)
        error = self.target_pitch - self.current_pitch

        # Incorporate visual horizon override bias
        error += self.visual_bias_y

        # Deadband check: If very close to target and no visual bias, prevent crosshair jitter
        if abs(error) < self.deadband and abs(self.visual_bias_y) < 0.01:
            return 0

        # Spring-damper force calculation:
        # F = k_spring * error - k_damper * velocity
        restoring_torque = (self.k_spring * error) - (self.k_damper * self.pitch_velocity)

        # Convert torque to mouse displacement rate
        # 1.0 pitch error corresponds to full range (pitch_range_pixels)
        desired_dy_rate = restoring_torque / self.pitch_scale
        desired_dy_frame = desired_dy_rate * dt

        # Accumulate sub-pixel values
        self.subpixel_accum_y += desired_dy_frame

        # Clamp output to prevent jarring snaps
        dy_to_dispatch = int(self.subpixel_accum_y)
        if dy_to_dispatch != 0:
            dy_to_dispatch = int(np.clip(dy_to_dispatch, -self.max_correction_dy, self.max_correction_dy))
            self.subpixel_accum_y -= float(dy_to_dispatch)

        return dy_to_dispatch

    def on_mouse_moved(self, dy: int, dt: float = 0.033) -> None:
        """
        Feeds back actual dispatched mouse dy into the internal state with anti-windup clamping.
        
        Args:
            dy: Relative mouse vertical displacement actually dispatched.
            dt: Frame elapsed time.
        """
        if dy == 0:
            return

        dt = max(0.005, min(0.1, float(dt)))
        delta_pitch = float(dy) * self.pitch_scale

        # Anti-Windup Clamping:
        # If already at ceiling (current_pitch <= -1.0) and dy < 0 (moving up), DO NOT integrate!
        if delta_pitch < 0 and self.current_pitch <= -1.0:
            self.pitch_velocity = 0.0
            return
        # If already at floor (current_pitch >= 1.0) and dy > 0 (moving down), DO NOT integrate!
        if delta_pitch > 0 and self.current_pitch >= 1.0:
            self.pitch_velocity = 0.0
            return

        # Update angle and velocity
        old_pitch = self.current_pitch
        self.current_pitch = float(np.clip(self.current_pitch + delta_pitch, -1.0, 1.0))
        self.pitch_velocity = (self.current_pitch - old_pitch) / dt

    def reset(self, initial_pitch: float = 0.0) -> None:
        """Resets haltere state to horizon (e.g. upon game pause or window focus switch)."""
        self.current_pitch = float(np.clip(initial_pitch, -1.0, 1.0))
        self.target_pitch = 0.0
        self.pitch_velocity = 0.0
        self.glance_remaining = 0.0
        self.subpixel_accum_y = 0.0
        self.visual_bias_y = 0.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns telemetry status dictionary."""
        return {
            "current_pitch": round(self.current_pitch, 3),
            "target_pitch": round(self.target_pitch, 3),
            "is_glancing": self.glance_remaining > 0.0,
            "glance_remaining_sec": round(self.glance_remaining, 2),
            "pitch_degrees_est": round(self.current_pitch * 85.0, 1)
        }
