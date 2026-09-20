"""
Unit tests for VirtualHaltere (Biyolojik Jiroskop & Gaze Stabilization Reflex)
Verifies spring-damper auto-horizon restitution, anti-windup clamping, deadband filtering,
transient saccadic glances, and visual horizon corrections.
"""

import unittest
import numpy as np
from motor.haltere import VirtualHaltere
from motor.locomotion import LocomotionController
from core.platform.base import ActionKey
from core.platform.mock import MockInputDriver


class TestVirtualHaltere(unittest.TestCase):

    def setUp(self):
        self.haltere = VirtualHaltere(
            pitch_range_pixels=140.0,
            k_spring=4.0,
            k_damper=0.6,
            deadband=0.025,
            max_correction_dy=8
        )

    def test_spring_damper_returns_to_horizon(self):
        """Verifies that tilted pitch naturally recovers to level horizon (0.0)."""
        # Start tilted UP at ceiling (-0.60)
        self.haltere.current_pitch = -0.60

        # Simulate 1 second of simulation (30 frames at dt=0.033)
        dt = 0.033
        for _ in range(30):
            dy = self.haltere.compute_step(dt=dt)
            self.haltere.on_mouse_moved(dy=dy, dt=dt)

        # After 1 second of spring-damper action, pitch should have significantly recovered towards 0
        self.assertGreater(self.haltere.current_pitch, -0.15)
        self.assertLessEqual(abs(self.haltere.current_pitch), 0.15)

    def test_deadband_suppresses_micro_jitter(self):
        """Verifies zero mouse commands when within deadband of horizon."""
        self.haltere.current_pitch = 0.01  # Inside deadband (0.025)
        dy = self.haltere.compute_step(dt=0.033)
        self.assertEqual(dy, 0, "Haltere must output zero dy within deadband to avoid jitter")

    def test_anti_windup_clamping(self):
        """Verifies anti-windup prevents integrator drift beyond ceiling / floor limits."""
        # Force to ceiling
        self.haltere.current_pitch = -1.0

        # Try pushing further up (dy < 0)
        self.haltere.on_mouse_moved(dy=-50, dt=0.033)
        self.assertEqual(self.haltere.current_pitch, -1.0, "Pitch must not wind up below -1.0")

        # Now apply downward recovery movement (dy > 0)
        self.haltere.on_mouse_moved(dy=10, dt=0.033)
        # Should immediately begin recovery without delay
        expected = -1.0 + (10.0 / 140.0)
        self.assertAlmostEqual(self.haltere.current_pitch, expected, places=3)

    def test_transient_glance_lifecycle(self):
        """Verifies transient glance looks up, then automatically decays back to horizon."""
        # Request a glance up (-0.35) for 0.15 seconds
        self.haltere.request_glance(target_pitch=-0.35, duration=0.15)
        self.assertTrue(self.haltere.glance_remaining > 0.0)
        self.assertEqual(self.haltere.target_pitch, -0.35)

        # First frame should produce negative dy (look up towards target)
        dy = self.haltere.compute_step(dt=0.033)
        self.assertLess(dy, 0, "Glance up must produce negative dy")
        self.haltere.on_mouse_moved(dy=dy, dt=0.033)

        # Simulate 0.25 seconds (glance expires)
        for _ in range(8):
            dy = self.haltere.compute_step(dt=0.033)
            self.haltere.on_mouse_moved(dy=dy, dt=0.033)

        # Glance should have expired and target returned to 0.0
        self.assertEqual(self.haltere.glance_remaining, 0.0)
        self.assertEqual(self.haltere.target_pitch, 0.0)

        # Simulate another 20 frames to let spring level the pitch
        for _ in range(20):
            dy = self.haltere.compute_step(dt=0.033)
            self.haltere.on_mouse_moved(dy=dy, dt=0.033)

        self.assertLess(abs(self.haltere.current_pitch), 0.10)

    def test_visual_horizon_override(self):
        """Verifies dorsal light ceiling detector forces downward gaze correction."""
        self.haltere.current_pitch = 0.0  # Level

        # Visual detector spots ceiling (high brightness, low floor edges)
        self.haltere.update_visual_horizon(ceiling_score=0.85, floor_score=0.0, vertical_flow=0.0)

        # Virtual haltere should detect tilt discrepancy and command downward correction (dy > 0)
        dy = self.haltere.compute_step(dt=0.033)
        self.assertGreater(dy, 0, "Ceiling detection must force positive dy to look down away from ceiling")

    def test_locomotion_controller_integration(self):
        """Verifies LocomotionController operates closed-loop haltere and explicit pitch."""
        mock_driver = MockInputDriver(dry_run=True)
        controller = LocomotionController(mock_driver, mouse_gain_y=50.0)

        # Case 1: Normal walking with pitch=0.0 -> haltere maintains horizon
        controller.apply({"turn": 0.0, "forward": 0.5, "pitch": 0.0, "is_firing": False})
        diag = controller.haltere.get_diagnostics()
        self.assertIn("current_pitch", diag)
        self.assertIn("target_pitch", diag)

        # Case 2: Explicit pitch still works (backwards compatibility for tests/overrides)
        controller.apply({"turn": 0.0, "forward": 0.0, "pitch": 0.4, "is_firing": False})
        moves = [act for act in mock_driver.logged_actions if act[0] == "move"]
        self.assertTrue(len(moves) > 0)
        self.assertEqual(moves[-1][2], 20)  # 0.4 * 50


if __name__ == "__main__":
    unittest.main()
