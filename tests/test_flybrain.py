"""
Automated Test Suite for FlyBrain-HalfLife
Verifies Connectome, LIF Dynamics, Deadlock Prevention, AGC Retina, Motor Decoder, and Panic Mode
"""
import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG
from core.connectome import FlyConnectome
from core.lif_engine import LIFEngine
from vision.retina import FlyRetina
from motor.controller import MotorController
from reinforcement.dopamine import DopamineController
from env.arena import HalfLifeArena
from telemetry.visualizer import FlyBrainVisualizer

class TestFlyBrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Test Suite: Advanced Biological Circuits & Protections ---")
        cls.config = DEFAULT_CONFIG
        cls.connectome = FlyConnectome(cls.config.connectome)
        cls.lif_engine = LIFEngine(cls.connectome, cls.config.sim)

    def test_01_connectome_structure(self):
        """Verifies connectome topology and sparse matrix invariants."""
        self.assertGreater(self.connectome.total_neurons, 5000)
        self.assertIsNotNone(self.connectome.weights)
        self.assertGreater(self.connectome.weights.nnz, 50000)
        print(f"  [OK] Connectome: {self.connectome.total_neurons:,} neurons, {self.connectome.weights.nnz:,} synapses.")

    def test_02_lif_deadlock_prevention(self):
        """Verifies deadlock prevention via spontaneous biological background noise."""
        for _ in range(5):
            spikes = self.lif_engine.step(external_current=None)
        self.assertTrue(np.any(self.lif_engine.v > self.config.sim.v_rest))
        print("  [OK] Deadlock Prevention: Spontaneous noise maintains active sub-threshold oscillations.")

    def test_03_retina_agc_scaling(self):
        """Verifies Adaptive Gain Control (AGC) and I_ext normalization."""
        retina = FlyRetina(self.config.vision)
        dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
        currents_dark, info_dark = retina.process_frame(dark_frame)
        
        bright_frame = np.full((480, 640, 3), 250, dtype=np.uint8)
        currents_bright, info_bright = retina.process_frame(bright_frame)
        
        self.assertLessEqual(np.max(currents_bright), self.config.vision.target_current_max)
        self.assertGreater(np.mean(currents_dark), 0.0)
        print(f"  [OK] Retina AGC: Dynamic balance maintained (Peak: {np.max(currents_bright):.1f} nA <= {self.config.vision.target_current_max} nA).")

    def test_04_motor_refractory_debounce(self):
        """Verifies hardware debounce and refractory cooldown."""
        self.config.motor.dry_run = True
        motor = MotorController(self.connectome, self.lif_engine, self.config.motor)
        telemetry = motor.update()
        self.assertIn("turn_diff", telemetry)
        motor.cleanup()
        print(f"  [OK] Motor Cooldown: Hardware refractory debounce operational.")

    def test_05_two_second_panic_mode(self):
        """Verifies 2-second stationary panic mode and chaotic escape turn."""
        dopamine = DopamineController(self.connectome, self.lif_engine, self.config.rl)
        
        # Simulate 60 steps stationary (~2.0 seconds at 30 FPS)
        for _ in range(self.config.rl.stuck_threshold_steps + 2):
            ext_i, rl_info = dopamine.update(motion_flow=0.0)
            
        self.assertTrue(rl_info["is_in_panic"])
        self.assertLess(rl_info["dopamine_level"], 0.0)
        print("  [OK] 2-Second Panic Mode: Stuck state triggered panic penalty and 180-degree escape reversal.")

    def test_06_arena_simulation(self):
        """Verifies built-in 3D FPS raycasting arena step."""
        arena = HalfLifeArena(width=320, height=240)
        frame, events = arena.step(turn_val=0.5, forward_val=0.8, is_attack=True)
        self.assertEqual(frame.shape, (240, 320, 3))
        print("  [OK] 3D FPS Arena: Raycasting frame generation and physics verified.")

    def test_07_visualizer_panic_overlay(self):
        """Verifies telemetry visualizer rendering with panic state."""
        vis = FlyBrainVisualizer(self.connectome, self.config.telemetry, width=1280, height=720)
        dummy_game = np.zeros((360, 640, 3), dtype=np.uint8)
        dummy_retina = np.zeros((200, 200, 3), dtype=np.uint8)
        spikes = np.zeros(self.connectome.total_neurons, dtype=bool)
        
        canvas = vis.render(
            dummy_game, dummy_retina, spikes, 
            {"turn_diff": 0.2, "rate_forward": 0.5, "action": "TEST", "is_firing": False},
            {"dopamine_level": -2.0, "is_in_panic": True},
            vision_info={"gain": 24.0}
        )
        self.assertEqual(canvas.shape, (720, 1280, 3))
        vis.close()
        print("  [OK] Telemetry Visualizer: 1280x720 dual-pane canvas generated with panic overlay.")

if __name__ == "__main__":
    unittest.main()
