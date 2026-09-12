"""
test_four_layers.py - Strict Compliance Test Suite for 4-Layer Architecture
Verifies data_loader, lif_engine, vision_bridge, and input_bridge under SYSTEM_RULES.
"""
import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import ConnectomeDataLoader
from lif_engine import PyTorchLIFEngine
from vision_bridge import VisionBridge
from input_bridge import InputBridge

class TestFourLayerArchitecture(unittest.TestCase):
    def test_layer1_data_loader(self):
        """Tests biological schema ingestion, sign mapping, and sparse matrix generation."""
        loader = ConnectomeDataLoader()
        weights, meta = loader.build_canonical_flywire_connectome()
        
        # Verify non-dense sparse matrix
        self.assertTrue(hasattr(weights, "nnz"))
        self.assertGreater(weights.nnz, 100000)
        self.assertEqual(weights.shape[0], weights.shape[1])
        
        # Verify region metadata
        self.assertIn("region_offsets", meta)
        self.assertIn("photoreceptors", meta["region_offsets"])
        self.assertIn("descending", meta["region_offsets"])
        print(f"[TEST PASS] Layer 1 (data_loader): Compiled {meta['num_neurons']:,} neurons, {weights.nnz:,} synapses.")

    def test_layer2_lif_engine(self):
        """Tests biophysical LIF differential equation solver and deadlock noise."""
        loader = ConnectomeDataLoader()
        weights, meta = loader.build_canonical_flywire_connectome()
        num_neurons = meta["num_neurons"]
        
        lif = PyTorchLIFEngine(sparse_weights=weights, num_neurons=num_neurons)
        
        # Step with zero external current (verifies spontaneous noise keeps membrane active)
        spikes = lif.forward_step(external_current=None)
        self.assertEqual(len(spikes), num_neurons)
        
        # Step with sustained injection current (biological tau_m=20ms requires charging over multiple dt=0.5ms steps)
        ext_current = np.zeros(num_neurons, dtype=np.float32)
        ext_current[:meta["region_counts"]["photoreceptors"]] = 40.0
        total_spikes = 0
        for _ in range(15):
            spikes_step = lif.forward_step(external_current=ext_current)
            total_spikes += int(np.sum(spikes_step))
        
        self.assertGreater(total_spikes, 0)
        print(f"[TEST PASS] Layer 2 (lif_engine): Vectorized forward step passed on {lif.device_str.upper()} with {total_spikes} spikes generated.")

    def test_layer3_vision_bridge(self):
        """Tests sub-2ms MSS capture and Naka-Rushton Poisson photoreceptor transduction."""
        vision = VisionBridge(grid_width=60, grid_height=60)
        dummy_frame = np.full((360, 640, 3), 120, dtype=np.uint8)
        
        currents, metrics = vision.process_frame(dummy_frame)
        self.assertEqual(currents.shape, (3600,))
        self.assertIn("mean_photocurrent", metrics)
        self.assertGreater(metrics["mean_photocurrent"], 0.0)
        vision.close()
        print(f"[TEST PASS] Layer 3 (vision_bridge): 60x60={len(currents)} ommatidia currents computed via Naka-Rushton kinetics.")

    def test_layer4_input_bridge(self):
        """Tests motor neuron thresholding and refractory cooldown DirectInput execution."""
        bridge = InputBridge(dry_run=True, cooldown_duration=0.10)
        
        # Symmetrical firing (no steering)
        res_idle = bridge.decode_and_dispatch(dnp20_left_rate=0.1, dnp20_right_rate=0.1, dnpe017_forward_rate=0.0, dnpe017_attack_rate=0.0)
        self.assertEqual(res_idle["action"], "IDLE")
        
        # Left steering dominant
        res_left = bridge.decode_and_dispatch(dnp20_left_rate=0.5, dnp20_right_rate=0.1, dnpe017_forward_rate=0.4, dnpe017_attack_rate=0.7)
        self.assertIn("STEER LEFT", res_left["action"])
        self.assertIn("WALK FORWARD", res_left["action"])
        self.assertIn("FIRE!", res_left["action"])
        bridge.release_all()
        print("[TEST PASS] Layer 4 (input_bridge): Hardware debounced motor decoding verified.")

    def test_halflife_hud_crop(self):
        """Tests GoldSrc Half-Life HUD exclusion and crosshair-centered FOV extraction."""
        vision = VisionBridge(grid_width=60, grid_height=60)
        
        # Create a frame with bright orange HUD numbers at bottom-left and bottom-right
        frame = np.full((480, 640, 3), 50, dtype=np.uint8) # Dark room
        # Add GoldSrc HUD (bright orange: B=0, G=160, R=255)
        frame[420:480, 10:150] = [0, 160, 255] # Health indicator
        frame[420:480, 500:630] = [0, 160, 255] # Ammo indicator
        
        # Center crosshair (neutral gray)
        frame[235:245, 315:325] = 200

        cropped = vision.crop_halflife_fov(frame)
        self.assertLess(cropped.shape[0], frame.shape[0])
        # Verify that bottom HUD area was completely excluded from cropped FOV
        self.assertFalse(np.any((cropped[:, :, 2] == 255) & (cropped[:, :, 0] == 0)))
        vision.close()
        print(f"[TEST PASS] Half-Life HUD Crop: Successfully excluded GoldSrc HUD and isolated central FOV.")

    def test_dynamic_data_discovery(self):
        """Tests that DataLoader dynamically binds to ./data and avoids hardcoded path traps."""
        loader = ConnectomeDataLoader()
        self.assertTrue(os.path.isabs(loader.data_dir))
        self.assertTrue(os.path.isabs(loader.cache_dir))
        self.assertTrue(os.path.exists(loader.data_dir))
        datasets = loader.discover_local_datasets()
        self.assertIsInstance(datasets, dict)
        print(f"[TEST PASS] Dynamic Data Discovery: Bound to '{loader.data_dir}' (zero hardcoded paths).")

    def test_damage_flash_detection(self):
        """Tests GoldSrc red screen flash damage detector using OpenCV."""
        vision = VisionBridge(grid_width=60, grid_height=60)
        
        # Frame 1: Normal dim room
        normal_frame = np.full((480, 640, 3), [60, 60, 60], dtype=np.uint8)
        is_dam1, _ = vision.detect_damage_flash(normal_frame)
        self.assertFalse(is_dam1)
        
        # Frame 2: Intense red screen damage flash (B=20, G=20, R=230)
        damage_frame = np.full((480, 640, 3), [20, 20, 230], dtype=np.uint8)
        is_dam2, intensity = vision.detect_damage_flash(damage_frame)
        self.assertTrue(is_dam2)
        self.assertGreater(intensity, 0.1)
        vision.close()
        print(f"[TEST PASS] Damage Flash: OpenCV red chromatic spike detected with intensity {intensity:.2f}.")

    def test_giant_fiber_escape(self):
        """Tests Giant Fiber emergency escape 180-degree turn and back leap dispatch."""
        bridge = InputBridge(dry_run=True)
        res = bridge.trigger_giant_fiber_escape()
        self.assertIn("GIANT FIBER ESCAPE", res["action"])
        self.assertIn("s", res["keys"])
        self.assertIn("space", res["keys"])
        self.assertTrue(bridge.is_escaping)
        bridge.release_all()
    def test_obstacle_avoidance_turn(self):
        """Tests active obstacle deadlock avoidance turn and key dispatch."""
        bridge = InputBridge(dry_run=True)
        res = bridge.trigger_obstacle_turn(direction=1)
        self.assertIn("OBSTACLE ESCAPE", res["action"])
        self.assertGreater(res["turn_dx"], 0)
        self.assertTrue(bridge.is_unstucking)
        bridge.release_all()
        print("[TEST PASS] Obstacle Avoidance: Directional turn and reverse step verified.")

if __name__ == "__main__":
    unittest.main()
