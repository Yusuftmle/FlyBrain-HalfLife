"""
test_mdn_color.py - Unit tests for MDN Moonwalker Descending Neurons and R8 Color Vision
"""
import unittest
import numpy as np

from data_loader import ConnectomeDataLoader
from input_bridge import InputBridge
from vision_bridge import VisionBridge


class TestMDNAndColorVision(unittest.TestCase):

    def test_mdn_and_dnp09_data_loading(self):
        """Verifies MDN and DNp09 descending neurons and kc_indices are mapped in connectome metadata."""
        loader = ConnectomeDataLoader()
        weights, meta = loader.build_canonical_flywire_connectome()
        readouts = meta["readouts"]
        readout_types = [r["type"] for r in readouts]
        self.assertIn("MDN", readout_types)
        self.assertIn("DNp09", readout_types)
        
        mdn_indices = [r["index"] for r in readouts if r["type"] == "MDN"]
        dnp09_indices = [r["index"] for r in readouts if r["type"] == "DNp09"]
        dn_off = meta["region_offsets"]["descending"]
        self.assertEqual(mdn_indices, [dn_off + 4, dn_off + 5])
        self.assertEqual(dnp09_indices, [dn_off + 6, dn_off + 7])
        self.assertIn("kc_indices", meta)
        self.assertGreater(len(meta["kc_indices"]), 0)

    def test_mdn_backward_dispatch(self):
        """Verifies that high MDN firing rate activates backward stepping in InputBridge."""
        bridge = InputBridge(dry_run=True)
        
        # 1. Normal forward walking: forward rate high, MDN low
        res_fwd = bridge.decode_and_dispatch(
            dnp20_left_rate=0.05,
            dnp20_right_rate=0.05,
            dnpe017_forward_rate=0.45,
            dnpe017_attack_rate=0.0,
            mdn_backward_rate=0.01,
            is_obstacle_close=False
        )
        self.assertIn("FORWARD", res_fwd["action"])
        self.assertGreater(res_fwd["net_forward"], 0.0)

        # 2. Obstacle / Moonwalker retro-stepping: MDN backward rate dominates
        res_bwd = bridge.decode_and_dispatch(
            dnp20_left_rate=0.05,
            dnp20_right_rate=0.05,
            dnpe017_forward_rate=0.02,
            dnpe017_attack_rate=0.0,
            mdn_backward_rate=0.38,
            is_obstacle_close=True
        )
        self.assertIn("MDN", res_bwd["action"])
        self.assertLess(res_bwd["net_forward"], 0.0)
        self.assertEqual(res_bwd["backward_rate"], 0.38)

    def test_r8_color_photoreceptor_kinetics(self):
        """Verifies R8y (Green) and R8p (Blue) extraction with Naka-Rushton saturation."""
        vision = VisionBridge(grid_width=60, grid_height=60)
        
        # Test 1: Dominant Green frame (e.g. Health Kit)
        green_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        green_frame[:, :, 1] = 220  # Strong green channel in BGR
        
        _, metrics_green = vision.process_frame(green_frame)
        self.assertIn("r8y_green", metrics_green)
        self.assertIn("r8y_current", metrics_green)
        self.assertGreater(metrics_green["r8y_green"], metrics_green["r8p_blue"])
        self.assertGreater(metrics_green["r8y_current"], 3.0)

        # Test 2: Dominant Blue frame (e.g. Armor Battery)
        blue_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blue_frame[:, :, 0] = 230  # Strong blue channel in BGR
        
        _, metrics_blue = vision.process_frame(blue_frame)
        self.assertIn("r8p_blue", metrics_blue)
        self.assertIn("r8p_current", metrics_blue)
        self.assertGreater(metrics_blue["r8p_blue"], metrics_blue["r8y_green"])
        self.assertGreater(metrics_blue["r8p_current"], 3.0)


if __name__ == "__main__":
    unittest.main()
