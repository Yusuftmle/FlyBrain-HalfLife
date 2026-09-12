"""
test_stdp_and_broadcaster.py - Unit tests for STDP Plasticity & Live Web Broadcaster
"""
import unittest
import urllib.request
import json
import numpy as np
import scipy.sparse as sp
import torch

from lif_engine import PyTorchLIFEngine
from server.broadcast_server import FlyBrainWebBroadcaster

class TestSTDPAndBroadcaster(unittest.TestCase):

    def test_stdp_plasticity_bounds(self):
        """Validates that Kenyon Cell -> MBON synapses exhibit STDP plasticity bounded by [0.1x, 2.0x]."""
        num_neurons = 200
        # Create synthetic bipartite connections from KC (0..99) to MBON (100..199)
        rows = []
        cols = []
        data = []
        for i in range(100):
            for j in range(100, 150):
                rows.append(i)
                cols.append(j)
                data.append(0.5)
        csr = sp.csr_matrix((data, (rows, cols)), shape=(num_neurons, num_neurons), dtype=np.float32)

        engine = PyTorchLIFEngine(sparse_weights=csr, num_neurons=num_neurons, device="cpu")
        kc_idx = np.arange(100)
        mbon_idx = np.arange(100, 150)
        engine.init_plastic_synapses(kc_idx, mbon_idx)

        self.assertIsNotNone(engine.plastic_edge_indices)
        self.assertEqual(len(engine.plastic_edge_indices), 100 * 50)

        # Baseline check
        stats_0 = engine.get_plasticity_telemetry()
        self.assertTrue(stats_0["enabled"])
        self.assertAlmostEqual(stats_0["mean_efficacy"], 1.0, places=3)

        # Drive neurons to spike
        drive = np.zeros(num_neurons, dtype=np.float32)
        drive[0:100] = 30.0 # Force KC spikes
        drive[100:150] = 30.0 # Force MBON spikes
        for _ in range(5):
            engine.forward_step(drive)

        # Potentiation test: Positive dopamine
        for _ in range(20):
            engine.apply_stdp_update(dopamine=2.0, eta=0.05)
        
        stats_pot = engine.get_plasticity_telemetry()
        self.assertGreater(stats_pot["mean_efficacy"], 1.0)
        self.assertLessEqual(stats_pot["max_efficacy"], 2.0001, "Plasticity must not exceed 2.0x ceiling")

        # Depression test: Negative dopamine shock
        for _ in range(50):
            engine.apply_stdp_update(dopamine=-3.0, eta=0.1)

        stats_dep = engine.get_plasticity_telemetry()
        self.assertGreaterEqual(stats_dep["min_efficacy"], 0.0999, "Plasticity must not drop below 0.1x floor")
        self.assertEqual(len(stats_dep["histogram"]), 10)

    def test_web_broadcaster_endpoints(self):
        """Validates that FlyBrainWebBroadcaster serves HTTP HTML dashboard and /state JSON."""
        test_port = 8799
        broadcaster = FlyBrainWebBroadcaster(host="127.0.0.1", port=test_port)
        broadcaster.start()
        self.assertTrue(broadcaster.is_running)

        try:
            # Send sample telemetry
            dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            dummy_telemetry = {
                "status": "active",
                "action": "WALK_FORWARD",
                "fps": 28.5,
                "forward_rate": 0.12,
                "turn_diff": -0.04,
                "dopamine": 0.45,
                "spikes": 310,
                "sugar_active": True,
                "plasticity": {"mean_efficacy": 1.08}
            }
            broadcaster.update(dummy_frame, dummy_telemetry)

            # Test GET /
            url_root = f"http://127.0.0.1:{test_port}/"
            with urllib.request.urlopen(url_root, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
                self.assertIn("FlyBrain", html)

            # Test GET /state
            url_state = f"http://127.0.0.1:{test_port}/state"
            with urllib.request.urlopen(url_state, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(body["action"], "WALK_FORWARD")
                self.assertEqual(body["spikes"], 310)
                self.assertTrue(body["sugar_active"])
                self.assertAlmostEqual(body["dopamine"], 0.45, places=2)

        finally:
            broadcaster.stop()
            self.assertFalse(broadcaster.is_running)

if __name__ == "__main__":
    unittest.main()
