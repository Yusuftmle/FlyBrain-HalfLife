import unittest
import time
import numpy as np
from config import DEFAULT_CONFIG
from data_loader import ConnectomeDataLoader
from lif_engine import PyTorchLIFEngine
from vision_bridge import VisionBridge
from input_bridge import InputBridge

class TestPerformanceFPS(unittest.TestCase):
    """Verifies that the core agent loop maintains high throughput (>60 FPS)."""

    def test_core_loop_throughput(self):
        loader = ConnectomeDataLoader()
        sparse_weights, meta = loader.build_canonical_flywire_connectome()
        num_neurons = meta["num_neurons"]

        engine = PyTorchLIFEngine(
            num_neurons=num_neurons,
            sparse_weights=sparse_weights,
            dt=0.001,
            device="cuda"
        )
        vision = VisionBridge(grid_width=60, grid_height=60)
        inp = InputBridge(dry_run=True)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_ext = np.zeros(num_neurons, dtype=np.float32)

        # Warmup
        for _ in range(5):
            photocurrents, motion = vision.process_frame(dummy_frame)
            _ = engine.forward_substeps(dummy_ext, num_substeps=2)

        # Timed 100 steps
        t0 = time.perf_counter()
        for _ in range(100):
            photocurrents, motion = vision.process_frame(dummy_frame)
            spikes = engine.forward_substeps(dummy_ext, num_substeps=2)
            rates = engine.get_firing_rates()
            _ = inp.decode_and_dispatch(
                dnp20_left_rate=float(rates[0]),
                dnp20_right_rate=float(rates[1]),
                dnpe017_forward_rate=float(rates[2]),
                dnpe017_attack_rate=float(rates[3]),
                mdn_backward_rate=float(rates[4]),
                is_obstacle_close=False,
                vertical_pitch_rate=0.0
            )
        elapsed = time.perf_counter() - t0
        fps = 100.0 / elapsed

        print(f"\n[TestPerformanceFPS] 100-step core pipeline speed: {fps:.1f} FPS ({elapsed*10:.2f} ms/frame)")
        self.assertGreater(fps, 60.0, f"Expected >60 FPS, got {fps:.1f} FPS")

if __name__ == "__main__":
    unittest.main()
