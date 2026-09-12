"""
broadcast_server.py - Live HTTP Web Broadcaster for FlyBrain-HalfLife
Exposes real-time MJPEG gameplay stream and biological telemetry on http://localhost:8766.
Zero external server dependencies (pure Python threading & http.server).
"""
import io
import json
import time
import logging
import threading
from typing import Dict, Any, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np

logger = logging.getLogger("FlyBrain.Broadcaster")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class BroadcastState:
    """Thread-safe storage for the latest encoded video frame and telemetry data with async worker."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.new_jpeg_event = threading.Event()
        self.pending_frame: Optional[np.ndarray] = None
        self.is_running = True

        # Generate initial standby frame so browser stream immediately receives valid JPEG
        init_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(init_frame, "FLYBRAIN-HALFLIFE STREAM ACTIVE", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 102), 2)
        cv2.putText(init_frame, "Connecting to Game Window & Connectome...", (95, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        _, init_buf = cv2.imencode('.jpg', init_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.latest_jpeg: Optional[bytes] = init_buf.tobytes()
        self.latest_telemetry: Dict[str, Any] = {
            "status": "CONNECTING",
            "action": "IDLE",
            "dopamine": 0.0,
            "forward_rate": 0.0,
            "backward_rate": 0.0,
            "turn_diff": 0.0,
            "spikes": 0,
            "active_spikes": [],
            "raster": [],
            "circuits": {
                "retina": 0.0,
                "optic_lobe": 0.0,
                "central_complex": 0.0,
                "mushroom_body": 0.0,
                "descending": 0.0,
                "dopamine": 0.0,
                "turn_diff": 0.0,
                "forward_rate": 0.0,
                "backward_rate": 0.0,
                "attack_rate": 0.0,
                "is_firing": False,
                "is_escaping": False,
                "is_damage": False
            },
            "plasticity": {}
        }
        self.last_update_time: float = time.time()

        # Dedicated asynchronous background thread for JPEG compression (zero main-loop latency!)
        self.encode_thread = threading.Thread(target=self._encode_worker, daemon=True)
        self.encode_thread.start()

    def _encode_worker(self):
        """Dedicated background worker that encodes JPEGs asynchronously without blocking main game loop."""
        while self.is_running:
            self.frame_event.wait(timeout=0.08)
            self.frame_event.clear()
            
            frame_to_encode = None
            with self.lock:
                if self.pending_frame is not None:
                    frame_to_encode = self.pending_frame
                    self.pending_frame = None

            if frame_to_encode is not None:
                try:
                    fh, fw = frame_to_encode.shape[:2]
                    # Optimize stream resolution for web: cap max width to 960px to maintain crisp 60 FPS
                    if fw > 960:
                        scale = 960.0 / fw
                        frame_scaled = cv2.resize(frame_to_encode, (960, int(fh * scale)), interpolation=cv2.INTER_LINEAR)
                    else:
                        frame_scaled = frame_to_encode

                    success, buffer = cv2.imencode('.jpg', frame_scaled, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    if success:
                        jpeg_bytes = buffer.tobytes()
                        with self.lock:
                            self.latest_jpeg = jpeg_bytes
                        self.new_jpeg_event.set()
                except Exception as e:
                    logger.debug(f"Async JPEG encode error: {e}")

    def update(self, frame_bgr: np.ndarray, telemetry: Dict[str, Any]):
        """Zero-latency update: copies frame reference and signals background worker (0.01 ms)."""
        try:
            with self.lock:
                self.pending_frame = frame_bgr
                self.latest_telemetry = telemetry
            self.frame_event.set()
        except Exception as e:
            logger.debug(f"Broadcast update error: {e}")

    def stop(self):
        self.is_running = False
        self.frame_event.set()

    def get_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_telemetry(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest_telemetry)

GLOBAL_BROADCAST_STATE = BroadcastState()

class FlyBrainRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for HTML dashboard, MJPEG stream, and JSON telemetry."""

    def log_message(self, format, *args):
        # Silence routine HTTP request logging to keep console clean
        pass

    def do_GET(self):
        clean_path = self.path.split("?")[0]
        if clean_path in ["/", "/index.html"]:
            self._serve_dashboard()
        elif clean_path == "/stream":
            self._serve_mjpeg()
        elif clean_path in ["/frame.jpg", "/snapshot", "/current.jpg"]:
            self._serve_single_frame()
        elif clean_path == "/state":
            self._serve_state()
        elif clean_path in ["/connectome_coords", "/coords"]:
            self._serve_coords()
        elif clean_path == "/toggle_pause":
            self._handle_toggle_pause()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/toggle_pause":
            self._handle_toggle_pause()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_coords(self):
        conn = getattr(GLOBAL_BROADCAST_STATE, "connectome", None)
        if conn is not None and hasattr(conn, "positions"):
            pos = conn.positions
            reg = conn.region_labels
            step = 1 if len(pos) <= 6000 else 2
            sampled_pos = pos[::step]
            sampled_reg = reg[::step]
            data = {
                "count": len(sampled_pos),
                "step": step,
                "positions": np.round(sampled_pos, 1).tolist(),
                "regions": [int(r) for r in sampled_reg]
            }
        else:
            data = {"count": 0, "positions": [], "regions": []}
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_toggle_pause(self):
        bridge = getattr(GLOBAL_BROADCAST_STATE, "input_bridge", None)
        is_paused = False
        if bridge is not None:
            is_paused = bridge.toggle_pause()
        body = json.dumps({"status": "success", "is_paused": is_paused}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_dashboard(self):
        import os
        static_html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
        content = b"<h1>FlyBrain-HalfLife Web Dashboard</h1>"
        if os.path.exists(static_html_path):
            with open(static_html_path, "rb") as f:
                content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_state(self):
        data = GLOBAL_BROADCAST_STATE.get_telemetry()
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_single_frame(self):
        jpeg = GLOBAL_BROADCAST_STATE.get_frame()
        if jpeg is None:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(jpeg)))
        self.end_headers()
        self.wfile.write(jpeg)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while GLOBAL_BROADCAST_STATE.is_running:
                # Wait for new frame event or max 33ms (up to 60 FPS streaming)
                GLOBAL_BROADCAST_STATE.new_jpeg_event.wait(timeout=0.033)
                GLOBAL_BROADCAST_STATE.new_jpeg_event.clear()

                jpeg = GLOBAL_BROADCAST_STATE.get_frame()
                if jpeg is not None:
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    )
                    self.wfile.write(header)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

class FlyBrainWebBroadcaster:
    """Multi-threaded background Web Broadcaster running on port 8766."""
    def __init__(self, host: str = "0.0.0.0", port: int = 8766, input_bridge: Any = None, connectome: Any = None):
        self.host = host
        self.port = port
        self.input_bridge = input_bridge
        self.connectome = connectome
        GLOBAL_BROADCAST_STATE.input_bridge = input_bridge
        GLOBAL_BROADCAST_STATE.connectome = connectome
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.server_8080: Optional[HTTPServer] = None
        self.thread_8080: Optional[threading.Thread] = None
        self.is_running = False

    def start(self):
        if self.is_running:
            return
        try:
            self.server = HTTPServer((self.host, self.port), FlyBrainRequestHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.info(f"🌐 Live Web Stream active: http://localhost:{self.port} (and http://[YOUR_IP]:{self.port})")
        except Exception as e:
            logger.error(f"Failed to start web broadcaster on port {self.port}: {e}")

        # Spin up mirror listener on port 8080 if primary is not 8080
        if self.port != 8080:
            try:
                self.server_8080 = HTTPServer((self.host, 8080), FlyBrainRequestHandler)
                self.thread_8080 = threading.Thread(target=self.server_8080.serve_forever, daemon=True)
                self.thread_8080.start()
                logger.info("🌐 Dual-Port Mirror active: http://localhost:8080 (and http://[YOUR_IP]:8080)")
            except Exception as e:
                logger.debug(f"Port 8080 mirror not active: {e}")

    def update(self, frame_bgr: np.ndarray, telemetry: Dict[str, Any]):
        GLOBAL_BROADCAST_STATE.update(frame_bgr, telemetry)

    def stop(self):
        GLOBAL_BROADCAST_STATE.stop()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False
        if self.server_8080:
            try:
                self.server_8080.shutdown()
                self.server_8080.server_close()
            except Exception:
                pass
        logger.info("Web broadcaster stopped.")
