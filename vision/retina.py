"""
Fly Retina & Photoreceptor Model
Maps 2D Game Frames to Drosophila Ommatidia (R1-R6, R8) & T4/T5 Motion Circuits
Includes Adaptive Gain Control (AGC) to Prevent Neural Saturation or Silence
"""
import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from config import VisionConfig, DEFAULT_CONFIG

class FlyRetina:
    """
    Drosophila Retinal Ommatidia & Elementary Motion Detection System.
    Input: 2D BGR Game Frame (e.g. 1920x1080 or internal 3D arena)
    Output: Photoreceptor Injection Currents (R1-R6, R8) & T4/T5 Motion Signals
    
    Feature: Adaptive Gain Control (AGC) prevents network saturation during bright flashes
    and maintains responsiveness in dark underground corridors.
    """
    def __init__(self, cfg: VisionConfig = DEFAULT_CONFIG.vision):
        self.cfg = cfg
        self.w = cfg.grid_width
        self.h = cfg.grid_height
        self.total_ommatidia = self.w * self.h
        
        # Motion correlation frame buffer
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_motion_vector: Tuple[float, float] = (0.0, 0.0)
        
        # Last processed retinotopic layers
        self.last_r1_r6: np.ndarray = np.zeros((self.h, self.w), dtype=np.float32)
        self.last_r8: np.ndarray = np.zeros((self.h, self.w), dtype=np.float32)
        self.optical_flow_magnitude: float = 0.0
        
        # Adaptive Gain Control (AGC) Multiplier
        self.current_gain: float = 22.0
        self.running_mean_luminance: float = 0.35

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Processes game frame and returns AGC-scaled photoreceptor currents and motion vectors.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros(self.total_ommatidia, dtype=np.float32), {"dx": 0.0, "dy": 0.0, "flow": 0.0, "gain": self.current_gain}

        # 1. Downsample to 60x60 Ommatidia Lattice
        resized = cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # 2. R1-R6 Luminance Channel (Broadband Green + Blue)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.last_r1_r6 = gray

        # 3. R8 Spectral / Color Contrast Channel (Red threat & Health pickup contrast)
        b, g, r = cv2.split(resized.astype(np.float32) / 255.0)
        color_contrast = np.clip(r - 0.5 * (g + b), 0.0, 1.0)
        self.last_r8 = color_contrast

        # 4. T4/T5 Hassenstein-Reichardt Elementary Motion Detection
        dx, dy, flow_mag = self._compute_elementary_motion(gray)

        # 5. Composite Visual Signal
        combined = (self.cfg.r1_r6_weight * gray + self.cfg.r8_weight * color_contrast)
        
        # Direction-selective motion modulation
        if dx > 0.05: # Rightward flow (fly turns left)
            combined[:, :self.w // 2] *= (1.0 + self.cfg.motion_sensitivity * dx)
        elif dx < -0.05: # Leftward flow (fly turns right)
            combined[:, self.w // 2:] *= (1.0 + self.cfg.motion_sensitivity * abs(dx))

        # 6. Adaptive Gain Control (AGC - I_ext Normalization)
        mean_lum = float(np.mean(combined))
        self.running_mean_luminance = 0.95 * self.running_mean_luminance + 0.05 * mean_lum
        
        if self.cfg.use_agc:
            target_gain = self.cfg.target_current_mean / max(0.08, self.running_mean_luminance)
            self.current_gain = (1.0 - self.cfg.agc_adaptation_rate) * self.current_gain + self.cfg.agc_adaptation_rate * target_gain
            self.current_gain = np.clip(self.current_gain, 8.0, 40.0)

        raw_currents = combined.flatten() * self.current_gain
        photoreceptor_currents = np.clip(raw_currents, 0.0, self.cfg.target_current_max).astype(np.float32)

        motion_info = {
            "dx": float(dx),
            "dy": float(dy),
            "flow": float(flow_mag),
            "gain": float(self.current_gain),
            "mean_current": float(np.mean(photoreceptor_currents))
        }

        return photoreceptor_currents, motion_info

    def _compute_elementary_motion(self, current_gray: np.ndarray) -> Tuple[float, float, float]:
        """Calculates gradient-based optical flow and directional velocity."""
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return 0.0, 0.0, 0.0

        diff = current_gray - self.prev_gray
        grad_x = cv2.Sobel(current_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(current_gray, cv2.CV_32F, 0, 1, ksize=3)

        denom = grad_x**2 + grad_y**2 + 1e-4
        u = - (diff * grad_x) / denom
        v = - (diff * grad_y) / denom

        mean_dx = float(np.mean(u))
        mean_dy = float(np.mean(v))
        flow_mag = float(np.sqrt(mean_dx**2 + mean_dy**2))

        self.prev_gray = current_gray
        self.prev_motion_vector = (mean_dx, mean_dy)
        self.optical_flow_magnitude = flow_mag

        return mean_dx, mean_dy, flow_mag

    def get_retina_visualization(self) -> np.ndarray:
        """Returns false-color BGR rendering of what the fly compound eyes perceive."""
        disp_lum = (self.last_r1_r6 * 255).astype(np.uint8)
        disp_col = (self.last_r8 * 255).astype(np.uint8)
        retina_bgr = cv2.merge([disp_lum, (disp_lum * 0.7).astype(np.uint8), disp_col])
        return cv2.resize(retina_bgr, (200, 200), interpolation=cv2.INTER_NEAREST)
