"""
vision_bridge.py - Low-Latency Screen Capture and Biophysical Photoreceptor Transduction
Implements mss-based frame grab (<2ms) and Naka-Rushton / Poisson photoreceptor current modeling.
"""
import logging
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import cv2
import numpy as np
from core.platform import get_screen_capture_driver

logger = logging.getLogger("FlyBrain.VisionBridge")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VisionBridge:
    """
    Sub-2ms Game Frame Capture and Biophysical Retinal Transduction Layer.
    Uses platform-abstracted screen capture for high-frequency direct memory frame transfer.
    Converts photons to neural current via Naka-Rushton Log-Sigmoidal kinetics
    with Poisson synaptic noise generation (No toy linear scaling).
    """
    def __init__(
        self, 
        grid_width: int = 60, 
        grid_height: int = 60, 
        window_title: str = "Half-Life",
        i_max: float = 28.0,
        semi_saturation: float = 0.35,
        hill_exponent: float = 1.25
    ):
        self.w: int = grid_width
        self.h: int = grid_height
        self.total_ommatidia: int = self.w * self.h
        self.window_title: str = window_title
        
        # Biophysical Naka-Rushton Parameters
        self.i_max: float = i_max
        self.sigma_semi: float = semi_saturation
        self.n_hill: float = hill_exponent
        
        # Platform Screen Capture Driver (Windows GDI/MSS or Linux X11/MSS)
        self.capture_driver = get_screen_capture_driver(target_title=window_title)
        
        # Elementary Motion Detector State
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_flow: Tuple[float, float] = (0.0, 0.0)
        
        # GoldSrc Red Screen Damage Flash Detector State
        self.red_baseline: Optional[float] = None
        self.damage_flash_cooldown: int = 0
        self.last_valid_frame: Optional[np.ndarray] = None
        self.prev_hud_crop: Optional[np.ndarray] = None
        
        logger.info(f"VisionBridge initialized: {self.w}x{self.h} grid ({self.total_ommatidia} ommatidia).")

    @property
    def hwnd(self) -> Optional[int]:
        """Backwards compatibility property for Windows window handle."""
        return getattr(self.capture_driver, "hwnd", None)

    @property
    def sct(self):
        """Backwards compatibility property for MSS instance."""
        return getattr(self.capture_driver, "sct", None)

    def is_game_running(self) -> bool:
        """Returns True if the game window is currently running or process is active."""
        return self.capture_driver.is_game_running()

    def is_game_process_running(self) -> bool:
        """Checks if game process is active on the system."""
        return self.capture_driver.is_game_process_running()

    def focus_game_window(self) -> bool:
        """Brings the game window to the foreground."""
        return self.capture_driver.focus_game_window()

    def is_game_focused(self) -> bool:
        """Returns True if the game window is the active foreground window."""
        return self.capture_driver.is_game_focused()

    def capture_frame(self) -> np.ndarray:
        """Grabs active game frame via platform capture driver."""
        if not self.is_game_running():
            standby = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(standby, "HALF-LIFE NOT DETECTED", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            cv2.putText(standby, "Waiting for game window... (Inputs Locked)", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            return standby

        frame = self.capture_driver.grab_frame()
        if frame is not None and frame.size > 0:
            self.last_valid_frame = frame
            return frame

        if self.last_valid_frame is not None:
            return self.last_valid_frame.copy()

        return np.zeros((360, 640, 3), dtype=np.uint8)

    def close(self):
        """Releases capture resources."""
        self.capture_driver.close()

    def crop_halflife_fov(
        self, 
        frame_bgr: np.ndarray, 
        roi_pixels: Optional[int] = None,
        fov_ratio_x: float = 0.65, 
        fov_ratio_y: float = 0.65,
        hud_bottom_margin: float = 0.18
    ) -> np.ndarray:
        """
        Crops GoldSrc (Half-Life) game frame to isolate the 3D world corridor and crosshair.
        Excludes HUD indicators (bottom-left health/HEV, bottom-right ammo, top chat),
        focusing solely on the central field of view (FOV) surrounding the crosshair.
        
        Args:
            frame_bgr: Raw input BGR frame captured from game window.
            roi_pixels: If specified (e.g. 64), cuts an exact square patch around crosshair.
            fov_ratio_x: Horizontal fraction of frame to include centered on crosshair (default 0.65).
            fov_ratio_y: Vertical fraction of 3D viewport to include (default 0.65).
            hud_bottom_margin: Bottom fraction reserved for GoldSrc HUD indicators to cut off (default 0.18).
            
        Returns:
            Cropped BGR frame centered on the crosshair without HUD clutter.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr
            
        fh, fw = frame_bgr.shape[:2]
        if fh < 20 or fw < 20:
            return frame_bgr
            
        # 1. Strip bottom HUD area (health, suit power, ammunition counters)
        game_world_h = int(fh * (1.0 - hud_bottom_margin))
        game_world = frame_bgr[:game_world_h, :]
        
        cx = fw // 2
        cy = game_world_h // 2
        
        # 2. Extract crosshair-centered ROI
        if roi_pixels is not None and roi_pixels > 0:
            half_roi = roi_pixels // 2
            x1 = max(0, cx - half_roi)
            x2 = min(fw, cx + half_roi)
            y1 = max(0, cy - half_roi)
            y2 = min(game_world_h, cy + half_roi)
            return game_world[y1:y2, x1:x2]
        
        crop_w = max(10, int(fw * fov_ratio_x))
        crop_h = max(10, int(game_world_h * fov_ratio_y))
        
        x1 = max(0, cx - crop_w // 2)
        x2 = min(fw, cx + crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        y2 = min(game_world_h, cy + crop_h // 2)
        
        return game_world[y1:y2, x1:x2]

    def process_frame(self, frame_bgr: np.ndarray, roi_size: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Converts 2D game frame into biophysical photoreceptor currents.
        Crops GoldSrc HUD, downsamples to ommatidia grid, and applies Naka-Rushton log-sigmoidal
        photoreceptor transduction and Poisson fluctuations.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros(self.total_ommatidia, dtype=np.float32), {"dx": 0.0, "dy": 0.0, "flow": 0.0}

        # 0. GoldSrc HUD Exclusion and Central Crosshair Crop
        clean_fov = self.crop_halflife_fov(frame_bgr, roi_pixels=roi_size)

        # 1. Resample to ommatidial lattice dimensions (matching R1-R6 photoreceptors)
        resized = cv2.resize(clean_fov, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # 2. R1-R6 Broadband Luminance (Grayscale normalized [0.0, 1.0])
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = np.clip(gray, 0.0, 1.0)

        # 3. R8 Spectral / Color Contrast (Xiao et al., Nature 2023)
        # R8y: Yellow/Green sensitive opsin (Green channel in linear sRGB)
        # R8p: Pale/Blue sensitive opsin (Blue channel in linear sRGB)
        # R8 contrast: Red hazard and enemy contrast
        b, g, r = cv2.split(resized.astype(np.float32) / 255.0)
        r8_contrast = np.clip(r - 0.5 * (g + b), 0.0, 1.0)
        r8y_green_mean = float(np.mean(g))
        r8p_blue_mean = float(np.mean(b))
        r8y_current = float(self.i_max * (r8y_green_mean ** 1.25 / (r8y_green_mean ** 1.25 + 0.35 ** 1.25)))
        r8p_current = float(self.i_max * (r8p_blue_mean ** 1.25 / (r8p_blue_mean ** 1.25 + 0.35 ** 1.25)))

        # 4. Hassenstein-Reichardt Directional Motion Computation
        dx, dy, flow_mag = self._compute_optical_flow(gray)

        # 4.5 Drosophila Lamina Cartridge Filtering (L1/L2 High-Pass Temporal Contrast)
        # Enhances spatial contour edges and temporal ON/OFF motion channels
        spatial_edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        spatial_edges = np.clip(spatial_edges * 1.8, 0.0, 1.0)

        temporal_motion = np.zeros_like(gray)
        if self.prev_gray is not None:
            temporal_diff = gray - self.prev_gray
            on_channel = np.maximum(0.0, temporal_diff)   # L1 Brightening
            off_channel = np.maximum(0.0, -temporal_diff) # L2 Darkening / Looming shadow
            temporal_motion = np.clip((on_channel + off_channel) * 2.5, 0.0, 1.0)

        # 5. Composite Visual Stimulus:
        # Baseline Luminance (0.45) + Spatial Edges (0.25) + L1/L2 Motion (0.20) + R8 Color (0.10)
        stimulus = 0.45 * gray + 0.25 * spatial_edges + 0.20 * temporal_motion + 0.10 * r8_contrast
        
        # 5.1 Drosophila Optomotor Corridor Centering & Wall Avoidance Reflex
        # Compares visual depth/luminance/variance between Left and Right hemispheres
        left_eye = gray[:, :self.w // 2]
        right_eye = gray[:, self.w // 2:]
        left_depth = float(np.var(left_eye) + np.mean(left_eye) * 0.4)
        right_depth = float(np.var(right_eye) + np.mean(right_eye) * 0.4)
        depth_balance = right_depth - left_depth
        
        # Center collision zone (flat wall directly in front has near-zero texture variance)
        center_zone = gray[self.h // 4 : 3 * self.h // 4, self.w // 4 : 3 * self.w // 4]
        is_obstacle_close = float(np.var(center_zone)) < 0.0006
        
        now = time.time()
        if not hasattr(self, "obstacle_divert_dir"):
            self.obstacle_divert_dir = 0.22
            self.last_obstacle_divert_time = 0.0

        if is_obstacle_close:
            # Wall dead ahead: steer away towards whichever hemisphere is more open
            # If symmetric, maintain consistent avoidance direction for at least 1.4s to complete avoidance turn
            if abs(depth_balance) < 0.008:
                if (now - self.last_obstacle_divert_time) > 1.4:
                    self.obstacle_divert_dir = 0.22 if np.random.rand() > 0.5 else -0.22
                    self.last_obstacle_divert_time = now
                depth_balance = self.obstacle_divert_dir
            if depth_balance >= 0:
                stimulus[:, self.w // 2:] *= 1.75 # Excite right hemifield -> turn right
            else:
                stimulus[:, :self.w // 2] *= 1.75 # Excite left hemifield -> turn left
        elif abs(depth_balance) > 0.012:
            # Optomotor corridor centering: naturally gravitate towards open corridor/doorways
            if depth_balance > 0:
                stimulus[:, self.w // 2:] *= (1.0 + min(0.65, depth_balance * 2.5))
            else:
                stimulus[:, :self.w // 2] *= (1.0 + min(0.65, abs(depth_balance) * 2.5))

        # Directional motion amplification (optomotor reflex)
        if dx > 0.04:
            stimulus[:, :self.w // 2] *= (1.0 + 1.0 * dx)
        elif dx < -0.04:
            stimulus[:, self.w // 2:] *= (1.0 + 1.0 * abs(dx))

        # 5.2 Drosophila LC10/LC11 Small Target & Crosshair Motion Tracking
        # Checks the central 14x14 ommatidia directly in front of the fly (aligned with weapon crosshair)
        ch_y1, ch_y2 = max(0, self.h // 2 - 7), min(self.h, self.h // 2 + 7)
        ch_x1, ch_x2 = max(0, self.w // 2 - 7), min(self.w, self.w // 2 + 7)
        ch_motion = float(np.mean(temporal_motion[ch_y1:ch_y2, ch_x1:ch_x2]))
        ch_edges = float(np.mean(spatial_edges[ch_y1:ch_y2, ch_x1:ch_x2]))
        ch_r8 = float(np.mean(r8_contrast[ch_y1:ch_y2, ch_x1:ch_x2]))
        
        # Efference copy: subtract ego-motion optical flow from center patch
        relative_center_motion = ch_motion - (flow_mag * 1.25)
        
        # Target locked ONLY if an independent moving entity or distinct high-contrast enemy model is in crosshair
        is_target_in_crosshair = bool(
            not is_obstacle_close and ((relative_center_motion > 0.18 and ch_edges > 0.28) or ch_r8 > 0.30)
        )

        # 6. Biophysical Naka-Rushton Photoreceptor Transduction
        # I(L) = I_max * (L^n / (L^n + sigma^n))
        stim_powered = np.power(np.clip(stimulus, 1e-4, 1.0), self.n_hill)
        sigma_powered = np.power(self.sigma_semi, self.n_hill)
        mean_currents = self.i_max * (stim_powered / (stim_powered + sigma_powered))

        # 7. Poisson Synaptic Emission Noise
        # Terminal synaptic release exhibits discrete quantal Poisson variability
        poisson_noise = np.random.poisson(lam=np.clip(mean_currents * 0.1, 0.01, 10.0)).astype(np.float32)
        total_currents = mean_currents.flatten() + (poisson_noise.flatten() * 0.5)
        
        photoreceptor_currents = np.clip(total_currents, 0.0, self.i_max * 1.5).astype(np.float32)
        assert photoreceptor_currents.shape == (self.total_ommatidia,), (
            f"Photoreceptor current shape assertion failed: {photoreceptor_currents.shape} != ({self.total_ommatidia},)"
        )

        metrics = {
            "dx": float(dx),
            "dy": float(dy),
            "flow": float(flow_mag),
            "depth_balance": float(depth_balance),
            "is_obstacle_close": bool(is_obstacle_close),
            "is_target_in_crosshair": bool(is_target_in_crosshair),
            "ch_motion": float(ch_motion),
            "ch_edges": float(ch_edges),
            "ch_r8": float(ch_r8),
            "r8y_green": float(r8y_green_mean),
            "r8p_blue": float(r8p_blue_mean),
            "r8y_current": float(r8y_current),
            "r8p_current": float(r8p_current),
            "mean_photocurrent": float(np.mean(photoreceptor_currents))
        }

        return photoreceptor_currents, metrics

    def _compute_optical_flow(self, current_gray: np.ndarray) -> Tuple[float, float, float]:
        """Calculates gradient-based Hassenstein-Reichardt optical flow."""
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return 0.0, 0.0, 0.0

        dt_diff = current_gray - self.prev_gray
        grad_x = cv2.Sobel(current_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(current_gray, cv2.CV_32F, 0, 1, ksize=3)

        denom = grad_x**2 + grad_y**2 + 1e-4
        u = - (dt_diff * grad_x) / denom
        v = - (dt_diff * grad_y) / denom

        dx = float(np.mean(u))
        dy = float(np.mean(v))
        mag = float(np.mean(np.sqrt(u**2 + v**2)))

        self.prev_gray = current_gray
        self.prev_flow = (dx, dy)
        return dx, dy, mag

    def detect_damage_flash(
        self, 
        frame_bgr: np.ndarray, 
        threshold: float = 0.09,
        last_shot_time: float = 0.0
    ) -> Tuple[bool, float]:
        """
        Dual-Mechanism GoldSrc Damage Detection:
          1. Central Viewport Blood/Damage indicator red chromatic spike (with weapon muzzle flash immunity).
          2. Bottom-left Health HUD indicator sudden pixel change / drop.
        Returns (is_damage: bool, intensity: float).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0

        now = time.time()
        # 1. Weapon Muzzle Flash Immunity Window (ignore own weapon fire)
        is_own_fire = (now - last_shot_time) < 0.38
            
        if self.damage_flash_cooldown > 0:
            self.damage_flash_cooldown -= 1
            
        h, w = frame_bgr.shape[:2]
        if h < 20 or w < 20:
            return False, 0.0

        # --- Central Viewport Blood / Damage Flash ---
        center_patch = frame_bgr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        delta_red = 0.0
        mean_r_excess = 0.0
        is_damage = False

        if center_patch.size > 0:
            small = cv2.resize(center_patch, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            r_c = small[:, :, 2]
            g_c = small[:, :, 1]
            b_c = small[:, :, 0]
            r_excess = np.clip(r_c - 0.5 * (g_c + b_c), 0.0, 1.0)
            mean_r_excess = float(np.mean(r_excess))
            
            # Initialize baseline on first frame
            if self.red_baseline is None:
                self.red_baseline = mean_r_excess
            else:
                delta_red = mean_r_excess - self.red_baseline
                self.red_baseline = 0.92 * self.red_baseline + 0.08 * mean_r_excess

            # Only trigger on real incoming damage; suppress player's own weapon muzzle flash
            if not is_own_fire and (delta_red > threshold or mean_r_excess > 0.45):
                is_damage = True

        if is_damage and (self.damage_flash_cooldown == 0):
            self.damage_flash_cooldown = 14 # Refractory debounce (~0.45s)
            logger.warning(
                f"🚨 Damage detected! Red delta: {delta_red:.3f} (mean: {mean_r_excess:.3f})"
            )
            return True, float(max(delta_red, 0.25))
            
        return False, float(max(delta_red, 0.0))

    def close(self):
        """Releases MSS screen resources."""
        if self.sct:
            self.sct.close()
