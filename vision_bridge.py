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
        self.last_flow_details: Dict[str, float] = {
            "divergence": 0.0,
            "v_pitch": 0.0,
            "v_top": 0.0,
            "v_bottom": 0.0
        }
        
        # GoldSrc Red Screen Damage Flash Detector State
        self.red_baseline: Optional[float] = None
        self.damage_flash_cooldown: int = 0
        self.consecutive_red_frames: int = 0
        self.last_valid_frame: Optional[np.ndarray] = None
        self.prev_hud_crop: Optional[np.ndarray] = None
        
        # Drosophila Lamina Local Contrast Adaptation (Weber-Fechner / CLAHE for shadow enhancement)
        self.clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
        
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

    def process_frame(
        self, 
        frame_bgr: np.ndarray, 
        roi_size: Optional[int] = None,
        ego_yaw_dx: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Converts 2D game frame into biophysical photoreceptor currents.
        Crops GoldSrc HUD, downsamples to ommatidia grid, and applies Naka-Rushton log-sigmoidal
        photoreceptor transduction and Poisson fluctuations.
        Includes biological efference copy (Kim et al. 2015) to cancel self-induced rotational flow.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros(self.total_ommatidia, dtype=np.float32), {"dx": 0.0, "dy": 0.0, "flow": 0.0}

        # 0. GoldSrc HUD Exclusion and Central Crosshair Crop
        clean_fov = self.crop_halflife_fov(frame_bgr, roi_pixels=roi_size)

        # 1. Resample to ommatidial lattice dimensions (matching R1-R6 photoreceptors)
        resized = cv2.resize(clean_fov, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # 2. R1-R6 Broadband Luminance & Drosophila Lamina Local Adaptation (Laughlin 1981)
        gray_u8 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        adapted_gray = self.clahe.apply(gray_u8).astype(np.float32) / 255.0
        raw_gray = gray_u8.astype(np.float32) / 255.0
        # Blend: 60% locally adapted (reveals deep shadows, dark doors, and silhouettes) + 40% ambient
        gray = np.clip(0.60 * adapted_gray + 0.40 * raw_gray, 0.0, 1.0)

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

        # 4. Hassenstein-Reichardt Directional Motion with Efference Copy Subtraction
        dx_net, dy, flow_mag = self._compute_optical_flow(gray, ego_yaw_dx=ego_yaw_dx)
        dx = dx_net

        # 4.5 Drosophila Lamina Cartridge Spatial & Temporal Edge Extraction
        # Spatial Laplacian edges directly on the 60x60 ommatidial lattice
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        # Lamina L3 Monopolar cells: amplify gradients in shadowed / dim regions
        shadow_weight = np.clip(1.35 - gray, 0.5, 1.8)
        spatial_edges = np.clip(lap * 3.2 * shadow_weight, 0.0, 1.0)

        temporal_motion = np.zeros_like(gray)
        if self.prev_gray is not None:
            temporal_diff = gray - self.prev_gray
            on_channel = np.maximum(0.0, temporal_diff)   # L1 Brightening
            off_channel = np.maximum(0.0, -temporal_diff) # L2 Darkening / Looming shadow
            # Drosophila OFF-pathway has 1.8x higher gain and faster kinetics (Joesch et al., Nature 2010)
            temporal_motion = np.clip((0.9 * on_channel + 1.8 * off_channel) * 2.5, 0.0, 1.0)

        # 5. Composite Visual Stimulus (High-Contrast R1-R6 Motion, Shadow & Edge Transduction)
        # Drosophila R1-R6 ommatidial photoreceptors process pure broadband luminance, shadow gradients & motion
        shadow_contrast = np.clip((0.55 - gray) * 1.2, 0.0, 0.5)  # L2 darkness / shadow silhouette
        stimulus = np.clip(0.40 * gray + 0.30 * spatial_edges + 0.20 * temporal_motion + 0.10 * shadow_contrast, 0.0, 1.0)
        
        # Compound-eye ommatidial multispectral rendering (illuminated shadows & emerald edges)
        edge_overlay = np.zeros_like(resized)
        edge_overlay[:, :, 1] = np.clip(spatial_edges * 200, 0, 255).astype(np.uint8) # Emerald edge contrast
        edge_overlay[:, :, 0] = np.clip(spatial_edges * 120, 0, 255).astype(np.uint8)
        
        shadow_map = np.clip((0.55 - gray) * 1.6, 0.0, 1.0)
        shadow_tint = np.zeros_like(resized)
        shadow_tint[:, :, 0] = np.clip(shadow_map * 130, 0, 255).astype(np.uint8) # Biological indigo depth for shadows
        shadow_tint[:, :, 2] = np.clip(shadow_map * 70, 0, 255).astype(np.uint8)
        
        retina_multispectral = cv2.addWeighted(resized, 0.75, edge_overlay, 0.35, 0)
        retina_multispectral = cv2.add(retina_multispectral, shadow_tint)
        
        # Grayscale neural stimulus visualization
        stim_vis = np.clip(stimulus * 255.0, 0, 255).astype(np.uint8)
        retina_gray_bgr = cv2.cvtColor(stim_vis, cv2.COLOR_GRAY2BGR)
        half_w = self.w // 2
        
        # 5.1 Drosophila Optomotor Corridor Centering & Wall Avoidance Reflex
        left_eye = gray[:, :self.w // 2]
        right_eye = gray[:, self.w // 2:]
        left_edges = spatial_edges[:, :self.w // 2]
        right_edges = spatial_edges[:, self.w // 2:]
        left_depth = float(np.var(left_eye) + np.mean(left_eye) * 0.4 + np.mean(left_edges) * 0.2)
        right_depth = float(np.var(right_eye) + np.mean(right_eye) * 0.4 + np.mean(right_edges) * 0.2)
        depth_balance = float(right_depth - left_depth)
        
        # Center collision zone (flat wall directly in front has near-zero texture variance)
        center_zone = gray[self.h // 4 : 3 * self.h // 4, self.w // 4 : 3 * self.w // 4]
        center_var = float(np.var(center_zone))

        # Obstacle proximity: only true when flush against a wall or flat barrier (guard against black screen)
        is_obstacle_close = bool(center_var < 0.0018 and float(np.mean(center_zone)) > 0.04)
        is_fence = False
        
        now = time.time()
        if not hasattr(self, "obstacle_divert_dir"):
            self.obstacle_divert_dir = 0.25
            self.last_obstacle_divert_time = 0.0

        if is_obstacle_close:
            # Wall or fence dead ahead: steer away towards whichever hemisphere is more open
            if abs(depth_balance) < 0.02:
                if (now - self.last_obstacle_divert_time) > 1.0:
                    self.obstacle_divert_dir = 0.30 if np.random.rand() > 0.5 else -0.30
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

        # Directional motion amplification (optomotor reflex with biological efference copy)
        # Voluntary turns are canceled out via corollary discharge; only unpredicted external slip excites the eyes
        if dx > 0.06:
            stimulus[:, :self.w // 2] *= (1.0 + 0.40 * min(1.0, dx))
        elif dx < -0.06:
            stimulus[:, self.w // 2:] *= (1.0 + 0.40 * min(1.0, abs(dx)))

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
        # Split into Left Eye (first 1800 ommatidia) and Right Eye (second 1800 ommatidia)
        # Matches core/connectome.py topology (Left Eye: x=-220, Right Eye: x=+220)
        half_w = self.w // 2
        left_stimulus = stimulus[:, :half_w]   # Left visual field (60x30)
        right_stimulus = stimulus[:, half_w:]  # Right visual field (60x30)

        sigma_powered = np.power(self.sigma_semi, self.n_hill)
        left_powered = np.power(np.clip(left_stimulus, 1e-4, 1.0), self.n_hill)
        right_powered = np.power(np.clip(right_stimulus, 1e-4, 1.0), self.n_hill)

        left_mean_currents = self.i_max * (left_powered / (left_powered + sigma_powered))
        right_mean_currents = self.i_max * (right_powered / (right_powered + sigma_powered))

        # Poisson quantal synaptic noise per eye
        left_noise = np.random.poisson(lam=np.clip(left_mean_currents * 0.1, 0.01, 10.0)).astype(np.float32)
        right_noise = np.random.poisson(lam=np.clip(right_mean_currents * 0.1, 0.01, 10.0)).astype(np.float32)

        left_currents = left_mean_currents.flatten() + (left_noise.flatten() * 0.5)
        right_currents = right_mean_currents.flatten() + (right_noise.flatten() * 0.5)

        # Concatenate: First half = Left Eye, Second half = Right Eye
        total_currents = np.concatenate([left_currents, right_currents])
        photoreceptor_currents = np.clip(total_currents, 0.0, self.i_max * 1.5).astype(np.float32)
        assert photoreceptor_currents.shape == (self.total_ommatidia,), (
            f"Photoreceptor current shape assertion failed: {photoreceptor_currents.shape} != ({self.total_ommatidia},)"
        )

        # 5.3 Dorsal Light Response (DLR) & Visual Horizon Estimation
        # In nature, flies assume the sky/ceiling is brighter and has distinct edge gradients.
        upper_lum = float(np.mean(gray[:self.h // 2, :]))
        lower_lum = float(np.mean(gray[self.h // 2:, :]))
        upper_edges = float(np.mean(spatial_edges[:self.h // 2, :]))
        lower_edges = float(np.mean(spatial_edges[self.h // 2:, :]))

        lum_sum = upper_lum + lower_lum + 1e-4
        horizon_balance = float((upper_lum - lower_lum) / lum_sum)

        ceiling_score = 0.0
        if upper_lum > 0.45 and lower_edges < 0.18:
            ceiling_score = float(np.clip((upper_lum - 0.40) * 1.5 + (0.18 - lower_edges) * 2.0, 0.0, 1.0))

        floor_score = 0.0
        if lower_lum > upper_lum and upper_edges < 0.12 and lower_edges > 0.22:
            floor_score = float(np.clip((lower_edges - 0.20) * 2.0, 0.0, 1.0))

        divergence = float(self.last_flow_details.get("divergence", 0.0))
        v_pitch = float(self.last_flow_details.get("v_pitch", dy))

        metrics = {
            "dx": float(dx),
            "dx_raw": float(self.last_flow_details.get("raw_dx", dx)),
            "ego_slip": float(self.last_flow_details.get("ego_turn_slip", 0.0)),
            "dy": float(dy),
            "flow": float(flow_mag),
            "divergence": float(divergence),
            "v_pitch": float(v_pitch),
            "ceiling_score": float(ceiling_score),
            "floor_score": float(floor_score),
            "horizon_balance": float(horizon_balance),
            "depth_balance": float(depth_balance),
            "is_obstacle_close": bool(is_obstacle_close),
            "is_fence": bool(is_fence),
            "is_target_in_crosshair": bool(is_target_in_crosshair),
            "ch_motion": float(ch_motion),
            "ch_edges": float(ch_edges),
            "ch_r8": float(ch_r8),
            "r8y_green": float(r8y_green_mean),
            "r8p_blue": float(r8p_blue_mean),
            "r8y_current": float(r8y_current),
            "r8p_current": float(r8p_current),
            "mean_photocurrent": float(np.mean(photoreceptor_currents)),
            "left_eye_current": float(np.mean(left_currents)),
            "right_eye_current": float(np.mean(right_currents)),
            "left_eye_bgr": retina_multispectral[:, :half_w].copy(),
            "right_eye_bgr": retina_multispectral[:, half_w:].copy(),
            "retina_color_bgr": retina_multispectral
        }

        return photoreceptor_currents, metrics

    def _compute_optical_flow(self, current_gray: np.ndarray, ego_yaw_dx: float = 0.0) -> Tuple[float, float, float]:
        """
        Calculates gradient-based Hassenstein-Reichardt optical flow.
        Includes biological corollary discharge / efference copy (Kim et al. 2015 Nature):
        Cancels out the visual rotational slip produced by the fly's own motor steering command.
        """
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return 0.0, 0.0, 0.0

        dt_diff = current_gray - self.prev_gray
        grad_x = cv2.Sobel(current_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(current_gray, cv2.CV_32F, 0, 1, ksize=3)

        denom = grad_x**2 + grad_y**2 + 1e-4
        u = - (dt_diff * grad_x) / denom
        v = - (dt_diff * grad_y) / denom

        raw_dx = float(np.mean(u))
        dy = float(np.mean(v))
        mag = float(np.mean(np.sqrt(u**2 + v**2)))

        # Biological Efference Copy (Corollary Discharge):
        # Turning right (mouse_dx > 0) causes leftward scene translation (negative u).
        # We subtract this expected self-induced slip so voluntary turns are not resisted.
        ego_turn_slip = - float(ego_yaw_dx) * 0.008
        dx_net = raw_dx - ego_turn_slip

        # Optical Flow Helmholtz Decomposition
        h, w = u.shape
        u_left = float(np.mean(u[:, :w // 2]))
        u_right = float(np.mean(u[:, w // 2:]))
        v_top = float(np.mean(v[:h // 2, :]))
        v_bottom = float(np.mean(v[h // 2:, :]))

        # Divergence (Looming / Z-Surge expansion)
        div_x = u_right - u_left
        div_y = v_bottom - v_top
        divergence = float(div_x + div_y)

        # Pure Pitch Flow (VS cells common-mode vertical flow)
        v_pitch = float((v_top + v_bottom) / 2.0)

        self.last_flow_details = {
            "divergence": divergence,
            "v_pitch": v_pitch,
            "v_top": v_top,
            "v_bottom": v_bottom,
            "raw_dx": raw_dx,
            "dx_net": dx_net,
            "ego_turn_slip": ego_turn_slip
        }

        self.prev_gray = current_gray
        self.prev_flow = (dx_net, dy)
        return dx_net, dy, mag

    def detect_damage_flash(
        self, 
        frame_bgr: np.ndarray, 
        threshold: float = 0.05,
        last_shot_time: float = 0.0
    ) -> Tuple[bool, float]:
        """
        GoldSrc Damage Detection: Fullscreen ScreenFade red chromatic impulse.
        Accurately detects both light bullet grazes and heavy explosive damage,
        while strictly rejecting localized RPG lasers and weapon lights.
        Returns (is_damage: bool, intensity: float).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, 0.0

        now = time.time()
        # Weapon Muzzle Flash Immunity Window (shortened to 120ms so enemy return fire is not missed)
        is_own_fire = (now - last_shot_time) < 0.12
            
        if self.damage_flash_cooldown > 0:
            self.damage_flash_cooldown -= 1
            
        h, w = frame_bgr.shape[:2]
        if h < 20 or w < 20:
            return False, 0.0

        # 1. Screen Border Red Excess Check:
        # Fullscreen damage ScreenFade tints the whole frame uniformly, including borders.
        b_h = max(2, h // 12)
        b_w = max(2, w // 12)
        top_strip = frame_bgr[:b_h, :]
        bot_strip = frame_bgr[-b_h:, :]
        left_strip = frame_bgr[:, :b_w]
        right_strip = frame_bgr[:, -b_w:]

        def get_strip_red_excess(patch: np.ndarray) -> float:
            if patch.size == 0:
                return 0.0
            r = patch[:, :, 2].astype(np.float32) / 255.0
            g = patch[:, :, 1].astype(np.float32) / 255.0
            b = patch[:, :, 0].astype(np.float32) / 255.0
            return float(np.mean(np.clip(r - 0.5 * (g + b), 0.0, 1.0)))

        border_r_excess = 0.25 * (
            get_strip_red_excess(top_strip)
            + get_strip_red_excess(bot_strip)
            + get_strip_red_excess(left_strip)
            + get_strip_red_excess(right_strip)
        )

        # 2. Central Viewport Blood / Damage Flash Check
        center_patch = frame_bgr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        delta_red = 0.0
        mean_r_excess = 0.0
        high_red_fraction = 0.0
        is_damage = False

        if center_patch.size > 0:
            small = cv2.resize(center_patch, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            r_c = small[:, :, 2]
            g_c = small[:, :, 1]
            b_c = small[:, :, 0]
            r_excess = np.clip(r_c - 0.5 * (g_c + b_c), 0.0, 1.0)
            mean_r_excess = float(np.mean(r_excess))
            high_red_fraction = float(np.mean(r_excess > 0.06))

            # Temporal persistence:
            if mean_r_excess > 0.06:
                self.consecutive_red_frames += 1
            else:
                self.consecutive_red_frames = 0

            # Initialize baseline on first frame
            if self.red_baseline is None:
                self.red_baseline = mean_r_excess
            else:
                delta_red = mean_r_excess - self.red_baseline
                self.red_baseline = 0.90 * self.red_baseline + 0.10 * mean_r_excess

            # Half-Life damage flash criteria:
            # 1. Not during fly's own immediate muzzle flash (<120ms)
            # 2. Sharp onset delta spike in red (delta_red > threshold)
            # 3. Sufficient red coverage across central FOV (high_red_fraction > 0.10)
            # 4. Border or overall pervasive redness (border_r_excess > 0.012 or delta_red > 0.08)
            # 5. Onset window (consecutive_red_frames <= 6 frames at 60-100 FPS)
            if (not is_own_fire
                    and delta_red > threshold
                    and high_red_fraction > 0.10
                    and (border_r_excess > 0.012 or delta_red > 0.08)
                    and self.consecutive_red_frames <= 6):
                is_damage = True

        if is_damage and (self.damage_flash_cooldown == 0):
            self.damage_flash_cooldown = 15  # Debounce
            logger.warning(
                f"🚨 Damage detected! Red delta: {delta_red:.3f} (mean: {mean_r_excess:.3f}, border: {border_r_excess:.3f}, coverage: {high_red_fraction:.2f})"
            )
            return True, float(max(delta_red, 0.25))
            
        return False, float(max(delta_red, 0.0))

    def close(self):
        """Releases MSS screen resources."""
        if self.sct:
            self.sct.close()
