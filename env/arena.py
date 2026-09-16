"""
Retro 3D FPS Raycasting Arena (Built-in Half-Life/Doom Simulator)
Enables standalone simulation, training, and visualization without requiring Half-Life installed.
Features DDA raycasting, textured industrial corridors, Headcrab enemies, health pickups,
and an integrated 2D tactical radar minimap with route trail tracing.
"""
import numpy as np
import cv2
import math
from typing import Tuple, Dict, Any, List

class TargetEntity:
    def __init__(self, x: float, y: float, entity_type: str = "enemy"):
        self.x = x
        self.y = y
        self.type = entity_type  # "enemy" or "health"
        self.alive = True

class HalfLifeArena:
    """
    3D Raycasting FPS Simulation Arena.
    Features Half-Life industrial corridors, hazard collisions,
    headcrab enemies, health supply crates, and an integrated real-time tactical radar minimap.
    The fly brain perceives this 3D environment and navigates the maze.
    """
    def __init__(self, width: int = 640, height: int = 480):
        self.w = width
        self.h = height
        
        # 16x16 Maze Grid Map (1: Solid Wall, 0: Walkable Space)
        self.map = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
            [1,0,1,1,0,0,1,0,1,1,1,0,1,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1],
            [1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,1],
            [1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1],
            [1,1,1,0,1,0,0,1,0,1,1,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,1,1,0,1,1,1,1,0,1,0,1,0,1],
            [1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,1],
            [1,0,1,0,1,1,0,1,0,1,0,1,1,1,0,1],
            [1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1],
            [1,0,1,1,1,0,1,1,1,1,1,1,0,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ], dtype=np.int32)
        self.map_h, self.map_w = self.map.shape

        # Player State (Spawn in open corridor at 1.5, 1.5 facing East down hallway)
        self.player_x = 1.5
        self.player_y = 1.5
        self.player_angle = 0.0  # Radians
        self.fov = math.pi / 3.0  # 60-degree field of view
        self.health = 100.0
        self.score = 0
        self.kills = 0
        self.muzzle_flash_timer = 0
        
        # Route breadcrumb history (capped at 200 points for trail visualization)
        self.trail: List[Tuple[float, float]] = [(self.player_x, self.player_y)]
        self._step_counter = 0

        # Targets and Pickups
        self.entities: List[TargetEntity] = []
        self._spawn_entities()

    def _spawn_entities(self):
        """Spawns enemies and health packs inside valid walkable spaces."""
        self.entities = [
            TargetEntity(4.5, 1.5, "enemy"),
            TargetEntity(9.5, 3.5, "enemy"),
            TargetEntity(3.5, 7.5, "enemy"),
            TargetEntity(10.5, 9.5, "enemy"),
            TargetEntity(13.5, 13.5, "enemy"),
            TargetEntity(6.5, 11.5, "enemy"),
            TargetEntity(1.5, 5.5, "health"),
            TargetEntity(14.5, 1.5, "health")
        ]

    def _is_walkable(self, x: float, y: float, radius: float = 0.22) -> bool:
        """Collision check preventing camera from clipping flush against walls."""
        for dx in [-radius, radius]:
            for dy in [-radius, radius]:
                mx, my = int(x + dx), int(y + dy)
                if mx < 0 or mx >= self.map_w or my < 0 or my >= self.map_h:
                    return False
                if self.map[my, mx] == 1:
                    return False
        return True

    def step(self, turn_val: float, forward_val: float, is_attack: bool) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Executes motor commands:
        turn_val: [-1.0, 1.0] angular yaw delta
        forward_val: [0.0, 1.0] forward velocity
        is_attack: True if firing primary weapon
        """
        events = []
        self._step_counter += 1

        # 1. Yaw Angle Update
        rot_speed = 0.08
        self.player_angle += turn_val * rot_speed
        self.player_angle %= (2 * math.pi)

        # 2. Movement (Forward & Backward) with sliding collision
        move_val = max(-1.0, min(1.0, forward_val))
        move_speed = 0.12 * move_val
        dx = math.cos(self.player_angle) * move_speed
        dy = math.sin(self.player_angle) * move_speed

        new_x = self.player_x + dx
        new_y = self.player_y + dy

        # Slide along X
        if self._is_walkable(new_x, self.player_y):
            self.player_x = new_x

        # Slide along Y
        if self._is_walkable(self.player_x, new_y):
            self.player_y = new_y

        if abs(move_speed) > 0.03:
            events.append({"type": "forward" if move_speed > 0 else "backward", "mag": 0.05})

        # Record route breadcrumbs every 3 frames
        if self._step_counter % 3 == 0:
            if not self.trail or math.hypot(self.player_x - self.trail[-1][0], self.player_y - self.trail[-1][1]) > 0.04:
                self.trail.append((self.player_x, self.player_y))
                if len(self.trail) > 150:
                    self.trail.pop(0)

        # 3. Fire and Hit Target
        if is_attack:
            self.muzzle_flash_timer = 2
            hit = self._check_gun_shot()
            if hit:
                events.append({"type": "kill", "mag": 1.0})
                self.kills += 1
                self.score += 100

        if self.muzzle_flash_timer > 0:
            self.muzzle_flash_timer -= 1

        # 4. Entity Interactions
        for ent in self.entities:
            if not ent.alive:
                continue
            dist = math.hypot(ent.x - self.player_x, ent.y - self.player_y)
            if dist < 0.6:
                if ent.type == "health":
                    self.health = min(100.0, self.health + 30.0)
                    ent.alive = False
                    events.append({"type": "forward", "mag": 0.5})
                elif ent.type == "enemy":
                    self.health = max(0.0, self.health - 15.0)
                    events.append({"type": "damage", "mag": 1.0})

        # Auto-respawn at start of maze if player dies
        if self.health <= 0:
            self.player_x = 1.5
            self.player_y = 1.5
            self.player_angle = 0.0
            self.health = 100.0
            self.trail = [(self.player_x, self.player_y)]

        # Respawn entities if all enemies defeated
        if not any(e.alive for e in self.entities if e.type == "enemy"):
            self._spawn_entities()

        # 5. Render 3D Frame
        frame_bgr = self._render_frame()

        return frame_bgr, events

    def _check_gun_shot(self) -> bool:
        """Evaluates whether crosshair ray intersects an enemy."""
        for ent in self.entities:
            if not ent.alive or ent.type != "enemy":
                continue
            dx = ent.x - self.player_x
            dy = ent.y - self.player_y
            dist = math.hypot(dx, dy)
            if dist > 8.0:
                continue
            angle_to = math.atan2(dy, dx) - self.player_angle
            while angle_to > math.pi: angle_to -= 2 * math.pi
            while angle_to < -math.pi: angle_to += 2 * math.pi
            
            if abs(angle_to) < 0.18:
                ent.alive = False
                return True
        return False

    def _render_frame(self) -> np.ndarray:
        """Renders 3D perspective projection via DDA raymarching with textures & radar."""
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Ceiling (Industrial dark metal gradient)
        ceiling_grad = np.linspace(24, 10, self.h // 2, dtype=np.uint8)
        frame[:self.h//2, :, 0] = ceiling_grad[:, None]
        frame[:self.h//2, :, 1] = (ceiling_grad * 0.9).astype(np.uint8)[:, None]
        frame[:self.h//2, :, 2] = (ceiling_grad * 0.85).astype(np.uint8)[:, None]

        # 2. Floor (Textured concrete gradient with perspective lines)
        floor_grad = np.linspace(25, 60, self.h - self.h // 2, dtype=np.uint8)
        frame[self.h//2:, :, 0] = (floor_grad * 0.85).astype(np.uint8)[:, None]
        frame[self.h//2:, :, 1] = (floor_grad * 0.95).astype(np.uint8)[:, None]
        frame[self.h//2:, :, 2] = floor_grad[:, None]

        # Floor grid lines for motion feedback
        for line_y in range(self.h // 2 + 15, self.h, 25):
            cv2.line(frame, (0, line_y), (self.w, line_y), (35, 42, 48), 1)

        # 3. DDA Raycaster
        num_rays = 240
        ray_step = self.fov / num_rays
        start_angle = self.player_angle - self.fov / 2.0
        slice_w = max(1, math.ceil(self.w / num_rays))

        z_buffer = [999.0] * num_rays

        for i in range(num_rays):
            angle = start_angle + i * ray_step
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)

            # DDA Step
            map_x = int(self.player_x)
            map_y = int(self.player_y)

            delta_dist_x = abs(1.0 / (cos_a + 1e-9))
            delta_dist_y = abs(1.0 / (sin_a + 1e-9))

            if cos_a < 0:
                step_x = -1
                side_dist_x = (self.player_x - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1.0 - self.player_x) * delta_dist_x

            if sin_a < 0:
                step_y = -1
                side_dist_y = (self.player_y - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1.0 - self.player_y) * delta_dist_y

            hit = False
            side = 0
            while not hit:
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1

                if 0 <= map_x < self.map_w and 0 <= map_y < self.map_h:
                    if self.map[map_y, map_x] == 1:
                        hit = True
                else:
                    hit = True

            if side == 0:
                perp_dist = (map_x - self.player_x + (1 - step_x) / 2) / (cos_a + 1e-9)
                wall_x = self.player_y + perp_dist * sin_a
            else:
                perp_dist = (map_y - self.player_y + (1 - step_y) / 2) / (sin_a + 1e-9)
                wall_x = self.player_x + perp_dist * cos_a

            wall_x -= math.floor(wall_x)
            perp_dist = max(0.12, perp_dist)
            z_buffer[i] = perp_dist

            # Wall height clamped so floor/ceiling always retain perspective
            raw_wall_h = int((self.h * 0.75) / max(0.35, perp_dist))
            wall_h = min(int(self.h * 0.94), raw_wall_h)
            top = max(0, self.h // 2 - wall_h // 2)
            bottom = min(self.h - 1, self.h // 2 + wall_h // 2)

            # Distance Shading and Orientation Shading
            shade = max(0.12, min(1.0, 1.0 - (perp_dist / 13.0)))
            if side == 1:
                shade *= 0.72  # North-South wall darkening for 3D depth

            # Industrial wall panel pattern (3 panels per tile)
            panel_rel = (wall_x * 3.0) % 1.0
            is_panel_seam = (panel_rel < 0.05) or (panel_rel > 0.95)
            if is_panel_seam:
                shade *= 0.60

            base_b = int(60 * shade)
            base_g = int(95 * shade)
            base_r = int(120 * shade)

            x_start = int(i * (self.w / num_rays))
            x_end = min(self.w, x_start + slice_w)
            cv2.rectangle(frame, (x_start, top), (x_end, bottom), (base_b, base_g, base_r), -1)

            # Top and bottom hazard/metal trims on walls
            trim_h = max(2, int(wall_h * 0.08))
            if top + trim_h < bottom:
                cv2.rectangle(frame, (x_start, top), (x_end, top + trim_h), 
                              (int(base_b * 0.6), int(base_g * 0.6), int(base_r * 0.6)), -1)
                cv2.rectangle(frame, (x_start, bottom - trim_h), (x_end, bottom), 
                              (int(base_b * 0.4), int(base_g * 0.4), int(base_r * 0.4)), -1)

        # 4. Draw Sprites (Enemies and Pickups)
        for ent in self.entities:
            if not ent.alive:
                continue
            dx = ent.x - self.player_x
            dy = ent.y - self.player_y
            dist = math.hypot(dx, dy)
            if dist < 0.3 or dist > 14.0:
                continue

            angle_to = math.atan2(dy, dx) - self.player_angle
            while angle_to > math.pi: angle_to -= 2 * math.pi
            while angle_to < -math.pi: angle_to += 2 * math.pi

            if abs(angle_to) < self.fov / 1.4:
                screen_x = int((self.w / 2) + math.tan(angle_to) * (self.w / 2))
                sprite_size = int(self.h / (dist * 1.1))
                top_y = self.h // 2 - sprite_size // 2
                
                ray_idx = int((screen_x / self.w) * num_rays)
                if 0 <= ray_idx < num_rays and dist < z_buffer[ray_idx]:
                    if ent.type == "enemy":
                        # Alien Headcrab Sprite (Glowing biological hazard)
                        cv2.circle(frame, (screen_x, top_y + sprite_size//2), sprite_size//2, (20, 50, 220), -1)
                        cv2.circle(frame, (screen_x, top_y + sprite_size//2), sprite_size//4, (60, 160, 255), -1)
                        cv2.circle(frame, (screen_x - sprite_size//6, top_y + sprite_size//3), max(1, sprite_size//10), (255, 255, 255), -1)
                    else:
                        # Health Supply Crate
                        cv2.rectangle(frame, (screen_x - sprite_size//3, top_y), 
                                      (screen_x + sprite_size//3, top_y + sprite_size), (220, 200, 40), -1)
                        # Red Cross on crate
                        cw = max(2, sprite_size // 8)
                        cv2.line(frame, (screen_x - sprite_size//5, top_y + sprite_size//2), 
                                 (screen_x + sprite_size//5, top_y + sprite_size//2), (0, 0, 255), cw)
                        cv2.line(frame, (screen_x, top_y + sprite_size//3), 
                                 (screen_x, top_y + 2*sprite_size//3), (0, 0, 255), cw)

        # 5. Crosshair (compact 6px crosshair)
        cx, cy = self.w // 2, self.h // 2
        cv2.line(frame, (cx - 4, cy), (cx + 4, cy), (0, 255, 255), 1)
        cv2.line(frame, (cx, cy - 4), (cx, cy + 4), (0, 255, 255), 1)

        # 6. Muzzle Flash
        if self.muzzle_flash_timer > 0:
            cv2.circle(frame, (cx + 60, cy + 80), 35, (80, 220, 255), -1)
            cv2.circle(frame, (cx + 60, cy + 80), 18, (255, 255, 255), -1)

        # 7. Draw 2D Tactical Radar Minimap (Maze, Trail, FOV, and Entities)
        self._draw_minimap(frame)

        # 8. Bottom HUD Status
        cv2.putText(frame, f"HEALTH: {int(self.health)}", (20, self.h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"KILLS: {self.kills}", (self.w - 140, self.h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def _draw_minimap(self, frame: np.ndarray):
        """Renders top-down 2D radar overlay showing maze layout, trail, FOV cone, and entities."""
        map_size = 140
        pad = 12
        mx0 = self.w - map_size - pad
        my0 = pad
        
        # Semi-transparent dark background
        sub = frame[my0:my0+map_size, mx0:mx0+map_size]
        bg = np.full_like(sub, 18)
        cv2.addWeighted(sub, 0.25, bg, 0.75, 0, sub)
        cv2.rectangle(frame, (mx0, my0), (mx0+map_size, my0+map_size), (80, 85, 105), 1)

        tile_s = map_size / float(self.map_w)

        # Draw maze walls
        for r in range(self.map_h):
            for c in range(self.map_w):
                if self.map[r, c] == 1:
                    x1 = int(mx0 + c * tile_s)
                    y1 = int(my0 + r * tile_s)
                    x2 = int(mx0 + (c + 1) * tile_s)
                    y2 = int(my0 + (r + 1) * tile_s)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (75, 80, 95), -1)

        # Draw route trail (breadcrumbs)
        for i in range(1, len(self.trail)):
            pt1 = (int(mx0 + self.trail[i-1][0] * tile_s), int(my0 + self.trail[i-1][1] * tile_s))
            pt2 = (int(mx0 + self.trail[i][0] * tile_s), int(my0 + self.trail[i][1] * tile_s))
            cv2.line(frame, pt1, pt2, (0, 200, 255), 1)

        # Draw entities
        for ent in self.entities:
            if not ent.alive:
                continue
            ex = int(mx0 + ent.x * tile_s)
            ey = int(my0 + ent.y * tile_s)
            if ent.type == "enemy":
                cv2.circle(frame, (ex, ey), 3, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (ex, ey), 3, (255, 220, 0), -1)

        # Draw player dot and FOV cone
        px = int(mx0 + self.player_x * tile_s)
        py = int(my0 + self.player_y * tile_s)
        
        cone_len = 16
        a1 = self.player_angle - self.fov / 2.0
        a2 = self.player_angle + self.fov / 2.0
        p_fov1 = (int(px + math.cos(a1) * cone_len), int(py + math.sin(a1) * cone_len))
        p_fov2 = (int(px + math.cos(a2) * cone_len), int(py + math.sin(a2) * cone_len))
        cv2.line(frame, (px, py), p_fov1, (0, 255, 255), 1)
        cv2.line(frame, (px, py), p_fov2, (0, 255, 255), 1)
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        # Minimap title
        cv2.putText(frame, "RADAR", (mx0 + 6, my0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 185, 200), 1)
