"""
Retro 3D FPS Raycasting Arena (Built-in Half-Life/Doom Simulator)
Enables standalone simulation, training, and visualization without requiring Half-Life installed.
"""
import numpy as np
import cv2
import math
from typing import Tuple, Dict, Any, List

class TargetEntity:
    def __init__(self, x: float, y: float, entity_type: str = "enemy"):
        self.x = x
        self.y = y
        self.type = entity_type # "enemy" or "health"
        self.alive = True

class HalfLifeArena:
    """
    3D Raycasting FPS Simulation Arena.
    Features Half-Life industrial aesthetics, corridors, hazard collisions,
    headcrab enemies, and health supply crates.
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

        # Player State
        self.player_x = 2.5
        self.player_y = 2.5
        self.player_angle = 0.0 # Radians
        self.fov = math.pi / 3.0 # 60-degree field of view
        self.health = 100.0
        self.score = 0
        self.kills = 0
        self.muzzle_flash_timer = 0
        
        # Targets and Pickups
        self.entities: List[TargetEntity] = []
        self._spawn_entities()

    def _spawn_entities(self):
        """Spawns enemies and health packs inside the arena."""
        self.entities = [
            TargetEntity(4.5, 3.5, "enemy"),
            TargetEntity(9.5, 3.5, "enemy"),
            TargetEntity(3.5, 7.5, "enemy"),
            TargetEntity(10.5, 9.5, "enemy"),
            TargetEntity(13.5, 13.5, "enemy"),
            TargetEntity(7.5, 11.5, "enemy"),
            TargetEntity(1.5, 5.5, "health"),
            TargetEntity(14.5, 1.5, "health")
        ]

    def step(self, turn_val: float, forward_val: float, is_attack: bool) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Executes motor commands:
        turn_val: [-1.0, 1.0] angular yaw delta
        forward_val: [0.0, 1.0] forward velocity
        is_attack: True if firing primary weapon
        """
        events = []

        # 1. Yaw Angle Update
        rot_speed = 0.08
        self.player_angle += turn_val * rot_speed
        self.player_angle %= (2 * math.pi)

        # 2. Forward Movement
        move_speed = 0.12 * min(1.0, max(0.0, forward_val))
        dx = math.cos(self.player_angle) * move_speed
        dy = math.sin(self.player_angle) * move_speed

        new_x = self.player_x + dx
        new_y = self.player_y + dy

        # Collision with solid walls
        if self.map[int(self.player_y), int(new_x)] == 0:
            self.player_x = new_x
        else:
            events.append({"type": "damage", "mag": 0.2})
            
        if self.map[int(new_y), int(self.player_x)] == 0:
            self.player_y = new_y
        else:
            events.append({"type": "damage", "mag": 0.2})

        if move_speed > 0.04:
            events.append({"type": "forward", "mag": 0.05})

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
        """Renders 3D perspective projection via DDA raymarching."""
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Ceiling and Floor
        frame[:self.h//2, :] = [30, 25, 35]
        frame[self.h//2:, :] = [45, 55, 65]

        num_rays = 120
        ray_step = self.fov / num_rays
        start_angle = self.player_angle - self.fov / 2.0
        slice_w = self.w // num_rays + 1

        z_buffer = [999.0] * num_rays

        for i in range(num_rays):
            angle = start_angle + i * ray_step
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)

            dist = 0.05
            hit_wall = False
            while dist < 16.0:
                cx = self.player_x + cos_a * dist
                cy = self.player_y + sin_a * dist
                map_x, map_y = int(cx), int(cy)
                if 0 <= map_x < self.map_w and 0 <= map_y < self.map_h:
                    if self.map[map_y, map_x] == 1:
                        hit_wall = True
                        break
                dist += 0.08

            corrected_dist = dist * math.cos(angle - self.player_angle)
            z_buffer[i] = corrected_dist

            wall_h = int(self.h / (corrected_dist + 1e-4))
            top = max(0, self.h // 2 - wall_h // 2)
            bottom = min(self.h - 1, self.h // 2 + wall_h // 2)

            # Distance shading
            shade = max(0.1, min(1.0, 1.0 - (dist / 14.0)))
            b = int(70 * shade)
            g = int(120 * shade)
            r = int(140 * shade)

            x_start = i * (self.w // num_rays)
            cv2.rectangle(frame, (x_start, top), (x_start + slice_w, bottom), (b, g, r), -1)

        # Draw Sprites
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

            if abs(angle_to) < self.fov / 1.5:
                screen_x = int((self.w / 2) + math.tan(angle_to) * (self.w / 2))
                sprite_size = int(self.h / (dist * 1.2))
                top_y = self.h // 2 - sprite_size // 2
                
                ray_idx = int((screen_x / self.w) * num_rays)
                if 0 <= ray_idx < num_rays and dist < z_buffer[ray_idx]:
                    if ent.type == "enemy":
                        # Alien Headcrab Sprite (Red/Orange sphere)
                        cv2.circle(frame, (screen_x, top_y + sprite_size//2), sprite_size//2, (20, 50, 220), -1)
                        cv2.circle(frame, (screen_x, top_y + sprite_size//2), sprite_size//4, (60, 160, 255), -1)
                    else:
                        # Health Supply Crate (Cyan/Green Box)
                        cv2.rectangle(frame, (screen_x - sprite_size//3, top_y), 
                                      (screen_x + sprite_size//3, top_y + sprite_size), (220, 200, 40), -1)

        # HUD and Crosshair
        cx, cy = self.w // 2, self.h // 2
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)

        # Muzzle Flash
        if self.muzzle_flash_timer > 0:
            cv2.circle(frame, (cx + 60, cy + 80), 35, (80, 220, 255), -1)
            cv2.circle(frame, (cx + 60, cy + 80), 18, (255, 255, 255), -1)

        # Bottom HUD Status
        cv2.putText(frame, f"HEALTH: {int(self.health)}", (20, self.h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"KILLS: {self.kills}", (self.w - 140, self.h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame
