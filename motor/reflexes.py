"""
reflexes.py - Biological Emergency Reflexes & Anti-Stuck FSM Manager
Manages high-priority emergency escapes, damage retaliation, and obstacle recovery maneuvers.
"""
import time
from typing import Dict, Any, Optional
from core.platform.base import BaseInputDriver, ActionKey


class ReflexState:
    NORMAL = "NORMAL"
    OBSTACLE_SACCADE = "OBSTACLE_SACCADE"
    COMBAT_RETALIATION = "COMBAT_RETALIATION"
    GIANT_FIBER = "GIANT_FIBER"


class ReflexManager:
    """
    Finite State Machine (FSM) for biological emergency reflexes:
      1. Obstacle Deadlock Saccade: 90-120 degree spin + backstep to clear flat wall collisions.
      2. Combat Retaliation: Instant counter-fire + evasive backstep and strafe hop.
      3. Giant Fiber Escape: 180 degree jump-turn away from lethal looming threats.
    """
    def __init__(self, input_driver: BaseInputDriver):
        self.driver = input_driver
        self.current_state: str = ReflexState.NORMAL
        self.state_start_time: float = 0.0
        self.state_duration: float = 0.0
        self.last_retaliation_time: float = 0.0
        self.last_obstacle_saccade_time: float = 0.0
        self.active_reflex_keys = []

    def is_active(self) -> bool:
        """Returns True if an emergency reflex currently has control override."""
        return self.current_state != ReflexState.NORMAL

    def update(self, now: Optional[float] = None) -> bool:
        """
        Updates active reflex timer. Automatically recovers held reflex keys upon expiration.
        Returns True if a reflex is still actively executing.
        """
        if now is None:
            now = time.time()

        if self.current_state != ReflexState.NORMAL:
            if (now - self.state_start_time) >= self.state_duration:
                self._clear_reflex_state()
                return False
            return True
        return False

    def trigger_obstacle_saccade(self, direction: int = 1, flick_pixels: int = 180) -> Dict[str, Any]:
        """
        Executes an unstuck saccade maneuver to break flat wall deadlock:
          - Backs up for 350ms
          - Controlled yaw mouse flick in escape direction (180px instead of 550px instant snap)
          - Strafe step away from wall
        """
        now = time.time()
        if (now - self.last_obstacle_saccade_time) < 0.8:
            return {"action": "SUPPRESSED (Cooldown)", "state": self.current_state}

        self._clear_reflex_state()
        self.current_state = ReflexState.OBSTACLE_SACCADE
        self.state_start_time = now
        self.state_duration = 0.35
        self.last_obstacle_saccade_time = now

        # Direction: 1 = Right, -1 = Left
        flick_dx = flick_pixels if flick_pixels else (120 if direction >= 0 else -120)
        self.driver.mouse_move_relative(dx=flick_dx, dy=0)

        # Brief backstep + lateral strafe step to break contact
        self.driver.press_action(ActionKey.BACKWARD)
        strafe_act = ActionKey.STRAFE_RIGHT if direction >= 0 else ActionKey.STRAFE_LEFT
        self.driver.press_action(strafe_act)
        self.active_reflex_keys = [ActionKey.BACKWARD, strafe_act]

        side = "RIGHT" if direction >= 0 else "LEFT"
        desc = f"OBSTACLE SACCADE {side} (yaw={flick_dx} + S + {strafe_act.value})"
        return {"action": desc, "flick_dx": flick_dx, "state": self.current_state}

    def trigger_combat_retaliation(self, direction: int = 1, turn_whip: bool = True, flick_pixels: int = 200) -> Dict[str, Any]:
        """
        Executes immediate combat counter-fire, 180° whip turn, and evasive jump-strafe upon taking damage:
          - Instant yaw whip turn (180° or 140° snap) to face/scan threat source.
          - Rapid counter-fire burst (sustained mouse click).
          - Evasive backstep + lateral strafe sprint + jump hop to dodge enemy fire.
          - 380ms sustained evasion state.
        """
        now = time.time()
        if (now - self.last_retaliation_time) < 0.40:
            return {"action": "SUPPRESSED (Cooldown)", "is_firing": False}

        self._clear_reflex_state()
        self.current_state = ReflexState.COMBAT_RETALIATION
        self.state_start_time = now
        self.state_duration = 0.38
        self.last_retaliation_time = now

        # 1. Immediate 140°-180° combat yaw whip turn to face attacker / scan threat
        flick_dx = flick_pixels if direction >= 0 else -flick_pixels
        if turn_whip:
            self.driver.mouse_move_relative(dx=flick_dx, dy=-10)

        # 2. Sustained weapon counter-fire (100ms hold ensures GoldSrc weapon discharges)
        self.driver.mouse_click(duration=0.10)

        # 3. Evasive backstep + lateral strafe sprint + bunnyhop dodge
        self.driver.press_action(ActionKey.BACKWARD)
        strafe_act = ActionKey.STRAFE_RIGHT if direction >= 0 else ActionKey.STRAFE_LEFT
        self.driver.press_action(strafe_act)
        self.driver.press_action(ActionKey.JUMP)
        self.active_reflex_keys = [ActionKey.BACKWARD, strafe_act, ActionKey.JUMP]

        side = "RIGHT" if direction >= 0 else "LEFT"
        return {
            "action": f"COMBAT RETALIATION (180° Whip {side} + Counter-Fire + Jump-Strafe)",
            "is_firing": True,
            "flick_dx": flick_dx if turn_whip else 0,
            "state": self.current_state
        }

    def _clear_reflex_state(self):
        """Releases keys held by the reflex and restores NORMAL state."""
        for act in self.active_reflex_keys:
            self.driver.release_action(act)
        self.active_reflex_keys.clear()
        self.current_state = ReflexState.NORMAL
        self.state_duration = 0.0

    def release_all(self):
        """Failsafe release of all reflex actions."""
        self._clear_reflex_state()
