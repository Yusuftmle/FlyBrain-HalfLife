"""
Descending Motor Neuron Controller (DNp20 & DNpe017 Decoder)
Decodes Drosophila Descending Premotor Pathways into FPS Controls
Includes Hardware Debounce & Refractory Cooldown to Prevent Input Buffer Flooding
"""
import numpy as np
import time
from typing import Dict, Any
from config import MotorConfig, DEFAULT_CONFIG
from core.connectome import FlyConnectome
from core.lif_engine import LIFEngine
from motor.direct_input import DirectInputDriver, DIK_W, DIK_A, DIK_S, DIK_D

class MotorController:
    """
    Decodes DNp20 (Steering) and DNpe017 (Forward Walking & Attacking) descending neurons.
    Maintains hardware debounce and refractory cooldown for natural game input.
    """
    def __init__(self, connectome: FlyConnectome, lif_engine: LIFEngine, cfg: MotorConfig = DEFAULT_CONFIG.motor):
        self.cfg = cfg
        self.lif_engine = lif_engine
        self.indices = connectome.get_motor_neuron_indices()
        self.driver = DirectInputDriver(dry_run=cfg.dry_run)
        
        # Debounce / Refractory Cooldown Timestamps
        self.last_press_time: Dict[str, float] = {
            "turn_l": 0.0,
            "turn_r": 0.0,
            "forward": 0.0,
            "attack": 0.0
        }
        
        # Real-time telemetry state
        self.current_turn_delta: float = 0.0
        self.current_forward_rate: float = 0.0
        self.current_attack_rate: float = 0.0
        self.active_action_label: str = "IDLE"
        self.is_firing: bool = False

    def update(self) -> Dict[str, Any]:
        """
        Retrieves latest firing rates from LIF engine and executes debounced key actions.
        """
        now = time.time()
        rates = self.lif_engine.get_firing_rates()
        
        rate_dnp20_l = float(rates[self.indices["dnp20_left"]])
        rate_dnp20_r = float(rates[self.indices["dnp20_right"]])
        rate_forward = float(rates[self.indices["dnpe017_forward"]])
        rate_attack = float(rates[self.indices["dnpe017_attack"]])
        
        # 1. DNp20 Differential Steering: Left vs Right firing rate
        turn_diff = rate_dnp20_l - rate_dnp20_r
        self.current_turn_delta = turn_diff
        self.current_forward_rate = rate_forward
        self.current_attack_rate = rate_attack

        actions_taken = []

        # Turn Left (DNp20-Left)
        if turn_diff > self.cfg.turn_threshold:
            if (now - self.last_press_time["turn_l"]) > self.cfg.cooldown_duration:
                self.driver.release_key(DIK_D)
                self.driver.press_key(DIK_A)
                self.driver.mouse_move_relative(dx=-10, dy=0)
                self.last_press_time["turn_l"] = now
                actions_taken.append("STEER LEFT (DNp20-L)")
        elif turn_diff < -self.cfg.turn_threshold: # Turn Right (DNp20-Right)
            if (now - self.last_press_time["turn_r"]) > self.cfg.cooldown_duration:
                self.driver.release_key(DIK_A)
                self.driver.press_key(DIK_D)
                self.driver.mouse_move_relative(dx=10, dy=0)
                self.last_press_time["turn_r"] = now
                actions_taken.append("STEER RIGHT (DNp20-R)")
        else:
            if (now - self.last_press_time["turn_l"]) > self.cfg.press_duration:
                self.driver.release_key(DIK_A)
            if (now - self.last_press_time["turn_r"]) > self.cfg.press_duration:
                self.driver.release_key(DIK_D)

        # Forward Walking (DNpe017)
        if rate_forward > self.cfg.walk_threshold:
            if (now - self.last_press_time["forward"]) > self.cfg.cooldown_duration:
                self.driver.press_key(DIK_W)
                self.last_press_time["forward"] = now
                actions_taken.append("WALK FORWARD (DNpe017)")
        else:
            if (now - self.last_press_time["forward"]) > self.cfg.press_duration:
                self.driver.release_key(DIK_W)

        # Attack / Primary Fire (DNpe017 Attack Target Lock)
        if rate_attack > self.cfg.attack_threshold:
            if (now - self.last_press_time["attack"]) > (self.cfg.cooldown_duration * 1.5):
                self.driver.mouse_click(duration=self.cfg.press_duration)
                self.last_press_time["attack"] = now
                self.is_firing = True
                actions_taken.append("FIRE! [LEFT CLICK]")
            else:
                self.is_firing = False
        else:
            self.is_firing = False

        if not actions_taken:
            self.active_action_label = "IDLE / OBSERVING"
        else:
            self.active_action_label = " + ".join(actions_taken)

        return {
            "turn_diff": turn_diff,
            "rate_forward": rate_forward,
            "rate_attack": rate_attack,
            "rate_dnp20_l": rate_dnp20_l,
            "rate_dnp20_r": rate_dnp20_r,
            "action": self.active_action_label,
            "is_firing": self.is_firing
        }

    def cleanup(self):
        """Releases held keys on shutdown."""
        self.driver.release_all()
