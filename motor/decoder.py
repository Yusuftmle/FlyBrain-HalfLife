"""
decoder.py - DOOMFLY-Inspired Neural Decoder
Converts raw descending neuron firing rates into smooth analog locomotion controls
using Exponential Moving Average (EMA) low-pass filtering.
"""
import math
from typing import Dict, Any, Optional
import numpy as np


class NeuralDecoder:
    """
    Bio-inspired neural decoder modeled after DOOMFLY's NeuralControls.
    Applies an exponential moving average (tau = 100ms) to filter high-frequency
    synaptic noise and spikes, producing smooth, continuous control signals:
      - turn: [-1.0, 1.0] (Differential DNp20 Steering)
      - forward: [-1.0, 1.0] (Net DNpe017 Forward vs MDN Backward)
      - pitch: [-1.0, 1.0] (Vertical View Angle / Gaze Control)
      - attack: bool (Primary Weapon Trigger)
    """
    def __init__(
        self,
        tau_ms: float = 100.0,
        turn_gain: float = 5.5,
        forward_gain: float = 3.5,
        turn_deadzone: float = 0.010,
        walk_threshold: float = 0.02,
        attack_threshold: float = 0.35,
        decision_interval: float = 0.08
    ):
        self.tau_sec = max(0.01, tau_ms / 1000.0)
        self.turn_gain = turn_gain
        self.forward_gain = forward_gain
        self.turn_deadzone = turn_deadzone
        self.walk_threshold = walk_threshold
        self.attack_threshold = attack_threshold
        self.decision_interval = decision_interval

        # Smoothed rate states (EMA)
        self.smooth_dnp20_l: float = 0.0
        self.smooth_dnp20_r: float = 0.0
        self.smooth_forward: float = 0.0
        self.smooth_backward: float = 0.0
        self.smooth_pitch: float = 0.0
        self.smooth_attack: float = 0.0

        # MochisRice 100ms Decision Sampling & Directional Hysteresis State
        self.last_decision_time: float = 0.0
        self.sampled_turn: float = 0.0
        self.current_turn: float = 0.0
        self.active_turn_dir: int = 0  # +1 right, -1 left, 0 neutral
        self.elapsed_time: float = 0.0

    def reset(self):
        """Resets smoothed rate history and decision sampler state."""
        self.smooth_dnp20_l = 0.0
        self.smooth_dnp20_r = 0.0
        self.smooth_forward = 0.0
        self.smooth_backward = 0.0
        self.smooth_pitch = 0.0
        self.smooth_attack = 0.0
        self.last_decision_time = 0.0
        self.sampled_turn = 0.0
        self.current_turn = 0.0
        self.active_turn_dir = 0
        self.elapsed_time = 0.0

    def decode(
        self,
        dnp20_left_rate: float,
        dnp20_right_rate: float,
        dnpe017_forward_rate: float,
        dnpe017_attack_rate: float,
        mdn_backward_rate: float = 0.0,
        vertical_pitch_rate: float = 0.0,
        dt: float = 0.02,
        now: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Decodes descending neuron rates with low-pass EMA filtering and 100ms MochisRice decision sampling.
        """
        dt = max(0.01, min(0.2, dt))
        if now is None:
            self.elapsed_time += dt
            t_eval = self.elapsed_time
        else:
            t_eval = now

        # Exponential smoothing factor: alpha = 1 - exp(-dt / tau)
        alpha = 1.0 - math.exp(-dt / self.tau_sec)

        # Update EMA low-pass filtered rates
        self.smooth_dnp20_l += alpha * (float(dnp20_left_rate) - self.smooth_dnp20_l)
        self.smooth_dnp20_r += alpha * (float(dnp20_right_rate) - self.smooth_dnp20_r)
        self.smooth_forward += alpha * (float(dnpe017_forward_rate) - self.smooth_forward)
        self.smooth_backward += alpha * (float(mdn_backward_rate) - self.smooth_backward)
        self.smooth_attack += alpha * (float(dnpe017_attack_rate) - self.smooth_attack)
        self.smooth_pitch += alpha * (float(vertical_pitch_rate) - self.smooth_pitch)

        # Biological Reciprocal Inhibition: MDN overrides forward drive
        if mdn_backward_rate > self.walk_threshold and mdn_backward_rate > dnpe017_forward_rate:
            self.smooth_forward = min(self.smooth_forward, float(dnpe017_forward_rate))
            self.smooth_backward = max(self.smooth_backward, float(mdn_backward_rate))

        # 1. Differential Steering (Right - Left) with MochisRice 100ms Decision Sampling & Hysteresis
        raw_turn_diff = self.smooth_dnp20_r - self.smooth_dnp20_l

        # Zero-latency initiation from neutral; 100ms (10 Hz) hold window once actively turning
        is_interval_ready = (t_eval - self.last_decision_time) >= self.decision_interval or self.last_decision_time == 0.0
        can_decide = is_interval_ready or (self.active_turn_dir == 0 and abs(raw_turn_diff) > self.turn_deadzone)

        if can_decide:
            self.last_decision_time = t_eval

            # Directional Hysteresis:
            # Prevents micro-oscillations and high-frequency sign flipping
            if self.active_turn_dir > 0:
                # Currently steering RIGHT: require significant opposite drive to flip LEFT
                if raw_turn_diff < -self.turn_deadzone * 1.4:
                    self.active_turn_dir = -1
                    self.sampled_turn = float(np.clip(raw_turn_diff * self.turn_gain, -1.0, 1.0))
                elif raw_turn_diff > self.turn_deadzone:
                    self.sampled_turn = float(np.clip(raw_turn_diff * self.turn_gain, -1.0, 1.0))
                else:
                    self.active_turn_dir = 0
                    self.sampled_turn = 0.0
            elif self.active_turn_dir < 0:
                # Currently steering LEFT: require significant opposite drive to flip RIGHT
                if raw_turn_diff > self.turn_deadzone * 1.4:
                    self.active_turn_dir = 1
                    self.sampled_turn = float(np.clip(raw_turn_diff * self.turn_gain, -1.0, 1.0))
                elif raw_turn_diff < -self.turn_deadzone:
                    self.sampled_turn = float(np.clip(raw_turn_diff * self.turn_gain, -1.0, 1.0))
                else:
                    self.active_turn_dir = 0
                    self.sampled_turn = 0.0
            else:
                # Starting from Neutral: initiate immediately with zero latency
                if raw_turn_diff > self.turn_deadzone:
                    self.active_turn_dir = 1
                    self.sampled_turn = float(np.clip(raw_turn_diff * self.turn_gain, -1.0, 1.0))
                elif raw_turn_diff < -self.turn_deadzone:
                    self.active_turn_dir = -1
                    self.sampled_turn = float(np.clip(raw_turn_diff * self.turn_gain, -1.0, 1.0))
                else:
                    self.sampled_turn = 0.0

        # When starting from neutral, adopt turn immediately; smooth during active transitions
        if self.active_turn_dir != 0 and abs(self.current_turn) < self.turn_deadzone:
            self.current_turn = self.sampled_turn
        else:
            self.current_turn += 0.40 * (self.sampled_turn - self.current_turn)

        turn_action = float(np.clip(self.current_turn, -1.0, 1.0))

        # 2. Net Forward Locomotion (Forward - Backward)
        net_fwd = self.smooth_forward - self.smooth_backward
        forward_action = float(np.clip(net_fwd * self.forward_gain, -1.0, 1.0))

        # 3. Vertical Pitch (Up / Down Gaze)
        pitch_action = float(np.clip(self.smooth_pitch, -1.0, 1.0))

        # 4. Primary Attack Trigger (Instant response on spike burst or sustained high rate)
        is_firing = bool(dnpe017_attack_rate > self.attack_threshold or self.smooth_attack > self.attack_threshold)

        return {
            "turn": turn_action,             # [-1.0, 1.0] (Negative=Left, Positive=Right)
            "forward": forward_action,       # [-1.0, 1.0] (Positive=Forward, Negative=Backward)
            "pitch": pitch_action,           # [-1.0, 1.0] (Positive=Down, Negative=Up)
            "is_firing": is_firing,
            "raw_turn_diff": raw_turn_diff,
            "net_forward": net_fwd,
            "rates": {
                "dnp20_l": self.smooth_dnp20_l,
                "dnp20_r": self.smooth_dnp20_r,
                "forward": self.smooth_forward,
                "backward": self.smooth_backward,
                "attack": self.smooth_attack,
                "pitch": self.smooth_pitch,
            }
        }
