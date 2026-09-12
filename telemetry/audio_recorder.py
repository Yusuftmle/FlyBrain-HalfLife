"""
audio_recorder.py - Biophysical Neural Audio Sonification & Muxing Engine
Synthesizes sample-accurate 44.1kHz electrophysiology audio (spikes, dopamine, actions, reflexes)
and muxes it into MP4 recordings via ffmpeg.
"""
import os
import wave
import time
import subprocess
import numpy as np
from typing import Dict, Any, Optional


class NeuralAudioRecorder:
    """
    Real-Time Electrophysiology Audio Synthesizer and Video-Audio Muxer.
    Captures live action potential crackles, dopamine mood harmonics, weapon clicks,
    and panic reflex alarms, then integrates them into high-fidelity MP4 video recordings.
    """
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate: int = sample_rate
        self.is_recording: bool = False
        self.audio_chunks = []
        self.current_wav_path: str = ""
        self.final_mp4_path: str = ""
        self.raw_video_path: str = ""
        
        # Audio synthesis continuous phase
        self.phase_da: float = 0.0

    def start(self, raw_video_path: str, final_mp4_path: str):
        """Initializes audio recording synchronized with video recording."""
        self.raw_video_path = raw_video_path
        self.final_mp4_path = final_mp4_path
        base, _ = os.path.splitext(raw_video_path)
        self.current_wav_path = f"{base}_audio.wav"
        self.audio_chunks = []
        self.is_recording = True
        self.phase_da = 0.0

    def record_frame(self, dt: float, spikes: Optional[np.ndarray], motor_info: Dict[str, Any], rl_info: Dict[str, Any]):
        """Synthesizes PCM audio samples for the duration of the current video frame."""
        if not self.is_recording:
            return

        # Frame duration clamp between 5ms and 100ms
        dt_clamped = min(0.1, max(0.005, dt))
        n_samples = max(64, int(self.sample_rate * dt_clamped))
        t = np.linspace(0, n_samples / self.sample_rate, n_samples, endpoint=False)
        samples = np.zeros(n_samples, dtype=np.float32)

        # 1. Real Biological Spike Crackle (Action Potential clicks from connectome)
        num_spikes = int(np.sum(spikes)) if spikes is not None else 0
        if num_spikes > 0:
            spike_density = min(1.0, num_spikes / 250.0)
            noise = np.random.uniform(-0.16, 0.16, n_samples) * spike_density
            pop = np.diff(noise, prepend=0) * 1.6
            samples += pop

        # 2. Dopamine Valence Tone Modulation
        da = float(rl_info.get("dopamine_level", 0.0))
        if da > 0.12:
            # Harmonic high-tone chime on reward / progress (C5 523Hz ~ G5 784Hz)
            freq = 523.25 + min(260.0, da * 60.0)
            vol = min(0.07, da * 0.035)
            wave_da = np.sin(2 * np.pi * freq * t + self.phase_da) * vol
            self.phase_da = (self.phase_da + 2 * np.pi * freq * (n_samples / self.sample_rate)) % (2 * np.pi)
            samples += wave_da
        elif da < -0.3:
            # Low-pitch warning drone / brown buzz on wall-stuck / panic
            freq = 115.0
            vol = min(0.10, abs(da) * 0.04)
            wave_buzz = (np.sin(2 * np.pi * freq * t) + 0.45 * np.sin(4 * np.pi * freq * t)) * vol
            samples += wave_buzz

        # 3. Weapon Discharge (DNpe017 Attack trigger)
        if motor_info.get("is_firing", False):
            click_len = min(n_samples, int(self.sample_rate * 0.025))
            click_t = t[:click_len]
            samples[:click_len] += np.sin(2 * np.pi * 340 * click_t) * 0.26 * np.exp(-np.linspace(0, 6, click_len))

        # 4. Giant Fiber Escape / Damage Alarm
        if motor_info.get("is_damage", False) or motor_info.get("is_escaping", False):
            samples += np.sin(2 * np.pi * 170 * t) * 0.22

        # 5. Moonwalker Backward Locomotion Acoustic Indicator (MDN Reverse Gear)
        if "MDN" in motor_info.get("action", "") or "BACKWARD" in motor_info.get("action", ""):
            samples += np.sin(2 * np.pi * 440 * t) * 0.05 * (np.sin(2 * np.pi * 12 * t) > 0)

        # Normalize and convert to 16-bit PCM
        samples = np.clip(samples, -0.95, 0.95)
        pcm = (samples * 32767).astype(np.int16)
        self.audio_chunks.append(pcm.tobytes())

    def stop_and_mux(self) -> Optional[str]:
        """Finalizes audio file and muxes video+audio into final MP4."""
        if not self.is_recording:
            return None
        self.is_recording = False

        if not self.audio_chunks:
            return None

        # 1. Write WAV file
        try:
            with wave.open(self.current_wav_path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(self.audio_chunks))
        except Exception as e:
            print(f"[AudioRecorder] Error writing WAV: {e}")
            return None

        # 2. Use ffmpeg to mux video and audio
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", self.raw_video_path,
                "-i", self.current_wav_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                self.final_mp4_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            if res.returncode == 0 and os.path.exists(self.final_mp4_path):
                # Clean up temporary raw video and audio files
                try:
                    if os.path.abspath(self.raw_video_path) != os.path.abspath(self.final_mp4_path):
                        os.remove(self.raw_video_path)
                    if os.path.exists(self.current_wav_path):
                        os.remove(self.current_wav_path)
                except Exception:
                    pass
                print(f"[AudioRecorder] 🎬 Audio-video mux successful: {self.final_mp4_path}")
                return self.final_mp4_path
        except Exception as e:
            print(f"[AudioRecorder] Muxing error: {e}")
        return None
