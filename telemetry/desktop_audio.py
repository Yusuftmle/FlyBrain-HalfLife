"""
desktop_audio.py - Zero-Latency Local PC Audio Engine
Plays real-time neural spikes, dopamine rewards, weapon fires, and panic alerts directly through PC speakers/headphones.
Zero web/browser dependency - 100% native desktop sound via pygame.mixer.
"""
import time
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class DesktopAudioPlayer:
    """
    Plays live electrophysiology audio through the user's headphones/speakers.
    - Geiger-style action potential clicks as the fly's connectome fires.
    - Pleasant chime when dopamine is high / items collected.
    - Warning buzz when stuck against a wall (dopamine drop / panic).
    - Sharp mechanical click when weapon fires.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PYGAME_AVAILABLE
        self.last_spike_sound_time: float = 0.0
        self.last_da_sound_time: float = 0.0
        self.was_in_panic: bool = False
        
        if not self.enabled:
            return
            
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._init_sounds()
            print("[DesktopAudio] 🔊 Yerel PC Ses Sistemi Aktif (Kulaklık/Hoparlör).")
        except Exception as e:
            print(f"[DesktopAudio] Ses başlatma uyarısı: {e}")
            self.enabled = False

    def _init_sounds(self):
        sr = 44100
        
        # 1. Nöron Spike Çıtırtısı (~12ms Geiger tıklaması)
        t_spike = np.linspace(0, 0.012, int(sr * 0.012), endpoint=False)
        pop = (np.sin(2 * np.pi * 1050 * t_spike) * np.exp(-t_spike * 500) * 11000).astype(np.int16)
        stereo_spike = np.column_stack([pop, pop])
        self.snd_spike = pygame.sndarray.make_sound(stereo_spike)
        self.snd_spike.set_volume(0.12)

        # 2. Dopamin Ödül Çanı (~90ms harmonik çift ton)
        t_da = np.linspace(0, 0.09, int(sr * 0.09), endpoint=False)
        chime = (np.sin(2 * np.pi * 587 * t_da) * 0.5 + np.sin(2 * np.pi * 880 * t_da) * 0.5) * np.exp(-t_da * 22) * 14000
        stereo_chime = np.column_stack([chime.astype(np.int16), chime.astype(np.int16)])
        self.snd_reward = pygame.sndarray.make_sound(stereo_chime)
        self.snd_reward.set_volume(0.22)

        # 3. Panik / Duvara Toslama Uyarısı (~120ms tok bas vızıltı)
        t_panic = np.linspace(0, 0.12, int(sr * 0.12), endpoint=False)
        buzz = (np.sin(2 * np.pi * 115 * t_panic) * 0.7 + np.sin(2 * np.pi * 230 * t_panic) * 0.3) * 15000
        stereo_buzz = np.column_stack([buzz.astype(np.int16), buzz.astype(np.int16)])
        self.snd_panic = pygame.sndarray.make_sound(stereo_buzz)
        self.snd_panic.set_volume(0.25)

        # 4. Silah Ateşi Kliği
        t_fire = np.linspace(0, 0.025, int(sr * 0.025), endpoint=False)
        fire = (np.sin(2 * np.pi * 320 * t_fire) * np.exp(-t_fire * 250) * 16000).astype(np.int16)
        stereo_fire = np.column_stack([fire, fire])
        self.snd_fire = pygame.sndarray.make_sound(stereo_fire)
        self.snd_fire.set_volume(0.24)

    def play_frame(self, spikes_count: int, dopamine: float, is_firing: bool, is_panic: bool, is_damage: bool):
        """Called every simulation step to emit instantaneous audio to PC speakers."""
        if not self.enabled:
            return
        now = time.perf_counter()
        
        # Silah ateşi
        if is_firing:
            self.snd_fire.play()
            
        # Duvara toslama / panik / hasar
        if (is_panic or is_damage) and not self.was_in_panic:
            self.snd_panic.play()
        self.was_in_panic = is_panic or is_damage
        
        # Dopamin artışı ödülü
        if dopamine > 0.25 and (now - self.last_da_sound_time > 0.45):
            self.snd_reward.play()
            self.last_da_sound_time = now

        # Nöron aksiyon potansiyeli çatırtısı (aşırı yüklemeyi önlemek için 25ms aralıkla)
        if spikes_count > 6 and (now - self.last_spike_sound_time > 0.028):
            self.snd_spike.play()
            self.last_spike_sound_time = now
