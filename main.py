"""
main.py - FlyBrain-HalfLife Master Pipeline
Integrates the 4 Mandatory Biological Layers:
  1. data_loader.py   (FlyWire Connectome & Neurotransmitter Signs)
  2. lif_engine.py    (PyTorch GPU Sparse LIF Differential Equation Solver)
  3. vision_bridge.py (Sub-2ms MSS Capture & Naka-Rushton Photoreceptor Transduction)
  4. input_bridge.py  (DirectX Hardware DirectInput & Refractory Cooldown)
"""
import sys
import os
import time
import argparse
import logging
import numpy as np
import cv2
import ctypes

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data_loader import ConnectomeDataLoader
from lif_engine import PyTorchLIFEngine
from vision_bridge import VisionBridge
from input_bridge import InputBridge
from env.arena import HalfLifeArena
from env.doom_arena import DoomArena, VIZDOOM_AVAILABLE
from dopamine import DopamineController
from telemetry.visualizer import FlyBrainVisualizer
from telemetry.desktop_audio import DesktopAudioPlayer
from server.broadcast_server import FlyBrainWebBroadcaster
from core.connectome import FlyConnectome
from config import DEFAULT_CONFIG

logger = logging.getLogger("FlyBrain.Main")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def run_simulation(args):
    print("=" * 74)
    print("   FLYBRAIN-HALFLIFE: MaleCNS v1.0 & FlyWire GPU Autonomous Agent")
    print("=" * 74)
    print(f"[*] Execution Mode : {args.mode.upper()}")
    print(f"[*] Visualizer     : {'ENABLED' if not args.no_vis else 'DISABLED (Headless)'}")
    print(f"[*] Web Broadcaster: {'ENABLED (http://localhost:' + str(args.stream_port) + ')' if getattr(args, 'web_stream', True) else 'DISABLED'}")
    print(f"[*] Hardware Input : {'DRY-RUN (Simulated - Physical Keys Disabled)' if (args.dry_run or args.mode in ['arena', 'doom']) else 'LIVE DIRECTINPUT (PHYSICAL KEYS ACTIVE: W, A, S, D, Mouse)'}")
    if args.record:
        print(f"[*] Video Output   : {args.record}")
    print("-" * 74)

    # LAYER 1: Load Biological Connectome and Neurotransmitter Signs
    loader = ConnectomeDataLoader()
    sparse_weights, meta = loader.build_canonical_flywire_connectome()
    num_neurons = meta["num_neurons"]
    region_offsets = meta["region_offsets"]
    region_counts = meta["region_counts"]

    # LAYER 2: PyTorch GPU Vectorized LIF Engine
    device_req = "cuda" if not args.cpu else "cpu"
    lif_engine = PyTorchLIFEngine(
        sparse_weights=sparse_weights,
        num_neurons=num_neurons,
        device=device_req
    )

    # Initialize Kenyon Cell -> MBON STDP plastic synapses (DOOMFLY Associative Learning)
    if "mushroom_body_kc" in region_offsets and "mbon" in region_offsets:
        kc_start = region_offsets["mushroom_body_kc"]
        kc_end = kc_start + region_counts["mushroom_body_kc"]
        mbon_start = region_offsets["mbon"]
        mbon_end = mbon_start + region_counts["mbon"]
        kc_indices = np.arange(kc_start, kc_end, dtype=np.int32)
        mbon_indices = np.arange(mbon_start, mbon_end, dtype=np.int32)
        lif_engine.init_plastic_synapses(kc_indices, mbon_indices)

    # Benchmark Mode
    if args.mode == "benchmark":
        run_benchmark(lif_engine, num_neurons, sparse_weights.nnz, steps=args.steps or 1000)
        return

    # LAYER 3: Vision Bridge (MSS Sub-2ms Capture & Naka-Rushton Photoreceptors)
    vision_bridge = VisionBridge(
        grid_width=60, 
        grid_height=60, 
        window_title=DEFAULT_CONFIG.vision.capture_window_title
    )

    # LAYER 4: Input Bridge (DirectInput Hardware Scancodes, Window Lock, & Safety Guard)
    input_bridge = InputBridge(
        turn_threshold=0.03,
        walk_threshold=0.05,
        attack_threshold=0.35,
        dry_run=args.dry_run or (args.mode in ["arena", "doom"]), 
        cooldown_duration=0.08,
        target_hwnd=vision_bridge.hwnd if args.mode == "live" else None
    )

    # Reinforcement Dopamine Circuit & Anti-Stuck State
    core_connectome = FlyConnectome(DEFAULT_CONFIG.connectome)
    dopamine = DopamineController(core_connectome, lif_engine, DEFAULT_CONFIG.rl)

    # Live HTTP / MJPEG Web Broadcaster (Zero-dependency embedded server with Stop/Resume control)
    broadcaster = None
    if getattr(args, "web_stream", True):
        broadcaster = FlyBrainWebBroadcaster(
            host="0.0.0.0", 
            port=args.stream_port, 
            input_bridge=input_bridge,
            connectome=core_connectome
        )
        broadcaster.start()

    # Environment Selection
    if args.mode == "doom":
        if not VIZDOOM_AVAILABLE:
            print("[!] ViZDoom not installed. Falling back to retro raycaster arena.")
            arena = HalfLifeArena(width=640, height=480)
        else:
            scenario_name = getattr(args, "doom_scenario", "deadly_corridor")
            arena = DoomArena(scenario=scenario_name, width=640, height=480)
            print(f"[*] Genuine 3D ViZDoom Arena Active: scenario='{scenario_name}'.")
    elif args.mode == "arena":
        arena = HalfLifeArena(width=640, height=480)
        print("[*] Autonomous 3D Half-Life Raycasting Arena Active.")
    else:
        arena = None
        print("[*] Live Game Mode: Listening to Half-Life window...")
        vision_bridge.focus_game_window()
        input_bridge.target_hwnd = vision_bridge.hwnd
        print("[*] 🔒 SAFETY GUARD ACTIVE:")
        print("    - Sinek sadece aktif odak Half-Life'tayken tuşlara basar (Masaüstü/VS Code güvende).")
        print("    - F10: Sinek Girişlerini DURDUR / DEVAM ET (Toggle Pause/Resume).")
        print("    - Half-Life'ta ESC (menü) veya ~ (konsol) açıkken sinek otomatik duraklar.")

    # Telemetry Visualizer
    visualizer = None
    if not args.no_vis:
        res_map = {
            "900p": (1600, 900),
            "1080p": (1920, 1080),
            "1440p": (2560, 1440),
            "2k": (2560, 1440),
        }
        res_key = getattr(args, "resolution", "1080p").lower()
        vis_w, vis_h = res_map.get(res_key, (1920, 1080))
        visualizer = FlyBrainVisualizer(
            core_connectome, 
            DEFAULT_CONFIG.telemetry, 
            record_path=args.record,
            width=vis_w,
            height=vis_h
        )
        win_name = "FlyBrain - MaleCNS Connectome Telemetry"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, vis_w, vis_h)
        try:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        if getattr(args, "auto_record", False) and not visualizer.is_recording:
            visualizer.toggle_recording()

    # Pre-calculated Motor Neuron Indices
    dn_offset = region_offsets["descending"]
    idx_dnp20_l = dn_offset + 0
    idx_dnp20_r = dn_offset + 1
    idx_dnpe017_fwd = dn_offset + 2
    idx_dnpe017_atk = dn_offset + 3
    idx_mdn_l = dn_offset + 4
    idx_mdn_r = dn_offset + 5
    idx_dnp09_l = dn_offset + 6
    idx_dnp09_r = dn_offset + 7

    # Pre-sampled 128 neurons across circuits for real-time Spike Raster / EEG
    pr_s, pr_l = region_offsets["photoreceptors"], region_counts["photoreceptors"]
    ol_s, ol_l = region_offsets["optic_lobe"], region_counts["optic_lobe"]
    cx_s, cx_l = region_offsets["central_complex"], region_counts["central_complex"]
    mb_s, mb_l = region_offsets["mushroom_body_kc"], region_counts["mushroom_body_kc"]
    dn_s, dn_l = region_offsets["descending"], region_counts["descending"]
    raster_indices = np.concatenate([
        np.linspace(pr_s, pr_s + pr_l - 1, 16, dtype=int),
        np.linspace(ol_s, ol_s + ol_l - 1, 32, dtype=int),
        np.linspace(cx_s, cx_s + cx_l - 1, 32, dtype=int),
        np.linspace(mb_s, mb_s + mb_l - 1, 32, dtype=int),
        np.linspace(dn_s, dn_s + dn_l - 1, 16, dtype=int),
    ])

    # Local PC Speaker / Headphone Audio Synthesizer (Zero Latency)
    desktop_audio = DesktopAudioPlayer(enabled=True)

    print("[*] Simulation loop engaged. Press 'Q' or 'ESC' in visualizer to exit.")
    
    step = 0
    start_time = time.time()
    turn_val = 0.0
    fwd_val = 0.5
    is_attack = False
    was_in_panic = False
    last_frame_time = time.perf_counter()
    last_global_hotkey_time = 0.0
    win_styles_applied = False

    try:
        while True:
            t_now = time.perf_counter()
            dt_frame_ms = (t_now - last_frame_time) * 1000.0
            last_frame_time = t_now

            # 1. Low-Latency Frame Capture (<2ms memory grab)
            if args.mode in ["arena", "doom"]:
                game_frame, env_events = arena.step(turn_val, fwd_val, is_attack)
                for ev in env_events:
                    dopamine.register_event(ev["type"], ev["mag"])
            else:
                game_frame = vision_bridge.capture_frame()
                if not input_bridge.target_hwnd and vision_bridge.hwnd:
                    input_bridge.target_hwnd = vision_bridge.hwnd

            # 1.5. GoldSrc Red Screen Flash & Combat Retaliation Circuit
            # Supply last_shot_time so the fly's own weapon muzzle flash is never mistaken for incoming damage!
            last_shot = getattr(input_bridge, "last_fire_time", 0.0)
            is_damage, red_intensity = vision_bridge.detect_damage_flash(game_frame, last_shot_time=last_shot)

            # 2. Retinal Transduction (HUD-free Central FOV & Naka-Rushton Kinetics)
            photocurrents, motion_metrics = vision_bridge.process_frame(game_frame)

            # 3. Dopamine Circuit & Combat Retaliation / Obstacle Deadlock Circuit
            if is_damage:
                dopamine.register_event("damage", magnitude=max(1.5, red_intensity * 4.0))
                # Combat Retaliation: Immediate return fire and evasive combat move!
                input_bridge.trigger_combat_retaliation()
            elif getattr(lif_engine, "is_flatlined", False):
                # Respawn Resuscitation
                lif_engine.trigger_defibrillation()
                input_bridge.handle_respawn()

            da_current, rl_info = dopamine.update(motion_metrics["flow"])
            if is_damage:
                rl_info["giant_fiber_active"] = True

            # Unstuck Maneuver: Trigger once upon obstacle deadlock detection
            is_panic_now = bool(rl_info.get("is_in_panic", False))
            if is_panic_now and not was_in_panic:
                input_bridge.trigger_obstacle_turn(direction=rl_info.get("escape_direction", 1))
            was_in_panic = is_panic_now

            # 4. Inject Photocurrents, Lamina Tonic Drive, and Neuromodulation
            total_ext_current = np.copy(da_current)
            pr_start = region_offsets["photoreceptors"]
            pr_len = region_counts["photoreceptors"]
            total_ext_current[pr_start : pr_start + pr_len] += photocurrents

            # Inject DOOMFLY biological tonic drive into Lamina monopolar units (L1-L5)
            if "lamina_indices" in meta and len(meta["lamina_indices"]) > 0:
                total_ext_current[meta["lamina_indices"]] += 12.0
            elif "optic_lobe" in region_offsets:
                ol_start = region_offsets["optic_lobe"]
                ol_count = min(1200, region_counts.get("optic_lobe", 1200))
                total_ext_current[ol_start : ol_start + ol_count] += 12.0

            # Drosophila Optomotor Corridor Centering & Visual Premotor Drive (DNp20 & DNpe017)
            # Compares Left vs Right retinal hemifields + visual depth balance
            pc_grid = photocurrents.reshape(60, 60)
            left_eye_drive = float(np.mean(pc_grid[:, :30]))
            right_eye_drive = float(np.mean(pc_grid[:, 30:]))
            depth_bal = float(motion_metrics.get("depth_balance", 0.0))
            is_obs = bool(motion_metrics.get("is_obstacle_close", False))
            
            # Positive turn_diff (> 0.03) triggers STEER LEFT (A + Left Arrow / Mouse dx < 0)
            # Negative turn_diff (< -0.03) triggers STEER RIGHT (D + Right Arrow / Mouse dx > 0)
            steer_bias = (left_eye_drive - right_eye_drive) * 1.2 - (depth_bal * 26.0)
            if is_obs:
                obs_sign = 1.0 if depth_bal <= 0 else -1.0
                steer_bias += obs_sign * 22.0

            dnp20_l_stim = max(0.0, steer_bias) * 1.4 + 10.0
            dnp20_r_stim = max(0.0, -steer_bias) * 1.4 + 10.0
            
            # Forward drive: ease off during sharp steering or obstacle avoidance so rotation takes effect
            if is_obs:
                fwd_stim = 4.0   # Minimal forward crawl during obstacle avoidance
                # MDN Moonwalker Descending Neuron activation: depolarize bilateral MDN to trigger backward stepping
                mdn_stim = 36.0
            elif abs(steer_bias) > 12.0:
                fwd_stim = 10.0  # Slow down during sharp corridor turns to allow clean rotation
                mdn_stim = 0.0
            else:
                fwd_stim = 22.0  # Smooth forward walking in open corridors
                mdn_stim = 0.0

            total_ext_current[idx_dnp20_l] += dnp20_l_stim
            total_ext_current[idx_dnp20_r] += dnp20_r_stim
            total_ext_current[idx_dnpe017_fwd] += fwd_stim
            total_ext_current[idx_mdn_l] += mdn_stim
            total_ext_current[idx_mdn_r] += mdn_stim

            # R8 Photoreceptor Chromatic Pathways (Xiao et al., Nature 2023)
            # R8y (Green) -> aMe12 -> KCγ-d (health pack / ally detection)
            # R8p (Blue) -> aMe12 -> KCγ-d (armor / HEV battery detection)
            r8y_cur = float(motion_metrics.get("r8y_current", 0.0))
            r8p_cur = float(motion_metrics.get("r8p_current", 0.0))
            kc_indices = meta.get("kc_indices", [])
            if len(kc_indices) >= 100:
                if r8y_cur > 3.5:
                    total_ext_current[kc_indices[:50]] += r8y_cur * 2.5
                    dopamine.register_event("health_pack", magnitude=0.4)
                if r8p_cur > 3.5:
                    total_ext_current[kc_indices[50:100]] += r8p_cur * 2.5

            # Gustatory Sugar Reward Circuit (LB3c sweet taste sensation)
            sugar_active = False
            if "sugar_indices" in meta and len(meta["sugar_indices"]) > 0:
                if dopamine.dopamine_level > 0.2:
                    total_ext_current[meta["sugar_indices"]] += 18.0
                    sugar_active = True

            if is_damage:
                # Combat Retaliation: Direct emergency burst depolarization into DNpe017 Attack neuron!
                total_ext_current[idx_dnpe017_atk] += 65.0
                dn_start = region_offsets["descending"]
                dn_len = min(10, region_counts["descending"])
                total_ext_current[dn_start : dn_start + dn_len] += 30.0

            # 5. Low-Latency Vectorized LIF Forward Pass
            # 4 substeps (4ms on CUDA) guarantees snappy ~50ms biological reflex propagation without FPS loss
            if getattr(args, "sync_dt", False):
                num_substeps = max(2, min(6, int(round(dt_frame_ms / max(0.1, lif_engine.dt * 10.0)))))
            else:
                num_substeps = 4 if not args.cpu else 2

            for _ in range(num_substeps):
                spikes = lif_engine.forward_step(total_ext_current)

            # 5.5. DOOMFLY 3-Factor Hebbian STDP Learning (KC -> MBON, modulated by PPL1/PAM dopamine)
            lif_engine.apply_stdp_update(dopamine=dopamine.dopamine_level, eta=0.002)

            # 6. Read Motor Firing Rates (DNp20, DNpe017, MDN)
            rates = lif_engine.get_firing_rates()
            rate_dnp20_l = float(rates[idx_dnp20_l])
            rate_dnp20_r = float(rates[idx_dnp20_r])
            rate_forward = float(rates[idx_dnpe017_fwd])
            rate_attack = float(rates[idx_dnpe017_atk])
            rate_backward = float(rates[idx_mdn_l] + rates[idx_mdn_r]) / 2.0

            # 7. Hardware Debounced Input Dispatch
            motor_info = input_bridge.decode_and_dispatch(
                dnp20_left_rate=rate_dnp20_l,
                dnp20_right_rate=rate_dnp20_r,
                dnpe017_forward_rate=rate_forward,
                dnpe017_attack_rate=rate_attack,
                mdn_backward_rate=rate_backward,
                is_obstacle_close=is_obs
            )

            # Pass controls to next step
            turn_val = -float(motor_info["turn_diff"]) * 2.0
            fwd_val = float(motor_info.get("net_forward", rate_forward - rate_backward)) * 2.5
            is_attack = bool(motor_info["is_firing"])

            # 7.5. Live Desktop Audio Playback (PC Speakers / Headphones)
            desktop_audio.play_frame(
                spikes_count=int(np.sum(spikes)),
                dopamine=float(dopamine.dopamine_level),
                is_firing=bool(motor_info["is_firing"]),
                is_panic=bool(is_panic_now),
                is_damage=bool(is_damage)
            )

            # 8. Render Live Telemetry
            if visualizer is not None:
                # 60x60 retina visualization
                retina_vis = cv2.resize(
                    cv2.cvtColor((photocurrents.reshape(60, 60) * 8.0).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                    (200, 200),
                    interpolation=cv2.INTER_NEAREST
                )
                motor_vis_data = {
                    "turn_diff": motor_info["turn_diff"],
                    "rate_forward": rate_forward,
                    "rate_backward": rate_backward,
                    "net_forward": motor_info.get("net_forward", rate_forward - rate_backward),
                    "action": motor_info["action"],
                    "is_firing": motor_info["is_firing"],
                    "is_damage": is_damage,
                    "rate_attack": rate_attack,
                    "r8y_green": motion_metrics.get("r8y_green", 0.0),
                    "r8p_blue": motion_metrics.get("r8p_blue", 0.0)
                }
                canvas = visualizer.render(
                    game_frame, retina_vis, spikes, motor_vis_data, rl_info,
                    vision_info={"gain": motion_metrics.get("mean_photocurrent", 12.0)}
                )
                cv2.imshow("FlyBrain - MaleCNS Connectome Telemetry", canvas)
                if not win_styles_applied:
                    visualizer.apply_win32_window_styles("FlyBrain - MaleCNS Connectome Telemetry", topmost=True, no_activate=True)
                    win_styles_applied = True

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('v') or key == ord('V'):
                    visualizer.toggle_recording()
                elif key == ord('t') or key == ord('T'):
                    visualizer.toggle_topmost("FlyBrain - MaleCNS Connectome Telemetry")
                elif key == ord('r') or key == ord('R'):
                    dopamine.register_event("kill", 1.5)
                elif key == ord('p') or key == ord('P') or key == 32:
                    input_bridge.toggle_pause()

                # Global Hotkeys (Works even when Half-Life has foreground focus!)
                if sys.platform == "win32":
                    try:
                        user32_hk = ctypes.windll.user32
                        t_now_hk = time.time()
                        if (t_now_hk - last_global_hotkey_time) > 0.35:
                            # F8 (0x77): Toggle Lossless MP4 Recording
                            if user32_hk.GetAsyncKeyState(0x77) & 0x8000:
                                last_global_hotkey_time = t_now_hk
                                visualizer.toggle_recording()
                            # F11 (0x7A): Toggle Window Pinned Topmost
                            elif user32_hk.GetAsyncKeyState(0x7A) & 0x8000:
                                last_global_hotkey_time = t_now_hk
                                visualizer.toggle_topmost("FlyBrain - MaleCNS Connectome Telemetry")
                            # F6 (0x75): Refocus Half-Life Game Window
                            elif user32_hk.GetAsyncKeyState(0x75) & 0x8000:
                                last_global_hotkey_time = t_now_hk
                                vision_bridge.focus_game_window()
                    except Exception:
                        pass

            # 8.5. Live Web Broadcaster Update (MJPEG Stream & Biological Telemetry)
            if broadcaster and broadcaster.is_running:
                fps_instant = 1000.0 / max(1.0, dt_frame_ms)
                plasticity_stats = lif_engine.get_plasticity_telemetry()
                if getattr(lif_engine, "is_flatlined", False):
                    current_status = "FLATLINE (BRAIN DEATH)"
                elif input_bridge.user_paused:
                    current_status = "PAUSED [F10 / STOP BUTONU]"
                elif not input_bridge.is_game_focused():
                    current_status = "PAUSED [ODAK YOK - MASAÜSTÜ]"
                elif input_bridge.is_cursor_visible():
                    current_status = "PAUSED [MENÜ / KONSOL AÇIK]"
                elif getattr(input_bridge, "is_bout_paused", False):
                    current_status = "OBSERVING (PAUSED)"
                else:
                    current_status = "ACTIVE"

                # Calculate Biological Circuit Firing Rates
                pr_off = region_offsets["photoreceptors"]
                pr_cnt = region_counts["photoreceptors"]
                ol_off = region_offsets["optic_lobe"]
                ol_cnt = region_counts["optic_lobe"]
                cx_off = region_offsets["central_complex"]
                cx_cnt = region_counts["central_complex"]
                mb_off = region_offsets["mushroom_body_kc"]
                mb_cnt = region_counts["mushroom_body_kc"]
                mbon_off = region_offsets["mbon"]
                mbon_cnt = region_counts["mbon"]
                da_off = region_offsets["dopaminergic"]
                da_cnt = region_counts["dopaminergic"]
                dn_off = region_offsets["descending"]
                dn_cnt = region_counts["descending"]

                pr_rate = float(np.mean(spikes[pr_off : pr_off + pr_cnt]))
                ol_rate = float(np.mean(spikes[ol_off : ol_off + ol_cnt]))
                cx_rate = float(np.mean(spikes[cx_off : cx_off + cx_cnt]))
                mb_rate = float(np.mean(spikes[mb_off : mb_off + mb_cnt]))
                dn_rate = float(np.mean(spikes[dn_off : dn_off + dn_cnt]))

                # Representative active spikes across all 7 functional circuits (mapped to 3D point indices)
                sampled_active = []
                for s_off, s_cnt in [
                    (pr_off, pr_cnt), (ol_off, ol_cnt), (cx_off, cx_cnt),
                    (mb_off, mb_cnt), (mbon_off, mbon_cnt), (da_off, da_cnt), (dn_off, dn_cnt)
                ]:
                    active_sub = np.where(spikes[s_off : s_off + s_cnt] > 0)[0]
                    if len(active_sub) > 0:
                        chosen = (active_sub[:50] + s_off) // 2
                        sampled_active.extend(chosen.tolist())

                raster_spikes = spikes[raster_indices].astype(int).tolist()

                broadcaster.update(
                    game_frame,
                    {
                        "status": current_status,
                        "action": motor_info["action"],
                        "fps": float(fps_instant),
                        "forward_rate": float(rate_forward),
                        "backward_rate": float(rate_backward),
                        "turn_diff": float(motor_info["turn_diff"]),
                        "dopamine": float(dopamine.dopamine_level),
                        "spikes": int(np.sum(spikes)),
                        "active_spikes": sampled_active,
                        "raster": raster_spikes,
                        "circuits": {
                            "retina": round(pr_rate * 100.0, 1),
                            "optic_lobe": round(ol_rate * 100.0, 1),
                            "central_complex": round(cx_rate * 100.0, 1),
                            "mushroom_body": round(mb_rate * 100.0, 1),
                            "descending": round(dn_rate * 100.0, 1),
                            "dopamine": round(dopamine.dopamine_level * 100.0, 1),
                            "turn_diff": float(motor_info["turn_diff"]),
                            "forward_rate": float(rate_forward),
                            "backward_rate": float(rate_backward),
                            "attack_rate": float(rate_attack),
                            "is_firing": bool(motor_info["is_firing"]),
                            "is_escaping": bool(motor_info.get("is_escaping", False) or rl_info.get("giant_fiber_active", False)),
                            "is_damage": bool(is_damage)
                        },
                        "sugar_active": sugar_active,
                        "is_paused": bool(input_bridge.user_paused or not input_bridge.is_game_focused() or input_bridge.is_cursor_visible()),
                        "is_flatlined": bool(getattr(lif_engine, "is_flatlined", False)),
                        "plasticity": plasticity_stats
                    }
                )

            step += 1
            if args.max_steps and step >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        input_bridge.release_all()
        vision_bridge.close()
        if visualizer:
            visualizer.close()
            cv2.destroyAllWindows()
        if broadcaster:
            broadcaster.stop()
            
        elapsed = time.time() - start_time
        fps_real = step / (elapsed + 1e-4)
        print(f"[*] Simulation ended. Total Steps: {step:,} | Elapsed: {elapsed:.2f}s | Average FPS: {fps_real:.1f}")

def run_benchmark(lif_engine: PyTorchLIFEngine, num_neurons: int, num_synapses: int, steps: int = 1000):
    """Executes high-throughput CUDA LIF benchmark."""
    print(f"\n[BENCHMARK] Executing {steps} steps on {lif_engine.device_str.upper()}...")
    print(f"[BENCHMARK] Neurons : {num_neurons:,} | Synapses : {num_synapses:,}")

    dummy_ext = np.full(num_neurons, 15.0, dtype=np.float32)
    
    total_spikes = 0
    t0 = time.time()
    for _ in range(steps):
        spikes = lif_engine.forward_step(dummy_ext)
        total_spikes += np.sum(spikes)
    t1 = time.time()

    elapsed = t1 - t0
    fps = steps / (elapsed + 1e-6)
    mean_deg = num_synapses / num_neurons
    msyn_s = (total_spikes * mean_deg) / (elapsed + 1e-6) / 1e6

    print("-" * 50)
    print(f"  Device              : {lif_engine.device_str.upper()}")
    print(f"  Elapsed Time        : {elapsed:.3f} s")
    print(f"  Simulation Speed    : {fps:.1f} FPS")
    print(f"  Total Spikes Fired  : {int(total_spikes):,}")
    print(f"  Synaptic Throughput : {msyn_s:.2f} Million Synapses/Sec (MSyn/s)")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyBrain-HalfLife Autonomous Agent")
    parser.add_argument("--mode", choices=["arena", "doom", "live", "benchmark"], default="live", help="Mode: live (real Half-Life game), doom (genuine ViZDoom 3D arena), arena (retro raycaster), benchmark")
    parser.add_argument("--doom-scenario", type=str, default="deadly_corridor", choices=["deadly_corridor", "my_way_home", "defend_the_center", "basic"], help="ViZDoom scenario: deadly_corridor, my_way_home, defend_the_center, basic")
    parser.add_argument("--dry-run", action="store_true", help="Simulate motor commands without sending physical keystrokes")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Explicitly enable live hardware DirectInput scancodes")
    parser.set_defaults(dry_run=False)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of CUDA")
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--auto-record", action="store_true", help="Automatically start lossless video recording on launch")
    parser.add_argument("--resolution", choices=["900p", "1080p", "1440p", "2k"], default="1080p", help="Recording & visualizer resolution: 1080p (Full HD 1920x1080, default), 1440p (2K QHD 2560x1440), 900p (1600x900)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sync-dt", action="store_true", help="Synchronize biological LIF dt with physical wall-clock frame interval")
    parser.add_argument("--web-stream", action="store_true", default=True, help="Enable live web broadcaster on port 8766")
    parser.add_argument("--no-web-stream", dest="web_stream", action="store_false", help="Disable live web broadcaster")
    parser.add_argument("--stream-port", type=int, default=8766, help="Port for live web broadcaster (default: 8766)")
    
    args = parser.parse_args()
    run_simulation(args)
