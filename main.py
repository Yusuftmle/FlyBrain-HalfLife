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
from reinforcement.mushroom_body import MushroomBodyController
from telemetry.visualizer import FlyBrainVisualizer
from telemetry.desktop_audio import DesktopAudioPlayer
from server.broadcast_server import FlyBrainWebBroadcaster
from core.connectome import FlyConnectome
from config import DEFAULT_CONFIG, Config, apply_preset

logger = logging.getLogger("FlyBrain.Main")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def run_simulation(args):
    is_lite = (getattr(args, "preset", "default").lower() in ["cpu-lite", "lite"]) or getattr(args, "cpu_lite", False)
    if is_lite:
        args.cpu = True
        if getattr(args, "vis_fps", 30) == 30:
            args.vis_fps = 15

    sim_cfg = apply_preset(Config(), "cpu-lite" if is_lite else "default")

    print("=" * 74)
    print("   FLYBRAIN-HALFLIFE: MaleCNS v1.0 & FlyWire GPU Autonomous Agent")
    print("   Drosophila Melanogaster Connectome Navigation System")
    print("=" * 74)
    print(f"[*] Execution Mode : {args.mode.upper()}")
    print(f"[*] Preset         : {'CPU-LITE (3,170 Neurons, 30x30 Eye)' if is_lite else 'CANONICAL (12,260 Neurons, 60x60 Eye)'}")
    print(f"[*] Visualizer     : {'ENABLED' if not args.no_vis else 'DISABLED (Headless)'}")
    print(f"[*] Web Broadcaster: {'ENABLED (http://localhost:' + str(args.stream_port) + ')' if getattr(args, 'web_stream', True) else 'DISABLED'}")
    print(f"[*] Hardware Input : {'DRY-RUN (Simulated - Physical Keys Disabled)' if (args.dry_run or args.mode in ['arena', 'doom']) else 'LIVE DIRECTINPUT (PHYSICAL KEYS ACTIVE: W, A, S, D, Mouse)'}")
    if args.record:
        print(f"[*] Video Output   : {args.record}")
    print("-" * 74)

    # LAYER 1: Load Biological Connectome and Neurotransmitter Signs
    loader = ConnectomeDataLoader()
    sparse_weights, meta = loader.build_canonical_flywire_connectome(is_lite=is_lite)
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
        # Bipolar valence: Avoidance (-1.0) on first half, Approach (+1.0) on second half
        mbon_valences = np.ones(len(mbon_indices), dtype=np.float32)
        mbon_valences[:len(mbon_indices) // 2] = -1.0
        lif_engine.init_plastic_synapses(kc_indices, mbon_indices, post_valences=mbon_valences)

    # Benchmark Mode
    if args.mode == "benchmark":
        run_benchmark(lif_engine, num_neurons, sparse_weights.nnz, steps=args.steps or 1000)
        return

    # LAYER 3: Vision Bridge (MSS Sub-2ms Capture & Naka-Rushton Photoreceptors)
    vision_bridge = VisionBridge(
        grid_width=sim_cfg.vision.grid_width, 
        grid_height=sim_cfg.vision.grid_height, 
        window_title=sim_cfg.vision.capture_window_title
    )

    # LAYER 4: Input Bridge (DirectInput Hardware Scancodes, Window Lock, & Safety Guard)
    input_bridge = InputBridge(
        turn_threshold=0.03,
        walk_threshold=0.05,
        attack_threshold=DEFAULT_CONFIG.motor.attack_threshold,
        dry_run=args.dry_run or (args.mode in ["arena", "doom"]), 
        cooldown_duration=0.08,
        target_hwnd=vision_bridge.hwnd if args.mode == "live" else None
    )

    # Reinforcement Dopamine Circuit & Anti-Stuck State
    core_connectome = FlyConnectome(sim_cfg.connectome)
    dopamine = DopamineController(core_connectome, lif_engine, sim_cfg.rl)

    # Mushroom Body (Mantar Cisimciği) Valence & Associative Learning Circuit
    kc_cnt = region_counts.get("mushroom_body_kc", 2400)
    mbon_cnt = region_counts.get("mbon", 120)
    kc_off = region_offsets.get("mushroom_body_kc", 9600)
    mbon_off = region_offsets.get("mbon", 12000)
    mushroom_body = MushroomBodyController(
        num_kc=kc_cnt,
        num_mbon=mbon_cnt,
        kc_offset=kc_off,
        mbon_offset=mbon_off,
        sparse_ratio=0.05,
        learning_rate=0.018,
        extinction_rate=0.0001
    )

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
    last_vis_render_time = 0.0
    last_broadcast_time = 0.0
    win_styles_applied = False
    # Damage evasion turn memory: after taking damage, persist a strong turn for 1.5s
    damage_escape_dir = 1    # +1 = turn right, -1 = turn left
    last_damage_time = 0.0   # timestamp of last damage event
    last_steer_sign = 0          # which way we were steering last frame
    last_dispatched_mouse_dx = 0 # biological efference copy tracking
    # Virtual Haltere gaze glance timers: periodic vertical exploration without getting stuck looking up
    last_obs_glance_time = 0.0
    last_stuck_glance_time = 0.0
    filtered_steer_bias = 0.0
    # Clean Collision-Free Forward Exploration Streak Tracker
    clean_streak_time = 0.0
    last_streak_reward_time = 0.0
    sugar_boost_until = 0.0
    # Mushroom Body Memory Recall Cache
    mb_info = {
        "valence": 0.0, 
        "is_avoidance_recall": False, 
        "is_approach_recall": False, 
        "mdn_stim": 0.0, 
        "steer_bias": 0.0, 
        "forward_suppression": 1.0
    }

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

            # 2. Retinal Transduction (HUD-free Central FOV & Biological Efference Copy)
            photocurrents, motion_metrics = vision_bridge.process_frame(game_frame, ego_yaw_dx=last_dispatched_mouse_dx)

            # 3. Dopamine Circuit & Combat Retaliation / Obstacle Deadlock Circuit
            if is_damage:
                dopamine.register_event("damage", magnitude=max(1.5, red_intensity * 4.0))
                # Pick a persistent escape direction on damage hit
                if (time.time() - last_damage_time) > 1.2:
                    damage_escape_dir = 1 if np.random.rand() > 0.5 else -1
                last_damage_time = time.time()
                # Combat Retaliation: Immediate 180° whip turn, return fire, and evasive combat move!
                input_bridge.trigger_combat_retaliation(direction=damage_escape_dir)
            elif getattr(lif_engine, "is_flatlined", False):
                # Respawn Resuscitation
                lif_engine.trigger_defibrillation()
                input_bridge.handle_respawn()

            is_obs = bool(motion_metrics.get("is_obstacle_close", False))
            cur_flow = float(motion_metrics.get("flow", 0.0))
            
            # Genuine Spatial Forward Exploration Streak Reinforcement
            # Requires:
            # 1. Real optical flow above stationary weapon idle breathing (> 0.00060)
            # 2. Genuine forward optical flow expansion (divergence > 0.00008)
            # 3. Active locomotion forward drive (input_bridge.is_walking)
            # 4. No close frontal obstacles or damage
            divergence = float(motion_metrics.get("divergence", 0.0))
            is_genuine_forward = (
                (cur_flow > 0.00060)
                and (divergence > 0.00008)
                and input_bridge.is_walking
                and (not is_obs)
                and (not is_damage)
            )
            if is_genuine_forward:
                dt_sec = max(0.01, min(0.1, dt_frame_ms / 1000.0))
                clean_streak_time += dt_sec
                # Every 1.8 seconds of sustained uninterrupted exploration -> Dopamine & Sugar Reward Surge!
                if (clean_streak_time - last_streak_reward_time) >= 1.8:
                    last_streak_reward_time = clean_streak_time
                    streak_mag = min(2.5, 0.8 + (clean_streak_time / 4.0))
                    dopamine.register_event("streak", magnitude=streak_mag)
                    sugar_boost_until = time.time() + 0.40  # 400ms LB3c sweet taste sensation
                    logger.info(f"🌟 [EXPLORATION REWARD] {clean_streak_time:.1f}s Temiz İlerleme! Dopamin + Şeker Ödülü (+{streak_mag:.2f} DA)")
            else:
                if is_obs or is_damage or cur_flow < 0.00045:
                    clean_streak_time = 0.0
                    last_streak_reward_time = 0.0

            da_current, rl_info = dopamine.update(cur_flow, is_obstacle=is_obs)
            rl_info["clean_streak"] = clean_streak_time
            rl_info["sugar_active"] = (time.time() < sugar_boost_until)
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
            pc_grid = photocurrents.reshape(vision_bridge.h, vision_bridge.w)
            left_eye_drive = float(np.mean(pc_grid[:, : vision_bridge.w // 2]))
            right_eye_drive = float(np.mean(pc_grid[:, vision_bridge.w // 2 :]))
            depth_bal = float(motion_metrics.get("depth_balance", 0.0))
            is_obs = bool(motion_metrics.get("is_obstacle_close", False))
            is_fence = bool(motion_metrics.get("is_fence", False))
            
            # Positive steer_bias (> 0) stimulates DNp20-L (turn left)
            # Negative steer_bias (< 0) stimulates DNp20-R (turn right)
            # depth_bal > 0 means right side is more open -> steer_bias becomes negative -> turn right!
            retinal_contrast_steer = (left_eye_drive - right_eye_drive) * 1.2
            depth_steer = -(depth_bal * 24.0)  # Balanced open corridor centering
            raw_steer_bias = retinal_contrast_steer + depth_steer

            # Mushroom Body Learned Threat Evasion (Avoidance Turn Bias)
            if mb_info.get("is_avoidance_recall", False):
                raw_steer_bias += mb_info.get("steer_bias", 0.0)

            # Biological Optomotor Low-Pass Damping (Optic Lobe / Haltere time constant ~120ms)
            # Alpha 0.28: responsive steering while preventing limit-cycle camera jitter
            filtered_steer_bias += 0.28 * (raw_steer_bias - filtered_steer_bias)
            final_steer_bias = filtered_steer_bias

            # Responsive DNp20 LIF Stimulus (Channels 1 & 2):
            # Base 10.0 gives wide dynamic differential headroom
            base_dnp20 = 10.0
            dnp20_l_stim = float(np.clip(base_dnp20 + max(0.0, final_steer_bias) * 1.5, 0.0, 95.0))
            dnp20_r_stim = float(np.clip(base_dnp20 + max(0.0, -final_steer_bias) * 1.5, 0.0, 95.0))

            # Biological DNp09 Evasive Saccade Neurons (Descending Channels 7 & 8)
            # Rayshubskiy et al. (2020) / Ache et al. (2019): Evasive body saccades triggered by looming obstacles
            dnp09_l_stim = 0.0
            dnp09_r_stim = 0.0
            if is_obs:
                if depth_bal > 0.02:
                    dnp09_r_stim = 50.0  # Open corridor on right -> saccade right
                elif depth_bal < -0.02:
                    dnp09_l_stim = 50.0  # Open corridor on left -> saccade left
                else:
                    # Symmetric wall in front: saccade toward slightly clearer side or alternate
                    if np.random.rand() > 0.5:
                        dnp09_r_stim = 50.0
                    else:
                        dnp09_l_stim = 50.0
            
            # Continuous Locomotor CPG Forward Drive (DOOMFLY / Drosophila biology)
            # In nature, flies walk forward continuously; obstacles are navigated by steering (DNp20),
            # while MDN backward stepping is reserved strictly for damage evasion and deadlock panic.
            in_damage_evasion = (time.time() - last_damage_time) < 0.8
            if is_damage or in_damage_evasion:
                fwd_stim = 0.0
                mdn_stim = 40.0
                if damage_escape_dir > 0:
                    dnp20_r_stim = max(dnp20_r_stim, 85.0)
                    dnp20_l_stim = 5.0
                    dnp09_r_stim = max(dnp09_r_stim, 60.0)
                else:
                    dnp20_l_stim = max(dnp20_l_stim, 85.0)
                    dnp20_r_stim = 5.0
                    dnp09_l_stim = max(dnp09_l_stim, 60.0)
            elif is_panic_now:
                # In panic: stop forward push, let DNp20 rotate out without moonwalking
                fwd_stim = 0.0
                mdn_stim = 0.0
            else:
                # Steady forward locomotion modulated by turning & obstacles
                if is_obs:
                    fwd_stim = 4.0   # Slow crawl while executing sharp yaw turn away from obstacle
                elif abs(final_steer_bias) > 12.0:
                    fwd_stim = 14.0  # Ease forward speed on corners for tight camera arc
                else:
                    fwd_stim = 28.0  # Confident corridor speed
                mdn_stim = 0.0

            # Mushroom Body Valence Modulation on Forward/Backward locomotion
            if mb_info.get("is_avoidance_recall", False):
                fwd_stim *= mb_info.get("forward_suppression", 0.65)
                # Avoidance memory steers away via DNp20; never frantically moonwalks backwards in open corridors
                mdn_stim = 0.0
            elif mb_info.get("is_approach_recall", False):
                fwd_stim += 15.0 * mb_info.get("valence", 0.0)

            # Positive dopamine reinforces confident forward locomotion (PPL1 -> MBON -> DNpe017)
            # Only boost forward when NOT stuck against a wall
            da_fwd_boost = 0.0
            if not is_obs and not is_panic_now:
                da_fwd_boost = max(0.0, float(dopamine.dopamine_level)) * 12.0

            total_ext_current[idx_dnp20_l] += dnp20_l_stim
            total_ext_current[idx_dnp20_r] += dnp20_r_stim
            total_ext_current[idx_dnpe017_fwd] += fwd_stim + da_fwd_boost
            total_ext_current[idx_mdn_l] += mdn_stim
            total_ext_current[idx_mdn_r] += mdn_stim
            total_ext_current[idx_dnp09_l] += dnp09_l_stim
            total_ext_current[idx_dnp09_r] += dnp09_r_stim

            # Mushroom Body 16-D Sensory Scene Projection & Sparse 5% Kenyon Cell Transduction
            mb_feat = mushroom_body.extract_feature_vector(motion_metrics, is_damage=is_damage, red_excess=red_intensity)
            kc_stim = mushroom_body.step_sensory_projection(mb_feat)
            total_ext_current[kc_off : kc_off + kc_cnt] += kc_stim

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
                is_sugar_reward = (dopamine.dopamine_level > 0.2) or (time.time() < sugar_boost_until)
                if is_sugar_reward:
                    sugar_stim = 25.0 if (time.time() < sugar_boost_until) else 18.0
                    total_ext_current[meta["sugar_indices"]] += sugar_stim
                    sugar_active = True

            # Visual Target Lock & Combat Retaliation Attack Drive (DNpe017-Atk)
            is_target = bool(motion_metrics.get("is_target_in_crosshair", False))
            if is_damage or in_damage_evasion:
                # Combat Retaliation: Sustained aggressive counter-attack during evasion window!
                atk_burst = 70.0 if is_damage else 45.0
                total_ext_current[idx_dnpe017_atk] += atk_burst
                dn_start = region_offsets["descending"]
                dn_len = min(10, region_counts["descending"])
                total_ext_current[dn_start : dn_start + dn_len] += 25.0
            elif is_target:
                # Predatory Visual Strike: Target locked in central ommatidial crosshair
                total_ext_current[idx_dnpe017_atk] += 65.0

            # 5. Low-Latency Vectorized LIF Forward Pass
            if getattr(args, "sync_dt", False):
                num_substeps = max(2, min(4, int(round(dt_frame_ms / max(0.1, lif_engine.dt * 10.0)))))
            else:
                num_substeps = 2 if not args.cpu else 2

            if hasattr(lif_engine, "forward_substeps"):
                spikes = lif_engine.forward_substeps(total_ext_current, num_substeps=num_substeps)
            else:
                for _ in range(num_substeps):
                    spikes = lif_engine.forward_step(total_ext_current)

            # 5.5. DOOMFLY 3-Factor Hebbian STDP Learning (modulated by PPL1/PAM dopamine)
            if step % 2 == 0:
                lif_engine.apply_stdp_update(dopamine=dopamine.dopamine_level, eta=0.002)
                mushroom_body.apply_dopamine_plasticity(dopamine.dopamine_level, escape_dir=damage_escape_dir)

            # 6. Read Motor Firing Rates (DNp20, DNpe017, MDN, DNp09) & Mushroom Body Valence
            rates = lif_engine.get_firing_rates()
            rate_dnp20_l = float(rates[idx_dnp20_l])
            rate_dnp20_r = float(rates[idx_dnp20_r])
            rate_forward = float(rates[idx_dnpe017_fwd])
            rate_attack = float(rates[idx_dnpe017_atk])
            rate_backward = float(rates[idx_mdn_l] + rates[idx_mdn_r]) / 2.0
            rate_dnp09_l = float(rates[idx_dnp09_l])
            rate_dnp09_r = float(rates[idx_dnp09_r])
            
            # Compute Mushroom Body net valence and associative memory recall
            mb_info = mushroom_body.compute_valence_and_recall(rates)
            # 6.5 Drosophila Gaze Stabilization & Closed-Loop Virtual Haltere
            # Visual horizon cues (Dorsal Light & Optic Flow Helmholtz Decomposition)
            visual_horizon_data = {
                "ceiling_score": float(motion_metrics.get("ceiling_score", 0.0)),
                "floor_score": float(motion_metrics.get("floor_score", 0.0)),
                "vertical_flow": float(motion_metrics.get("v_pitch", 0.0))
            }

            # Fixed level horizon (matching DOOMFLY reference: pitch = 0)
            glance_req = None
            glance_dur = 0.0

            # 7. Hardware Debounced Input Dispatch (DOOMFLY-inspired EMA Modular Decoder & Locomotion)
            motor_info = input_bridge.decode_and_dispatch(
                dnp20_left_rate=rate_dnp20_l,
                dnp20_right_rate=rate_dnp20_r,
                dnpe017_forward_rate=rate_forward,
                dnpe017_attack_rate=rate_attack,
                mdn_backward_rate=rate_backward,
                is_obstacle_close=is_obs,
                vertical_pitch_rate=0.0,
                visual_horizon=visual_horizon_data,
                glance_pitch=glance_req,
                glance_duration=glance_dur,
                dnp09_left_rate=rate_dnp09_l,
                dnp09_right_rate=rate_dnp09_r
            )

            # Pass controls to next step
            turn_val = -float(motor_info["turn_diff"]) * 2.0
            fwd_val = float(motor_info.get("net_forward", rate_forward - rate_backward)) * 2.5
            is_attack = bool(motor_info["is_firing"])
            last_dispatched_mouse_dx = int(motor_info.get("mouse_dx", 0))

            # 7.5. Live Desktop Audio Playback (PC Speakers / Headphones)
            desktop_audio.play_frame(
                spikes_count=int(np.sum(spikes)),
                dopamine=float(dopamine.dopamine_level),
                is_firing=bool(motor_info["is_firing"]),
                is_panic=bool(is_panic_now),
                is_damage=bool(is_damage)
            )

            # 8. Render Live Telemetry (Decoupled GUI refresh for 100+ FPS simulation)
            vis_target_fps = getattr(args, "vis_fps", 30)
            vis_interval_s = 1.0 / max(1, vis_target_fps)
            t_now_s = time.time()
            should_render_vis = (visualizer is not None) and ((t_now_s - last_vis_render_time) >= vis_interval_s)

            if should_render_vis:
                last_vis_render_time = t_now_s
                # 60x60 Drosophila Multispectral Ommatidial Color Viewport
                if "retina_color_bgr" in motion_metrics and motion_metrics["retina_color_bgr"] is not None:
                    retina_vis = cv2.resize(
                        motion_metrics["retina_color_bgr"],
                        (200, 200),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    half_pr = len(photocurrents) // 2
                    gw = vision_bridge.w
                    gh = vision_bridge.h
                    left_cur = photocurrents[:half_pr].reshape(gh, gw // 2)
                    right_cur = photocurrents[half_pr:].reshape(gh, gw // 2)
                    reconstructed = np.hstack([left_cur, right_cur])
                    retina_vis = cv2.resize(
                        cv2.cvtColor((reconstructed * 8.0).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
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
                    "r8p_blue": motion_metrics.get("r8p_blue", 0.0),
                    "mushroom_body": mushroom_body.get_diagnostics()
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
                            if visualizer:
                                visualizer.toggle_recording()
                        # F11 (0x7A): Toggle Window Pinned Topmost
                        elif user32_hk.GetAsyncKeyState(0x7A) & 0x8000:
                            last_global_hotkey_time = t_now_hk
                            if visualizer:
                                visualizer.toggle_topmost("FlyBrain - MaleCNS Connectome Telemetry")
                        # F6 (0x75): Refocus Half-Life Game Window
                        elif user32_hk.GetAsyncKeyState(0x75) & 0x8000:
                            last_global_hotkey_time = t_now_hk
                            vision_bridge.focus_game_window()
                except Exception:
                    pass

            # 8.5. Live Web Broadcaster Update (Decoupled at 25 FPS)
            should_broadcast = (broadcaster and broadcaster.is_running) and ((t_now_s - last_broadcast_time) >= 1.0 / 25.0)
            if should_broadcast:
                last_broadcast_time = t_now_s
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
                        "mushroom_body": mushroom_body.get_diagnostics(),
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
    parser.add_argument("--preset", choices=["default", "cpu-lite"], default="default", help="Operational preset: default (canonical 12,260 neurons) or cpu-lite (3,170 neurons, 30x30 eye, high CPU FPS)")
    parser.add_argument("--cpu-lite", dest="cpu_lite", action="store_true", help="Shortcut for --preset cpu-lite")
    parser.add_argument("--vis-fps", type=int, default=30, help="Visualizer HUD refresh rate in FPS (default: 30, leaves full headroom for 100+ FPS simulation)")
    
    args = parser.parse_args()
    run_simulation(args)
