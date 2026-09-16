# Biological Experiments & Behavioral Paradigms

This directory outlines experimental neuroscience paradigms adapted from biological *Drosophila melanogaster* research for the `FlyBrain-HalfLife` connectome engine. Each paradigm is grounded in empirical literature and maps directly to observable behaviors in Half-Life and ViZDoom environments.

---

## Table of Contents
1. [Experiment 01: In Silico Optogenetics (Targeted Circuit Activation)](#experiment-01-in-silico-optogenetics-targeted-circuit-activation)
2. [Experiment 02: Looming Collision Detection & Ballistic Evasion](#experiment-02-looming-collision-detection--ballistic-evasion)
3. [Experiment 03: Optomotor Pursuit & Biological Target Locking](#experiment-03-optomotor-pursuit--biological-target-locking)
4. [Experiment 04: Accelerated Pavlovian Conditioning via Mushroom Body Plasticity](#experiment-04-accelerated-pavlovian-conditioning-via-mushroom-body-plasticity)
5. [Experiment 05: Pharmacological Neuromodulation (Caffeine & Ethanol Perturbation)](#experiment-05-pharmacological-neuromodulation-caffeine--ethanol-perturbation)

---

## Experiment 01: In Silico Optogenetics (Targeted Circuit Activation)

### Biological Grounding
In laboratory neurogenetics, Channelrhodopsin (ChR2) and Crimson are expressed in specific GAL4/UAS driver lines to activate designated neuronal subsets with targeted light pulses (Bidaye et al., 2014; Allen et al., 2006). For example:
- **Moonwalker Descending Neurons (MDN):** Optogenetic excitation of MDNs instantly halts forward walking and drives persistent backward stepping.
- **Giant Fiber (GF):** Activation of GF interneurons triggers a reflexive jump-escape motor program.

### Connectome Mechanism
- **Target Subsets:**
  - `MDN` pair: `descending` region indices 4 and 5.
  - `Giant Fiber (GF)`: `descending` region escape triggers.
- **Intervention:** A direct depolarizing current pulse ($I_{opto} = +25.0\,\text{nA}$) is injected into the membrane potential equation ($dV/dt$) of the selected neuron indices for a fixed duration (e.g., 200 ms).
- **Telemetry Indication:** 3D GCaMP visualizer displays intense cyan/blue photostimulation bloom over the targeted neuropil.

### Behavioral Objective
- When the virtual photostimulation is triggered on MDN, the in-game character immediately suppresses forward locomotion and exhibits backward moonwalking.
- When photostimulation hits GF, the agent triggers an immediate jump-turn evasion.

---

## Experiment 02: Looming Collision Detection & Ballistic Evasion

### Biological Grounding
Flies possess specialized visual projection neurons (LC4 and LPLC2) sensitive to radially expanding visual patterns (looming stimuli) that signify incoming predators or approaching surfaces (von Reyn et al., 2014; Ache et al., 2019). The firing rate scales with angular velocity and projected time-to-collision ($TTC$).

### Connectome Mechanism
- **Visual Input:** Radially symmetrical luminance changes detected across the 3,600-ommatidia retina.
- **Circuit Pathway:** 
  $$\eta(t) = \theta(t) \cdot \exp(-\alpha \cdot \frac{\dot{\theta}(t)}{\theta(t)})$$
  Feedforward excitation from LC4/LPLC2 channels directly into the Giant Fiber descending motor pathway when the looming expansion rate exceeds threshold.
- **Telemetry Indication:** Red looming warning badge and rapid spike bursts across the visual projection channels.

### Behavioral Objective
- When a high-velocity projectile approaches (e.g., Doom Imp fireball, Half-Life rocket, or grenade), the agent calculates time-to-collision and executes a duck-and-strafe evasive leap before impact.

---

## Experiment 03: Optomotor Pursuit & Biological Target Locking

### Biological Grounding
Flies track moving conspecifics and navigate using Lobula Plate Tangential Cells (LPTCs, including Horizontal System HS and Vertical System VS neurons) that integrate wide-field and small-field optic flow (Borst et al., 2010; Land & Collett, 1974). Flies perform menotaxis, maintaining a target at a fixed retinal bearing.

### Connectome Mechanism
- **Visual Input:** Central foveal ommatidia (inner $12 \times 12$ array) dedicated to detecting high-contrast, moving foreground entities.
- **Circuit Pathway:** Central optic flow asymmetry drives differential excitation between bilateral DNp20 descending neurons:
  $$\Delta \text{Turn} = k_{pursuit} \cdot (I_{\text{optic}, L} - I_{\text{optic}, R})$$
- **Telemetry Indication:** Central crosshair foveal reticle turns green upon target acquisition, with bilateral DNp20 gauges dynamically reflecting the steering differential.

### Behavioral Objective
- As an enemy or moving entity crosses the agent's field of view, the optomotor loop locks orientation onto the entity, continuously centering the target in the crosshairs.

---

## Experiment 04: Accelerated Pavlovian Conditioning via Mushroom Body Plasticity

### Biological Grounding
Associative olfactory and visual learning in *Drosophila* occurs within the Mushroom Body (Aso et al., 2014; Hige et al., 2015). Sparse odor/visual patterns are represented by ~2,000 Kenyon Cells (KCs). Coincident activation of KCs and dopaminergic neurons (DANs from PPL1 or PAM clusters) induces Spike-Timing-Dependent Plasticity (STDP) at KC-to-MBON synapses, altering the valence (approach vs. avoidance) of sensory cues.

### Connectome Mechanism
- **Plasticity Rate:** Elevated learning rate ($\eta = 0.15$) for clear within-session behavioral adaptation.
- **Conditioning Stimulus (CS):** Specific environmental chromatic features (e.g., green toxic sludge in ViZDoom, explosive red barrels in Half-Life).
- **Unconditioned Stimulus (US):** Negative dopamine surge ($DA = -2.0$) upon damage or toxicity.
- **Synaptic Rule:** 3-factor Hebbian update on KC $\to$ MBON synapses:
  $$\Delta W = \eta \cdot \text{DA} \cdot S_{pre} \cdot S_{post}$$
- **Telemetry Indication:** Real-time histogram and line plot of KC $\to$ MBON synaptic weight distribution showing selective depression of hazard-associated pathways.

### Behavioral Objective
- On initial exposure, the agent navigates indifferently into hazard zones. After taking damage, the specific sensory pattern is associated with negative valence. Subsequent approaches trigger pre-emptive turning away from the learned hazard.

---

## Experiment 05: Pharmacological Neuromodulation (Caffeine & Ethanol Perturbation)

### Biological Grounding
Pharmacological studies evaluate how psychoactive compounds alter Drosophila circadian activity, motor coordination, and synaptic transmission (Bainton et al., 2000; Nall et al., 2016):
- **Caffeine (Adenosine Antagonist):** Blocks inhibitory adenosine tone, elevating spontaneous sub-threshold noise, increasing firing frequency, and shortening refractory periods.
- **Ethanol (GABAergic Agonist / NMDA Antagonist):** Enhances central inhibition, suppresses high-frequency transmission, induces motor ataxia, and causes wandering drift.

### Connectome Mechanism
- **Caffeine Mode:**
  - Spontaneous noise std: $\sigma_{noise} \times 3.0$
  - Refractory period: $\tau_{ref} \times 0.5$
  - Synaptic transmission gain: $1.4\times$
- **Ethanol Mode:**
  - Global synaptic conductance: $0.6\times$
  - Directional motor damping: High random angular drift variance ($\sigma_{drift} = 0.4$)
  - Membrane time constant: $\tau_m \times 1.5$ (sluggish depolarization)
- **Telemetry Indication:** Dynamic substance badge ("NEUROMOD: CAFFEINE (HYPERKINETIC)" or "NEUROMOD: ETHANOL (ATAXIA)") with corresponding EEG / spike raster shifts.

### Behavioral Objective
- **Caffeine:** The agent exhibits hyperkinetic jitter, rapid twitch-turns, hyper-frequent weapon discharge, and hypersensitive wall bounces.
- **Ethanol:** The agent displays sluggish response times, staggering walk trajectories, drifting against walls, and delayed reaction to enemy encounters.
