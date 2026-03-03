# QVolutionHackathon-NativePhotonicQRC
## Hybrid GRU + Photonic Memory (Feedback-Driven QRC)

We have produced a hybrid native photonic time-series forecast with a classical GRU backbone to learn the low frequency structure of the swaption surface with a photonic quantum resevoir to supply nonlinear features with a fading memory, used to correct the GRU backbone via a residual head. 

The key novelty to this approach is the photonic-native quantum resevoir which is of low circuit depth and utilises native photonic gates such that it can provide immediate utility with current photonic NISQ systems which are too noisy for universal computation.

### Key references (where the ideas come from)

- *R. Di Bartolo et al.*, **“Time-series forecasting with multiphoton quantum states and integrated photonics”**, arXiv:2512.02928 (2025).  
  Core idea we borrow: **encode the input signal into optical phase(s)** of a reconfigurable linear-optical circuit; extract features from measurement statistics.

- *Çağın Ekici*, **“A Programmable Linear Optical Quantum Reservoir with Measurement Feedback for Time Series Analysis”**, arXiv:2602.17440 (2026).  
  Core idea we borrow: introduce **memory/recurrence** by **feeding back** (a function of) measured features to update a *subset* of programmable phases, producing controllable fading memory and nonlinear temporal processing.

  These papers propose the resevoir methodology we use, but use end-to-end QRC which performs worse than hybrid methods due to the current noisy landscape of NISQ systems. We extend their methodology by wrapping it with a classical GRU via a residual head to reduce the quantity of quantum calculatons and consider more extended forms of time-series data than the narrow scope of Mackey glasses considered by the authors.

---

## Architecture overview

At each time step `t` (or each observation index), we maintain:

- `x_t`: classical input features (e.g., PCA factors of the vol surface, micro-features, calendar features)
- `r_t`: **photonic memory state**, produced by a photonic feature map with **feedback**
- `ŷ_t,h`: prediction for horizon `h` 

High-level dataflow:


Where `α_h` is a **per-horizon gate** (often crucial in practice): it lets the model use strong correction on short horizons while suppressing harmful long-horizon residual corrections.

---

## Photonic encoding (what the “photonic memory” computes)

### 1) Encode classical inputs into photonic circuit parameters
We map the current input vector `x_t` into a set of circuit phases:

- `θ_in(t) = E(x_t)` where `E(·)` is a simple encoding map (typically linear + scaling + wrap/clamp into `[0, 2π)`).
- Practically, the encoded phases drive a reconfigurable interferometer / phase shifters.

This is the “temporal photonic encoding” / “phase modulation” concept: the time-series drives the photonic device *through its phases*.

### 2) Fixed interferometer + multiphoton feature readout
We propagate a (simulated) multiphoton input state through a fixed linear-optical network `U` and extract coarse-grained measurement features:

- single-click / marginal stats (if used)
- **two-photon coincidence features** (commonly the most informative and stable in our implementation)

We denote the extracted feature vector as:

- `φ_t = Φ(U, θ(t))`

where `φ_t` is a high-dimensional nonlinear embedding of the current input.

### 3) Measurement-conditioned feedback = “photonic memory”
To turn a feedforward photonic embedding into a **recurrent reservoir**, we add a feedback update that uses the previous reservoir output/state:

One convenient abstraction consistent with our implementation:

- Maintain an internal reservoir state `r_t` (or equivalently a subset of feedback phases).
- Update it using a leaky integration + feedback strength:


- `β` controls the fading-memory timescale (“leak rate”).
- `fb_strength` controls how strongly the previous state perturbs the next circuit configuration.
- `W_fb` is a fixed (often random) projection selecting a *budgeted subset* of phases to update.
- `g(·)` is a simple nonlinearity / clipping to keep the feedback stable.

**Intuition:** the circuit’s configuration at `t+1` depends on what it “saw” at `t`, creating recurrence and temporal feature mixing *without backprop training the photonic internals*.

---

## GRU backbone + residual correction
The GRU learns the more straightforward dynamics: smooth temporal evolution, autocorrelation structure, and stable low-dimensional dynamics.

### Photonic residual head (fast linear readout)
Compute residual targets from GRU using the training data.
Train a simple readout model (ridge regression) on the photonic memory state.
This is cheap and stable

---

## Training procedure we use (typical)

1. **Train GRU** on the training split to minimize RMSE (or MAE/RMSE combo).
2. **Freeze GRU**, compute residuals `e(t, h)` on train.
3. **Generate photonic memory states** `r_t` by running the photonic feature map with feedback through the sequence.
4. **Fit ridge regression** readouts `W_out(h)` to predict `e(t, h)` from `r_t`.
5. **Tune** hyperparameters `fb_strength`, ridge `λ`, leak `β`, and **per-horizon gates** `α_h` on validation.
6. Fix hyperparameters and evaluate against classical models on test.

---

## Novelty
- **Photonic memory as a modular drop-in:** we treat a measurement-feedback photonic quantum reservoir as an explicit *memory module* that augments a classical recurrent model.
- **Recurrence without training the quantum block:** feedback introduces memory/temporal processing while keeping the photonic part essentially “fixed + programmable”, avoiding heavy gradient-based training through a quantum simulator.
- **Residualization + gating:** the photonic block only needs to explain what the GRU misses, and the gate `α_h` prevents long-horizon degradation—this is often the difference between “quantum hurts” and “quantum helps”.



