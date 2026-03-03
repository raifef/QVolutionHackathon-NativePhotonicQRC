# QVolutionHackathon-NativePhotonicQRC
## Hybrid GRU + Photonic Memory (Feedback-Driven QRC)

We have produced a hybrid native photonic time-series forecast with a classical GRU backbone to learn the low frequency structure of the swaption surface with a photonic quantum resevoir to supply nonlinear features with a fading memory, used to correct the GRU backbone via a residual head. 

The key novelty to this approach is the photonic-native quantum resevoir which is of low circuit depth and utilises native photonic gates such that it can provide immediate utility with current photonic NISQ systems which are too noisy for universal computation.

---
## Key Plots

---

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



