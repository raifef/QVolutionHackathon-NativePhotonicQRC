# QVolutionHackathon-NativePhotonicQRC
## Hybrid GRU + Photonic Memory (Feedback-Driven QRC)

We have produced a hybrid native photonic time-series forecast with a classical GRU backbone to learn the low frequency structure of the swaption surface with a photonic quantum resevoir to supply nonlinear features with a fading memory, used to correct the GRU backbone via a residual head. 

The key novelty to this approach is the photonic-native quantum resevoir which is of low circuit depth and utilises native photonic gates such that it can provide immediate utility with current photonic NISQ systems which are too noisy for universal computation. Despite the significantly lower circuit depth than other QRC implementations for time-series data (Li, Qingyu, et al, 2025), indicative of lower quantum error rates when run at scale on real quantum hardware, we find more accurate forecasting for lower (noisy) quantum computational cost.

Our results show a clear increase in predictive power (reduction in rmse error on test data for a 6 day horizon) over classical methods, while providing better error scaling with noise than other QRC implementations, directly compared against that in Li, Qingyu, et al. "Quantum reservoir computing for realized volatility forecasting." arXiv preprint arXiv:2505.13933 (2025). Full wall-time analyses and runs on real quantum hardware were not able to be completed due to time constraints but would be necessary to truly show the computational advantage over classical methods and error-scaling advantage over similar quantum implementations, however are results are indicative of better error scaling due to lower gate count and lower forecasting error.

---
## Key Plots
![alt text](https://github.com/raifef/QVolutionHackathon-NativePhotonicQRC/blob/main/images/Screenshot%202026-03-03%20at%2012.51.58.png)
Bar + horizon summaries on external test data showing our quantum/photonic hybrid beats the classical comparator on RMSE and MAE in 2/3 random seeds and on average, with the horizon plot indicating the win is driven mainly by stronger gains at longer horizons.

![alt text](https://github.com/raifef/QVolutionHackathon-NativePhotonicQRC/blob/main/images/02_single_seed_horizon_rmse_seed_2024.png)
The horizon-by-horizon curves show the GRU+photonic memory has lower RMSE than GRU alone for horizons 2–6 (especially 4–6), and the lower panel’s edge bars quantify exactly how much the hybrid improves each horizon.

![alt text](https://github.com/raifef/QVolutionHackathon-NativePhotonicQRC/blob/main/images/04_advantage_ratio.png)
A hardware-proxy comparison against Li et al. (2025) plotting RMSE_paper / RMSE_native under photonic-loss and 2q-gate-error sweeps, where the key takeaway is the ratio stays >1 and grows strongly with two-qubit error, supporting the claim that photonic-native, low-depth operations degrade more gracefully under noise.

---
## Motivation

Modeling of full temporal volatility surfaces is not feasible on current NISQ quantum systems due to error propagation with circuit scaling, which scales with feature dimension.

We solve this problem by implementing a hybrid algorithm. Classical GRU is used to learn the low-frequency structure of the data, while our photonic resevoir computing provides nonlinearity on top of this, implemented with a fading memory via photonic feedback which provides lower forecasting errors. Our photonic resevoir is native to a photonic quantum computer, minimising the gate count and therefore total computational error on real quantum hardware.

Feedback creates photonic memory by taking a small summary of the measured reservoir features at time t and feeding it back to update a subset of the programmable phases before processing x_{t+1}. This makes the next photonic feature map depend on the recent past (a controllable fading memory set by fb_strength/leak), giving nonlinear temporal context without training the quantum block end-to-end.

All results at this stage are currently run purely as local simulation with added shot noise but otherwise idealistic quantum circuits, due to time and quantum budget constraints in this hackathon preventing full runs on real quantum hardware.

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
- `r_t`: photonic memory state, produced by a photonic feature map with feedback
- `ŷ_t,h`: prediction for horizon `h` 

High-level dataflow:
`α_h` is a per-horizon gate which lets the model use strong correction on short horizons while suppressing harmful long-horizon residual corrections.

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
5. **Tune** hyperparameters `fb_strength`, ridge `λ`, leak `β`, and per-horizon gates `α_h` on validation.
6. Fix hyperparameters and evaluate against classical models on test.

---

## Novelty
- **Photonic memory as a modular drop-in:** we treat a measurement-feedback photonic quantum reservoir as an explicit *memory module* that augments a classical recurrent model.
- **Recurrence without training the quantum block:** feedback introduces memory/temporal processing while keeping the photonic part essentially “fixed + programmable”, avoiding heavy gradient-based training through a quantum simulator.
- **Residualization + gating:** the photonic block only needs to explain what the GRU misses and the gate `α_h` prevents long-horizon degradation



