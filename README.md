# Enhanced Audio Anti-Spoofing Detection on the ASVspoof 2019 Dataset

An end-of-year project submitted to the **Faculty of Sciences of Sfax**, Department of Computer Science and Communications.

**Authors:** Kechrida Mohamed Ali · Ben Youssef Aymen  
**Supervisor:** M. Zied Ouarda  
**Academic Year:** 2025–2026

---

## Project Structure

```
.
├── notebooks/
│   ├── eda.ipynb            # Exploratory Data Analysis
│   └── main.ipynb           # LCNN model training & evaluation
├── README.md
├── requirements.txt
└── data/
    └── README.md            # Dataset download instructions
```

**Run the notebooks in order:**
1. `notebooks/eda.ipynb` — data exploration, label distributions, audio feature analysis
2. `notebooks/main.ipynb` — LCNN model training, evaluation, DET curves

---

## Approach

### EDA (`notebooks/eda.ipynb`)
- Label and attack-type distribution across train/dev/eval splits (LA & PA)
- Per-speaker sample count analysis
- Audio duration analysis (4s fixed-length preprocessing)
- Multi-channel audio feature extraction: RMS, ZCR, spectral centroid, rolloff, bandwidth
- MFCC boxplots comparing bonafide vs. spoof samples

### Model (`notebooks/main.ipynb`)
- **Architecture:** Light CNN (LCNN) with Max Feature Map (MFM) activations and Channel Attention
- **Input:** 3-channel tensor — Mel-spectrogram (dB) + Δ + ΔΔ — 80 mel bins, 4s @ 16 kHz (shape `[B, 3, 80, T]`)
- **Augmentation:** SpecAugment applied before delta computation — 2× FrequencyMasking(15) + 2× TimeMasking(50); also mild Gaussian noise (std 0.001–0.005, 25% prob.) and random gain perturbation (±3 dB, 30% prob.)
- **Class imbalance:** Weighted cross-entropy loss (weight = n_spoof / n_bonafide for bonafide class) + label smoothing (0.1)
- **Training:** Adam (lr=1e-4, weight decay=1e-4), cosine annealing LR, 20 epochs, batch size 64, early stopping (5 epochs), best checkpoint selected by lowest dev EER
- **Hardware:** NVIDIA Tesla T4 GPU on Kaggle
- **Metrics:** Equal Error Rate (EER) and minimum t-DCF per the official ASVspoof 2019 protocol

---

## Results

| Condition | Dev EER | Dev min t-DCF | Eval EER | Eval min t-DCF |
|-----------|---------|---------------|----------|----------------|
| LA        | 0.11%   | 0.0032        | 2.49%    | 0.1195         |
| PA        | 0.23%   | 0.0087        | 3.07%    | 0.1231         |

### Comparison with State-of-the-Art (LA Eval)

| System              | Feature       | Architecture       | LA EER↓ | LA t-DCF↓ |
|---------------------|---------------|--------------------|---------|-----------|
| **Ours (LCNN)**     | Mel+Δ+ΔΔ     | LCNN + SpecAugment | **2.49%** | **0.1195** |
| RawNet2             | Raw waveform  | SincNet + GRU      | 1.12%   | 0.0335    |
| AASIST              | Raw waveform  | Graph Attention    | 0.83%   | 0.0275    |
| T05 (top system)    | Ensemble      | CNN Ensemble       | 0.22%   | 0.0069    |


---

## Requirements

```bash
pip install -r requirements.txt
```

---

## References

- Yamagishi et al. (2019) — ASVspoof 2019 challenge
- Park et al. (2019) — SpecAugment
- Müller et al. (2021) — Dataset artifacts in ASVspoof-trained models
