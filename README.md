# Enhanced Audio Anti-Spoofing Detection on the ASVspoof 2019 Dataset

An end-of-year project submitted to the **Faculty of Sciences of Sfax**, Department of Computer Science and Communications.

| | |
|---|---|
| **Authors** | Kechrida Mohamed Ali · Ben Youssef Aymen |
| **Academic supervisor** | M. Zied Ouarda |
| **Examiner** | M. Ayman Massoudi |
| **Academic year** | 2025–2026 |

---

## Table of contents

1. [Overview](#overview)
2. [Background](#background)
3. [Project structure](#project-structure)
4. [Dataset](#dataset)
5. [Feature extraction](#feature-extraction)
6. [Model architecture](#model-architecture)
7. [Training strategy](#training-strategy)
8. [Results](#results)
9. [Comparison with other systems](#comparison-with-other-systems)
10. [Conclusion and future work](#conclusion-and-future-work)
11. [How to run](#how-to-run)
12. [References](#references)

---

## Overview

Automatic Speaker Verification (ASV) is a biometric technology that authenticates people from their voice. As it is used more and more in security-critical applications, ASV systems have become an attractive target for **spoofing attacks**: speech synthesis, voice conversion and replayed recordings that imitate a genuine speaker.

This project builds a **countermeasure** that decides whether an audio sample is **bonafide** (a real human voice) or **spoofed**. It is developed and evaluated on the standardized **ASVspoof 2019** benchmark, in both of its scenarios:

- **Logical Access (LA):** synthetic and voice-converted speech.
- **Physical Access (PA):** replayed recordings captured in different acoustic environments.

The system is a **Light CNN (LCNN)** with Max-Feature-Map activations and Channel Attention. It takes a three-channel spectral input and is trained with SpecAugment and a class-weighted loss. The goal is a model that is accurate, **lightweight**, and able to **generalize to attacks it has never seen**.

---

## Background

**From hand-crafted features to deep learning**

- *Traditional methods* relied on hand-crafted features such as CQCCs combined with Gaussian Mixture Models (GMMs). They generalized poorly to novel attacks.
- *The deep learning era* moved to CNNs that learn hierarchical features automatically from raw audio or spectrograms.

**Key technologies used in this project**

| Technology | Role |
|---|---|
| **Light CNN (LCNN)** | An efficient architecture designed for anti-spoofing that balances performance and computational cost. |
| **Max-Feature-Map (MFM)** | Selects the maximum of pairs of feature maps, which improves feature selection and suppresses noise better than standard ReLU. |
| **SpecAugment** | Data augmentation by frequency and time masking, which reduces overfitting and improves generalization. |

---

## Project structure

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

1. `notebooks/eda.ipynb`: data exploration, label distributions, audio feature analysis.
2. `notebooks/main.ipynb`: LCNN model training, evaluation, DET curves.

---

## Dataset

ASVspoof 2019 provides separate training, development and evaluation partitions for each scenario.

| Partition | Bonafide | Spoof | Spoof : bonafide ratio |
|-----------|---------:|------:|-----------------------:|
| LA Train  | 2,580    | 22,800  | 8.8× |
| LA Dev    | 2,548    | 22,296  | 8.7× |
| LA Eval   | 7,355    | 63,882  | 8.7× |
| PA Train  | 5,400    | 48,600  | 9.0× |
| PA Dev    | 5,400    | 24,300  | 4.5× |
| PA Eval   | 18,090   | 116,640 | 6.4× |
| **Overall** | **41,373** | **298,518** | **7.2×** |

**Class imbalance.** Spoofed samples outnumber genuine ones by roughly 7 to 1 overall (up to 9 to 1 in some partitions). Without a balancing strategy, a model would become biased towards the majority class, so training uses a weighted loss (see [Training strategy](#training-strategy)).

### Attack distribution

**Logical Access (LA)**

- **Seen attacks (A01–A06):** present in the train, dev and eval sets. The model is directly exposed to these synthesis methods during training.
- **Unseen attacks (A07–A19):** present **only in the evaluation set**. They test whether the model can generalize to novel speech-synthesis and voice-conversion systems. This is the main challenge in LA.

**Physical Access (PA)**

- All **27 attack configurations** are uniformly distributed across all partitions.
- The main difficulty is **environmental variability**: room acoustics and replay-device characteristics, rather than unseen algorithms.

### Exploratory data analysis (`notebooks/eda.ipynb`)

- Label and attack-type distribution across the train/dev/eval splits (LA and PA).
- Per-speaker sample count analysis.
- Audio duration analysis (leading to 4 s fixed-length preprocessing).
- Multi-channel audio feature extraction: RMS, zero-crossing rate (ZCR), spectral centroid, rolloff and bandwidth.
- MFCC boxplots comparing bonafide and spoofed samples.

---

## Feature extraction

Every recording is cut or padded to **4 seconds at 16 kHz (64,000 samples)** and transformed into a **3-channel tensor**. This captures both the static distribution of energy and how it changes over time.

| Channel | What it represents | Details |
|---|---|---|
| **Mel-spectrogram (dB)** | Energy distribution across perceptually scaled frequency bins | 80 mel bins, N_FFT = 1024, hop size = 160 |
| **Delta (Δ)** | First-order temporal derivative of the Mel-spectrogram: the rate of spectral change between frames | Context window of 9 frames |
| **Delta-delta (ΔΔ)** | Second-order temporal derivative: the acceleration of spectral changes | Gives the model deeper dynamic context |

The resulting input has shape `[B, 3, 80, T]`, where `B` is the batch size and `T` the number of time frames.

---

## Model architecture

The model is defined in `notebooks/main.ipynb`.

- **LCNN framework.** A Look-up based Convolutional Neural Network designed for efficiency and high performance in audio classification. It is lightweight and robust, which makes it suitable for real-time deployment in practical ASV systems.
- **MFM activations.** Max-Feature-Map activations make pairs of feature maps compete and keep the maximum. This selects the strongest features and suppresses noise and irrelevant signals, and works better than standard ReLU.
- **Channel Attention.** A module that dynamically re-weights channel-wise feature responses, so the network focuses on the most discriminative spectral patterns.
- **SpecAugment integration.** Frequency and time masking are built into the training pipeline. They force the model to learn spoofing cues that are invariant to missing parts of the spectrogram, which improves generalization.

---

## Training strategy

### Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR schedule | Cosine annealing |
| Batch size | 64 |
| Epochs | 20 maximum, with early stopping (patience of 5 epochs) |
| Input length | 4.0 s (64,000 samples) |
| Loss | Weighted cross-entropy with label smoothing (0.1) |
| Hardware | NVIDIA Tesla T4 GPU on Kaggle |

### Key strategies

- **Weighted cross-entropy.** The bonafide class receives a weight of `n_spoof / n_bonafide`. This mitigates the severe class imbalance (about 9:1) and stops the model from favouring spoofed samples. Label smoothing (0.1) reduces over-confidence.
- **Augmentation.**
  - *SpecAugment*, applied to the spectrogram **before the delta features are computed**: 2× frequency masking (15) and 2× time masking (50).
  - Mild Gaussian noise (std 0.001–0.005, 25% probability).
  - Random gain perturbation (±3 dB, 30% probability).
- **Model checkpointing.** The best checkpoint is selected by the **lowest development EER**, so the saved model is the one most likely to generalize to the evaluation set.

### Metrics

- **Equal Error Rate (EER):** the operating point where the false-acceptance rate equals the false-rejection rate. Lower is better.
- **Minimum t-DCF:** the tandem detection cost function, which measures the combined effect of the countermeasure and the ASV system. Lower is better.

Both follow the official ASVspoof 2019 evaluation protocol.

---

## Results

| Condition | Dev EER | Dev min t-DCF | Eval EER | Eval min t-DCF |
|-----------|--------:|--------------:|---------:|---------------:|
| LA        | 0.11%   | 0.0032        | **2.49%** | 0.1195 |
| PA        | 0.23%   | 0.0087        | **3.07%** | 0.1231 |

### Training behaviour

- **LA:** the EER converges rapidly within the first 5 epochs and stays very low and stable throughout training.
- **PA:** the EER fluctuates at the start, then stabilizes after epoch 15 and reaches a competitive final value.

### Analysis

- **Logical Access.** The model separates genuine speech from synthetic and voice-converted attacks very well. The 2.49% EER is obtained on an evaluation set that contains the **unseen attacks A07–A19**. This indicates that MFM activations and SpecAugment help the model learn invariant spoofing cues.
- **Physical Access.** The 3.07% EER shows that the model copes with replay attacks across diverse acoustic environments, which is the main challenge in PA.

---

## Comparison with other systems

Evaluation-set results for both scenarios:

| System | LA EER (%) | LA min t-DCF | PA EER (%) | PA min t-DCF |
|---|---:|---:|---:|---:|
| Baseline B01 (CQCC-GMM) | 9.57 | 0.2366 | 11.04 | 0.2454 |
| Baseline B02 (LFCC-GMM) | 8.09 | 0.2116 | 13.54 | 0.3017 |
| **Proposed LCNN** | **2.49** | **0.1195** | **3.07** | **0.1231** |
| RawNet2 (Tak et al.) | 1.12 | 0.0415 | 1.15 | 0.0244 |
| ASSERT (Lai et al.) | 1.18 | 0.0524 | 0.55 | 0.0186 |

**Reading the table**

- The proposed LCNN clearly outperforms the official challenge baselines in both scenarios, reducing the error rate by roughly a factor of three to four.
- More complex systems such as RawNet2 and ASSERT still reach lower error rates. The proposed model offers a good balance between accuracy and a lightweight architecture suited to real-time use.

---

## Conclusion and future work

### Conclusion

- **Robust performance.** The LCNN system with MFM activations achieved competitive EERs of **2.49% (LA)** and **3.07% (PA)**.
- **Generalization.** SpecAugment and the weighted loss addressed the class imbalance and improved robustness against unseen attacks.
- **Scalable architecture.** The lightweight design makes the model suitable for real-time deployment in practical ASV systems.

### Future work

- **Ensemble methods.** Combine several diverse models (for example, the LCNN with other architectures) to exploit their complementary strengths.
- **Cross-dataset evaluation.** Test the model on ASVspoof 2021 and other datasets to validate generalization across different recording conditions.
- **Self-supervised learning.** Use self-supervised feature extraction to learn powerful representations from large amounts of unlabelled audio.

---

## How to run

1. **Install the requirements:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset** by following the instructions in [`data/README.md`](data/README.md).
3. **Run the notebooks in order:**
   1. `notebooks/eda.ipynb` for data exploration.
   2. `notebooks/main.ipynb` for training, evaluation and DET curves.

Training was run on a Kaggle notebook with an NVIDIA Tesla T4 GPU. A GPU is recommended, because the dataset is large.

---

## References

- Yamagishi et al. (2019). ASVspoof 2019 challenge.
- Park et al. (2019). SpecAugment.
- Müller et al. (2021). Dataset artifacts in ASVspoof-trained models.
- Tak et al. (RawNet2) and Lai et al. (ASSERT), used as comparison systems.
