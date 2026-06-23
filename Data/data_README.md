# Data

The dataset used in this project is **ASVspoof 2019**, a benchmark for automatic speaker verification spoofing and countermeasure research.

The data is **not included** in this repository due to its large size. Follow the instructions below to download it.

---

## Download

The dataset is publicly available on Kaggle:  
🔗 https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

You can download it directly from Kaggle or use the Kaggle API:

```bash
kaggle datasets download -d awsaf49/asvpoof-2019-dataset
```

---

## Expected Structure

Once downloaded, the dataset should follow this structure:

```
data/
├── LA/
│   └── LA/
│       ├── ASVspoof2019_LA_train/flac/
│       ├── ASVspoof2019_LA_dev/flac/
│       ├── ASVspoof2019_LA_eval/flac/
│       └── ASVspoof2019_LA_cm_protocols/
│           ├── ASVspoof2019.LA.cm.train.trn.txt
│           ├── ASVspoof2019.LA.cm.dev.trl.txt
│           └── ASVspoof2019.LA.cm.eval.trl.txt
└── PA/
    └── PA/
        ├── ASVspoof2019_PA_train/flac/
        ├── ASVspoof2019_PA_dev/flac/
        ├── ASVspoof2019_PA_eval/flac/
        └── ASVspoof2019_PA_cm_protocols/
            ├── ASVspoof2019.PA.cm.train.trn.txt
            ├── ASVspoof2019.PA.cm.dev.trl.txt
            └── ASVspoof2019.PA.cm.eval.trl.txt
```

> **Note:** If you are running on Kaggle, the dataset will be available at `/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/` and no additional setup is needed.

---

## Dataset Statistics

| Partition | Bonafide | Spoof   | Spoof/Bonafide Ratio |
|-----------|----------|---------|----------------------|
| LA Train  | 2,580    | 22,800  | 8.8×                 |
| LA Dev    | 2,548    | 22,296  | 8.7×                 |
| LA Eval   | 7,355    | 63,882  | 8.7×                 |
| PA Train  | 5,400    | 48,600  | 9.0×                 |
| PA Dev    | 5,400    | 24,300  | 4.5×                 |
| PA Eval   | 18,090   | 116,640 | 6.4×                 |

---

## Attack Types

- **LA (Logical Access):** 19 spoofing systems (A01–A19). Attacks A01–A06 are *seen* (present in train/dev/eval), while A07–A19 are *unseen* (evaluation only), directly testing generalization to novel TTS/VC systems.
- **PA (Physical Access):** 27 replay attack configurations encoded as 3-character codes representing combinations of room acoustics, replay devices, and recording environments. All configurations are present across all splits.
