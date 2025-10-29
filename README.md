# 🧠 ConvoReleNet – Hybrid Deep Network for Motor Imagery EEG Classification

This repository contains the official implementation of **ConvoReleNet**, a hybrid deep neural network combining **Convolutional**, **Transformer**, and **Recurrent (BiLSTM)** modules for **Motor Imagery EEG (MI-EEG)** classification.

The code accompanies the paper:

> **ConvoReleNet: A Multi-Branch CNN–Transformer–LSTM Hybrid Network with Transfer Learning for Motor Imagery EEG Decoding**  
> (Frontiers in Neuroscience, 2025)

---

## 🌟 Overview

ConvoReleNet is designed to capture **spatial**, **temporal**, and **contextual** dependencies in EEG signals.  
It integrates three complementary modules:

- **Convolutional layers** → learn localized spectral–spatial features.  
- **Transformer encoder** → models long-range temporal dependencies via self-attention.  
- **BiLSTM** → refines sequential patterns and stabilizes temporal learning.

The network supports:
- **Training from scratch** or **transfer learning** between datasets.  
- **Subject-wise cross-validation (LOSO-like)** evaluation.  
- Multiple activation variants (ELU, ReLU, Tanh).

---

## 🧩 Environment Setup

Tested environment:

| Library | Version |
|----------|----------|
| Python | 3.8 |
| PyTorch | 1.10 |
| NumPy | ≥1.21 |
| SciPy | ≥1.7 |
| scikit-learn | ≥1.0 |
| MNE | ≥0.24 |
| matplotlib | ≥3.4 |

### Installation
```bash
pip install torch==1.10 numpy scipy scikit-learn mne matplotlib
```

You can also use a virtual environment:

```bash
python3 -m venv convoenv
source convoenv/bin/activate      # Linux / macOS
convoenv\Scripts\activate         # Windows
pip install torch==1.10 numpy scipy scikit-learn mne matplotlib

```

---

## 📦 Dataset Preparation

This project uses publicly available **Motor Imagery EEG datasets**.  
Download any or all of the following:

| Dataset | Description | Download Link |
|----------|--------------|---------------|
| **BCI Competition IV – 2a** | 4-class MI EEG (22 channels) | [Download](https://service.tib.eu/ldmservice/dataset/bci-competition-iv-2a) |
| **BCI Competition IV – 2b** | 2-class MI EEG (3 channels) | [Download](https://service.tib.eu/ldmservice/dataset/bci-competition-iv-2b) |
| **Weibo 2014 Dataset** | MI EEG benchmark | [Download](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Weibo2014.html) |
| **PhysioNet EEG Motor Movement/Imagery** | Open EEG dataset | [Download](https://physionet.org/content/eegmmidb/1.0.0/) |

After downloading, organize your folder as follows:

```
/datasets/
   ├── BNCI_IV_2a/
   ├── BNCI_IV_2b/
   ├── Weibo2014/
   └── PhysioNet_MMI/
```

The code automatically detects datasets from this directory.

---

## ⚙️ Running the Model

### 1️⃣ Train From Scratch
```bash
python ConvoReleNet_MIEEG.py --dataset IV2a --mode scratch
```

### 2️⃣ Transfer Learning (Cross-Dataset)
```bash
python ConvoReleNet_MIEEG.py --pretrained IV2b --finetune IV2a --mode transfer
```

### 3️⃣ Subject-Wise Cross-Validation
```bash
python ConvoReleNet_MIEEG.py --dataset IV2a --cv subjectwise
```

Add `--device cpu` if you do not have a GPU.

---

## 📊 Output

After training, the script generates:

- Accuracy, F1-score, and Cohen’s κ (per subject)  
- Confusion matrices  
- Training and validation curves  
- CSV summary files and plots in the `results/` directory

---

## 🧠 Notes for Beginners

- You do **not** need prior EEG experience. Follow the steps above step-by-step.  
- The model and preprocessing pipelines are fully implemented in PyTorch.  
- Training may take ~15–30 min on GPU, or longer on CPU.  
- You can change hyperparameters in the script directly (e.g., number of filters, heads, layers).  

---

## 🧾 Citation

If you use this code or derivative work, **you must cite** the associated paper:

```
@article{ConvoReleNet2025,
  title={ConvoReleNet: A Multi-Branch CNN–Transformer–LSTM Hybrid Network with Transfer Learning for Motor Imagery EEG Decoding},
  author={Your Name and Co-authors},
  journal={Frontiers in Neuroscience},
  year={2025}
}
```

---

## ⚖️ License

This work is licensed under a [**Creative Commons Attribution 4.0 International License (CC BY 4.0)**](https://creativecommons.org/licenses/by/4.0/).

You are free to use, share, and modify the code **as long as you give appropriate credit and cite the original publication**.

Example acknowledgment:

> “This research used the ConvoReleNet framework (Kyzyrkanov et al., 2025) available under CC BY 4.0.”

---

## 📫 Contact

**Author:** Your Name  
**Email:** your.email@example.com  
**Repository:** [https://github.com/YourUsername/ConvoReleNet-MI-EEG](https://github.com/YourUsername/ConvoReleNet-MI-EEG)

For questions or collaboration, feel free to open an issue or reach out via email.

---

### ✅ Summary

- Python 3.8 / Torch 1.10  
- Download EEG datasets (links above)  
- Run `ConvoReleNet_MIEEG.py` to reproduce results  
- Cite the paper if you use the code  

Enjoy experimenting with ConvoReleNet!
