# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 01:45:25 2025

@author: Zhenis
"""

# -*- coding: utf-8 -*-
import pickle
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import precision_recall_fscore_support


# 1) Band-pass helper
def bandpass_epoch(epoch, lowcut, highcut, fs, order=4):
    """
    epoch: 2D array (n_chans × n_times)
    returns filtered epoch of same shape
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # axis=1 filters over time dimension
    return filtfilt(b, a, epoch, axis=1)


# 2) Load your list of 9 subjects (each an MNE Epochs)
with open('aBNCI2014001R.pickle', 'rb') as f:
    epochs_list = pickle.load(f)

sfreq = 80.0   # sampling rate
low_cut = 8.0    # Hz
high_cut = 30.0   # Hz

# 3) Loop subjects, filter & standardize
standardized_data = []  # will hold tuples (X_std, events)

for subj_idx, epochs in enumerate(epochs_list, start=1):
    print(f"Processing Subject {subj_idx}")

    # raw data and labels
    X = epochs.get_data()       # shape (n_epochs, n_chans, n_times)
    events = epochs.events      # keep for downstream labels

    # band-pass filter each epoch
    X_filt = np.stack([
        bandpass_epoch(ep, low_cut, high_cut, sfreq)
        for ep in X
    ], axis=0)                  # (n_epochs, n_chans, n_times)

    # standardize per-subject: fit on all epochs, then transform
    n_epochs, n_chans, n_times = X_filt.shape
    X_flat = X_filt.reshape(n_epochs, -1)              # (n_epochs, n_chans*n_times)
    scaler = StandardScaler().fit(X_flat)
    X_std = scaler.transform(X_flat).reshape(n_epochs, n_chans, n_times)

    standardized_data.append((X_std, events))

print("All subjects filtered and standardized.")


# 4) ConvoReleNet: CNN → Transformer → Bi-LSTM with LayerNorm & Gradient Clipping
class ConvoReleNet(nn.Module):
    def __init__(self, n_chans, n_times, n_classes):
        super().__init__()

        # Deep4-style branch
        self.deep_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(10, 1), padding=(5, 0)),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(1, n_chans)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.4),
            nn.Conv2d(32, 64, kernel_size=(10, 1), padding=(5, 0)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.4),
        )

        # EEGNetv4-style shallow branch
        self.shallow_branch = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, n_chans), bias=False),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 80, kernel_size=(25, 1)),
            nn.BatchNorm2d(80),
            nn.ELU(),
            nn.AvgPool2d((3, 1)),
            nn.Dropout(0.4),
        )

        # Fuse branch outputs
        self.fuse_conv1d = nn.Sequential(
            nn.Conv1d(64 + 80, 128, kernel_size=1),
            nn.ELU(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)

        # Bi-LSTM
        self.bi_lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        d = self.deep_branch(x).squeeze(-1)
        s = self.shallow_branch(x).squeeze(-1)
        T = min(d.size(2), s.size(2))
        d, s = d[:, :, :T], s[:, :, :T]
        f = torch.cat([d, s], dim=1)
        f = self.fuse_conv1d(f)
        t = f.permute(2, 0, 1)
        t = self.transformer(t)
        t = t.permute(1, 0, 2)
        out, _ = self.bi_lstm(t)
        rep = out.mean(dim=1)
        return self.classifier(rep)


# 5) DataLoader helper
def make_loader(Xa, ya, batch_size, shuffle):
    tX = torch.from_numpy(Xa).float().unsqueeze(1).permute(0, 1, 3, 2).to(device)
    ty = torch.from_numpy(ya).long().to(device)
    return DataLoader(TensorDataset(tX, ty), batch_size=batch_size, shuffle=shuffle)


# 6) Hyperparameters
best_lr = 5e-4
best_bs = 32
best_epochs = 150
weight_decay = 1e-5
early_stopping_patience = 20
criterion = nn.CrossEntropyLoss()

# 7) Load Data from aBNCI2014004R
with open("aBNCI2014004R.pickle", "rb") as f:
    epochs_list = pickle.load(f)

standardized_data = [(ep.get_data(), ep.events) for ep in epochs_list]

# 8) Pretraining on all subjects
all_X, all_y = [], []
for X, events in standardized_data:
    all_X.append(X)
    all_y.append(events[:, 2] - 1)

all_X = np.concatenate(all_X, axis=0)
all_y = np.concatenate(all_y, axis=0)

n_t, n_ch, n_tm = all_X.shape
scaler_all = StandardScaler().fit(all_X.reshape(n_t, -1))
all_X = scaler_all.transform(all_X.reshape(n_t, -1)).reshape(n_t, n_ch, n_tm)

X_tr_all, X_te_all, y_tr_all, y_te_all = train_test_split(
    all_X, all_y, test_size=0.2, random_state=42, stratify=all_y)

train_loader_all = make_loader(X_tr_all, y_tr_all, best_bs, True)
test_loader_all = make_loader(X_te_all, y_te_all, best_bs, False)

# 9) Training Function with F1 Score Calculation
def train_model(model, train_loader, test_loader, epochs,
                optimizer, scheduler, criterion, device,
                early_stopping_patience=20):
    best_acc = 0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        acc = 100 * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

        print(f"Epoch {epoch:3d} | loss {total_loss/len(train_loader):.4f} | test acc {acc:.2f}% | F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping!")
            break

    return best_acc, best_state, f1


# 10) Pretraining
print("=== Pretraining on all subjects ===")
model_pre = ConvoReleNet(n_chans=n_ch, n_times=n_tm, n_classes=len(np.unique(all_y))).to(device)
opt_all = optim.AdamW(model_pre.parameters(), lr=best_lr, weight_decay=weight_decay)
sched_all = OneCycleLR(opt_all, max_lr=best_lr, steps_per_epoch=len(train_loader_all), epochs=best_epochs)

best_acc_all, best_state_all, f1_all = train_model(
    model_pre, train_loader_all, test_loader_all,
    best_epochs, opt_all, sched_all, criterion, device,
    early_stopping_patience
)

torch.save(best_state_all, "pretrained_model_4004R.pth")
print(f"Pretraining done: best acc {best_acc_all:.2f}% | F1: {f1_all:.4f}")

# 11) Fine-tuning per Subject
results = []
for idx, (X, events) in enumerate(standardized_data, start=1):
    print(f"\n=== Fine-tuning Subject {idx} ===")
    y = events[:, 2] - 1

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    n_t, n_ch, n_tm = X_tr.shape
    scaler = StandardScaler().fit(X_tr.reshape(n_t, -1))
    X_tr = scaler.transform(X_tr.reshape(n_t, -1)).reshape(n_t, n_ch, n_tm)
    X_te = scaler.transform(X_te.reshape(X_te.shape[0], -1)).reshape(X_te.shape[0], n_ch, n_tm)

    tr_loader = make_loader(X_tr, y_tr, best_bs, True)
    te_loader = make_loader(X_te, y_te, best_bs, False)

    model = ConvoReleNet(n_chans=n_ch, n_times=n_tm, n_classes=len(np.unique(y))).to(device)
    model.load_state_dict(torch.load("pretrained_model_4004R.pth"))

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=best_lr, weight_decay=weight_decay)
    sched = OneCycleLR(opt, max_lr=best_lr, steps_per_epoch=len(tr_loader), epochs=best_epochs)

    best_acc, best_state, f1 = train_model(
        model, tr_loader, te_loader, best_epochs,
        opt, sched, criterion, device, early_stopping_patience
    )

    torch.save(best_state, f"best_model_subj_{idx}_4004R.pth")
    results.append({"Subject": idx, "Accuracy": best_acc, "F1": f1})

print("\n===== Summary =====")
for r in results:
    print(f"Subject {r['Subject']}: Acc={r['Accuracy']:.2f}% | F1={r['F1']:.4f}")
