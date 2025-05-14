"""
author: Rushikesh
date: April 28, 2025

"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ========== Load and preprocess ==========
def load_and_label(file_path, phase):
    df = pd.read_csv(file_path, sep="\t")
    df["Phase"] = phase
    return df

# Load files
before_df = load_and_label("HO_Duration/ProcessedData/single-video-TEST1-M1_M7000_Before_HO.txt", "before")
during_df = load_and_label("HO_Duration/ProcessedData/single-video-TEST1-M1_M7000_During_HO.txt", "during")
after_df = load_and_label("HO_Duration/ProcessedData/single-video-TEST1-M1_M7000_After_HO.txt", "after")

# Features and target
feature_cols = [
    "PCell Serving SS-RSRP [dBm]",
    "PCell Serving SS-RSRQ [dB]",
    "PCell Serving SS-SINR [dB]",
    "PCell Pathloss [dB]",
    "API GPS Info Speed"
]
target_col = "Intra-NR HO Duration [sec]"

# Combine and filter
df = pd.concat([before_df, during_df, after_df], ignore_index=True)
df = df[feature_cols + [target_col]].dropna()
df = df[df[target_col] > 0]  # Remove invalid (zero or negative) durations

# Feature matrix and target vector
X = df[feature_cols].values
y = np.log1p(df[target_col].values)  # log-transform target

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature and target scaling
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))

# Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

# ========== Model definition ==========
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# ========== Training function ==========
def train_model(X_train, X_val, y_train, y_val):
    model = MLP(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience, wait = 20, 0
    train_losses, val_losses = [], []

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
        scheduler.step(val_loss)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            best_model = model.state_dict()
        else:
            wait += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ========== Evaluation ==========
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_val).numpy()

    pred_log = y_scaler.inverse_transform(pred_scaled).flatten()
    y_true_log = y_scaler.inverse_transform(y_val.numpy()).flatten()

    y_pred = np.expm1(pred_log)
    y_true = np.expm1(y_true_log)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\nFinal Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # ========== Visualizations ==========
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True HO Duration")
    plt.ylabel("Predicted HO Duration")
    plt.title("Prediction vs True")
    plt.grid(True)
    plt.show()

# ========== Run Training ==========
train_model(X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor)
