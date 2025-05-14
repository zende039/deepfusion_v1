import os, glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
import warnings

# ========== 1. Load all data and label ==========
def load_all_data(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.txt"))
    print(f"📁 Found {len(all_files)} txt files in: {folder_path}")
    all_dfs = []

    for file_path in all_files:
        filename = os.path.basename(file_path).lower()
        if "before" in filename:
            phase = "before"
        elif "during" in filename:
            phase = "during"
        elif "after" in filename:
            phase = "after"
        else:
            print(f"❌ Skipping unrecognized file: {filename}")
            continue

        device = "unknown"
        for tag in ["m7000", "m7001", "m7002", "ul-loop"]:
            if tag in filename:
                device = tag
                break

        try:
            df = pd.read_csv(file_path, sep="\t")
            if df.empty:
                continue
            df["Phase"] = phase
            df["Device"] = device
            all_dfs.append(df)
            print(f"✅ Loaded {filename} with shape {df.shape}")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    if not all_dfs:
        raise FileNotFoundError("No valid dataframes loaded.")

    return pd.concat(all_dfs, ignore_index=True)

# Load data
folder = "/home/zz-lab-autoware/Downloads/HO_Duration/HO_Duration/ProcessedData"
df = load_all_data(folder)

# ========== 2. Preprocessing ==========
base_features = [
    "PCell Serving SS-RSRP [dBm]", "PCell Serving SS-RSRQ [dB]",
    "PCell Serving SS-SINR [dB]", "PCell Pathloss [dB]", "API GPS Info Speed"
]
target_col = "Intra-NR HO Duration [sec]"
df = df[base_features + [target_col, "Phase", "Device"]].dropna()
df = df[df[target_col] > 0]

# ========== 3. Feature Engineering ==========
df["RSRP_minus_Pathloss"] = df["PCell Serving SS-RSRP [dBm]"] - df["PCell Pathloss [dB]"]
df["SINR_by_Speed"] = df["PCell Serving SS-SINR [dB]"] / (df["API GPS Info Speed"] + 1)
df["PCell Serving SS-SINR [dB]"] = df["PCell Serving SS-SINR [dB]"].replace([-np.inf, np.inf], np.nan).fillna(-1)
df["PCell Pathloss [dB]"] = df["PCell Pathloss [dB]"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df["log_SINR"] = np.log1p(np.clip(df["PCell Serving SS-SINR [dB]"], a_min=-1, a_max=None))
    df["log_Pathloss"] = np.log1p(np.clip(df["PCell Pathloss [dB]"], a_min=0, a_max=None))

num_before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"🧹 Dropped {num_before - len(df)} rows due to inf or NaN values")
df = df.reset_index(drop=True)

# One-hot encode
onehot = OneHotEncoder(sparse_output=False)
encoded = onehot.fit_transform(df[["Phase", "Device"]])
encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out(["Phase", "Device"]))

X_df = pd.concat([
    df[["PCell Serving SS-RSRP [dBm]", "PCell Serving SS-RSRQ [dB]", "API GPS Info Speed",
         "RSRP_minus_Pathloss", "SINR_by_Speed", "log_SINR", "log_Pathloss"]].reset_index(drop=True),
    encoded_df.reset_index(drop=True)
], axis=1)

X = X_df.values
y = df[target_col].values

# Plot imbalance and save
plt.figure(figsize=(8, 4))
plt.hist(y, bins=20, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.title("Distribution of HO Durations")
plt.xlabel("HO Duration (s)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("ho_duration_distribution.png")
plt.close()

# ========== 4. Handle imbalance via stratified binning ==========
bins = np.quantile(y, q=np.linspace(0, 1, 5))
y_bins = np.digitize(y, bins)

# Transform target
y_log = np.log1p(np.clip(y, a_min=0, a_max=None))
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y_log.reshape(-1, 1))

x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

X_train, X_val, y_train, y_val, yb_train, yb_val = train_test_split(
    X_scaled, y_scaled, y_bins, test_size=0.2, stratify=y_bins, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

sample_weights = compute_sample_weight(class_weight='balanced', y=yb_train)
sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

# ========== 5. Model ==========
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# ========== 6. Train & Evaluate ==========
def train_model(X_train, X_val, y_train, y_val):
    model = MLP(X_train.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=0.0007, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()  # Huber loss

    best_loss = float("inf")
    patience = 20
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train)
        loss_train = (criterion(pred_train, y_train) * sample_weights_tensor.view(-1, 1)).mean()
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            loss_val = criterion(pred_val, y_val)

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            best_model = model.state_dict()
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss_train.item():.4f}, Val Loss = {loss_val.item():.4f}")

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

    print(f"\n📊 Final Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    # ========== Plot ==========
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True HO Duration")
    plt.ylabel("Predicted HO Duration")
    plt.title("Prediction vs True")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_vs_true.png")
    plt.close()

    # Save CSV
    pd.DataFrame({
        "True_HO_Duration": y_true,
        "Predicted_HO_Duration": y_pred
    }).to_csv("predictions_output_optimized.csv", index=False)
    print("✅ Saved to predictions_output_optimized.csv")

# Run
train_model(X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor)
