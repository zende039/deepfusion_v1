#!/usr/bin/python3
'''
author: Rushikesh
date: March 18, 2025
    This is a test.
    Description:
         This file processes the OokleSpeedTest files
'''

import numpy as np
import pandas as pd
import sys, os
import shutil
from tqdm import tqdm

sys.path.append(os.getcwd())
myPath = str(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(myPath, os.pardir)))

# Add the Helpers directory
helpers_path = os.path.abspath(os.path.join(myPath, "Helpers"))
sys.path.append(helpers_path)

# Add the Script directory
scripts_path = os.path.abspath(os.path.join(myPath, "Scripts"))
sys.path.append(scripts_path)



from Helpers.helpers import *
from Helpers.utils import plotme
from Helpers.constants import *
from Helpers.set_paths import *

import matplotlib.pyplot as plt
import re
import seaborn as sns
import os
from datetime import datetime
from datetime import datetime
import matplotlib.gridspec as gridspec
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from learnable import *

if Process_Xcal:
    flag = 'Xcal/'
    camera = 'single'
    dataFiles = getFiles(flag='Process')
    df_lst = []
    for f in tqdm(dataFiles):
        if 'Before' in f and 'single-video-ul-loop' in f    :
            # continue
            print(f)
            df = loadData(f)
            df_lst.append(df)


df = pd.concat(df_lst)
# Keep only rows with non-null HO duration values
df = df[df["Intra-NR HO Duration [sec]"].notnull()]

# Target column
target_col = "Intra-NR HO Duration [sec]"

# Drop columns that are IDs, timestamps, tech labels, or not useful
drop_cols = [col for col in df.columns if "TIME_STAMP" in col or
             "ts" in col or "HO Result" in col or "Event" in col or
             "Tech" in col or "HO Attempt" in col or "Measure Report" in col or
             "HO Source to Target" in col]

df = df.drop(columns=drop_cols)

# Drop any columns with more than 50% missing values
df = df.dropna(thresh=len(df) * 0.5, axis=1)

# Forward-fill or fill remaining NaNs with column mean
df = df.fillna(df.mean(numeric_only=True))

# Filter only numeric columns
df = df.select_dtypes(include=[np.number])

# Get the correlation row for the target and turn it into a DataFrame
target_corr_df = df.corr()[[target_col]].sort_values(by=target_col, key=abs, ascending=False)




# Plot heatmap
plt.figure(figsize=(8, len(target_corr_df) * 0.6))
sns.set(font_scale=1.2)
sns.heatmap(target_corr_df, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=0.5, linecolor='gray')

plt.title(f'Correlation with Target: {target_col}', fontsize=16)
plt.tight_layout()
plt.show()

# Select top-k features correlated with the target
kbase = 10  # You can change this value
top_k_features = target_corr_df.drop(index=target_col).head(kbase).index.tolist()

# Filter dataset to use only these top-k features
X = df[top_k_features]
y = df[target_col]



# # Separate features and target
# X = df.drop(columns=[target_col])
# y = df[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)

print(f"Features shape: {X_train_tensor.shape}, Target shape: {y_train_tensor.shape}")



