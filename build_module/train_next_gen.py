import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import gc

# --- DNA Encoding Function ---
def encode_dna_seq(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [mapping.get(base, 0) for base in seq]

# --- Model Definition ---
class GenePredictorNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Setup ---
csv_dir = "/Users/bearcheung/Documents/Year3/FYP/sample_families"
csv_files = sorted(glob(os.path.join(csv_dir, "*.csv")))

chromosomes = [f"Chromosome_{i}" for i in range(1, 23)] + ["Chromosome_X", "Chromosome_Y"]
feature_columns = [f"P1_{c}" for c in chromosomes] + [f"P2_{c}" for c in chromosomes]
label_columns = [f"Child_{c}" for c in chromosomes]

# --- Load and Encode All Data ---
all_X = []
all_y = []

print("📦 Loading and encoding data...")
for csv_path in tqdm(csv_files, desc="Reading CSVs"):
    df = pd.read_csv(csv_path)
    for i in range(len(df)):
        X_row = np.concatenate([encode_dna_seq(df[col][i]) for col in feature_columns])
        y_row = np.concatenate([encode_dna_seq(df[col][i]) for col in label_columns])
        all_X.append(X_row)
        all_y.append(y_row)

X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.float32)

# --- Create Dataset & DataLoader ---
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# --- Initialize Model ---
input_size = X.shape[1]
output_size = y.shape[1]
model = GenePredictorNN(input_size=input_size, output_size=output_size)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# --- Train Model ---
print("🚀 Training model...")
for epoch in range(5):
    total_loss = 0
    model.train()
    for X_batch, y_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/5"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

# --- Save Model ---
torch.save(model.state_dict(), "next_gen_dna_nn_model.pth")
print("✅ Final model saved as next_gen_dna_nn_model.pth")
