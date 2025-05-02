import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import torch
import torch.nn as nn
import random
from collections import Counter

# -----------------------------
# MODEL ARCHITECTURES
# -----------------------------

class DNA_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(46 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1000 * 4)  # Output shape: (4000,)
        )

    def forward(self, x):
        return self.net(x)

class SymptomChecker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1000 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10),  # Output: 10 gene probabilities
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# CONSTANTS
# -----------------------------

GENES = ["DISC1", "TCF4", "BDNF", "DRD2", "COMT", "GRIN2B", "NRG1", "RELN", "DTNBP1", "HTR2A"]

# -----------------------------
# LOAD MODELS
# -----------------------------

dna_model = DNA_Generator()
dna_model.load_state_dict(torch.load("next_gen_dna_nn_model.pth", map_location="cpu"))
dna_model.eval()

checker_model = SymptomChecker()
checker_model.load_state_dict(torch.load("checker.pth", map_location="cpu"))
checker_model.eval()

# -----------------------------
# GUI SETUP
# -----------------------------

root = tk.Tk()
root.title("DNA Generator & Symptom Checker")

# -----------------------------
# FILE LOADING
# -----------------------------

def load_file():
    file_path = filedialog.askopenfilename(
        title="Select Parent DNA CSV",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        return

    try:
        df = pd.read_csv(file_path)
        messagebox.showinfo("Loaded", f"{len(df)} row(s) loaded.")
        start_prediction(df)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# -----------------------------
# DNA GENERATION
# -----------------------------

def generate_dna_sequences(parents_df, count=100):
    dna_outputs = []

    for _ in range(count):
        row = parents_df.sample(1).iloc[0]
        parent1 = row[2:46].values  # Chromosomes 1–22, X, Y (parent1)
        parent2 = row[48:92].values  # Chromosomes 1–22, X, Y (parent2)

        input_tensor = torch.tensor(list(parent1) + list(parent2), dtype=torch.float32)
        output = dna_model(input_tensor)  # Shape: [4000]
        one_hot = output.view(1000, 4)    # Shape: [1000, 4]

        probs = one_hot.softmax(dim=1)    # Apply softmax across ACGT
        sampled = torch.multinomial(probs, num_samples=1).squeeze()
        dna_string = ''.join(['ACGT'[i] for i in sampled.tolist()])
        dna_outputs.append((dna_string, probs.flatten()))

    return dna_outputs

# -----------------------------
# SYMPTOM CHECKING
# -----------------------------

def check_symptoms(flat_dna_probs_list):
    results = []
    for prob_vector in flat_dna_probs_list:
        pred = checker_model(prob_vector)
        binary = (pred > 0.5).int().tolist()
        results.append(dict(zip(GENES, binary)))
    return results

def calculate_symptom_percentages(results):
    gene_counts = Counter()
    total = len(results)
    for r in results:
        for gene, flag in r.items():
            gene_counts[gene] += flag
    return {gene: gene_counts[gene] / total * 100 for gene in GENES}

# -----------------------------
# RESULT DISPLAY
# -----------------------------

def display_percentages(percentages):
    win = tk.Toplevel()
    win.title("Symptom Detection Summary")
    tk.Label(win, text="Gene", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=10)
    tk.Label(win, text="Detected in (%)", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=10)

    for i, (gene, pct) in enumerate(sorted(percentages.items()), start=1):
        tk.Label(win, text=gene).grid(row=i, column=0, sticky='w', padx=10)
        tk.Label(win, text=f"{pct:.1f}%").grid(row=i, column=1, sticky='e', padx=10)

# -----------------------------
# MAIN PIPELINE
# -----------------------------

def start_prediction(parents_df):
    generated = generate_dna_sequences(parents_df)
    sequences, flat_probs = zip(*generated)
    results = check_symptoms(flat_probs)
    percentages = calculate_symptom_percentages(results)
    display_percentages(percentages)

# -----------------------------
# MAIN WINDOW
# -----------------------------

tk.Button(root, text="Load Parent DNA CSV", command=load_file, padx=20, pady=10).pack(pady=30)
root.mainloop()
