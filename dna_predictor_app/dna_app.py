import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import pandas as pd
import numpy as np

# --- Model Definitions ---

class GeneratorModel(torch.nn.Module):
    def __init__(self, input_size=48000):  # Updated dynamically later
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 48)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SymptomClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(48, 10)  # 10 output symptoms

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


# --- DNA Encoding Functions ---

def encode_dna_sequence(seq):
    base_map = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1]
    }
    return [base_map.get(base, [0, 0, 0, 0]) for base in seq]

def load_parent_dna(file_path):
    df = pd.read_csv(file_path)
    # Use all chromosome columns that are not from child data
    chromosome_cols = [col for col in df.columns if 'Chromosome' in col and not col.startswith("Child")]
    data = df.iloc[0][chromosome_cols]

    one_hot_encoded = []
    for seq in data:
        one_hot_encoded.extend(encode_dna_sequence(seq))

    one_hot_array = np.array(one_hot_encoded, dtype=np.float32).flatten()
    tensor = torch.tensor(one_hot_array, dtype=torch.float32).unsqueeze(0)
    return tensor


# --- Load Models Dynamically ---

def load_models(input_tensor_shape):
    input_size = input_tensor_shape[1]

    generator = GeneratorModel(input_size=input_size)
    generator.load_state_dict(torch.load("/Users/bearcheung/Documents/Year3/FYP/dna_predictor_app/modules/next_gen_dna_nn_model.pth", map_location="mps"))
    generator.eval()

    classifier = SymptomClassifier()
    classifier.load_state_dict(torch.load("/Users/bearcheung/Documents/Year3/FYP/dna_predictor_app/modules/checker.pth", map_location="mps"))
    classifier.eval()

    return generator, classifier


# --- Prediction Logic ---

def run_prediction(file_path, result_text):
    try:
        parent_tensor = load_parent_dna(file_path).to("mps")
        generator, classifier = load_models(parent_tensor.shape)

        result_text.delete("1.0", tk.END)
        for i in range(100):
            child = generator(parent_tensor)
            prediction = classifier(child)
            prediction_np = prediction.squeeze().detach().cpu().numpy()
            result_text.insert(tk.END, f"Child {i+1}: {np.round(prediction_np, 3)}\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# --- GUI Setup ---

root = tk.Tk()
root.title("Next Gen DNA Predictor")
root.geometry("1200x720")

file_path_var = tk.StringVar()

frame = tk.Frame(root)
frame.pack(pady=10)

def browse_file():
    file_path = filedialog.askopenfilename()
    file_path_var.set(file_path)

entry = tk.Entry(frame, textvariable=file_path_var, width=50)
entry.pack(side=tk.LEFT, padx=5)

browse_button = tk.Button(frame, text="Browse", command=browse_file)
browse_button.pack(side=tk.LEFT)

run_button = tk.Button(root, text="Run Prediction", command=lambda: run_prediction(file_path_var.get(), result_text))
run_button.pack(pady=10)

result_text = tk.Text(root, wrap=tk.WORD, height=25)
result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()
