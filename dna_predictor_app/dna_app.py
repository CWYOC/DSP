import torch
import pandas as pd
import numpy as np

# --- Model Definitions ---

class GeneratorModel(torch.nn.Module):
    def __init__(self, input_size=48000):
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
        self.fc = torch.nn.Linear(48, 10)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# --- DNA Encoding ---
def encode_dna_sequence(seq):
    base_map = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}
    return [base_map.get(base, [0, 0, 0, 0]) for base in seq]

def load_parent_dna(file_path):
    df = pd.read_csv(file_path)
    chromosome_cols = [col for col in df.columns if 'Chromosome' in col and not col.startswith("Child")]
    data = df.iloc[0][chromosome_cols]
    one_hot_encoded = [b for seq in data for b in encode_dna_sequence(seq)]
    one_hot_array = np.array(one_hot_encoded, dtype=np.float32).flatten()
    return torch.tensor(one_hot_array, dtype=torch.float32).unsqueeze(0)

# --- Load Models ---
def load_models(input_tensor_shape):
    input_size = input_tensor_shape[1]
    generator = GeneratorModel(input_size=input_size)
    generator.load_state_dict(torch.load("next_gen_dna_nn_model.pth", map_location="cpu"))
    generator.eval()

    classifier = SymptomClassifier()
    classifier.load_state_dict(torch.load("checker.pth", map_location="cpu"))
    classifier.eval()
    return generator, classifier

# --- Run Prediction ---
def run_prediction(file_path):
    parent_tensor = load_parent_dna(file_path)
    generator, classifier = load_models(parent_tensor.shape)

    for i in range(100):
        child = generator(parent_tensor)
        prediction = classifier(child)
        prediction_np = prediction.squeeze().detach().numpy()
        print(f"Child {i+1}: {np.round(prediction_np, 3)}")

# --- Entry Point ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python dna_predict_cli.py path/to/input.csv")
        sys.exit(1)

    run_prediction(sys.argv[1])
