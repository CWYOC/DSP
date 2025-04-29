import pandas as pd
import random
import csv

# Define schizophrenia-related genes with mutation probabilities
genes = {
    "DISC1": 0.015, "TCF4": 0.012, "BDNF": 0.010, "DRD2": 0.020,
    "COMT": 0.018, "GRIN2B": 0.014, "NRG1": 0.011, "RELN": 0.008,
    "DTNBP1": 0.013, "HTR2A": 0.016
}

# Chromosomes to include (fixed length of 1000 bp for simulation)
chromosome_names = [
    "Chromosome_1", "Chromosome_2", "Chromosome_3", "Chromosome_4", "Chromosome_5",
    "Chromosome_6", "Chromosome_7", "Chromosome_8", "Chromosome_9", "Chromosome_10",
    "Chromosome_11", "Chromosome_12", "Chromosome_13", "Chromosome_14", "Chromosome_15",
    "Chromosome_16", "Chromosome_17", "Chromosome_18", "Chromosome_19", "Chromosome_20",
    "Chromosome_21", "Chromosome_22", "Chromosome_X", "Chromosome_Y"
]

# Number of samples to generate
num_samples = 100000  # Adjust as needed

# Function to generate a 1000 base DNA sequence
def generate_dna_sequence(length=1000):
    bases = ["A", "T", "C", "G"]
    return ''.join(random.choices(bases, k=length))

# Output CSV path
file_path = "/Users/bearcheung/Documents/Year3/FYP/dna_gen/dna_100k.csv"

# Write data
with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Create headers
    headers = ["ID", "Gender"] + chromosome_names + list(genes.keys()) + ["Mental_Disorder"]
    writer.writerow(headers)
    
    for i in range(1, num_samples + 1):
        sample_id = f"Sample_{i}"
        gender = random.choice(["Male", "Female"])
        
        # Generate chromosome sequences (fixed length = 1000 bp)
        chromosomes = [generate_dna_sequence(1000) for _ in chromosome_names]
        
        # Gender-based chromosome_Y override
        if gender == "Male":
            chromosomes[-1] = generate_dna_sequence(1000)  # keep Y
        else:
            chromosomes[-1] = generate_dna_sequence(1000)  # simulate duplicate X
        
        # Generate gene mutation profile
        gene_mutations = [1 if random.random() < genes[g] else 0 for g in genes]
        
        # Mental disorder label: 1 if at least 3 mutations
        mental_disorder_label = 1 if sum(gene_mutations) >= 3 else 0
        
        writer.writerow([sample_id, gender] + chromosomes + gene_mutations + [mental_disorder_label])

print(f"✅ Dataset successfully saved as: {file_path}")
