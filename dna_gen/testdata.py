import random
import uuid
import pandas as pd
from tqdm import tqdm

# Full chromosome lengths
chromosome_lengths = {
    "Chromosome_1": 1000,
    "Chromosome_2": 1000,
    "Chromosome_3": 1000,
    "Chromosome_4": 1000,
    "Chromosome_5": 1000,
    "Chromosome_6": 1000,
    "Chromosome_7": 1000,
    "Chromosome_8": 1000,
    "Chromosome_9": 1000,
    "Chromosome_10": 1000,
    "Chromosome_11": 1000,
    "Chromosome_12": 1000,
    "Chromosome_13": 1000,
    "Chromosome_14": 1000,
    "Chromosome_15": 1000,
    "Chromosome_16": 1000,
    "Chromosome_17": 1000,
    "Chromosome_18": 1000,
    "Chromosome_19": 1000,
    "Chromosome_20": 1000,
    "Chromosome_21": 1000,
    "Chromosome_22": 1000,
    "Chromosome_X": 1000,
    "Chromosome_Y": 1000
}

def generate_dna(length):
    return ''.join(random.choices("ACGT", k=length))

def simulate_person(gender, prefix):
    chromosomes = {}
    for chrom, length in tqdm(chromosome_lengths.items(), desc=f"Generating {prefix}", unit="chrom"):
        chromosomes[chrom] = generate_dna(length)
    return {
        "ID": str(uuid.uuid4()),
        "Gender": gender,
        "Chromosomes": chromosomes
    }

# Generate parents with full genome
parent1 = simulate_person("M", "Parent 1")
parent2 = simulate_person("F", "Parent 2")

# Save to CSV
chroms = list(chromosome_lengths.keys())
header = (
    ['P1_ID', 'P1_Gender'] + [f"P1_{c}" for c in chroms] +
    ['P2_ID', 'P2_Gender'] + [f"P2_{c}" for c in chroms]
)

row = (
    [parent1["ID"], parent1["Gender"]] + [parent1["Chromosomes"][c] for c in chroms] +
    [parent2["ID"], parent2["Gender"]] + [parent2["Chromosomes"][c] for c in chroms]
)

df = pd.DataFrame([row], columns=header)
df.to_csv("test_parents_full.csv", index=False)

print("✅ Full-length parent DNA saved as 'test_parents_full.csv'")
