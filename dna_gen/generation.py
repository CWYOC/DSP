import random
import uuid
import pandas as pd
import os
import gc
from tqdm import tqdm

# --- Fixed Chromosome lengths (all 1000 base pairs) ---
chromosome_lengths = {
    f"Chromosome_{i}": 1000 for i in range(1, 23)
}
chromosome_lengths["Chromosome_X"] = 1000
chromosome_lengths["Chromosome_Y"] = 1000

# --- DNA generation ---
def generate_dna(length):
    return ''.join(random.choices('ACGT', k=length))

def mutate(dna, mutation_rate=0.00001):
    dna_list = list(dna)
    for i in range(len(dna_list)):
        if random.random() < mutation_rate:
            dna_list[i] = random.choice('ACGT')
    return ''.join(dna_list)

def crossover(chrom1, chrom2):
    length = min(len(chrom1), len(chrom2))
    if length < 2:
        return chrom1
    point = random.randint(1, length - 1)
    return mutate(chrom1[:point] + chrom2[point:])

# --- Person simulation ---
def simulate_parent(gender):
    return {
        'ID': str(uuid.uuid4()),
        'Gender': gender,
        'Chromosomes': {
            chrom: generate_dna(length) for chrom, length in tqdm(chromosome_lengths.items(), desc=f"Generating {gender}", leave=False)
        }
    }

def simulate_child(p1, p2):
    return {
        'ID': str(uuid.uuid4()),
        'Gender': random.choice(['M', 'F']),
        'Chromosomes': {
            chrom: crossover(p1['Chromosomes'][chrom], p2['Chromosomes'][chrom])
            for chrom in tqdm(chromosome_lengths.keys(), desc="Crossover Child", leave=False)
        }
    }

# --- Save function ---
def save_family_to_csv(index, p1, p2, child, output_dir):
    row = {
        'P1_ID': p1['ID'],
        'P1_Gender': p1['Gender'],
        **{f'P1_{c}': p1['Chromosomes'][c] for c in chromosome_lengths},
        'P2_ID': p2['ID'],
        'P2_Gender': p2['Gender'],
        **{f'P2_{c}': p2['Chromosomes'][c] for c in chromosome_lengths},
        'Child_ID': child['ID'],
        'Child_Gender': child['Gender'],
        **{f'Child_{c}': child['Chromosomes'][c] for c in chromosome_lengths}
    }
    df = pd.DataFrame([row])
    filename = os.path.join(output_dir, f"fam_{index+1}.csv")
    df.to_csv(filename, index=False)

# --- Main ---
def main():
    num_families = 10000
    output_dir = "sample_families"
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(num_families), desc="Generating Families"):
        p1_gender = random.choice(['M', 'F'])
        p2_gender = 'F' if p1_gender == 'M' else 'M'
        p1 = simulate_parent(p1_gender)
        p2 = simulate_parent(p2_gender)
        child = simulate_child(p1, p2)

        save_family_to_csv(i, p1, p2, child, output_dir)

        # Clear memory after each family
        del p1, p2, child
        gc.collect()

    print(f"\n✅ Done: {num_families} families saved to {output_dir}/")

if __name__ == "__main__":
    main()
