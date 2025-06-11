import pandas as pd
import subprocess
from collections import defaultdict
import sys
import os

# Read parallel.csv (columns 4 and 5, 0-based indices 3 and 4)
try:
    df = pd.read_csv("parallel.csv", header=None)
    df = df[[3, 4]]
    df.columns = ["English", "ZRL"]
except FileNotFoundError:
    print("Error: 'parallel.csv' not found in the current directory.")
    sys.exit(1)

# Write sentence pairs to parallel.txt for fast_align
parallel_file = "parallel.txt"
with open(parallel_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f"{row['English']} ||| {row['ZRL']}\n")

# Run fast_align to generate alignments
alignments_file = "alignments.txt"
try:
    # Use full path to fast_align if not in PATH, e.g., "/path/to/fast_align"
    subprocess.run(["fast_align", "-i", parallel_file, "-v", "-o", "-d"], 
                   stdout=open(alignments_file, "w", encoding="utf-8"), 
                   check=True)
except FileNotFoundError:
    print("Error: 'fast_align' executable not found. Install from https://github.com/clab/fast_align")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"Error running fast_align: {e}")
    sys.exit(1)

# Initialize dictionary for ZRL to English mappings
zrl_to_eng = defaultdict(set)

# Process alignments
try:
    with open(alignments_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            alignments = line.strip().split()
            eng_sent = df.iloc[index]["English"].lower()
            zrl_sent = df.iloc[index]["ZRL"].lower()
            eng_words = eng_sent.split()
            zrl_words = zrl_sent.split()
            for alignment in alignments:
                try:
                    i, j = map(int, alignment.split("-"))
                    if i < len(eng_words) and j < len(zrl_words):
                        eng_word = eng_words[i]
                        zrl_word = zrl_words[j]
                        zrl_to_eng[zrl_word].add(eng_word)
                except ValueError:
                    print(f"Skipping invalid alignment in line {index + 1}: {alignment}")
except FileNotFoundError:
    print(f"Error: Alignment file '{alignments_file}' not found.")
    sys.exit(1)

# Write dictionary to output.csv
output_file = "output.csv"
with open(output_file, "w", encoding="utf-8") as f:
    for zrl_word in sorted(zrl_to_eng.keys()):
        eng_words = sorted(zrl_to_eng[zrl_word])
        f.write(f"{zrl_word}," + ",".join(eng_words) + "\n")

print(f"Dictionary successfully written to '{output_file}'.")
