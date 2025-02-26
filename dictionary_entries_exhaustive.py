import os
import re
from collections import defaultdict

# Path to your Mac desktop
desktop_path = "/Users/drewcurley/Desktop/test/experiment/"
usfm_files = [f"{desktop_path}{i}.usfm" for i in range(41, 68)]  # Files 41.usfm to 67.usfm

# Dictionary to store word frequencies
word_freq = defaultdict(int)

# Regular expression for tokenizing words (ignores pure numerals)
word_pattern = re.compile(r"\b[a-zA-Z]+\b")

def process_file(file_path):
    """
    Processes a USFM file and updates the word frequency dictionary.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            # Find all words, normalize to lowercase
            words = word_pattern.findall(text.lower())
            for word in words:
                word_freq[word] += 1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process all USFM files
for file in usfm_files:
    if os.path.exists(file):
        process_file(file)
    else:
        print(f"File not found: {file}")

# Sort the word frequency dictionary by frequency
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# Save all word frequencies to a text file
output_file = f"{desktop_path}word_frequencies.txt"
with open(output_file, "w", encoding="utf-8") as out:
    for word, freq in sorted_word_freq:
        out.write(f"{word}: {freq}\n")

# Identify and save words that only occur once
words_occurring_once = [word for word, freq in word_freq.items() if freq == 1]
unique_words_file = f"{desktop_path}unique_words.txt"
with open(unique_words_file, "w", encoding="utf-8") as out:
    for word in words_occurring_once:
        out.write(f"{word}\n")

# Print summary
print(f"Word frequency dictionary saved to: {output_file}")
print(f"List of unique words (occurring once) saved to: {unique_words_file}")