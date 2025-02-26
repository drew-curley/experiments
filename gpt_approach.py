import re
from collections import Counter

# Path to the uploaded word list
file_path = "/Users/drewcurley/Desktop/test/experiment/word_frequencies.txt"

def load_word_list(file_path):
    """
    Load words from the document.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().split()
    return words

def detect_potential_misspellings(words):
    """
    Detect potential misspellings using heuristic rules.
    """
    potential_misspellings = []
    for word in words:
        # Rule 1: Unusually long words
        if len(word) > 15:
            potential_misspellings.append(word)
        # Rule 2: Excessive repetition of a single character
        elif any(word.count(char) > 3 for char in set(word)):
            potential_misspellings.append(word)
        # Rule 3: Words with non-alphabetic characters
        elif not re.match(r'^[a-zA-Z]+$', word):
            potential_misspellings.append(word)
    return potential_misspellings

def find_top_misspellings(words):
    """
    Find the top 30 likely misspellings by frequency and pattern.
    """
    misspellings = detect_potential_misspellings(words)
    misspelling_counts = Counter(misspellings)
    return misspelling_counts.most_common(30)

# Load the word list
words = load_word_list(file_path)

# Find the top 30 most likely misspellings
top_misspellings = find_top_misspellings(words)

# Print the results
print("Top 30 Likely Misspelled Words:")
for word, count in top_misspellings:
    print(f"{word}: {count}")