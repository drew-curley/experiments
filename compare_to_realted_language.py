import re
from difflib import SequenceMatcher
from collections import Counter

# Example: Dummy corpus with Bantu and unknown language
corpus = [
    ("muntu", "mantuu"),  # Example Bantu and unknown language pair
    ("mzuri", "mzzuri"),
    ("chakula", "chakla"),
    ("rafiki", "rafikii"),
    ("nyama", "nyyama")
]

# Define a function to calculate edit distance similarity
def similarity_ratio(word1, word2):
    return SequenceMatcher(None, word1, word2).ratio()

# Threshold for detecting potential misspellings
SIMILARITY_THRESHOLD = 0.7  # Adjust based on your corpus

# Identify potential misspellings
potential_misspellings = []
for bantu_word, unknown_word in corpus:
    similarity = similarity_ratio(bantu_word, unknown_word)
    if similarity < SIMILARITY_THRESHOLD:
        potential_misspellings.append((unknown_word, bantu_word, similarity))

# Print results
print("Potential Misspellings:")
for unknown_word, bantu_word, similarity in potential_misspellings:
    print(f"Unknown: {unknown_word}, Bantu: {bantu_word}, Similarity: {similarity:.2f}")

# Optionally save results to a file
with open("potential_misspellings.txt", "w", encoding="utf-8") as f:
    for unknown_word, bantu_word, similarity in potential_misspellings:
        f.write(f"Unknown: {unknown_word}, Bantu: {bantu_word}, Similarity: {similarity:.2f}\n")