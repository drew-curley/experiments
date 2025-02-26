# File path
file_path = "/Users/drewcurley/Desktop/test/experiment/unique_words.txt"

#### Trimming to unique words only
try:
    # Read the file
    with open(file_path, "r", encoding="utf-8") as file:
        # Read all lines into a list
        words = file.readlines()
        
    # Count the number of words (each word is on a new line)
    word_count = len(words)
    
    # Print the result
    print(f"The number of words in the file is: {word_count}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

#### Limiting the number of unique words to display

import re

# Path to the file containing the word list
file_path = "/Users/drewcurley/Desktop/test/experiment/unique_words.txt"

# Common patterns for potential typing errors
keystroke_patterns = [
    r"[zxqjk]{2,}",   # Uncommon bigrams or clusters
    r"(.)\1{2,}",     # Repeated characters (e.g., "worrdd")
    r"[^aeiouy]{5,}", # Long strings without vowels (e.g., "trhgh")
]

# Compile the patterns into a single regex
error_regex = re.compile("|".join(keystroke_patterns))

def is_likely_misspelled(word):
    """
    Checks if a word matches likely misspelling patterns.
    """
    return bool(error_regex.search(word))

try:
    # Read the file
    with open(file_path, "r", encoding="utf-8") as file:
        words = [line.strip() for line in file]

    # Filter the words for likely misspellings
    likely_misspellings = [word for word in words if is_likely_misspelled(word)]

    # Save the misspelled words to a new file
    output_path = "/Users/drewcurley/Desktop/test/experiment/likely_misspelled_words.txt"
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(likely_misspellings))

    # Print results
    print(f"Total words checked: {len(words)}")
    print(f"Likely misspelled words found: {len(likely_misspellings)}")
    print(f"Misspelled words saved to: {output_path}")
    
    # Print the new number of potentially misspelled words
    print(f"New count of potentially misspelled words: {len(likely_misspellings)}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

