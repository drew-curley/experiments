import os
import csv
import re

# Define input and output directories
INPUT_DIR = '/Users/drewcurley/Desktop/test/experiment/'
OUTPUT_FILE = '/Users/drewcurley/Desktop/test/experiment/output.csv'

# Regex patterns for chapters and verses
CHAPTER_PATTERN = re.compile(r'\\c (\d+)')
VERSE_PATTERN = re.compile(r'\\v (\d+) (.+)')

# Process each file
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    # Loop through files numbered 41-67
    for file_num in range(41, 68):
        file_path = os.path.join(INPUT_DIR, f'{file_num}.usfm')
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()

        # Extract chapters and verses
        current_chapter = None
        for line in content:
            line = line.strip()
            chapter_match = CHAPTER_PATTERN.match(line)
            if chapter_match:
                current_chapter = int(chapter_match.group(1))

            verse_match = VERSE_PATTERN.match(line)
            if verse_match and current_chapter is not None:
                verse_num = int(verse_match.group(1))
                verse_text = verse_match.group(2)

                # Construct 8-digit ID: XXCCCVVV
                id_str = f'{file_num:02d}{current_chapter:03d}{verse_num:03d}'

                # Combine ID and verse text in the same cell
                combined_text = f'{id_str} {verse_text}'

                # Write to CSV (each verse in its own row)
                writer.writerow([combined_text])

print(f'CSV file created successfully: {OUTPUT_FILE}')