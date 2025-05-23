# align_nt.py

import pandas as pd
import csv
import torch
from simalign import SentenceAligner

def main():
    # 1) Load your CSV (no header), pick cols 5 & 6 (0-based idx: 4 & 5)
    df = pd.read_csv("nt.csv", header=None, dtype=str, keep_default_na=False)
    src_sents = df.iloc[:,4].tolist()
    tgt_sents = df.iloc[:,5].tolist()

    # 2) Initialize SimAlign with a high-quality multilingual model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aligner = SentenceAligner(
        model   = "xlm-roberta-large",
        token_type       = "bpe",
        matching_methods = ["itermax"],
        device           = device,
        batch_size       = 16
    )

    # 3) Prepare output CSV
    with open("nt_word_alignments.csv", "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["src", "tgt", "alignment"])  # header

        # 4) Loop and align
        for src, tgt in zip(src_sents, tgt_sents):
            src_tokens = src.split()
            tgt_tokens = tgt.split()
            aligns = aligner.get_word_aligns(src_tokens, tgt_tokens)["itermax"]
            # `aligns` is a list of (src_idx, tgt_idx) tuples

            writer.writerow([src, tgt, aligns])

    print("✅ Done!  Alignments written to nt_word_alignments.csv")

if __name__ == "__main__":
    main()
