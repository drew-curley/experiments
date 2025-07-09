import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def detect_missing_tokens(eng_text, zrl_text, tokenizer, model, threshold=0.7, device="cuda"):
    tokenizer.src_lang = "eng_Latn"
    eng_inputs = tokenizer(eng_text, return_tensors="pt", add_special_tokens=False).to(device)
    tokenizer.src_lang = "zrl_ZRL"
    zrl_inputs = tokenizer(zrl_text, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        eng_embeds = model.get_input_embeddings()(eng_inputs.input_ids).squeeze(0)  # (n_eng, d)
        zrl_embeds = model.get_input_embeddings()(zrl_inputs.input_ids).squeeze(0)  # (n_zrl, d)

    # Normalize embeddings
    eng_norm = F.normalize(eng_embeds, dim=-1)
    zrl_norm = F.normalize(zrl_embeds, dim=-1)

    # Compute cosine similarities
    similarity = eng_norm @ zrl_norm.T  # (n_eng, n_zrl)
    max_scores, _ = similarity.max(dim=1)

    eng_tokens = tokenizer.convert_ids_to_tokens(eng_inputs.input_ids.squeeze(0))
    missing = [tok for tok, score in zip(eng_tokens, max_scores.cpu().tolist()) if score < threshold]

    return missing

def main():
    parser = argparse.ArgumentParser(description="Detect missing English tokens in ZRL translations using NLLB-3.3B embeddings.")
    parser.add_argument("--input_csv", required=True, help="Path to CSV with columns 'english' and 'zrl'.")
    parser.add_argument("--output_csv", required=True, help="Path to save output CSV with 'missing_tokens' column.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Cosine similarity threshold.")
    parser.add_argument("--device", default="cuda", help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df["missing_tokens"] = ""

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="eng_Latn", tgt_lang="zrl_ZRL")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B").to(args.device)
    model.eval()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            missing = detect_missing_tokens(
                row["english"], row["zrl"],
                tokenizer, model,
                threshold=args.threshold, device=args.device
            )
            df.at[idx, "missing_tokens"] = " ".join(missing)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            df.at[idx, "missing_tokens"] = "[ERROR]"

    df.to_csv(args.output_csv, index=False)
    print(f"Done. Output written to {args.output_csv}")

if __name__ == "__main__":
    main()

python detect_missing_tokens_nllb.py \
  --input_csv bitexts.csv \
  --output_csv missing_output.csv \
  --threshold 0.7 \
  --device cuda
