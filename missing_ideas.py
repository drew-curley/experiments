import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embeddings(texts, tokenizer, model, lang_code, device):
    tokenizer.src_lang = lang_code
    encoded = tokenizer(texts, return_offsets_mapping=True, padding=True, truncation=True,
                        return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(encoded.input_ids)
    return encoded, embeddings

def compute_similarity(eng_embeds, zrl_embeds):
    eng_norm = F.normalize(eng_embeds, dim=-1)
    zrl_norm = F.normalize(zrl_embeds, dim=-1)
    sim = torch.matmul(eng_norm, zrl_norm.transpose(1, 2))  # (batch_size, n_eng, n_zrl)
    max_scores, _ = sim.max(dim=2)  # Max over ZRL tokens
    return max_scores

def detect_missing_words_batch(eng_batch, zrl_batch, tokenizer, model, threshold, device):
    eng_encoded, eng_embeds = get_embeddings(eng_batch, tokenizer, model, "eng_Latn", device)
    zrl_encoded, zrl_embeds = get_embeddings(zrl_batch, tokenizer, model, "zrl_ZRL", device)

    max_sim = compute_similarity(eng_embeds, zrl_embeds)  # (batch_size, n_eng)
    missing_words_batch = []
    sim_scores_batch = []

    for i in range(len(eng_batch)):
        tokens = tokenizer.convert_ids_to_tokens(eng_encoded.input_ids[i])
        offsets = eng_encoded["offset_mapping"][i].cpu().tolist()
        text = eng_batch[i]
        scores = max_sim[i].cpu().tolist()

        missing_words = set()
        detailed_scores = []

        for j, score in enumerate(scores):
            if offsets[j][0] == offsets[j][1]:  # Skip padding or empty spans
                continue
            word = text[offsets[j][0]:offsets[j][1]]
            detailed_scores.append(f"{word}:{score:.2f}")
            if score < threshold:
                missing_words.add(word.strip())

        missing_words_batch.append(" ".join(missing_words))
        sim_scores_batch.append(" | ".join(detailed_scores))

    return missing_words_batch, sim_scores_batch

def main():
    parser = argparse.ArgumentParser(description="Batch detect missing English words in ZRL translations using NLLB-3.3B.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV with 'Source' and 'zrl' columns.")
    parser.add_argument("--output_csv", required=True, help="Path to save output CSV with 'missing_tokens' and 'token_similarities'.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Cosine similarity threshold.")
    parser.add_argument("--device", default="cuda", help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df["missing_tokens"] = ""
    df["token_similarities"] = ""

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B").to(args.device)
    model.eval()

    logger.info(f"Loaded model and tokenizer on {args.device}")

    for start in tqdm(range(0, len(df), args.batch_size)):
        end = start + args.batch_size
        batch = df.iloc[start:end]
        eng_batch = batch["Source"].tolist()
        zrl_batch = batch["zrl"].tolist()

        try:
            missing, scores = detect_missing_words_batch(eng_batch, zrl_batch, tokenizer, model,
                                                         args.threshold, args.device)
            df.loc[start:end - 1, "missing_tokens"] = missing
            df.loc[start:end - 1, "token_similarities"] = scores
        except Exception as e:
            logger.error(f"Error processing batch {start}-{end}: {e}")
            df.loc[start:end - 1, "missing_tokens"] = "[ERROR]"
            df.loc[start:end - 1, "token_similarities"] = "[ERROR]"

    df.to_csv(args.output_csv, index=False)
    print(f"✅ Done. Output saved to {args.output_csv}")

if __name__ == "__main__":
    main()
