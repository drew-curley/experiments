import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, EncoderDecoderCache
from datasets import Dataset
import sacrebleu
import logging
from torch.utils.data import DataLoader

from torch.nn.utils import clip_grad_norm_

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_tokenize(tokenizer, file_path, src_col, tgt_col, max_length=128, batch_size=64):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV with columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load CSV {file_path}: {str(e)}")
        return None

    if src_col >= len(df.columns) or tgt_col >= len(df.columns):
        logger.error(f"Invalid column indices: src_col={src_col}, tgt_col={tgt_col}, but CSV has {len(df.columns)} columns")
        return None

    df = df.rename(columns={df.columns[src_col]: 'source', df.columns[tgt_col]: 'target'})
    df['source'] = df['source'].astype(str).fillna('')
    df['target'] = df['target'].astype(str).fillna('')

    logger.info(f"Sample source data: {df['source'].head().tolist()}")
    logger.info(f"Sample target data: {df['target'].head().tolist()}")

    dataset = Dataset.from_pandas(df)

    def tokenize_batch(examples):
        source_texts = [str(text) for text in examples['source'] if str(text).strip()]
        target_texts = [str(text) for text in examples['target'] if str(text).strip()]

        if not source_texts or not target_texts:
            return {'input_ids': [], 'attention_mask': [], 'labels': []}

        inputs = tokenizer(source_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        labels = tokenizer(target_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        labels['input_ids'] = [[tok if tok != tokenizer.pad_token_id else -100 for tok in label] for label in labels['input_ids']]

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']
        }

    tokenized_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

def evaluate_and_save(model, tokenizer, eval_file, src_col, ref_col, output_path, device):
    try:
        df = pd.read_csv(eval_file)
        logger.info(f"Loaded evaluation CSV with columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load evaluation CSV {eval_file}: {str(e)}")
        return

    generated_texts = []
    bleu_scores = []

    model.eval()

    for i, row in df.iterrows():
        input_text = str(row.iloc[src_col]) if pd.notnull(row.iloc[src_col]) else ''

        if not input_text.strip():
            logger.warning(f"[{i}] Skipping empty input.")
            generated_texts.append('')
            bleu_scores.append(0.0)
            continue

        try:
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)

            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"[{i}] Error during generation: {str(e)}")
            generated = ''

        reference = str(row.iloc[ref_col]) if pd.notnull(row.iloc[ref_col]) else ''
        bleu = sacrebleu.sentence_bleu(generated, [reference]).score if reference.strip() else 0.0

        logger.info(f"[{i}]\nSRC: {input_text}\nGEN: {generated}\nREF: {reference}\nBLEU: {bleu:.2f}\n---")

        generated_texts.append(generated)
        bleu_scores.append(bleu)

    df['generated_zrl'] = generated_texts
    df['bleu_score'] = bleu_scores

    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Evaluation results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {str(e)}")

def main():
    try:
        import transformers
        import safetensors
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"Torch version: {torch.__version__}")
        logger.info(f"Safetensors version: {safetensors.__version__}")
    except ImportError as e:
        logger.error(f"Missing library: {e}")
        return

    data_dir = Path(".")
    train_file = Path("/home/curleyd/GitHub/New/input_text.csv")
    eval_file = Path("/home/curleyd/GitHub/New/eval_text.csv")
    output_dir = Path("/home/curleyd/GitHub/New/output")
    output_dir.mkdir(exist_ok=True)

    model_name = "facebook/nllb-200-3.3B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
        if "[zrl]" not in tokenizer.get_vocab():
            tokenizer.add_tokens(["[zrl]"])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)
        model.resize_token_embeddings(len(tokenizer))
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("[zrl]")
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {str(e)}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    train_dataloader = load_and_tokenize(tokenizer, train_file, src_col=4, tgt_col=5, batch_size=64)
    if train_dataloader is None:
        logger.error("Failed to create training dataloader")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_epochs = 10
    logging_steps = 50
    grad_accum_steps = 4
    step = 0
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    train_size = int(0.9 * len(train_dataloader.dataset))
    val_size = len(train_dataloader.dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataloader.dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64)

    total_steps = (len(train_loader) // grad_accum_steps) * num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_steps)

    logger.info("Starting fine-tuning for maximum quality...")
    model.train()
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accum_steps

            loss.backward()
            total_loss += loss.item()

            if (i + 1) % grad_accum_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {step}/{total_steps}, Loss: {avg_loss:.4f}")
                    total_loss = 0.0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = model(**inputs, labels=labels)
                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)
        logger.info(f"Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info("Validation loss improved — keeping model.")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation loss (patience {patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    try:
        model.save_pretrained(output_dir / "fine_tuned_model")
        tokenizer.save_pretrained(output_dir / "fine_tuned_model")
        logger.info(f"Model saved to {output_dir / 'fine_tuned_model'}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")

    output_eval = output_dir / "eval_results.csv"
    evaluate_and_save(
        model, tokenizer, eval_file,
        src_col=4, ref_col=5,
        output_path=output_eval,
        device=device
    )

if __name__ == "__main__":
    main()
