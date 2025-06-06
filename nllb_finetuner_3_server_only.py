import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
BASE_DIR = "/home/curleyd/GitHub/New"
TRAIN_FILE = os.path.join(BASE_DIR, "input_texts.csv")
EVAL_FILE = os.path.join(BASE_DIR, "eval_texts.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_model")
EVAL_OUTPUT_CSV = os.path.join(BASE_DIR, "eval_with_translations_run_1.csv")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Load model and tokenizer
def load_model_and_tokenizer():
    logger.info(f"Loading model and tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

# Load datasets
def load_datasets(train_file, eval_file):
    df_train = pd.read_csv(train_file)
    df_eval = pd.read_csv(eval_file)
    return Dataset.from_pandas(df_train), Dataset.from_pandas(df_eval)

# Tokenization
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["source"], truncation=True, padding="max_length", max_length=128)

# Compute BLEU
def compute_bleu_scores(references, hypotheses):
    smooth = SmoothingFunction().method1
    scores = [sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth) for ref, hyp in zip(references, hypotheses)]
    return scores

# Save model manually
def save_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False)

# Main training routine
def run_training():
    logger.info("\U0001F680 Starting training run 1")

    tokenizer, model = load_model_and_tokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    train_dataset, eval_dataset = load_datasets(TRAIN_FILE, EVAL_FILE)

    logger.info("Tokenizing datasets")
    train_tokenized = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    eval_tokenized = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized
    )

    logger.info("Starting training")
    trainer.train()
    trainer.evaluate()

    logger.info("Generating translations")
    sources = eval_dataset["source"]
    targets = eval_dataset["target"]
    generated = []

    for s in sources:
        inputs = tokenizer(s, return_tensors="pt", padding=True, truncation=True).to(device)
        output = model.generate(**inputs, max_length=200)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated.append(generated_text)

    bleu_scores = compute_bleu_scores(targets, generated)

    for i, score in enumerate(bleu_scores):
        logger.info(f"Row {i+1} BLEU score: {score:.4f}")

    logger.info(f"Average BLEU score: {sum(bleu_scores)/len(bleu_scores):.4f}")

    eval_df = pd.read_csv(EVAL_FILE)
    eval_df["generated"] = generated
    eval_df.to_csv(EVAL_OUTPUT_CSV, index=False)

    logger.info(f"Updated eval dataset saved to {EVAL_OUTPUT_CSV}")

    logger.info(f"Saving model to {OUTPUT_DIR}")
    save_model(model, OUTPUT_DIR)

if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
