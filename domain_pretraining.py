#!/usr/bin/env python3
"""
Domain-adaptive pre-training of RoBERTa.

Continues pre-training from the original weights with masked language modelling
on a domain corpus of .txt files.
"""

import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
import json
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """Set the random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class PreTrainingDataset(Dataset):
    """Text segments for pre-training."""

    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def load_training_data(data_dir='pre_training_document'):
    """Load the corpus and return its unique text segments."""
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    # Support long file paths on Windows
    if os.name == 'nt':
        txt_files_processed = []
        for f in txt_files:
            abs_path = os.path.abspath(f)
            if len(abs_path) > 200 and not abs_path.startswith('\\\\?\\'):
                txt_files_processed.append('\\\\?\\' + abs_path)
            else:
                txt_files_processed.append(abs_path)
        txt_files = txt_files_processed

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    print(f"Found {len(txt_files)} .txt files")

    all_texts = []
    skipped_files = 0

    for txt_file in tqdm(txt_files, desc="Loading files"):
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'gbk', 'iso-8859-1']
            content = None

            for encoding in encodings:
                try:
                    with open(txt_file, 'r', encoding=encoding) as f:
                        content = f.read()
                        break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if content is None:
                skipped_files += 1
                continue

            # Split into sentences and remove segments of 10 characters or fewer
            sentences = content.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

            all_texts.extend(sentences)

        except Exception:
            skipped_files += 1
            continue

    if skipped_files > 0:
        print(f"Skipped {skipped_files} files that could not be read")

    print(f"Loaded {len(all_texts)} text segments")

    unique_texts = list(set(all_texts))
    print(f"Unique text segments: {len(unique_texts)}")

    return unique_texts


def create_data_loaders(texts, tokenizer, batch_size=16, max_length=512, train_split=0.85):
    """Create the training and validation data loaders."""
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    print(f"Training segments: {len(train_texts)}")
    print(f"Validation segments: {len(val_texts)}")

    train_dataset = PreTrainingDataset(train_texts, tokenizer, max_length)
    val_dataset = PreTrainingDataset(val_texts, tokenizer, max_length)

    # Randomly mask 15% of the tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    return train_loader, val_loader


def train_model(model, tokenizer, train_loader, val_loader, device, model_name,
                num_epochs=5, learning_rate=2e-5, save_dir='pre_training_result'):
    """Train the model and save it after each epoch."""
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.15 * total_steps),
        num_training_steps=total_steps
    )

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            train_loss += loss.item()
            train_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                val_loss += outputs.loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)

        print(f"Training loss: {avg_train_loss:.4f}")
        print(f"Validation loss: {avg_val_loss:.4f}")

        model_dir = os.path.join(save_dir, model_name.replace('/', '_'))
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f"epoch{epoch+1}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Saved the model to {model_save_path}")

    return train_losses, val_losses


def save_training_info(model_name, train_losses, val_losses, save_dir='pre_training_result',
                       learning_rate=2e-5, max_length=512, batch_size=16):
    """Save the losses and training settings."""
    info = {
        'model_name': model_name,
        'training_losses': train_losses,
        'validation_losses': val_losses,
        'best_val_loss': min(val_losses) if val_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'total_epochs': len(train_losses),
        'training_config': {
            'learning_rate': learning_rate,
            'max_length': max_length,
            'batch_size': batch_size,
            'mlm_probability': 0.15,
            'warmup_ratio': 0.15,
            'weight_decay': 0.01,
            'gradient_clip': 0.5
        }
    }

    model_dir = os.path.join(save_dir, model_name.replace('/', '_'))
    os.makedirs(model_dir, exist_ok=True)
    info_file = os.path.join(model_dir, "training_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Saved the training information to {info_file}")


def main():
    parser = argparse.ArgumentParser(description='Domain-adaptive pre-training')
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--data_dir', type=str, default='pre_training_document')
    parser.add_argument('--save_dir', type=str, default='pre_training_result')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    texts = load_training_data(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)

    train_loader, val_loader = create_data_loaders(
        texts, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    train_losses, val_losses = train_model(
        model, tokenizer, train_loader, val_loader, device, args.model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )

    save_training_info(args.model_name, train_losses, val_losses, args.save_dir,
                       learning_rate=args.learning_rate, max_length=args.max_length,
                       batch_size=args.batch_size)


if __name__ == "__main__":
    main()
