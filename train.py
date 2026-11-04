"""
Train and evaluate DAML-Net for bridge component condition rating.

Task 1 predicts the condition rating and Task 2 predicts the component type.
"""

import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from daml_net import DAMLNet

warnings.filterwarnings('ignore')


def set_seed(seed):
    """Set the random seeds used by Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


class MultiTaskDataset(Dataset):
    """Inspection descriptions with rating and component labels."""

    def __init__(self, descriptions, task1_labels, task2_labels, tokenizer, max_len=512):
        self.descriptions = descriptions
        self.task1_labels = task1_labels
        self.task2_labels = task2_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.descriptions[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_task1': torch.tensor(self.task1_labels[idx], dtype=torch.long),
            'labels_task2': torch.tensor(self.task2_labels[idx], dtype=torch.long)
        }


def load_data(train_path, val_path, test_path):
    """Load the three subsets and map both label columns to class indices."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    print(f"Records: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    task1_to_idx = {label: idx for idx, label in enumerate(sorted(all_df['rating'].unique()))}
    task2_to_idx = {label: idx for idx, label in enumerate(sorted(all_df['component'].unique()))}

    for df in (train_df, val_df, test_df):
        df['task1_idx'] = df['rating'].map(task1_to_idx)
        df['task2_idx'] = df['component'].map(task2_to_idx)

    return train_df, val_df, test_df, task1_to_idx, task2_to_idx


def create_data_loader(df, tokenizer, batch_size, max_len, generator=None):
    """Create a data loader. Passing a generator shuffles the records."""
    dataset = MultiTaskDataset(
        df['description'].values,
        df['task1_idx'].values,
        df['task2_idx'].values,
        tokenizer,
        max_len
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=generator is not None,
                      generator=generator)


def predict(model, data_loader, device):
    """Return the mean loss, labels and predictions for one subset."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    predictions_task1, labels_task1 = [], []
    predictions_task2, labels_task2 = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels_task1 = batch['labels_task1'].to(device)
            batch_labels_task2 = batch['labels_task2'].to(device)

            logits_task1, logits_task2 = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = (model.task1_weight * criterion(logits_task1, batch_labels_task1)
                    + model.get_effective_task2_weight()
                    * criterion(logits_task2, batch_labels_task2))
            total_loss += loss.item()

            predictions_task1.extend(torch.argmax(logits_task1, dim=1).cpu().numpy())
            labels_task1.extend(batch_labels_task1.cpu().numpy())
            predictions_task2.extend(torch.argmax(logits_task2, dim=1).cpu().numpy())
            labels_task2.extend(batch_labels_task2.cpu().numpy())

    return total_loss / len(data_loader), labels_task1, predictions_task1, labels_task2, predictions_task2


def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate,
                patience, save_dir):
    """Train the model and keep the weights with the best Task 1 validation accuracy."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * num_epochs
    )

    best_val_accuracy = 0
    best_epoch = 0
    patience_counter = 0
    model_file = os.path.join(save_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        model.set_epoch(epoch)
        train_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_task1 = batch['labels_task1'].to(device)
            labels_task2 = batch['labels_task2'].to(device)

            model.zero_grad()

            logits_task1, logits_task2 = model(input_ids=input_ids, attention_mask=attention_mask)

            # The Task 2 weight is 0 during the freeze epochs
            loss = (model.task1_weight * criterion(logits_task1, labels_task1)
                    + model.get_effective_task2_weight()
                    * criterion(logits_task2, labels_task2))
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss, val_labels_task1, val_predictions_task1, val_labels_task2, val_predictions_task2 = predict(
            model, val_loader, device
        )
        val_accuracy_task1 = accuracy_score(val_labels_task1, val_predictions_task1)
        val_accuracy_task2 = accuracy_score(val_labels_task2, val_predictions_task2)

        print(f"Train loss: {train_loss / len(train_loader):.4f}")
        print(f"Val loss: {val_loss:.4f}, Task 1 accuracy: {val_accuracy_task1:.4f}, "
              f"Task 2 accuracy: {val_accuracy_task2:.4f}")

        # Early stopping on Task 1 validation accuracy
        if val_accuracy_task1 > best_val_accuracy:
            best_val_accuracy = val_accuracy_task1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), model_file)
            print("Saved the best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Best epoch: {best_epoch}, Task 1 validation accuracy: {best_val_accuracy:.4f}")
    return model_file, best_epoch, best_val_accuracy


def compute_metrics(labels, predictions, idx_to_label):
    """Compute the test metrics for one task."""
    true_labels = [idx_to_label[label] for label in labels]
    predicted_labels = [idx_to_label[prediction] for prediction in predictions]
    report = classification_report(true_labels, predicted_labels, output_dict=True)

    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'classification_report': report
    }


def evaluate_model(model, test_loader, device, task1_to_idx, task2_to_idx):
    """Evaluate both tasks on the test set."""
    idx_to_task1 = {idx: str(label) for label, idx in task1_to_idx.items()}
    idx_to_task2 = {idx: label for label, idx in task2_to_idx.items()}

    _, labels_task1, predictions_task1, labels_task2, predictions_task2 = predict(
        model, test_loader, device
    )

    task1_metrics = compute_metrics(labels_task1, predictions_task1, idx_to_task1)
    task2_metrics = compute_metrics(labels_task2, predictions_task2, idx_to_task2)

    return {'task1_rating': task1_metrics, 'task2_component': task2_metrics}


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate DAML-Net')

    parser.add_argument('--train_path', type=str, default='dataset/multi/train.csv')
    parser.add_argument('--val_path', type=str, default='dataset/multi/val.csv')
    parser.add_argument('--test_path', type=str, default='dataset/multi/test.csv')
    parser.add_argument('--bert_model', type=str, default='pre_training_result/roberta-base/epoch5',
                        help='Path to the encoder saved by domain_pretraining.py')
    parser.add_argument('--save_dir', type=str, default='save/daml_net')

    parser.add_argument('--cnn_num_filters', type=int, default=100)
    parser.add_argument('--cnn_filter_sizes', type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_len', type=int, default=512)

    parser.add_argument('--task1_weight', type=float, default=0.8)
    parser.add_argument('--task2_freeze_epochs', type=int, default=1)
    parser.add_argument('--task2_feature_warmup_epochs', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_df, val_df, test_df, task1_to_idx, task2_to_idx = load_data(
        args.train_path, args.val_path, args.test_path
    )

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # A dedicated generator keeps the training order fixed for a given seed
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = create_data_loader(train_df, tokenizer, args.batch_size, args.max_len, generator)
    val_loader = create_data_loader(val_df, tokenizer, args.batch_size, args.max_len)
    test_loader = create_data_loader(test_df, tokenizer, args.batch_size, args.max_len)

    model = DAMLNet(
        bert_model_name=args.bert_model,
        num_classes_task1=len(task1_to_idx),
        num_classes_task2=len(task2_to_idx),
        task1_weight=args.task1_weight,
        cnn_num_filters=args.cnn_num_filters,
        cnn_filter_sizes=tuple(args.cnn_filter_sizes),
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        task2_feature_warmup_epochs=args.task2_feature_warmup_epochs,
        task2_freeze_epochs=args.task2_freeze_epochs
    )
    model.to(device)

    start_time = time.time()
    model_file, best_epoch, best_val_accuracy = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        save_dir=args.save_dir
    )
    training_time = time.time() - start_time

    model.load_state_dict(torch.load(model_file))
    # The test set always uses the real Task 2 probabilities.
    # The best epoch is expected to fall after the warm-up epochs.
    model.set_epoch(args.task2_feature_warmup_epochs)
    metrics = evaluate_model(model, test_loader, device, task1_to_idx, task2_to_idx)

    results = {
        'best_epoch': best_epoch,
        'best_val_accuracy_task1': float(best_val_accuracy),
        'training_time_seconds': training_time,
        'test_metrics': metrics,
        'args': vars(args)
    }
    results_file = os.path.join(args.save_dir, 'results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    task1_metrics = metrics['task1_rating']
    print(f"\nTest results for Task 1 (rating):")
    print(f"  Accuracy: {task1_metrics['accuracy']:.4f}")
    print(f"  Macro recall: {task1_metrics['macro_recall']:.4f}, Macro F1: {task1_metrics['macro_f1']:.4f}, "
          f"Weighted F1: {task1_metrics['weighted_f1']:.4f}")
    print(f"Test accuracy for Task 2 (component): {metrics['task2_component']['accuracy']:.4f}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
