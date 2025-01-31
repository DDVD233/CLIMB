import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, hamming_loss, accuracy_score
from src.datasets.multimodal.mimic_iv import MIMIC_IV
from torch.utils.data import Dataset
from fusion.multimodal_modal import MultimodalMedicalModel
import os
import wandb
import argparse
from datetime import datetime
import logging
from heavyball import PrecondScheduleForeachSOAP
from scipy.io import loadmat
from scipy.signal import resample


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def downsample_waves(wave, new_size):
    return resample(wave, new_size, axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train multimodal medical model')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and validation')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate in classifier')
    parser.add_argument('--task', type=str, default="diagnosis_classification",
                        help='Task (diagnosis_classification, length_of_stay_regression, or survival_classification)')
    parser.add_argument('--freeze_vision', action='store_true',
                        help='Freeze vision encoder during training')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze all encoders during training')
    parser.add_argument('--vision_checkpoint', type=str, default='/mnt/8T/convnext/model_56000.pt',
                        help='Path to ConvNextV2 checkpoint')
    parser.add_argument('--ecg_checkpoint', type=str, default='/mnt/8T/ecg/ecg_encoder.pth',
                        help='Path to ECG encoder checkpoint')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='Path to save model checkpoints')
    parser.add_argument('--fusion_technique', type=str, default='concatenate',
                        help='Fusion technique (concatenate or cross_attention)')
    parser.add_argument('--base_dir', type=str, default='/mnt/8T/high_modality/',
                        help='Base directory for dataset')
    parser.add_argument('--fusion_stage', type=str, default='early',
                        help='Fusion stage (early or late)')
    parser.add_argument('--few_shot', type=int, default=-1,
                        help='Number of samples per class for few-shot learning (-1 for no few-shot)')

    return parser.parse_args()


class MIMICMultimodalDataset(Dataset):
    def __init__(self, mimic_dataset: MIMIC_IV, task="diagnosis_classification", transform=None, few_shot=-1):
        self.mimic = mimic_dataset
        self.task = task
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.num_classes = mimic_dataset.__num_classes__()
        self.original_size = len(self.mimic)

        # Filter dataset for few-shot learning if specified
        if few_shot >= 1:
            # Get few-shot indices
            few_shot_indices = self._filter_few_shot(few_shot)
            # Oversample to match original dataset size
            self.indices = self._oversample_indices(few_shot_indices)
        else:
            self.indices = list(range(len(self.mimic)))

        # Calculate class weights based on filtered dataset
        self.class_weights = self._calculate_class_weights()

    def _oversample_indices(self, few_shot_indices):
        """
        Oversample the few-shot indices to match original dataset size and shuffle
        """
        import random
        # Calculate how many times we need to repeat the indices
        num_repeats = self.original_size // len(few_shot_indices) + 1
        # Repeat indices
        oversampled_indices = few_shot_indices * num_repeats
        # Trim to match original size
        oversampled_indices = oversampled_indices[:self.original_size]
        # Shuffle indices
        random.shuffle(oversampled_indices)
        return oversampled_indices

    def _filter_few_shot(self, k):
        """
        Filter dataset to keep only k samples per class/bin
        Returns list of valid indices
        """
        if self.task == "length_of_stay_regression":
            return self._filter_few_shot_regression(k)
        else:
            return self._filter_few_shot_classification(k)

    def _filter_few_shot_classification(self, k):
        """
        Filter classification dataset to keep k samples per class
        """
        # Initialize counters for each class
        class_counts = {i: 0 for i in range(self.num_classes)}
        selected_indices = []

        # For each sample in dataset
        for idx in range(len(self.mimic)):
            data = self.mimic[idx]

            if self.task == "diagnosis_classification":
                label_indices = data['diagnoses']
                # Check if any class in this sample needs more examples
                needs_sample = False
                for label_idx in label_indices:
                    if class_counts[label_idx] < k:
                        needs_sample = True
                        break

                if needs_sample:
                    selected_indices.append(idx)
                    # Update counts for all classes in this sample
                    for label_idx in label_indices:
                        if class_counts[label_idx] < k:
                            class_counts[label_idx] += 1

            elif self.task == "survival_classification":
                survived = data['survived']
                if class_counts[survived] < k:
                    selected_indices.append(idx)
                    class_counts[survived] += 1

            # Check if we have enough samples for all classes
            if all(count >= k for count in class_counts.values()):
                break

        return selected_indices

    def _filter_few_shot_regression(self, k):
        """
        Filter regression dataset to keep k samples per day bin
        """
        # Create bins for length of stay (rounded to days)
        bin_counts = {}
        selected_indices = []

        # First pass: count samples per bin
        for idx in range(len(self.mimic)):
            data = self.mimic[idx]
            los = data['length_of_stay']
            # Round to nearest day for binning
            day_bin = round(los)

            if day_bin not in bin_counts:
                bin_counts[day_bin] = 0

            if bin_counts[day_bin] < k:
                selected_indices.append(idx)
                bin_counts[day_bin] += 1

        return selected_indices

    def _calculate_class_weights(self):
        cache_file = 'class_weights.json'
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                class_weights = json.load(f)

            if self.task in class_weights:
                return torch.tensor(class_weights[self.task])

        if self.task == "diagnosis_classification":
            # Initialize counters for each class
            label_counts = torch.zeros(self.num_classes)
            total_samples = len(self.indices)

            # Count occurrences of each label in filtered dataset
            for idx in self.indices:
                data = self.mimic[idx]
                label_indices = data['diagnoses']
                for idx in label_indices:
                    label_counts[idx] += 1

            # Calculate weights (inverse of frequency)
            eps = 1e-5
            weights = total_samples / (label_counts + eps)
            weights = weights / weights.sum() * self.num_classes
            pos_weight = weights

        elif self.task == "survival_classification":
            survived_count = 0
            total_samples = len(self.indices)

            for idx in self.indices:
                data = self.mimic[idx]
                survived_count += data['survived']

            pos_weight = (total_samples - survived_count) / (survived_count + 1e-5)
            pos_weight = torch.tensor([pos_weight])
        else:
            return None

        class_weights[self.task] = pos_weight.numpy().tolist()
        with open(cache_file, 'w') as f:
            json.dump(class_weights, f)

        return pos_weight

    def get_class_weights(self):
        return self.class_weights

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual index from filtered indices
        actual_idx = self.indices[idx]
        data = self.mimic[actual_idx]

        # Load and process image
        image_paths = data['image_paths']
        if not image_paths or len(image_paths) == 0:
            image = torch.zeros(3, 224, 224)
        else:
            try:
                image_path = image_paths[0]['full_path']
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
            except:
                image = torch.zeros(3, 224, 224)

        ecg_records = data['ecg_data']
        if not ecg_records or len(ecg_records) == 0:
            ecg = None
        else:
            ecg_record = ecg_records[-1]
            ecg_path = ecg_record['file_path']
            ecg_full_path = os.path.join(self.mimic.mimiciv_root, ecg_path)
            ecg = loadmat(ecg_full_path)['val']
            ecg = np.concatenate((ecg[:2, :], ecg[6:, :]), axis=0)
            ecg = np.nan_to_num(ecg)
            ecg = downsample_waves(ecg, 2500)

        # Convert label indices to multi-hot encoding
        if self.task == "diagnosis_classification":
            label_indices = data['diagnoses']
            multi_hot_labels = torch.zeros(self.num_classes)
            multi_hot_labels[label_indices] = 1.0
            labels = multi_hot_labels
        elif self.task == "survival_classification":
            labels = torch.tensor([1 - data['48_ihm']], dtype=torch.float)
        else:  # length_of_stay_regression
            labels = torch.tensor(data['length_of_stay'], dtype=torch.float)

        return {
            'image': image,
            'patient_history': data['input'],
            'labels': labels,
            'survived': torch.tensor(data['survived'], dtype=torch.float),
            'length_of_stay': torch.tensor(data['length_of_stay'], dtype=torch.float),
            'ecg': ecg
        }


def collate_fn(batch):
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'patient_histories': [item['patient_history'] for item in batch],
        'labels': torch.stack([item['labels'] for item in batch]),
        'survived': torch.stack([item['survived'] for item in batch]),
        'length_of_stay': torch.stack([item['length_of_stay'] for item in batch]),
        'ecg': torch.stack([torch.tensor(item['ecg']) for item in batch])
    }


def calculate_metrics(y_true, y_pred, y_pred_proba, task="diagnosis_classification"):
    """Calculate performance metrics for multi-label classification."""
    from sklearn.metrics import confusion_matrix

    if task == "length_of_stay_regression":
        return {
            'mae': np.mean(np.abs(y_true - y_pred)),
        }
    else:

        if len(y_true.shape) == 1:  # Binary classification case
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
            y_pred_proba = y_pred_proba.reshape(-1, 1)

        # AUC-ROC (one-vs-rest)
        try:
            auc = roc_auc_score(y_true, y_pred_proba, average='macro')
        except:
            auc = 0  # Handle cases where some classes might not have both positive and negative samples

        # Hamming loss (accuracy)
        ham_loss = hamming_loss(y_true, y_pred)
        accuracy = 1 - ham_loss  # Convert hamming loss to accuracy

        # Calculate sensitivity and specificity per class
        n_classes = y_true.shape[1]
        sensitivities = []
        specificities = []

        # Calculate metrics for each class
        for i in range(n_classes):
            try:
                tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()

                # Calculate sensitivity (recall) for this class
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

                # Calculate specificity for this class
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                sensitivities.append(sensitivity)
                specificities.append(specificity)
            except:
                # Handle cases where a class might not have any positive/negative samples
                continue

        # Calculate macro-averaged metrics
        sensitivity = np.mean(sensitivities) if sensitivities else 0
        specificity = np.mean(specificities) if specificities else 0

        return {
            'auc': auc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        }


def train_multimodal():
    args = parse_args()
    shots = args.few_shot
    task = args.task
    technique = args.fusion_technique if args.fusion_stage == "early" else "late"
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize wandb
    wandb.init(
        project="mimic-multimodal",
        name=f"{task}_{technique}_{shots}shots",
        config=vars(args)
    )

    # Initialize datasets
    # base_dir = '/mnt/8T/high_modality/'
    train_dataset = MIMIC_IV(base_root=args.base_dir, split='train')
    valid_dataset = MIMIC_IV(base_root=args.base_dir, split='valid')

    # Create data loaders
    train_dataset = MIMICMultimodalDataset(train_dataset, task=args.task, few_shot=args.few_shot)
    valid_dataset = MIMICMultimodalDataset(valid_dataset, task=args.task)

    class_weights = train_dataset.get_class_weights()
    if class_weights is not None:
        wandb.log({"class_weights": wandb.Histogram(class_weights.numpy())})

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = train_dataset.num_classes
    model = MultimodalMedicalModel(
        output_size=num_classes,
        convnext_checkpoint=args.vision_checkpoint,
        ecg_checkpoint=args.ecg_checkpoint,
        task=args.task,
        fusion_technique=args.fusion_technique
    ).to(device)

    if args.freeze_vision:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    if args.freeze_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        for param in model.ecg_encoder.parameters():
            param.requires_grad = False
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    # model = model.to(torch.bfloat16)

    if args.model_checkpoint is not None:
        checkpoint = torch.load(args.model_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Training parameters
    if args.task == "diagnosis_classification":
        class_weights = train_dataset.get_class_weights().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    elif args.task == "survival_classification":
        class_weights = train_dataset.get_class_weights().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        criterion = nn.MSELoss()
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = PrecondScheduleForeachSOAP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_auc = 0 # for classification
    best_loss = float("inf") # for regression
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_preds_proba = []
        train_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}')
        for step, batch in enumerate(pbar):
            images = batch['images'].to(device).to(torch.bfloat16)
            histories = batch['patient_histories']
            ecg = batch['ecg'].to(device)

            # Forward pass
            logits = model(args.task, images, histories, ecg, args.fusion_technique, args.fusion_stage)

            if args.task == "diagnosis_classification":
                labels = batch['labels'].to(device)
            elif args.task == "length_of_stay_regression":
                labels = batch['length_of_stay'].to(device).float()
                logits = logits.squeeze(1)
                # filter
            elif args.task == "survival_classification":
                labels = batch['survived'].to(device)

            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Store predictions and labels for metrics
            if args.task == "length_of_stay_regression":
                preds = logits
                train_preds.extend(preds.detach().cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                train_preds_proba.extend(preds.detach().cpu().numpy())
            else:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_preds_proba.extend(probs.float().detach().cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            # evaluate every batch
            train_metrics = calculate_metrics(
                np.array(train_labels),
                np.array(train_preds),
                np.array(train_preds_proba),
                task=args.task
            )
            if args.task == "length_of_stay_regression":
                wandb.log({
                    'step': step,
                    'train/loss': train_loss / (pbar.n + 1) / args.batch_size,
                    'train/mae': train_metrics['mae']
                })
            else:
                wandb.log({
                    'step': step,
                    'train/loss': train_loss / (pbar.n + 1) / args.batch_size,
                    'train/auc': train_metrics['auc'],
                    'train/accuracy': train_metrics['accuracy'],
                    'train/sensitivity': train_metrics['sensitivity'],
                    'train/specificity': train_metrics['specificity']
                })

            if pbar.n % 500 == 0 and pbar.n > 0:
                model.eval()
                labels = validate(args, best_auc, best_loss, criterion, device, step, labels, model, optimizer, train_labels,
                                    train_loader, train_loss, train_preds, train_preds_proba, valid_loader, num_steps=500)
                model.train()

            pbar.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        labels = validate(args, best_auc, best_loss, criterion, device, step, labels, model, optimizer, train_labels,
                          train_loader, train_loss, train_preds, train_preds_proba, valid_loader)
        model.train()

    wandb.finish()


def validate(args, best_auc, best_loss, criterion, device, step, labels, model, optimizer, train_labels, train_loader,
             train_loss, train_preds, train_preds_proba, valid_loader, num_steps=-1):
    model.eval()
    valid_loss = 0
    valid_preds = []
    valid_preds_proba = []
    valid_labels = []
    this_step = 0

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc='Validation'):
            images = batch['images'].to(device).to(torch.bfloat16)
            histories = batch['patient_histories']
            ecg = batch['ecg'].to(device)


            logits = model(args.task, images, histories, ecg, args.fusion_technique)
            if args.task == "diagnosis_classification":
                labels = batch['labels'].to(device)
            elif args.task == "length_of_stay_regression":
                labels = batch['length_of_stay'].to(device).float()
                logits = logits.squeeze(1)
            elif args.task == "survival_classification":
                labels = batch['survived'].to(device)

            loss = criterion(logits, labels)
            valid_loss += loss.item()

            if args.task == "length_of_stay_regression":
                preds = logits
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().to(torch.float32).numpy())
                valid_preds_proba.extend(preds.cpu().to(torch.float32).numpy())
            else:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                valid_preds.extend(preds.cpu().numpy())
                valid_preds_proba.extend(probs.cpu().to(torch.float32).numpy())
                valid_labels.extend(labels.cpu().to(torch.float32).numpy())
            if num_steps > 0 and this_step > num_steps:
                break
            this_step += 1

    train_metrics = calculate_metrics(
        np.array(train_labels),
        np.array(train_preds),
        np.array(train_preds_proba),
        task=args.task
    )
    valid_metrics = calculate_metrics(
        np.array(valid_labels),
        np.array(valid_preds),
        np.array(valid_preds_proba),
        task=args.task
    )
    # Calculate metrics
    if args.task == "diagnosis_classification" or args.task == "survival_classification":
        # Log metrics
        wandb.log({
            'step': step,
            'train/loss': train_loss / len(train_loader) / args.batch_size,
            'train/auc': train_metrics['auc'],
            'train/accuracy': train_metrics['accuracy'],
            'train/sensitivity': train_metrics['sensitivity'],
            'train/specificity': train_metrics['specificity'],
            'valid/loss': valid_loss / len(valid_loader) / args.batch_size,
            'valid/auc': valid_metrics['auc'],
            'valid/accuracy': valid_metrics['accuracy'],
            'valid/sensitivity': valid_metrics['sensitivity'],
            'valid/specificity': valid_metrics['specificity']
        })

        print(f'Train Loss: {train_loss / len(train_loader):.4f}')
        print(f'Train Metrics - AUC: {train_metrics["auc"]:.4f}, '
              f'Accuracy: {train_metrics["accuracy"]:.4f}, '
              f'Sensitivity: {train_metrics["sensitivity"]:.4f}, '
              f'Specificity: {train_metrics["specificity"]:.4f}')
        print(f'Valid Loss: {valid_loss / len(valid_loader):.4f}')
        print(f'Valid Metrics - AUC: {valid_metrics["auc"]:.4f}, '
              f'Accuracy: {valid_metrics["accuracy"]:.4f}, '
              f'Sensitivity: {valid_metrics["sensitivity"]:.4f}, '
              f'Specificity: {valid_metrics["specificity"]:.4f}')
    elif args.task == "length_of_stay_regression":
        wandb.log({
            'step': step,
            'train/loss': train_loss / len(train_loader) / args.batch_size,
            'valid/loss': valid_loss / len(valid_loader) / args.batch_size,
            'train/mae': train_metrics['mae'],
            'valid/mae': valid_metrics['mae']
        })
        print(f'Train Loss: {train_loss / len(train_loader):.4f}')

    # Save best model based on loss
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_name = f'multimodal_models_{timestamp}.pt'
    if valid_loss < best_loss:
        best_loss = valid_loss
        # Remove previous
        for f in os.listdir('.'):
            if f.startswith('multimodal_models_'):
                os.remove(f)
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'args': args,
        }, save_name)
        print(f'Saved best model with loss {best_loss:.4f} to {save_name}')
    return labels


if __name__ == '__main__':
    train_multimodal()
