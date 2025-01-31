import os
import re
from collections import Counter
from time import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_medical_model(device):
    """Load a medical-domain-tuned BERT model for better medical text understanding"""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = model.to(device)
    return tokenizer, model


def get_embeddings(texts, tokenizer, model, device, batch_size=32):
    """Generate embeddings for a list of texts using GPU acceleration"""
    embeddings_list = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        # Keep embeddings on GPU if needed
        batch_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings_list.append(batch_embeddings)

    # Concatenate all embeddings
    return torch.cat(embeddings_list, dim=0)


def save_checkpoint(results, checkpoint_dir, checkpoint_num):
    """Save intermediate results to a checkpoint file"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'medical_images_checkpoint_{checkpoint_num}.csv')
    df = pd.DataFrame(results)
    df.to_csv(checkpoint_path, index=False)
    print(f"Saved checkpoint {checkpoint_num} with {len(results)} results")


def calculate_similarities(caption_embeddings, keyword_embedding):
    """Calculate cosine similarity between caption embeddings and single keyword embedding"""
    # Normalize embeddings for cosine similarity
    caption_embeddings_norm = torch.nn.functional.normalize(caption_embeddings, p=2, dim=1)
    keyword_embedding_norm = torch.nn.functional.normalize(keyword_embedding, p=2, dim=1)

    # Calculate similarities
    similarities = cosine_similarity(caption_embeddings_norm, keyword_embedding_norm)

    return similarities


def filter_medical_images(dataset, keywords, num_images=1000000, cache_dir=None, batch_size=1024,
                          checkpoint_interval=100):
    """
    Filter LAION dataset for medical images based on keywords with GPU acceleration and batch processing.

    Args:
        dataset: LAION dataset
        keywords: List of medical keywords
        num_images: Number of images to select
        cache_dir: Cache directory for the dataset
        batch_size: Size of batches for processing
        checkpoint_interval: Save checkpoint after this many new results
    """
    # Set up GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading medical language model...")
    tokenizer, model = load_medical_model(device)

    concatenated_keywords = ", ".join(keywords)
    print("Creating keyword embedding...")
    keyword_embedding = get_embeddings([concatenated_keywords], tokenizer, model, device)


    results = []
    checkpoint_dir = "checkpoints"
    checkpoint_count = 0
    last_save_count = 0

    # Create iterator with batch size
    dataset_iterator = dataset.iter(batch_size=batch_size)

    # Estimate total number of batches for tqdm
    estimated_total = num_images // batch_size * 2  # Rough estimate

    print("Processing dataset...")
    pbar = tqdm(total=num_images)
    last_results_count = 0

    try:
        for batch in dataset_iterator:
            # Process batch of captions
            batch_captions = batch['caption']

            # Get embeddings for captions
            caption_embeddings = get_embeddings(batch_captions, tokenizer, model, device, batch_size=64)

            # Calculate similarity scores using GPU
            max_similarities = calculate_similarities(caption_embeddings, keyword_embedding)

            # Store results with metadata
            for idx, (caption, similarity) in enumerate(zip(batch_captions, max_similarities)):
                if similarity > 0.3:  # Threshold to filter relevant content
                    results.append({
                        'text': caption,
                        'url': batch['url'][idx],
                        'similarity_score': float(similarity)
                    })

            # Update progress bar with new results
            new_results = len(results) - last_results_count
            pbar.update(new_results)
            last_results_count = len(results)

            # Save checkpoint if needed
            if len(results) - last_save_count >= checkpoint_interval:
                checkpoint_count += 1
                save_checkpoint(results, checkpoint_dir, checkpoint_count)
                last_save_count = len(results)

            # Early stopping if we have enough results
            if len(results) >= num_images * 2:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current results...")
    finally:
        pbar.close()

        # Final processing
        print("Processing final results...")
        df = pd.DataFrame(results)

        # Sort by similarity score and take top N
        df = df.sort_values('similarity_score', ascending=False)
        selected_images = df.head(num_images)

        # Save final results
        final_path = os.path.join(checkpoint_dir, 'medical_images_final.csv')
        selected_images.to_csv(final_path, index=False)

        return selected_images


def load_dataset_with_retry(dataset_name, cache_dir, max_retries=3):
    """Load dataset with retry mechanism"""
    for attempt in range(max_retries):
        try:
            return load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                streaming=True
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(5)


if __name__ == "__main__":
    # Configuration
    cache_dir = "<file_path>"
    BATCH_SIZE = 1024  # Adjust based on your GPU memory
    CHECKPOINT_INTERVAL = 100

    # Your keywords list
    medical_keywords = [
        "akiec", "atelectasis", "basal cell carcinoma", "benign",
        "birad 1", "birad 2", "birad 3", "birad 4", "birad 5",
        "cardiomegaly", "consolidation", "edema", "enlarged cardiomediastinum",
        "fracture", "has nodule", "lung lesion", "lung opacity", "malignant",
        "melanoma", "mild", "moderate", "nevus", "no nodule", "pleural effusion",
        "pleural other", "pneumonia", "pneumothorax", "proliferative",
        "support devices", "EHR", "Radiology Reports", "ECG", "EEG",
        "Vital Sign", "CXR", "X-ray", "Mammo", "Dermoscopy", "Fundus",
        "Ultrasound", "Retinal", "Polyp", "CT Scan", "MRI", "PET/CT", "Endoscopic"
    ]

    start_time = time()

    # Load dataset with retry mechanism
    print("Loading dataset...")
    dataset = load_dataset_with_retry(
        "laion/relaion2B-en-research-safe",
        cache_dir=cache_dir
    )

    # Filter and select images
    selected_images = filter_medical_images(
        dataset['train'],
        medical_keywords,
        num_images=1000000,
        cache_dir=cache_dir,
        batch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL
    )

    end_time = time()
    print(f"Processing completed in {(end_time - start_time) / 3600:.2f} hours")
    print(f"Selected {len(selected_images)} medical images")
    print(f"Results saved in checkpoints/medical_images_final.csv")
