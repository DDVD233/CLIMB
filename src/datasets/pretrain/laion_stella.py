import os
import re
from collections import Counter
from time import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_stella_model(device):
    """Load the Stella model for text embedding"""
    if device == "cuda":
        model = SentenceTransformer("dunzhang/stella_en_400M_v5",
                                    trust_remote_code=True,

                                    ).cuda()
    else:
        model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            trust_remote_code=True,
            device="cpu",
            config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
        )
    return model


def get_embeddings(texts, model, batch_size=32, is_query=False):
    """Generate embeddings for a list of texts using Stella model"""
    # Use s2s_query prompt for queries, no prompt for documents
    if is_query:
        embeddings = model.encode(texts, batch_size=batch_size, prompt_name="s2p_query")
    else:
        embeddings = model.encode(texts, batch_size=batch_size)

    return embeddings


def save_checkpoint(results, checkpoint_dir, checkpoint_num):
    """Save intermediate results to a checkpoint file"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'medical_images_checkpoint_{checkpoint_num}.csv')
    df = pd.DataFrame(results)
    df.to_csv(checkpoint_path, index=False)
    print(f"Saved checkpoint {checkpoint_num} with {len(results)} results")


def filter_medical_images(dataset, keywords, num_images=1000000, cache_dir=None, batch_size=1024,
                          checkpoint_interval=1000):
    """
    Filter LAION dataset for medical images based on keywords using Stella model.
    """
    # Set up GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Stella model...")
    model = load_stella_model(device)

    # Create keyword embedding using s2s_query prompt
    print("Creating keyword embeddings...")
    keyword_embedding = get_embeddings([", ".join(keywords)], model, is_query=True)

    results = []
    checkpoint_dir = "checkpoints"
    checkpoint_count = 0
    last_save_count = 0

    # Create iterator with batch size
    dataset_iterator = dataset.iter(batch_size=batch_size)

    print("Processing dataset...")
    pbar = tqdm(total=num_images)
    last_results_count = 0
    processed_images = 0

    try:
        for batch in dataset_iterator:
            # Process batch of captions
            batch_captions = batch['caption']

            # Get embeddings for captions
            caption_embeddings = get_embeddings(batch_captions, model, batch_size=1024)

            # Calculate similarities using Stella's similarity function
            similarities = model.similarity(
                keyword_embedding,
                caption_embeddings
            )
            processed_images += len(batch_captions)
            similarities = similarities[0]  # Get similarities for our single query

            # Store results with metadata
            for idx, (caption, similarity) in enumerate(zip(batch_captions, similarities)):
                if similarity > 0.37:  # Threshold to filter relevant content
                    results.append({
                        'text': caption,
                        'url': batch['url'][idx],
                        'similarity_score': float(similarity)
                    })

            # Update progress bar with new results
            new_results = len(results) - last_results_count
            pbar.update(new_results)
            last_results_count = len(results)
            description = f"Added {len(results)}/{processed_images} images"
            pbar.set_description(description)

            # Save checkpoint if needed
            if len(results) - last_save_count >= checkpoint_interval:
                checkpoint_count += 1
                save_checkpoint(results, checkpoint_dir, checkpoint_count)
                last_save_count = len(results)

            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
    cache_dir = "<path>"
    BATCH_SIZE = 8192  # Adjust based on your GPU memory
    CHECKPOINT_INTERVAL = 100

    # Your keywords list
    medical_keywords = [
        "akiec", "atelectasis", "basal cell carcinoma",
        "cardiomegaly", "consolidation", "edema", "enlarged cardiomediastinum",
        "fracture", "has nodule", "lung lesion", "lung opacity", "malignant",
        "melanoma", "nevus", "no nodule", "pleural effusion",
        "pneumonia", "pneumothorax", "proliferative",
        "support devices", "Radiology Reports", "ECG", "EEG",
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
