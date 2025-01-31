# CLIMB: Clinical Large-scale Integrative Multi-modal Benchmark

CLIMB is a comprehensive multimodal clinical benchmark that unifies diverse medical data across imaging, language, temporal, and genomic modalities. This framework enables holistic assessment of patient health by leveraging multiple data types and demonstrating significant improvements in clinical task performance.

This dataset framework is built on top of the [BenchMD](https://github.com/rajpurkarlab/BenchMD) repo, with added datasets from EEG, pathology, mammography, X-ray, and other clinical domains.

## Dataset Overview

CLIMB comprises:
- 4.51 million patient samples
- 19.01 terabytes of total data
- Data from 33 medical institutions
- 96 different clinical conditions
- 13 clinical domains

### Data Distribution

- 40.56% - 3D/Video samples (ultrasounds, CT scans, endoscopic images, MRI images)
- 22.90% - Multimodal data combinations
- 19.31% - 1D data (EHR, EEG, ECG, gait and genomic data)
- 15.68% - 2D imaging data (X-rays, dermoscopy, fundus images, pathology slides)
- 1.54% - Graph data (brain networks, molecules)

## Key Features and Findings

### Multitask Training Performance
- Significant performance improvements across clinical tasks
- Up to 32.54% AUC improvement in COVID ultrasound and other understudied areas
- General-domain architectures outperform specialized clinical models in multitask settings

### Few-shot Transfer Learning
- Enhanced generalization to novel clinical tasks with limited labeled data
- Up to 29% improvement in ultrasound tasks
- Up to 23% improvement in ECG tasks under few-shot settings

### Multimodal Fusion
- Improved performance through single-modality pretraining
- Enhanced multimodal learning with task-appropriate fusion strategies
- Comprehensive evaluation of different combination strategies for clinical data

## Available Resources

- Vision, EEG, and ECG encoders trained on CLIMB
- State-of-the-art performance on multitask clinical learning tasks
- Complete codebase for data collection, training, and evaluation
- Pre-trained model weights (Not included on Anonymous GitHub Due to Size Limitations)
- Detailed recommendations for model architecture selection
- Comprehensive pretraining strategies across clinical modalities

## Folder Structure

- `fusion`: Code for multimodal fusion strategies
- `models`: Vision, EEG, and ECG encoders
- `src`: Dataset collection and preprocessing scripts
