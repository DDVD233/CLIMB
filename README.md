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

A list of the datasets included in CLIMB can be found in the `Dataset_Info.md` file.

### Data Distribution

- 40.56% - 3D/Video samples (ultrasounds, CT scans, endoscopic images, MRI images)
- 22.90% - Multimodal data combinations
- 19.31% - 1D data (EHR, EEG, ECG, gait and genomic data)
- 15.68% - 2D imaging data (X-rays, dermoscopy, fundus images, pathology slides)
- 1.54% - Graph data (brain networks, molecules)

## Folder Structure

- `fusion`: Code for multimodal fusion strategies
- `models`: Vision, EEG, and ECG encoders
- `src`: Dataset collection and preprocessing scripts
