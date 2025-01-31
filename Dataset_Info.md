# CLIMB Dataset Information

This document provides detailed information about each dataset included in the Clinical Large-scale Integrative Multi-modal Benchmark (CLIMB). The datasets are organized by modality.

## ECG (Electrocardiogram)

### PTB-XL
- **Description**: 12-lead ECGs from 18,869 patients (10 seconds each)
- **Classes**: 7 (Normal, CD, HYP, MI, STTC, A. Fib/Aflutter, Other)
- **Size**: 21,837 records (17,476 train, 4,361 test)
- **Source**: PhysioNet
- **License**: Creative Commons Attribution 4.0 International
- **Access**: https://physionet.org/files/ptb-xl/1.0.3/

### Chapman-Shaoxing
- **Description**: 12-lead ECGs from 10,646 patients
- **Classes**: 7 (Same as PTB-XL)
- **Size**: 40,258 records (38,207 train, 2,051 test)
- **License**: Creative Commons Attribution 4.0 International
- **Access**: Available on Kaggle

### Georgia
- **Description**: 12-lead ECGs from 15,742 patients (10 seconds, 500 Hz)
- **Classes**: 7 (Same as PTB-XL)
- **Size**: 20,689 records (18,622 train, 2,067 test)
- **License**: Creative Commons Public Domain
- **Access**: Available on Kaggle

### CPSC
- **Description**: 12-lead Holter and 3-lead wearable ECG recordings
- **Classes**: 7 (Same as PTB-XL)
- **Size**: 6,192 records (4,815 train, 1,377 test)
- **License**: Creative Commons Attribution 4.0 International
- **Access**: Available on PhysioNet

## EEG (Electroencephalogram)

### IIIC
- **Description**: EEG samples from 2,711 patients, annotated by 124 raters
- **Classes**: 6 (SZ, LPD, GPD, LRDA, GRDA, Other)
- **Size**: 134,450 segments
- **License**: BDSP Restricted Health Data License 1.0.0
- **Access**: Available through BDSP
- **Split**: 60% train, 20% validation, 20% test (by patient groups)

### TUAB
- **Description**: 276 EEG recording sessions from 253 subjects
- **Classes**: 2 (Normal, Abnormal)
- **Size**: 409,500 total records
- **License**: Custom (requires application)
- **Access**: Available through Temple University

### TUEV
- **Description**: 518 EEG recording sessions from 368 subjects
- **Classes**: 6 (SPSW, GPED, PLED, EYEM, ARTF, BCKG)
- **Size**: 111,900 total records
- **License**: Custom (requires application)
- **Access**: Available through Temple University

## Chest X-ray

### MIMIC-CXR
- **Description**: Large-scale chest X-ray dataset
- **Classes**: 14 (Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pneumonia, Pneumothorax, Pleural Other, Support Devices, No Finding)
- **Size**: 356,225 records (348,516 train, 7,709 test)
- **License**: PhysioNet Credentialed Health Data License 1.5.0
- **Access**: Requires credentialed account and CITI training

### CheXpert
- **Description**: Chest radiographs from Stanford Hospital
- **Classes**: 14 (Same as MIMIC-CXR)
- **Size**: 212,498 records (212,243 train, 225 test)
- **License**: Stanford University Dataset Research Use Agreement
- **Access**: Requires registration

### VinDr-CXR
- **Description**: Adult chest X-rays from Vietnamese hospitals
- **Classes**: 6 (Lung tumor, Pneumonia, Tuberculosis, COPD, Other diseases, No finding)
- **Size**: 18,000 records (15,000 train, 3,000 test)
- **License**: PhysioNet Credentialed Health Data License 1.5.0
- **Access**: Requires credentialed account

### COVID-19
- **Description**: Chest X-ray dataset for COVID-19 related diseases
- **Classes**: 4 (Normal, Bacterial Pneumonia, COVID-19, Viral Pneumonia)
- **Size**: 2,990 records (2,002 train, 988 test)
- **License**: Creative Commons Attribution 4.0 International
- **Access**: Available on Kaggle

### CoronaHack
- **Description**: Chest X-ray dataset from University of Montreal
- **Classes**: 3 (Normal, Bacterial Pneumonia, Viral Pneumonia)
- **Size**: 5,908 records (5,284 train, 624 test)
- **License**: Creative Commons Attribution 4.0 International
- **Access**: Available on Kaggle

## Mammography

### VinDr-Mammo
- **Description**: Mammography from Vietnamese hospitals
- **Classes**: 5 (BI-RAD 1-5)
- **Size**: 20,000 records (16,000 train, 4,000 test)
- **License**: PhysioNet Restricted Health Data License 1.5.0
- **Access**: Requires data use agreement

### CBIS-DDSM
- **Description**: Curated subset of Digital Database for Screening Mammography
- **Classes**: 6 (BI-RAD 0-5)
- **Size**: 2,825 records (2,230 train, 595 test)
- **License**: Creative Commons Attribution 3.0 Unported
- **Access**: Available through Cancer Imaging Archive

### CMMD
- **Description**: Breast mammography dataset from China
- **Classes**: 2 (Benign, Malignant)
- **Size**: 1,872 records (1,404 train, 468 test)
- **License**: Creative Commons Attribution 4.0 International
- **Access**: Available through Cancer Imaging Archive
- **Note**: Currently pending expert verification

## Dermoscopy

### ISIC-2020
- **Description**: Dermoscopy of skin lesions from over 2000 patients
- **Classes**: 2 (Malignant, Benign)
- **Size**: 33,126 records (26,501 train, 6,625 test)
- **License**: Creative Commons Attribution-Noncommercial 4.0 International
- **Access**: Available through ISIC Archive

### HAM10000
- **Description**: Dermoscopy images from ISIC 2018 classification challenge
- **Classes**: 5 (MEL, NV, BCC, AKIEC, OTHER)
- **Size**: 10,015 records (8,012 train, 2,003 test)
- **License**: Creative Commons Attribution-Noncommercial-Sharealike 4.0 International
- **Access**: Available on Kaggle

### PAD-UFES-20
- **Description**: Dermoscopy images of 1641 skin lesions
- **Classes**: 5 (Same as HAM10000)
- **Size**: 2,298 records (1,839 train, 459 test)
- **License**: Creative Commons Attribution 4.0 International
- **Access**: Available on Kaggle

## Fundus

### Messidor-2
- **Description**: Diabetic Retinopathy examinations
- **Classes**: 5 (None, Mild DR, Moderate DR, Severe DR, PDR)
- **Size**: 1,744 records (1,394 train, 350 test)
- **License**: Creative Commons Public Domain
- **Access**: Available on Kaggle

### APTOS 2019
- **Description**: Fundus images from rural India
- **Classes**: 5 (No DR, Mild, Moderate, Severe, Proliferative DR)
- **Size**: 3,662 records (2,929 train, 733 test)
- **License**: Kaggle Competition Rules
- **Access**: Available on Kaggle

### Jichi
- **Description**: Posterior pole fundus photography dataset from Japan
- **Classes**: 3 (SDR, PPDR, PDR)
- **Size**: 9,939 records (7,950 train, 1,989 test)
- **License**: Creative Commons Attribution 4.0 International
- **Access**: Available through PMC

## CT (Computed Tomography)

### LNDb
- **Description**: Lung cancer CT scan dataset from Portugal
- **Classes**: 3 (nodule ≥3mm, nodule <3mm, non-nodule)
- **Size**: 5,561 records (4,130 train, 1,431 test)
- **License**: Creative Commons Attribution 4.0 International
- **Access**: Available through Zenodo

### INSPECT
- **Description**: Multi-modal dataset with CT images and reports
- **Classes**: 5 (No PE, Acute Subsegmental-only PE, Acute PE, Subsegmental-only PE, Chronic PE)
- **Size**: 23,240 records (17,434 train, 5,806 test)
- **License**: Stanford University Dataset Research Use Agreement
- **Access**: Requires registration

### KiTS23
- **Description**: Kidney Tumor Segmentation Challenge dataset
- **Classes**: 2 (Benign, Malignant)
- **Size**: 478 records (361 train, 117 test)
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
- **Access**: Available on GitHub

### Hemorrhage
- **Description**: Intracranial hemorrhage CT images
- **Classes**: 2 (No Hemorrhage, Has Hemorrhage)
- **Size**: 2,501 records (1,986 train, 515 test)
- **License**: Creative Commons Attribution 4.0 International
- **Access**: Available on Kaggle

### RSPECT
- **Description**: CT scans for Pulmonary Embolism
- **Classes**: 3 (No PE, Chronic PE, Acute PE)
- **Size**: 1,790,594 records (1,342,945 train, 447,649 test)
- **License**: Kaggle competition rules
- **Access**: Available on Kaggle

## Ultrasound

### EchoNet-Dynamic
- **Description**: Echocardiography videos from Stanford
- **Task**: Segmentation (not classification)
- **Size**: 10,247 records (8,196 train, 2,051 test)
- **License**: Stanford University Dataset Research Use Agreement
- **Access**: Requires registration

### BUSI
- **Description**: Breast cancer ultrasound images
- **Classes**: 3 (Normal, Malignant, Benign)
- **Size**: 780 records (583 train, 197 test)
- **License**: Creative Commons Public Domain
- **Access**: Available on Kaggle

### COVID-BLUES
- **Description**: Lung ultrasound videos for COVID-19
- **Classes**: 2 (Has COVID, No COVID)
- **Size**: 362 records (266 train, 96 test)
- **License**: Creative Commons Attribution-Noncommercial-NoDerivatives 4.0 International
- **Access**: Available on GitHub

### COVID-US
- **Description**: COVID-related lung ultrasound videos
- **Classes**: 3 (Covid, Pneumonia, Normal)
- **Size**: 99 records (74 train, 25 test)
- **License**: GNU Affero General Public License 3.0
- **Access**: Available on GitHub

## MRI (Magnetic Resonance Imaging)

### Brain Tumor
- **Description**: Brain MRI images for tumor classification
- **Classes**: 4 (No Tumor, Pituitary Tumor, Glioma Tumor, Meningioma Tumor)
- **Size**: 3,264 records (2,870 train, 394 test)
- **License**: MIT License
- **Access**: Available on Kaggle

### Brain MRI
- **Description**: Brain MRI dataset for tumor detection
- **Classes**: 2 (Yes, No - presence of tumors)
- **Size**: 253 records (202 train, 51 test)
- **License**: No specific license
- **Access**: Available on Kaggle

## BrainNet

### ABCD
- **Description**: Adolescent brain cognitive development study
- **Classes**: 2 (Normal, Abnormal)
- **Size**: 9,563 records (7,650 train, 1,613 test)
- **License**: Custom NIH license
- **Access**: Available through NDA

### ABIDE
- **Description**: Autism brain MRI diagnosis dataset
- **Classes**: 2 (ASD, Typical controls)
- **Size**: 1,009 records (807 train, 202 test)
- **License**: Creative Commons Attribution-NonCommercial-Share Alike
- **Access**: Requires account registration

### PPMI
- **Description**: Parkinson's disease progression study
- **Classes**: 2 (Control, PD patients)
- **Size**: 718 records (572 train, 143 test)
- **License**: Custom license
- **Access**: Available through PPMI

## Molecule

### PROTEINS
- **Description**: Protein structure graphs
- **Classes**: 2 (Enzyme, Not enzyme)
- **Size**: 1,113 records (890 train, 223 test)
- **License**: No specific license
- **Access**: Available through Papers with Code

### PPI
- **Description**: Protein-protein interaction dataset
- **Task**: Protein interaction prediction
- **Size**: 56,944 records (45,555 train, 11,389 test)
- **License**: MIT License
- **Access**: Available through Stanford SNAP

## Tissues

### LC25000
- **Description**: Lung and colon histopathological images
- **Classes**: 5 (Colon adenocarcinomas, Benign colon, Lung adenocarcinomas, Lung squamous cell carcinomas, Benign lung)
- **Size**: 18,750 records (15,000 train, 3,750 test)
- **License**: No specific license
- **Access**: Available on GitHub

### BCSS
- **Description**: Breast cancer slides
- **Classes**: 4 (Tumor, Stroma, Lymphocytic infiltrate, Necrosis/debris)
- **Size**: 5,264 records (4,211 train, 1,053 test)
- **License**: CC0 1.0 Universal
- **Access**: Available on GitHub

## Video

### Cholec80
- **Description**: Cholecystectomy surgery videos
- **Task**: Surgery phase annotations and tool labels
- **Size**: 7,200 records (5,760 train, 1,440 test)
- **License**: Creative Commons Attribution-Noncommercial-Sharealike 4.0 International
- **Access**: Requires request form

## IMU (Inertial Measurement Unit)

### HuGaDB
- **Description**: Human gait analysis data
- **Classes**: 4 (Sitting, Standing, Sitting down, Standing up)
- **Size**: 364 records (291 train, 73 test)
- **License**: No specific license
- **Access**: Available on GitHub

## Gene Expression

### Expression Atlas
- **Description**: RNA gene expression data across species and biological conditions
- **Task**: Expression analysis and classification
- **Size**: 4,506 records (3,605 train, 901 test)
- **License**: Creative Commons Attribution-Noncommercial-NoDerivatives 4.0 International
- **Access**: Available through EBI

### Geo
- **Description**: Functional genomics dataset supporting MIAME-compliant data submissions
- **Task**: Expression analysis and classification
- **Size**: 126,452 records (101,162 train, 25,290 test)
- **License**: Custom NCBI license
- **Access**: Available through NCBI

## Multimodal

### Vital
- **Description**: Medical image-language dataset based on PMC-15, with GPT-4 generated instructions
- **Task**: Multimodal understanding and instruction following
- **Size**: 210,000 records (42,000 train, 168,000 test)
- **License**: Apache License 2.0
- **Access**: Available through Hugging Face

### MIMIC-IV
- **Description**: Multimodal medical dataset from Beth Israel Deaconess Medical Center
- **Modalities**: EHR, vital signs, chest X-rays
- **Classes**: 2 (48 Hour In-Hospital-Mortality: Yes/No)
- **Size**: 
  - Full dataset: 800,000 records (640,000 train, 160,000 test)
  - Multimodal subset: 207,769 records (166,215 train, 41,554 test)
- **License**: PhysioNet Credentialed Health Data License 1.5.0
- **Access**: Requires credentialed account and CITI training

## Usage Guidelines
1. Respect all data usage agreements and licenses
2. Follow proper citation practices when using these datasets
3. Maintain data privacy and security standards
4. Check for updated versions of datasets
5. Review specific ethical guidelines for each dataset

## Citation
Please cite individual datasets as well as the CLIMB benchmark when using this data. Specific citation information is provided in the original paper.