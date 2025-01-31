# ECG-JEPA: Learning General Representation of 12-Lead Electrocardiogram With a Joint-Embedding Predictive Architecture
Official implementation for ECG-JEPA.
[\[arXiv\]](https://arxiv.org/pdf/2410.08559)

### Model Performance

<img src="./model_comparison.png" alt="Linear Evaluation on PTB-XL vs GPU Hours" style="width: 50%; height: auto;" />


### Installation
```console
conda create --name ecg_jepa python=3.9
conda activate ecg_jepa
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
git clone https://github.com/sehunfromdaegu/ECG_JEPA.git
cd ECG_JEPA
pip install -r requirements.txt
```

## Tutorial
tutorial.ipynb demonstrates how to use the pretrained ECG-JEPA.

## Pretraining
### Pretrain the Model Yourself
To pretrain the ECG-JEPA model, run one of the following commands in the terminal:

For random masking\
```console
python pretrain_ECG_JEPA.py --mask_type random --mask_scale 0.6 0.7 --batch_size 128 --lr 2.5e-5 --data_dir_shao PATH_TO_SHAOXING --data_dir_code15 PATH_TO_CODE15
```

For multi-block masking:\
```console
python pretrain_ECG_JEPA.py --mask_type block --mask_scale 0.175 0.225 --batch_size 64 --lr 5.5e-5 --data_dir_shao PATH_TO_SHAOXING --data_dir_code15 PATH_TO_CODE15
```

- `PATH_TO_SHAOXING` should be the path to the directory `path_to_data/ecg-arrhythmia/1.0.0/WFDBRecords`.
- `PATH_TO_CODE15` should be the directory containing files like `exams_part0.hdf5`, `exams_part1.hdf5`, ..., `exams_part17.hdf5`.

### Pretrained Checkpoints
Pretrained checkpoints are available at the following links:

- ECG-JEPA (random masking): [Download Link](https://drive.google.com/file/d/1mh-XL0XOvvhFbhvuZ9c2KnTHa9B4F3Wx/view?usp=drive_link)
- ECG-JEPA (multi-block masking): [Download Link](https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view?usp=drive_link)
  
Please download the pretrained checkpoints and save them in the ./weights/

## Downstream Datasets
Information about the datasets used in downstream tasks:

1. PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
2. CPSC2018: https://physionet.org/content/challenge-2020/1.0.2/sources/

Download the datasets using the following commands:

For PTB-XL:
```console
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```
For CPSC2018:
```console
wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/#files-panel
```


## Downstream Tasks
### Linear Evaluation 
For linear evaluation on the PTB-XL multi-label task:
```console
cd downstream_tasks
python linear_eval.py --ckpt_dir CKPT_DIR --dataset ptbxl --data_dir PATH_TO_PTBXL --task multilabel
```
`CKPT_DIR` is the file location of the pretrained weights.

Log files are saved in `./downstream_tasks/output/linear_eval/`.

- `PATH_TO_PTBXL`: Directory containing [records100, records500, index.html, LICENSE.txt, RECORDS]



Alternatively, replace `PATH_TO_PTBXL` with `PATH_TO_CPSC2018`, which should contain subdirectories `g1`, `g2`, ..., `g7`.

### Fine-Tuning
For fine-tuning on the PTB-XL multi-class task:

*ECG-JEPA with random masking:*
```console
cd downstream_tasks
python finetuning.py --model_name ejepa_random --dataset ptbxl --data_dir PATH_TO_PTBXL --task multiclass
```

*ECG-JEPA with multiblock masking:*
```console
cd downstream_tasks
python finetuning.py --model_name ejepa_multiblock --dataset ptbxl --data_dir PATH_TO_PTBXL --task multiclass
```
