from src.datasets.chest_xray import chexpert, chexpert_full, ddi, mimic_cxr, vindr_cxr, xray8
from src.datasets.ct import lidc, lndb, radfusion, rsna
from src.datasets.derm import HAM10000, isic2019, isic2020, pad_ufes_20
from src.datasets.ecg import ChapmanShaoxing, CPSC, Ga, ptbxl
from src.datasets.eeg import isruc, shhs
from src.datasets.fundus import aptos, jinchi, messidor2
from src.datasets.mammo import cbis, vindr

DATASET_DICT = {
    # Chest X-Rays.
    'chexpert': chexpert.CheXpert,
    'xray8': xray8.ChestXray8,
    'mimic-cxr': mimic_cxr.MIMIC_CXR,
    'chexpert_full': chexpert_full.CheXpert_Full,
    'vindr-cxr': vindr_cxr.VINDR_CXR,

    # Computed Tomography.
    'rsna': rsna.RSNADatasetWindow,
    'radfusion': radfusion.RadfusionDatasetWindow,
    'lidc': lidc.LIDCDatasetWindow,
    'lndb': lndb.LNDb,

    # Opthamology images.
    'aptos': aptos.Aptos,
    'jinchi': jinchi.Jinchi,
    'messidor2': messidor2.Messidor2,

    # Mammography
    'cbis': cbis.CBIS,
    'vindr': vindr.VINDR,

    # Dermatology images.
    'isic2019': isic2019.ISIC2019,
    'isic2020': isic2020.ISIC2020,
    'ddi': ddi.ddi,
    'ham10000': HAM10000.HAM10000,
    'pad_ufes_20': pad_ufes_20.pad_ufes_20,
    
    # ECG
    'ptbxl': ptbxl.ptbxl,
    'ChapmanShaoxing': ChapmanShaoxing.ChapmanShaoxing,
    'CPSC': CPSC.CPSC,
    'Ga': Ga.Ga,

    #EEG
    'shhs': shhs.SHHS,
    'isruc': isruc.ISRUC,
}

PRETRAINING_DATASETS = [
    'chexpert', 'mimic-cxr', 'isic2019', 'ptbxl', 'messidor2', 'vindr', 'rsna', 'lidc', 'aptos', 'shhs'
]

UNLABELED_DATASETS = ['wikitext103', 'librispeech', 'mc4']
MULTILABEL_DATASETS = ['chexpert', 'vqa', 'chexpert_full', 'mimic-cxr', 'vindr-cxr', 'lidc', 'lndb']

TRANSFER_DATASETS = [
    # Chest X-ray
    'chexpert',
    'xray8',
    'chexpert_full',
    'vindr-cxr',
    'mimic-cxr',
    #dermatology.
    'ddi',
    'ham10000',
    'isic2020',
    'pad_ufes_20',
    'isic2019',

    # Computed Tomography.
    'rsna',
    'radfusion',
    'lidc',
    'lndb',

    # Opthamology
    'aptos',
    'jinchi',
    'messidor2',

    # Mammography
    'cbis',
    'vindr',

    # ECG
    'ptbxl',
    'ChapmanShaoxing',
    'CPSC',
    'Ga',

    #EEG
    'shhs',
    'mros',
    'isruc'
]
