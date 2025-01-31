import gzip
import os
from typing import Any, Dict, List, Optional

import GEOparse
import numpy as np
import pandas as pd
import scipy.io as sio
from torch.utils.data import Dataset


def load_tsv(file_path):
    """Load TSV file"""
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path, sep='\t', index_col=0)
        else:
            # print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading TSV file: {e}")
        return None

def load_gzipped_tsv(file_path):
    """Load gzipped TSV file"""
    try:
        if os.path.exists(file_path):
            with gzip.open(file_path, 'rt') as f:
                return pd.read_csv(f, sep='\t')
        else:
            # print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading gzipped TSV file: {e}")
        return None

def load_xml(file_path):
    """Load XML file"""
    try:
        if os.path.exists(file_path):
            tree = ET.parse(file_path)
            return tree.getroot()
        else:
            print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading XML file: {e}")
        return None

def load_bedgraph(file_path):
    """Load BedGraph file"""
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path, sep='\t', header=None,
                               names=['chrom', 'start', 'end', 'value'])
        else:
            # print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading BedGraph file: {e}")
        return None


class ExpressionAtlas(Dataset):
    '''A dataset class for the ExpressionAtlas dataset (https://www.ebi.ac.uk/gxa/download).
    Data is publicly available and can be downloaded from the FTP site provided in the link.
    '''
    # Dataset information.
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'genom', 'expression_atlas', 'experiments')
        else:
            self.root = base_root
        super().__init__()
        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'
        self.ids = []
        self.x: Dict[str, pd.DataFrame] = {}  # file name -> data
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        # if no data is present, prompt the user to download it
        if len(os.listdir(self.root)) == 0:
            raise RuntimeError(
                """
                'Visit https://www.ebi.ac.uk/gxa/download to download the data'
                """
            )
        return self.root

    def build_index(self):
        print('Building index...')
        dirs = os.listdir(self.root)
        self.ids = []

        for experiment_name in dirs:
            if experiment_name.startswith('E-') and os.path.isdir(os.path.join(self.root, experiment_name)):
                self.ids.append(experiment_name)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Any:
        experiment_name = self.ids[index]
        base_path = os.path.join(self.root, experiment_name)

        file_paths = {
            # RNA-seq specific files
            'tpms': f"{experiment_name}-tpms.tsv",
            'fpkms': f"{experiment_name}-fpkms.tsv",
            'tpms_coexpressions': f"{experiment_name}-tpms-coexpressions.tsv.gz",
            'fpkms_coexpressions': f"{experiment_name}-fpkms-coexpressions.tsv.gz",

            # Microarray specific files
            'normalized_expressions': f"{experiment_name}_A-AFFY-2-normalized-expressions.tsv",
            'analytics': f"{experiment_name}_A-AFFY-2-analytics.tsv",

            # Common files
            'metadata': f"{experiment_name}.condensed-sdrf.tsv",
            'configuration': f"{experiment_name}-configuration.xml",

            # GSEA results
            'go_gsea_g1_g2': f"{experiment_name}.g1_g2.go.gsea.tsv",
            'go_gsea_g1_g3': f"{experiment_name}.g1_g3.go.gsea.tsv",
            'interpro_gsea_g1_g2': f"{experiment_name}.g1_g2.interpro.gsea.tsv",
            'interpro_gsea_g1_g3': f"{experiment_name}.g1_g3.interpro.gsea.tsv",
            'reactome_gsea_g1_g2': f"{experiment_name}.g1_g2.reactome.gsea.tsv",
            'reactome_gsea_g1_g3': f"{experiment_name}.g1_g3.reactome.gsea.tsv",

            # BedGraph files
            'log2foldchange_g1_g2': f"{experiment_name}.g1_g2.genes.log2foldchange.bedGraph",
            'pval_g1_g2': f"{experiment_name}.g1_g2.genes.pval.bedGraph",
            'log2foldchange_g1_g3': f"{experiment_name}.g1_g3.genes.log2foldchange.bedGraph",
            'pval_g1_g3': f"{experiment_name}.g1_g3.genes.pval.bedGraph"
        }

        data = {}

        for key, file_name in file_paths.items():
            full_path = os.path.join(base_path, file_name)
            if key == 'configuration':
                data[key] = load_xml(full_path)
            elif key.endswith(('g1_g2', 'g1_g3')) and 'gsea' not in key:
                data[key] = load_bedgraph(full_path)
            elif key.endswith('coexpressions'):
                data[key] = load_gzipped_tsv(full_path)
            else:
                data[key] = load_tsv(full_path)

        return data
