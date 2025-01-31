import json
import os
from typing import Any, Dict, List, Optional

import GEOparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Geo(Dataset):
    '''A dataset class for the Geo dataset (https://www.ncbi.nlm.nih.gov/geo/info/download.html).
    Data is publicly available and can be downloaded from the FTP site provided in the link.
    '''
    # Dataset information.
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True,
                 in_subfolder=True) -> None:
        if in_subfolder:
            self.root = os.path.join(base_root, 'genom', 'ftp.ncbi.nlm.nih.gov', 'geo')
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
                'Visit https://www.ncbi.nlm.nih.gov/geo/info/download.html to download the data'
                """
            )
        return self.root

    def build_index(self):
        print('Building index...')
        folders = os.listdir(os.path.join(self.root, 'datasets'))
        self.ids = []
        for folder in folders:
            if folder.startswith('GDS'):
                subfolders = os.listdir(os.path.join(self.root, 'datasets', folder))
                for subfolder in subfolders:
                    self.ids.append(f'{folder}/{subfolder}')

    def get_informative_genes(self, table, samples_info, n_genes=5):
        """
        Select informative genes using multiple strategies.
        Works with any genotype/variation categories present in the data.
        """
        from scipy import stats
        try:
            # Get unique genotypes
            genotypes = samples_info['genotype/variation'].unique()
            if len(genotypes) < 2:
                logger.warning("Only one genotype found, falling back to random selection")
                return table.sample(n=min(n_genes, len(table)))

            # Calculate metrics for each gene
            gene_metrics = []
            for idx, row in table.iterrows():
                try:
                    gene_data = row[row.index.difference(['ID_REF', 'IDENTIFIER'])]

                    # Calculate group-wise statistics
                    group_stats = []
                    for genotype in genotypes:
                        group_samples = samples_info[samples_info['genotype/variation'] == genotype].index
                        group_expr = gene_data[group_samples].astype(float)
                        group_stats.append({
                            'mean': group_expr.mean(),
                            'std': group_expr.std(),
                            'cv': group_expr.std() / group_expr.mean() if group_expr.mean() != 0 else float('inf')
                        })

                    # 1. Calculate maximum fold change between any two groups
                    fold_changes = []
                    for i in range(len(group_stats)):
                        for j in range(i + 1, len(group_stats)):
                            if group_stats[i]['mean'] != 0 and group_stats[j]['mean'] != 0:
                                fold_changes.append(abs(group_stats[i]['mean'] - group_stats[j]['mean']))
                    max_fold_change = max(fold_changes) if fold_changes else 0

                    # 2. Calculate overall F-statistic for difference between groups
                    group_expressions = [
                        gene_data[samples_info[samples_info['genotype/variation'] == g].index].astype(float)
                        for g in genotypes]
                    f_stat, p_val = stats.f_oneway(*group_expressions)

                    # 3. Calculate consistency score (average CV across groups)
                    consistency_score = np.mean([stat['cv'] for stat in group_stats])

                    gene_metrics.append({
                        'gene_id': idx,
                        'identifier': row.get('IDENTIFIER', 'Unknown'),
                        'fold_change': max_fold_change,
                        'p_value': p_val,
                        'f_statistic': f_stat,
                        'consistency_score': consistency_score
                    })
                except Exception as e:
                    logger.warning(f"Error calculating metrics for gene {idx}: {e}")
                    continue

            # Convert to DataFrame
            metrics_df = pd.DataFrame(gene_metrics)

            # Select genes based on different criteria
            selected_genes = []

            # 1. Top genes by fold change (40% of selections)
            n_fold_change = max(1, int(n_genes * 0.4))
            fold_change_genes = metrics_df.nlargest(n_fold_change, 'fold_change')['gene_id'].tolist()
            selected_genes.extend(fold_change_genes)

            # 2. Most statistically significant genes (40% of selections)
            n_significant = max(1, int(n_genes * 0.4))
            significant_genes = metrics_df[
                (metrics_df['gene_id'].isin(fold_change_genes) == False) &
                (metrics_df['p_value'] < 0.05)
                ].nlargest(n_significant, 'f_statistic')['gene_id'].tolist()
            selected_genes.extend(significant_genes)

            # 3. Most consistent genes (20% of selections)
            n_consistent = max(1, n_genes - len(selected_genes))
            consistent_genes = metrics_df[
                ~metrics_df['gene_id'].isin(selected_genes)
            ].nsmallest(n_consistent, 'consistency_score')['gene_id'].tolist()
            selected_genes.extend(consistent_genes)

            # Fill remaining slots with top fold change genes if needed
            while len(selected_genes) < n_genes:
                remaining = metrics_df[
                    ~metrics_df['gene_id'].isin(selected_genes)
                ].nlargest(1, 'fold_change')['gene_id'].tolist()
                if not remaining:
                    break
                selected_genes.extend(remaining)

            # Get the selected rows from original table
            selected_df = table.loc[selected_genes]

            return selected_df

        except Exception as e:
            logger.info(f"Error in gene selection: {e}")
            # Fallback to random selection if smart selection fails
            return table.sample(n=min(n_genes, len(table)))

    def get_classification_info(self, samples_info):
        """Helper function to determine classification type and categories based on available columns"""
        if 'cell type' in samples_info.columns:
            return {
                'column': 'cell type',
                'type': 'cell type',
                'categories': sorted(samples_info['cell type'].unique())
            }
        elif 'strain' in samples_info.columns:
            samples_info['strain'] = samples_info['strain'].str.strip()
            return {
                'column': 'strain',
                'type': 'strain',
                'categories': sorted(samples_info['strain'].unique())
            }
        elif 'disease state' in samples_info.columns:
            return {
                'column': 'disease state',
                'type': 'disease state',
                'categories': sorted(samples_info['disease state'].unique())
            }
        elif 'genotype/variation' in samples_info.columns:
            return {
                'column': 'genotype/variation',
                'type': 'genotype',
                'categories': samples_info['genotype/variation'].unique()
            }
        elif 'tissue' in samples_info.columns:
            if 'development stage' in samples_info.columns:
                def categorize_tissue_development(row):
                    tissue = str(row['tissue']).lower()
                    stage = str(row['development stage']).lower()
                    if 'embryo' in tissue:
                        return f"embryo ({stage})"
                    elif 'fetus' in tissue:
                        return f"fetus ({stage})"
                    elif 'fetal membrane' in tissue:
                        return f"fetal membrane ({stage})"
                    return f"{tissue} ({stage})"

                samples_info['tissue_stage'] = samples_info.apply(categorize_tissue_development, axis=1)
                return {
                    'column': 'tissue_stage',
                    'type': 'tissue and development stage',
                    'categories': sorted(samples_info['tissue_stage'].unique())
                }
            return {
                'column': 'tissue',
                'type': 'tissue',
                'categories': samples_info['tissue'].unique()
            }
        elif 'agent' in samples_info.columns:
            if 'time' in samples_info.columns:
                def categorize_response(row):
                    if row['agent'] == 'none':
                        return 'control response'
                    elif row['agent'] == 'saline':
                        return 'sham exposure response'
                    elif row['agent'] == 'EHC-93':
                        time_str = str(row['time']).lower()
                        if any(x in time_str for x in ['2 to 6', 'early']):
                            return 'early response'
                        elif any(x in time_str for x in ['15 to 21', 'intermediate']):
                            return 'intermediate response'
                        elif any(x in time_str for x in ['23 to 40', 'late']):
                            return 'late response'
                    return 'other response'

                samples_info['response_category'] = samples_info.apply(categorize_response, axis=1)
                return {
                    'column': 'response_category',
                    'type': 'response',
                    'categories': sorted(samples_info['response_category'].unique())
                }
            return {
                'column': 'agent',
                'type': 'agent',
                'categories': samples_info['agent'].unique()
            }
        else:
            return None

    def to_qa(self):
        """
        Convert the GEO dataset to a question answering dataset focusing on genotype/variation prediction.
        Adapted for GDS (Dataset) format rather than GSE (Series) format.
        """
        logging.getLogger('GEOparse').setLevel(logging.WARNING)

        qa_data = []

        for i in tqdm.tqdm(range(len(self))):
            try:
                data_item = self[i]
                gds = data_item['data']

                # Get sample information from the columns DataFrame
                if not hasattr(gds, 'columns') or gds.columns.empty:
                    logger.warning(f"Dataset {data_item['id']} has no sample information, skipping")
                    continue

                # Get samples metadata
                samples_info = gds.columns

                # Determine classification type and categories
                classification_info = self.get_classification_info(samples_info)
                if classification_info is None:
                    logger.warning(f"Dataset {data_item['id']} has no suitable classification columns, skipping")
                    continue

                # Remove samples without classification information
                samples_info = samples_info.dropna(subset=[classification_info['column']])

                # Skip if no samples have classification information
                if samples_info.empty:
                    logger.warning(f"Dataset {data_item['id']} has no valid classification information, skipping")
                    continue

                try:
                    # Get expression data
                    table = gds.table

                    # Ensure we have some expression data
                    if table.empty:
                        logger.warning(f"Dataset {data_item['id']} has no expression data, skipping")
                        continue

                    # Get informative genes
                    try:
                        informative_genes = self.get_informative_genes(table, samples_info)
                    except Exception as e:
                        logger.error(f"Error getting informative genes: {e}, falling back to random selection")
                        informative_genes = table.sample(n=min(5, len(table)))

                    # Create one question per sample
                    for sample_id in samples_info.index:
                        try:
                            # Format the expression data as a string
                            expression_context = "Gene expression values (selected informative genes):\n"
                            for idx, row in informative_genes.iterrows():
                                try:
                                    gene_id = idx
                                    identifier = row.get('IDENTIFIER', 'Unknown')
                                    values = []
                                    for col in row.index:
                                        if col not in ['ID_REF', 'IDENTIFIER']:
                                            try:
                                                value = float(row[col])
                                                values.append(f"{col}: {value:.1f}")
                                            except (ValueError, TypeError):
                                                values.append(f"{col}: NA")
                                    values_str = ", ".join(values)
                                    expression_context += f"{gene_id} ({identifier}): {values_str}\n"
                                except Exception as e:
                                    logger.warning(f"Error processing gene {idx}: {e}")
                                    continue

                            # Add information about unique categories
                            categories = classification_info['categories']
                            category_context = f"Possible {classification_info['type']} categories in the dataset:\n"
                            for category in categories:
                                sample_count = sum(samples_info[classification_info['column']] == category)
                                category_context += f"- {category} ({sample_count} samples)\n"

                            # Create the question
                            if classification_info['type'] == 'response':
                                question_suffix = "What is the response category of"
                            elif classification_info['type'] == 'genotype':
                                question_suffix = "What is the genotype/variation of"
                            else:
                                question_suffix = f"What is the {classification_info['type']} of"

                            question = (
                                f"Given the following metadata and gene expression data from a microarray experiment:\n\n"
                                f"Study title: {gds.metadata.get('title', ['No title'])[0]}\n"
                                f"Description: {gds.metadata.get('description', ['No description'])[0]}\n\n"
                                f"{category_context}\n"
                                f"{expression_context}\n"
                                f"{question_suffix} sample {sample_id}? Choose from:\n"
                                f"{', '.join(categories)}"
                            )

                            # Create annotation
                            qa_item = {
                                'id': f"{data_item['id']}_{sample_id}",
                                'explanation': '',
                                'conversations': [
                                    {'from': 'human', 'value': question},
                                    {'from': 'gpt', 'value': samples_info.loc[sample_id, classification_info['column']]}
                                ]
                            }

                            qa_data.append(qa_item)

                        except Exception as e:
                            logger.error(f"Error processing sample {sample_id} in dataset {data_item['id']}: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Error processing dataset {data_item['id']}: {e}")
                    continue

            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                continue

            # Save annotations if we have any
        if qa_data:
            try:
                os.makedirs(self.root, exist_ok=True)
                with open(os.path.join(self.root, f'annotation_{self.split.lower()}.jsonl'), 'w') as f:
                    for qa in qa_data:
                        json.dump(qa, f)
                        f.write('\n')
            except Exception as e:
                logger.error(f"Error saving annotations: {e}")

        else:
            logger.warning("No valid QA pairs were generated")

        return qa_data

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Any:
        id_ = self.ids[index]
        sub_id = id_.split('/')[1]

        soft_path = os.path.join(self.root, 'datasets', id_, 'soft', f'{sub_id}.soft.gz')
        gse = GEOparse.get_GEO(filepath=soft_path)

        data = {
            'id': id_,
            'data': gse
        }

        return data
