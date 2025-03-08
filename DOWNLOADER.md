# Currently supported downloaders:
- MIMIC CXR (MIMIC_CXR in src.datasets.chest_xray.mimic_cxr)
- VINDR CXR (VINDR_CXR in src.datasets.chest_xray.vindr_cxr)
- VINDR MAMMO (VINDR in src.datasets.mammo.vindr)
- BUSI (BUSI in src.datasets.ultrasound.busi)
- COVID19 (Covid19Dataset in src.datasets.chest_xray.covid19)
- Brain Tumor (BrainTumorDataset in src.datasets.mri.brain_tumor)
- Brain Tumor 2 (BinaryBrainTumorDataset in src.datasets.mri.brain_tumor_2)
- CoronaHack (CoronaHackDataset in src.datasets.chest_xray.coronahack)
- COVID BLUES (COVIDBLUES in src.datasets.ultrasound.covid_blue)
- COVID US (COVIDUS in src.datasets.ultrasound.covid_us)
# How to download:
## Downloading entire dataset
Create an object of the corresponding dataset class. An example is shown below.
```
d = BinaryBrainTumorDataset(base_root='data', download=True)
```
If we run this code from the root directory, the dataset will be downloaded to the 'data' directory.
Each dataset will be in its own directory, and the dataset directories are categorized by modality.
For example, Brain Tumor 2 will be downloaded to `data/mri/brain_tumor_2`.

## Downloading select files or directories
We support downloading select files from large datasets (MIMIC CXR, VINDR CXR, VINDR MAMMO).
In order to download select files, use the `file_list` parameter to pass in a list of directory/files (strings) to download.
Directories should only include the directory name and not the whole path.
Files should only include the file name without the extension (also no path). An example is shown below.
```
d = VINDR(download=True, base_root='data', file_list=['0025a5dc99fd5c742026f0b2b030d3e9',
                                                '16e58fc1d65fa7587247e6224ee96527',
                                                '7fc1f1bb8bb1a7efaf7104e49c4d8b86'])
```
`'0025a5dc99fd5c742026f0b2b030d3e9'` specifies everything inside the `0025a5dc99fd5c742026f0b2b030d3e9` directory.
`'16e58fc1d65fa7587247e6224ee96527'` specifies everything inside the `16e58fc1d65fa7587247e6224ee96527` directory.
`'7fc1f1bb8bb1a7efaf7104e49c4d8b86'` specifies the `7fc1f1bb8bb1a7efaf7104e49c4d8b86.dicom` file.

The downloader will not download a file multiple times, and will not download a file if it already exists.

Note that certain required files (e.g. metadata and licenses) will be downloaded (if they don't already exist)
even if not specified by the file_list parameter.
