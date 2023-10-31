# ML-SPM
Machine learning for scanning probe microscopy

## Installation

Install pre-requisites:
```
sudo apt install g++ git
```

Clone repository:
```
git clone https://github.com/SINGROUP/ml-spm.git
cd ml-spm
```

(Recommended) Create conda environment:
```
conda env create -f environment.yml
conda activate ml-spm
```

Install:
```
pip install .
```

## Papers
The [`papers`](papers) subdirectory contains training scripts and datasets for specific publications. Currently we have the following:
- [Structure discovery in Atomic Force Microscopy imaging of ice](papers/ice_structure_discovery)
