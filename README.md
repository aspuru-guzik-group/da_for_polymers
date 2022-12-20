Augmenting Polymer Datasets via Iterative Rearrangement

[![DOI](https://zenodo.org/badge/570637902.svg)](https://zenodo.org/badge/latestdoi/570637902)

Welcome to the GitHub repository for the paper: https://doi.org/10.26434/chemrxiv-2022-hxvcc

![alt text](https://github.com/stanlo229/da_for_polymers/TOC.png?raw=true)

1. Fork, Clone, or Download this repository.
2. Create a conda environment.
3. In this directory,  `pip install -e .`
4. Access raw data  ->  `data/raw/Dataset`
5. Access data augmentation tool ->  `data/input_representation/Dataset/manual_frag/augment_manual_frag.py`
6. Access processed data ->  `data/input_representation/Dataset`
6. Access prediction results -> `training`
7. Run the shell files in `ML_models/pytorch/Dataset` or `ML_models/sklearn/Dataset` to run models on your setup. To run specific models or molecular representations, uncomment or comment specific lines. Recommened to understand how to use argument parsers for these scripts.
8. To view or re-create figures, go to -> `visualization`. Each figure in the paper can be recreated with the corresponding python file.