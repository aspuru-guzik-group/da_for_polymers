# Augmenting Polymer Datasets via Iterative Rearrangement

[![DOI](https://zenodo.org/badge/570637902.svg)](https://zenodo.org/badge/latestdoi/570637902)

Welcome to the GitHub repository for the paper: https://doi.org/10.26434/chemrxiv-2022-hxvcc

Abstract: One of the biggest obstacles to successful polymer property prediction is an effective representation that accurately captures the sequence of repeat units in a polymer. Motivated by the successes of data augmentation in computer vision and natural language processing, we explore augmenting polymer data by rearranging the molecular representation while preserving the correct connectivity, revealing additional substructural information that is not present in a single representation. We evaluate the effects of this technique on the performance of machine learning models trained on three experimental polymer datasets and compare them to common molecular representations. Data augmentation improves deep learning property prediction performance compared to equivalent (non-augmented) representations. In datasets where the target property is primarily influenced by the polymer sequence rather than experimental parameters, this data augmentation technique provides the molecular embedding with more information to improve property prediction accuracy.

Keywords: data augmentation, machine learning, polymers, molecular representation

![alt text](https://github.com/stanlo229/da_for_polymers/blob/main/TOC_1.png?raw=true)
![alt text](https://github.com/stanlo229/da_for_polymers/blob/main/TOC_2.png?raw=true)


## Getting Started
1. Fork, Clone, or Download this repository.
2. Create a conda environment.
3. In this directory,  `pip install -e .`
4. Access raw data  ->  `da_for_polymers/data/raw/Dataset`
5. Access data augmentation tool ->  `da_for_polymers/data/input_representation/Dataset/manual_frag/augment_manual_frag.py`
6. Access processed data ->  `da_for_polymers/data/input_representation/Dataset`
6. Access prediction results -> `da_for_polymers/training`
7. Run the shell files in `da_for_polymers/ML_models/pytorch/Dataset` or `da_for_polymers/ML_models/sklearn/Dataset` to run models on your setup. To run specific models or molecular representations, uncomment or comment specific lines. Recommened to understand how to use argument parsers for these scripts.
<<<<<<< HEAD
8. To view or re-create figures, go to -> `da_for_polymers/visualization`. Each figure in the paper can be recreated with the `recreate_all_plots.sh` file (read comments). (Supplementary Figures are found in `da_for_polymers/data/exploration`)g
=======
8. To view or re-create figures, go to -> `da_for_polymers/visualization`. Each figure in the paper can be recreated with the `recreate_all_plots.sh` file (read comments). (Supplementary Figures are found in `da_for_polymers/data/exploration`)
>>>>>>> 3518320fe8131a4d5c99874c5d2194ecbf421006
