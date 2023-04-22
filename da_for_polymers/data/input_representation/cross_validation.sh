# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/CO2_Soleimani/automated_fragment/master_automated_fragment.csv --num_of_folds 7 --type_of_crossval StratifiedKFold --stratified_label Polymer

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/augmentation/train_aug_master.csv --num_of_folds 5 --type_of_crossval KFold

python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_fragment/master_automated_fragment_Egc.csv --num_of_folds 5 --type_of_crossval KFold

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/SMILES/dft_smiles.csv --num_of_folds 5 --type_of_crossval KFold

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/SMILES/master_smiles.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/SMILES/master_smiles.csv --num_of_folds 7 --type_of_crossval StratifiedKFold --stratified_label Solvent

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/fingerprint/pv_fingerprint.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/circular_fingerprint/pv_circular_fingerprint.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/ohe/master_ohe.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent