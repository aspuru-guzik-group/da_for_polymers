# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/CO2_Soleimani/manual_frag/master_manual_frag.csv --num_of_folds 7 --type_of_crossval StratifiedKFold --stratified_label Polymer

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_fragment/master_automated_fragment_Egc.csv --num_of_folds 5 --type_of_crossval KFold

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/manual_frag/master_manual_frag.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/manual_frag/master_manual_frag.csv --num_of_folds 7 --type_of_crossval StratifiedKFold --stratified_label Solvent

python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/fingerprint/ps_fingerprint.csv --num_of_folds 5 --type_of_crossval StratifiedKFold --stratified_label Solvent

python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/circular_fingerprint/ps_circular_fingerprint.csv --num_of_folds 5 --type_of_crossval StratifiedKFold --stratified_label Solvent

python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/ohe/master_ohe.csv --num_of_folds 5 --type_of_crossval StratifiedKFold --stratified_label Solvent