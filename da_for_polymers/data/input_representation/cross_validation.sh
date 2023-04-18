# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/CO2_Soleimani/manual_frag/master_manual_frag.csv --num_of_folds 7 --type_of_crossval StratifiedKFold --stratified_label Polymer

python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/automated_fragment/master_automated_fragment.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/manual_frag/master_manual_frag.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent

# python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/manual_frag/master_manual_frag.csv --num_of_folds 7 --type_of_crossval StratifiedKFold --stratified_label Solvent

python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/fingerprint/pv_fingerprint.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent

python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/circular_fingerprint/pv_circular_fingerprint.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent

python ./cross_validation.py --dataset_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/PV_Wang/ohe/master_ohe.csv --num_of_folds 6 --type_of_crossval StratifiedKFold --stratified_label Solvent