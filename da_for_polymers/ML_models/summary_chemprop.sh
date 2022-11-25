python summary_chemprop.py --ground_truth ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/augmentation/KFold/chemprop_test_*.csv --predictions ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/augmentation/KFold/predictions_*.csv --target calc_PCE_percent --features Augmented_SMILES

python summary_chemprop.py --ground_truth ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/manual_frag_aug_str/KFold/chemprop_test_*.csv --predictions ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/manual_frag_aug_str/KFold/predictions_*.csv --target calc_PCE_percent --features DA_manual_aug_str

python summary_chemprop.py --ground_truth ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/manual_frag_str/KFold/chemprop_test_*.csv --predictions ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/manual_frag_str/KFold/predictions_*.csv --target calc_PCE_percent --features DA_manual_str

python summary_chemprop.py --ground_truth ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/SMILES/KFold/chemprop_test_*.csv --predictions ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/SMILES/KFold/predictions_*.csv --target calc_PCE_percent --features DA_SMILES

python summary_chemprop.py --ground_truth ~/Research/Repos/polymer_chemprop_data/results/Swelling_Xu/augmentation/StratifiedKFold/chemprop_test_*.csv --predictions ~/Research/Repos/polymer_chemprop_data/results/Swelling_Xu/augmentation/StratifiedKFold/predictions_*.csv --target SD --features Augmented_SMILES

python summary_chemprop.py --ground_truth ~/Research/Repos/polymer_chemprop_data/results/Swelling_Xu/manual_frag_aug_str/StratifiedKFold/chemprop_test_*.csv --predictions ~/Research/Repos/polymer_chemprop_data/results/Swelling_Xu/manual_frag_aug_str/StratifiedKFold/predictions_*.csv --target SD --features PS_manual_aug_str

python summary_chemprop.py --ground_truth ~/Research/Repos/polymer_chemprop_data/results/Swelling_Xu/manual_frag_str/StratifiedKFold/chemprop_test_*.csv --predictions ~/Research/Repos/polymer_chemprop_data/results/Swelling_Xu/manual_frag_str/StratifiedKFold/predictions_*.csv --target SD --features PS_manual_str

python summary_chemprop.py --ground_truth ~/Research/Repos/polymer_chemprop_data/results/Swelling_Xu/SMILES/StratifiedKFold/chemprop_test_*.csv --predictions ~/Research/Repos/polymer_chemprop_data/results/Swelling_Xu/SMILES/StratifiedKFold/predictions_*.csv --target SD --features PS_SMILES