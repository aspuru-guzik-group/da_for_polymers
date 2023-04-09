model_types=('NN') # 'NN'
for model in "${model_types[@]}"
do
    # SMILES
    # python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/AUTOMATED_frag/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/AUTOMATED_frag/KFold/input_test_[0-9].csv --feature_names Polymer_SMILES --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/SMILES
    # # # AUGMENTED SMILES
    # python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/augmentation/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/augmentation/KFold/input_test_[0-9].csv --feature_names Augmented_SMILES --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/Augmented_SMILES
    # # BRICS
    # python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/BRICS/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/BRICS/KFold/input_test_[0-9].csv --feature_names Polymer_BRICS --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/BRICS
    # # # AUTOMATED FRAG
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/automated_frag
    # # AUGMENTED AUTOMATED FRAG
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/automated_frag_aug
    # AUGMENTED AUTOMATED FRAG STR
    # python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_recombined_SMILES --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/automated_frag_aug_recombined_SMILES
    # AUGMENTED AUTOMATED FRAG FINGERPRINT
    # python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_recombined_aug_fp --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/automated_frag_recombined_aug_fp
    # # FINGERPRINT
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/fingerprint/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/fingerprint/KFold/input_test_[0-9].csv --feature_names DFT_FP_radius_3_nbits_512 --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/fingerprint
    # # BigSMILES
    # python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/AUTOMATED_frag/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/AUTOMATED_frag/KFold/input_test_[0-9].csv --feature_names Polymer_BigSMILES --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/BigSMILES
    # SELFIES
    # python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/AUTOMATED_frag/KFold/input_train_[0-9].csv --test_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/DFT_Ramprasad/AUTOMATED_frag/KFold/input_test_[0-9].csv --feature_names Polymer_SELFIES --target_name value --model_type "$model" --model_config ../"$model"/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/DFT_Ramprasad/SELFIES --random_state 22
done