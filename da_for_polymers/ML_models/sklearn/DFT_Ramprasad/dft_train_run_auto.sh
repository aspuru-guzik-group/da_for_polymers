model_types=('RF' 'SVM' 'BRT') #'KRR'
for model in "${model_types[@]}"
do
    # AUGMENTED SMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/augmentation/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/augmentation/KFold/input_test_[0-9].csv --feature_names Augmented_SMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/Augmented_SMILES --random_state 22
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/BRICS/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/BRICS/KFold/input_test_[0-9].csv --feature_names polymer_BRICS --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/BRICS --random_state 22
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/fingerprint/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/fingerprint/KFold/input_test_[0-9].csv --feature_names DFT_FP_radius_3_nbits_512 --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/fingerprint --random_state 22
    
    # automated FRAG
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag --random_state 22
    
    # AUGMENTED automated FRAG
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug --random_state 22
    
    # AUGMENTED RECOMBINED FRAG SMILES
    python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_recombined_SMILES --target_name value --model_type "$model" --hyperparameter_optimization False --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug_recombined_SMILES --random_state 22
    
    # AUGMENTED RECOMBINED FRAG FINGERPRINT
    python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_recombined_fp --target_name value --model_type "$model" --hyperparameter_optimization False --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug_recombined_fp --random_state 22
    
    # automated FRAG STR
    python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_SMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_SMILES --random_state 22
    
    # AUGMENTED automated FRAG STR
    python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_SMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug_SMILES --random_state 22
    
    # SMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_test_[0-9].csv --feature_names polymer_SMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/SMILES --random_state 22
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_test_[0-9].csv --feature_names polymer_SELFIES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/SELFIES --random_state 22
    
    # BIGSMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_test_[0-9].csv --feature_names polymer_BigSMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/BigSMILES --random_state 22
    
done
