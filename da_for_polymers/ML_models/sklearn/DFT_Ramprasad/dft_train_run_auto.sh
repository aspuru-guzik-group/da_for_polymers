model_types=('KRR') #'KRR'
for model in "${model_types[@]}"
do
    # AUGMENTED SMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/augmentation/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/augmentation/KFold/input_test_[0-9].csv --feature_names Augmented_SMILES --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/Augmented_SMILES --random_state 22
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/BRICS/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/BRICS/KFold/input_test_[0-9].csv --feature_names polymer_BRICS --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/BRICS --random_state 22
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/fingerprint/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/fingerprint/KFold/input_test_[0-9].csv --feature_names polymer_FP_radius_3_nbits_512 --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/fingerprint --random_state 22
    
    # automated FRAG
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_test_[0-9].csv --feature_names polymer_automated --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag --random_state 22
    
    # AUGMENTED automated FRAG
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_test_[0-9].csv --feature_names polymer_automated_aug --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug --random_state 22
    
    # automated FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_test_[0-9].csv --feature_names polymer_automated_SMILES --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_SMILES --random_state 22
    
    # AUGMENTED automated FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_test_[0-9].csv --feature_names polymer_automated_aug_SMILES --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug_SMILES --random_state 22
    
    # AUGMENTED RECOMBINED FRAG SMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_test_[0-9].csv --feature_names polymer_automated_recombined_aug_SMILES --target_name Egc --model_type "$model" --hyperparameter_optimization False --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_recombined_aug_SMILES --random_state 22
    
    # AUGMENTED RECOMBINED FRAG FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_frag/KFold/input_test_[0-9].csv --feature_names polymer_automated_recombined_aug_FP --target_name Egc --model_type "$model" --hyperparameter_optimization False --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_recombined_aug_fingerprint --random_state 22
    
    # SMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_test_[0-9].csv --feature_names polymer_SMILES --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/SMILES --random_state 22
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_test_[0-9].csv --feature_names polymer_SELFIES --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/SELFIES --random_state 22
    
    # BIGSMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_test_[0-9].csv --feature_names polymer_BigSMILES --target_name Egc --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/BigSMILES --random_state 22
    
done
