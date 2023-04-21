model_types=('RF' 'BRT' 'SVM') #'KRR' # 'RF' 'SVM'
for model in "${model_types[@]}"
do
    # AUGMENTED SMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/augmentation/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/augmentation/KFold/input_test_[0-9].csv --feature_names Augmented_SMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/Augmented_SMILES --random_state 22
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/BRICS/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/BRICS/KFold/input_test_[0-9].csv --feature_names Polymer_BRICS --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/BRICS --random_state 22
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/fingerprint/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/fingerprint/KFold/input_test_[0-9].csv --feature_names DFT_FP_radius_3_nbits_512 --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/fingerprint --random_state 22
    
    # automated FRAG
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag --random_state 22
    
    # AUGMENTED automated FRAG
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug --random_state 22
    
    # AUGMENTED RECOMBINED FRAG SMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_recombined_SMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug_recombined_SMILES --random_state 22
    
    # AUGMENTED RECOMBINED FRAG FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_recombined_fp --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug_recombined_fp --random_state 22
    
    # automated FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_SMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_SMILES --random_state 22
    
    # AUGMENTED automated FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/automated_fragment/KFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_SMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/automated_frag_aug_SMILES --random_state 22
    
    # DIMER FP
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/circular_fingerprint/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/circular_fingerprint/KFold/input_test_[0-9].csv --feature_names 2mer_fp --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/dimer_fp --random_state 22
    
    # TRIMER FP
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/circular_fingerprint/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/circular_fingerprint/KFold/input_test_[0-9].csv --feature_names 3mer_fp --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/trimer_fp --random_state 22
    
    # POLYMER GRAPH FP
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/circular_fingerprint/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/circular_fingerprint/KFold/input_test_[0-9].csv --feature_names 3mer_circular_graph_fp --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/polymer_graph_fp --random_state 22
    
    # OHE
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/ohe/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/ohe/KFold/input_test_[0-9].csv --feature_names Polymer_ohe --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/ohe --random_state 22
    
    # SMILES
    python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_test_[0-9].csv --feature_names smiles --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/SMILES --random_state 22
    
    # SELFIES
    python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_test_[0-9].csv --feature_names selfies --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/SELFIES --random_state 22
    
    # BIGSMILES
    # python ../train.py --train_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_train_[0-9].csv --test_path ../../../data/input_representation/DFT_Ramprasad/SMILES/KFold/input_test_[0-9].csv --feature_names BigSMILES --target_name value --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./dft_hpo_space.json --results_path ../../../training/DFT_Ramprasad/BigSMILES --random_state 22
    
done
