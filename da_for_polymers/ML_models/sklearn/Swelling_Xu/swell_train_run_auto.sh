model_types=('RF' 'BRT' 'SVM' 'KRR') #'KRR'
for model in "${model_types[@]}"
do
    # AUGMENTED SMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/augmentation/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/augmentation/StratifiedKFold/input_test_[0-9].csv --feature_names Augmented_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/Augmented_SMILES --random_state 22
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/BRICS/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/BRICS/StratifiedKFold/input_test_[0-9].csv --feature_names PS_pair_BRICS --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/BRICS --random_state 22
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/fingerprint/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/fingerprint/StratifiedKFold/input_test_[0-9].csv --feature_names PS_FP_radius_3_nbits_512 --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/fingerprint --random_state 22
    
    # MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_frag --random_state 22
    
    # AUGMENTED MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_aug --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_frag_aug --random_state 22
    
    # MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_frag_SMILES --random_state 22
    
    # AUGMENTED MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_aug_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_frag_aug_SMILES --random_state 22
    
    # AUGMENTED RECOMBINED FRAG SMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_recombined_aug_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization False --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_recombined_aug_SMILES --random_state 22
    
    # AUGMENTED RECOMBINED FRAG FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_recombined_aug_FP --target_name SD --model_type "$model" --hyperparameter_optimization False --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_recombined_aug_fingerprint --random_state 22
    
    # SMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/SMILES --random_state 22
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_SELFIES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/SELFIES --random_state 22
    
    # BIGSMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_BigSMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/BigSMILES --random_state 22
    
    # SUM OF FRAGS
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Sum_of_Frags --target_name SD --model_type "$model" --hyperparameter_optimization False --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/Sum_of_Frags --random_state 22
done
