model_types=('RF' 'BRT' 'SVM') # 'RF'
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
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_recombined_aug_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_recombined_aug_SMILES --random_state 22
    
    # AUGMENTED RECOMBINED FRAG FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_recombined_aug_FP --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/manual_recombined_aug_fingerprint --random_state 22
    
    # AUTOMATED FRAG
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_test_[0-9].csv --feature_names polymer_automated_frag --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/automated_frag --random_state 22
    
    # AUTOMATED FRAG STR
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_test_[0-9].csv --feature_names polymer_automated_frag_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/automated_frag_SMILES --random_state 22
    
    # AUTOMATED AUGMENTED FRAG
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/automated_frag_aug --random_state 22
    
    # AUTOMATED AUGMENTED FRAG STR
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/automated_frag_aug_SMILES --random_state 22
    
    # AUTOMATED AUGMENTED RECOMBINED FRAG SMILES
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_recombined_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/automated_frag_aug_recombined_SMILES --random_state 22
    
    # AUTOMATED AUGMENTED RECOMBINED FRAG FINGERPRINT
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/automated_fragment/StratifiedKFold/input_test_[0-9].csv --feature_names polymer_automated_frag_aug_recombined_fp --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/automated_frag_aug_recombined_fp --random_state 22
    
    # DIMER FP
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/circular_fingerprint/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/circular_fingerprint/StratifiedKFold/input_test_[0-9].csv --feature_names 2mer_fp --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/dimer_fp --random_state 22
    
    # # TRIMER FP
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/circular_fingerprint/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/circular_fingerprint/StratifiedKFold/input_test_[0-9].csv --feature_names 3mer_fp --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/trimer_fp --random_state 22
    
    # # POLYMER GRAPH FP
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/circular_fingerprint/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/circular_fingerprint/StratifiedKFold/input_test_[0-9].csv --feature_names 3mer_circular_graph_fp --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/polymer_graph_fp --random_state 22
    
    # # OHE
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/ohe/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/ohe/StratifiedKFold/input_test_[0-9].csv --feature_names PS_ohe --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/ohe --random_state 22
    
    # SMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_SMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/SMILES --random_state 22
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_SELFIES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/SELFIES --random_state 22
    
    # BIGSMILES
    python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_BigSMILES --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/BigSMILES --random_state 22
    
    # SUM OF FRAGS
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Sum_of_Frags --target_name SD --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./swell_hpo_space.json --results_path ../../../training/Swelling_Xu/Sum_of_Frags --random_state 22
done
