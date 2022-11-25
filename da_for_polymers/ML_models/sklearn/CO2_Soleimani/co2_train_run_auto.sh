model_types=('RF' 'BRT' 'SVM')
for model in "${model_types[@]}"
do
    # AUGMENTED SMILES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/augmentation/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/augmentation/StratifiedKFold/input_test_[0-9].csv --feature_names Augmented_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/Augmented_SMILES --random_state 22
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_BRICS,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/BRICS --random_state 22
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/fingerprint/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/fingerprint/StratifiedKFold/input_test_[0-9].csv --feature_names CO2_FP_radius_3_nbits_512,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/fingerprint --random_state 22
    
    # MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_frag --random_state 22
    
    # AUGMENTED MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_aug,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_frag_aug --random_state 22
    
    # MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_frag_SMILES --random_state 22
    
    # AUGMENTED MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_aug_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_frag_aug_SMILES --random_state 22
    
    # AUGMENTED RECOMBINED FRAG SMILES
    python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_recombined_aug_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_recombined_aug_SMILES --random_state 22
    
    # AUGMENTED RECOMBINED FRAG FINGERPRINTS
    python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_recombined_aug_FP,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_recombined_aug_fingerprint --random_state 22
    
    #SMILES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/SMILES --random_state 22
    
    # BIGSMILES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_SELFIES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/SELFIES --random_state 22
    
    #SELFIES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_BigSMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/BigSMILES --random_state 22
    
    # BIGSMILES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_SELFIES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/SELFIES --random_state 22
    
    #SELFIES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_BigSMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/BigSMILES --random_state 22
done