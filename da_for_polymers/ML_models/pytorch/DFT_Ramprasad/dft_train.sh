#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/pytorch/CO2_Soleimani/slurm.out
#SBATCH --error=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/pytorch/CO2_Soleimani/slurm.err
#SBATCH --account=rrg-aspuru
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=4
#SBATCH --mem=12G
module load python/3.9.6
source /project/6025683/stanlo/opv_project/bin/activate
model_types=('LSTM' 'NN') #'NN'
for model in "${model_types[@]}"
do
    # SMILES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/SMILES
    
    # AUGMENTED SMILES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/augmentation/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/augmentation/StratifiedKFold/input_test_[0-9].csv --feature_names Augmented_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/Augmented_SMILES
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_BRICS,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/BRICS
    
    # MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/manual_frag
    
    # AUGMENTED MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_aug,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/manual_frag_aug
    
    # MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/manual_frag_SMILES
    
    # AUGMENTED MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_aug_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/manual_frag_aug_SMILES
    
    # AUGMENTED RECOMBINED FRAG SMILES
    python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_recombined_aug_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/manual_recombined_aug_SMILES
    
    # AUGMENTED RECOMBINED FRAG FINGERPRINTS
    python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_manual_recombined_aug_FP,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/manual_recombined_aug_fingerprint
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/fingerprint/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/fingerprint/StratifiedKFold/input_test_[0-9].csv --feature_names CO2_FP_radius_3_nbits_512,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/fingerprint
    
    # BigSMILES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_BigSMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/BigSMILES
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_SELFIES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/CO2_Soleimani/SELFIES --random_state 22
done