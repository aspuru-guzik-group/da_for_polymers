#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/pytorch/PV_Wang/slurm.out
#SBATCH --error=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/pytorch/PV_Wang/slurm.err
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
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_SMILES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/SMILES
    
    # # AUGMENTED SMILES
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/augmentation/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/augmentation/StratifiedKFold/input_test_[0-9].csv --feature_names Augmented_SMILES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/Augmented_SMILES
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/BRICS/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/BRICS/StratifiedKFold/input_test_[0-9].csv --feature_names PS_pair_BRICS,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/BRICS
    
    # # MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/manual_frag
    
    # AUGMENTED MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_aug,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/manual_frag_aug
    
    # MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_SMILES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/manual_frag_SMILES
    
    # AUGMENTED MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_aug_SMILES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/manual_frag_aug_SMILES
    
    # # AUGMENTED RECOMBINED FRAG SMILES
    python ../train.py --train_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_recombined_aug_SMILES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/manual_recombined_aug_SMILES
    
    # # AUGMENTED RECOMBINED FRAG FINGERPRINT
    python ../train.py --train_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_recombined_aug_FP,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/manual_recombined_aug_fingerprint
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/fingerprint/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/fingerprint/StratifiedKFold/input_test_[0-9].csv --feature_names PS_FP_radius_3_nbits_512,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/fingerprint
    
    # BigSMILES
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_BigSMILES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/BigSMILES
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_SELFIES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/PV_Wang/SELFIES --random_state 22
done