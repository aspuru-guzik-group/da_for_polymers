#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/pytorch/Swelling_Xu/slurm.out
#SBATCH --error=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/pytorch/Swelling_Xu/slurm.err
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
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names Polymer_SMILES --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/SMILES
    
    # AUGMENTED SMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/augmentation/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/augmentation/StratifiedKFold/input_test_[0-9].csv --feature_names Augmented_SMILES --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/Augmented_SMILES
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/BRICS/StratifiedKFold/input_train_[0].csv --test_path ../../../data/input_representation/Swelling_Xu/BRICS/StratifiedKFold/input_test_[0].csv --feature_names PS_pair_BRICS --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/BRICS
    
    # MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/manual_frag
    
    # AUGMENTED MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_aug --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/manual_frag_aug
    
    # MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_SMILES --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/manual_frag_SMILES
    
    # AUGMENTED MANUAL FRAG STR
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_aug_SMILES --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/manual_frag_aug_SMILES
    
    # AUGMENTED RECOMBINED FRAG SMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_recombined_aug_SMILES --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/manual_recombined_aug_SMILES
    
    # AUGMENTED RECOMBINED FRAG FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_test_[0-9].csv --feature_names PS_manual_recombined_aug_FP --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/manual_recombined_aug_fingerprint
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/fingerprint/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/fingerprint/StratifiedKFold/input_test_[0-9].csv --feature_names PS_FP_radius_3_nbits_512 --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/fingerprint
    
    # BigSMILES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_BigSMILES --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/BigSMILES
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --test_path ../../../data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_test_[0-9].csv --feature_names PS_SELFIES --target_name SD --model_type "$model" --model_config ../"$model"/model_config.json --results_path ../../../training/Swelling_Xu/SELFIES --random_state 22
done