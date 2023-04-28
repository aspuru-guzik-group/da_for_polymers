#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/sklearn/Swelling_Xu/slurm.out
#SBATCH --error=/project/6033559/stanlo/da_for_polymers/da_for_polymers/ML_models/sklearn/Swelling_Xu/slurm.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=12G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python swell_property_prediction.py