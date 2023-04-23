import numpy as np
import pandas as pd
import pkg_resources
from rdkit import Chem

AUTO_FRAG = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/automated_fragment/master_automated_fragment.csv",
)

MASTER_SMILES_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/SMILES/master_smiles.csv",
)


def combine_polymer_solvent(data_csv_path, master_smi_path):
    """
    Args:
        data_csv_path (str): filepath to .csv data with SMILES and features and target value
        master_smi_path (str): filepath to .csv data with combined Polymer and Solvent SMILES, SELFIES, BigSMILES

    Returns:
        .csv file with combined Polymer and Solvent SMILES, SELFIES, BigSMILES
    """

    data_df = pd.read_csv(data_csv_path)
    data_df = data_df.drop(
        columns=[
            "polymer_automated_frag",
            "polymer_automated_frag_SMILES",
            "polymer_automated_frag_aug",
            "polymer_automated_frag_aug_SMILES",
            "polymer_automated_frag_aug_recombined_SMILES",
            "polymer_automated_frag_aug_recombined_fp",
            "PS_automated_frag",
            "PS_automated_frag_SMILES",
            "PS_automated_frag_aug",
            "PS_automated_frag_aug_SMILES",
            "PS_automated_frag_aug_recombined_SMILES",
            "PS_automated_frag_aug_recombined_fp",
        ]
    )

    data_df["PS_SMILES"] = ""
    data_df["PS_SELFIES"] = ""
    data_df["PS_BigSMILES"] = ""

    for i, row in data_df.iterrows():
        data_df.at[i, "PS_SMILES"] = (
            data_df.at[i, "Polymer_SMILES"] + "." + data_df.at[i, "Solvent_SMILES"]
        )
        data_df.at[i, "PS_SELFIES"] = (
            data_df.at[i, "Polymer_SELFIES"] + "." + data_df.at[i, "Solvent_SELFIES"]
        )
        data_df.at[i, "PS_BigSMILES"] = (
            data_df.at[i, "Polymer_BigSMILES"] + "." + data_df.at[i, "Solvent_SMILES"]
        )
    data_df.to_csv(master_smi_path, index=False)


combine_polymer_solvent(AUTO_FRAG, MASTER_SMILES_DATA)
