import pkg_resources
import pandas as pd
import selfies as sf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

DFT_EXPT_RESULT = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/DFT_Ramprasad/dft_exptresults_Egc.csv"
)

DFT_SMILES = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/DFT_Ramprasad/SMILES/MASTER_Smiles.csv",
)

DFT_OHE_PATH = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/DFT_Ramprasad/ohe/master_ohe.csv",
)


def smi2selfies(filepath: str):
    """
    Function that will convert SMILES to SELFIES
    Args:
        filepath: path to data with experimental results from Polymer Swelling paper

    Returns:
        MASTER_Smiles.csv will have SELFIES
    """
    # initialize polymer swelling dataframe
    data: pd.DataFrame = pd.read_csv(filepath)
    polymer_selfies = data["smiles"].apply(lambda x: sf.encoder(x))
    data["selfies"] = polymer_selfies

    data.to_csv(filepath, index=False)


def create_master_ohe(filepath: str, dft_ohe_path: str):
    """
    Generate a function that will one-hot encode the all of the polymer and solvent molecules. Each unique molecule has a unique number.
    Create one new column for the polymer and solvent one-hot encoded data.
    """
    master_df: pd.DataFrame = pd.read_csv(filepath)
    polymer_ohe = OneHotEncoder()
    solvent_ohe = OneHotEncoder()
    polymer_ohe.fit(master_df["smiles"].values.reshape(-1, 1))
    polymer_ohe_data = polymer_ohe.transform(master_df["smiles"].values.reshape(-1, 1))
    # print(f"{polymer_ohe_data=}")
    master_df["Polymer_ohe"] = polymer_ohe_data.toarray().tolist()
    # print(f"{master_df.head()}")
    # combine polymer and solvent ohe data into one column
    master_df.to_csv(dft_ohe_path, index=False)


def cli_main():
    # smi2selfies(DFT_EXPT_RESULT)
    create_master_ohe(DFT_EXPT_RESULT, DFT_OHE_PATH)


if __name__ == "__main__":
    cli_main()
