import pkg_resources
import pandas as pd
import selfies as sf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import ast

DFT_EXPT_RESULT = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/DFT_Ramprasad/dft_exptresults_Egc.csv"
)

DFT_SMILES = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/DFT_Ramprasad/SMILES/dft_smiles.csv",
)

DFT_OHE_PATH = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/DFT_Ramprasad/ohe/master_ohe.csv",
)

DFT_AUTOMATED_FRAG = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/DFT_Ramprasad/automated_fragment/master_automated_fragment_Egc.csv",
)


def smi2selfies(filepath: str, smiles_path: str):
    """
    Function that will convert SMILES to SELFIES
    Args:
        filepath: path to data with experimental results from Polymer Swelling paper

    Returns:
        MASTER_Smiles.csv will have SELFIES
    """
    # initialize polymer swelling dataframe
    data: pd.DataFrame = pd.read_csv(filepath)
    # remove [*] and ([*]) from smiles
    data["smiles_no_dummy"] = data["smiles"].apply(lambda x: x.replace("([*])", "(C)"))
    data["smiles_no_dummy"] = data["smiles_no_dummy"].apply(
        lambda x: x.replace("[*]", "C")
    )
    polymer_selfies = data["smiles_no_dummy"].apply(lambda x: sf.encoder(x))
    data["selfies"] = polymer_selfies

    data.to_csv(smiles_path, index=False)


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


def bigsmiles_from_frag(dft_automated_frag: str, dft_smiles: str):
    """
    Function that takes ordered fragments (manually by hand) and converts it into BigSMILES representation, specifically block copolymers
    Args:
        dft_automated_frag: path to data with automated fragmented polymers

    Returns:
        concatenates fragments into BigSMILES representation and returns to data
    """
    # polymer/mixture BigSMILES
    data = pd.read_csv(dft_automated_frag)
    smi_data = pd.read_csv(dft_smiles)
    smi_data["BigSMILES"] = ""

    for index, row in data.iterrows():
        big_smi = "{[][<]"
        position = 0
        if len(ast.literal_eval(data["polymer_automated_frag"][index])) == 1:
            big_smi = ast.literal_eval(data["polymer_automated_frag"][index])[0]
        else:
            for frag in ast.literal_eval(data["polymer_automated_frag"][index]):
                big_smi += str(frag)
                if (
                    position
                    == len(ast.literal_eval(data["polymer_automated_frag"][index])) - 1
                ):
                    big_smi += "[>][]}"
                else:
                    big_smi += "[>][<]}{[>][<]"
                position += 1

        smi_data.at[index, "BigSMILES"] = big_smi

    smi_data.to_csv(dft_smiles, index=False)


def cli_main():
    # smi2selfies(DFT_EXPT_RESULT, DFT_SMILES)
    # create_master_ohe(DFT_EXPT_RESULT, DFT_OHE_PATH)
    bigsmiles_from_frag(DFT_AUTOMATED_FRAG, DFT_SMILES)


if __name__ == "__main__":
    cli_main()
