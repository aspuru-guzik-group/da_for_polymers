import pkg_resources
import pandas as pd
import selfies as sf
import numpy as np

DFT_EXPT_RESULT = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/DFT_Ramprasad/dft_exptresults_Egc.csv"
)

DFT_SMILES = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/DFT_Ramprasad/SMILES/MASTER_Smiles.csv",
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


def cli_main():
    smi2selfies(DFT_EXPT_RESULT)


if __name__ == "__main__":
    cli_main()
