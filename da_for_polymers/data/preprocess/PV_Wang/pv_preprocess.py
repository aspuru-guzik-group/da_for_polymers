import pkg_resources
import pandas as pd
import selfies as sf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

PV_INVENTORY = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/PV_Wang/pv_inventory.csv"
)

PV_EXPT_RESULT = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/PV_Wang/pv_exptresults.csv"
)

PV_OHE_PATH = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/PV_Wang/ohe/master_ohe.csv"
)


class Pervaporation:
    """
    Class that contains functions for processing data from Pervaporation from Wang et al.
    Functions such as SMILES -> BigSMILES/SELFIES, organizing data into one file
    """

    def __init__(self, pv_inventory_path, pv_expt_path):
        """
        Args:
            pv_inventory_path: path to data with appropriate Species (Polymer/Solvent) and SMILES
            pv_expt_path: path to data with experimental results from Pervaporation paper

        Returns:
            None
        """
        self.inventory = pd.read_csv(pv_inventory_path)
        self.data = pd.read_csv(pv_expt_path)

    def smi_match(self, pv_expt_path):
        """
        Function that will match Polymer and Solvent Label to appropriate SMILES to the Experimental CSV
        Args:
            pv_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            pv_exptresults.csv will have appropriate SMILES
        """
        # create dictionary of polymers and solvents w/ SMILES from inventory
        pv_dict = {}
        for index, row in self.inventory.iterrows():
            name = self.inventory.at[index, "Name"]
            smi = self.inventory.at[index, "SMILES"]
            if name not in pv_dict:
                pv_dict[name] = smi

        # iterate through experimental data and input SMILES for polymer and mixture(solvent)
        self.data["Polymer_SMILES"] = ""
        self.data["Solvent_SMILES"] = ""
        for index, row in self.data.iterrows():
            polymer = self.data.at[index, "Polymer"]
            mixture = self.data.at[index, "Solvent"]
            self.data.at[index, "Polymer_SMILES"] = pv_dict[polymer]
            self.data.at[index, "Solvent_SMILES"] = pv_dict[mixture]

        self.data.to_csv(pv_expt_path, index=False)

    def sum_of_frags(self, pv_expt_path):
        """
        Function that will match Polymer Label to appropriate SMILES to the Experimental CSV
        Args:
            pv_expt_path: path to data with experimental results from Pervaporation paper

        Returns:
            pv_exptresults.csv will have array with sum of fragments (as shown in paper)
        """
        # create dictionary of polymers and solvents w/ bag_of_frags from inventory
        pv_dict = {}
        for index, row in self.inventory.iterrows():
            name = self.inventory.at[index, "Name"]
            bag_of_frags = list(row[2:])
            pv_dict[name] = bag_of_frags

        # iterate through experimental data and compute sum_of_frags and input to data
        # for polymer + mixture(solvent)
        self.data["Sum_of_frags"] = ""
        for index, row in self.data.iterrows():
            polymer = self.data.at[index, "Polymer"]
            mixture = self.data.at[index, "Solvent_(w/o)"]
            polymer_frags = pv_dict[polymer]
            mixture_frags = pv_dict[mixture]
            sum_of_frags = [x + y for x, y in zip(polymer_frags, mixture_frags)]
            self.data.at[index, "Sum_of_frags"] = sum_of_frags

        self.data.to_csv(pv_expt_path, index=False)

    def smi2selfies(self, pv_expt_path):
        """
        Function that will convert SMILES to SELFIES
        Args:
            pv_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            pv_exptresults.csv will have SELFIES
        """
        # initialize polymer swelling dataframe
        self.data["Polymer_SELFIES"] = ""
        self.data["Solvent_SELFIES"] = ""
        for index, row in self.data.iterrows():
            polymer_selfies = sf.encoder(row["Polymer_SMILES"])
            mixture_selfies = sf.encoder(row["Solvent_SMILES"])
            self.data.at[index, "Polymer_SELFIES"] = polymer_selfies
            self.data.at[index, "Solvent_SELFIES"] = mixture_selfies

        self.data.to_csv(pv_expt_path, index=False)

    def create_master_ohe(self, pv_expt_path: str, pv_ohe_path: str):
        """
        Generate a function that will one-hot encode the all of the polymer and solvent molecules. Each unique molecule has a unique number.
        Create one new column for the polymer and solvent one-hot encoded data.
        """
        master_df: pd.DataFrame = self.data
        polymer_ohe = OneHotEncoder()
        solvent_ohe = OneHotEncoder()
        polymer_ohe.fit(master_df["Polymer"].values.reshape(-1, 1))
        solvent_ohe.fit(master_df["Solvent"].values.reshape(-1, 1))
        polymer_ohe_data = polymer_ohe.transform(
            master_df["Polymer"].values.reshape(-1, 1)
        )
        solvent_ohe_data = solvent_ohe.transform(
            master_df["Solvent"].values.reshape(-1, 1)
        )
        # print(f"{polymer_ohe_data=}")
        master_df["Polymer_ohe"] = polymer_ohe_data.toarray().tolist()
        master_df["Solvent_ohe"] = solvent_ohe_data.toarray().tolist()
        # print(f"{master_df.head()}")
        # combine polymer and solvent ohe data into one column
        master_df["PS_ohe"] = master_df["Polymer_ohe"] + master_df["Solvent_ohe"]
        master_df.to_csv(pv_ohe_path, index=False)


def cli_main():
    pv_data = Pervaporation(PV_INVENTORY, PV_EXPT_RESULT)
    # pv_data.smi_match(PV_EXPT_RESULT)
    # pv_data.sum_of_frags(PV_EXPT_RESULT)
    # pv_data.smi2selfies(PV_EXPT_RESULT)
    pv_data.create_master_ohe(PV_EXPT_RESULT, PV_OHE_PATH)


if __name__ == "__main__":
    cli_main()
