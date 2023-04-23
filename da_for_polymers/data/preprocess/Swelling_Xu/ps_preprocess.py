import pkg_resources
import pandas as pd
import selfies as sf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import ast

PS_BAG_OF_FRAGS = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/Swelling_Xu/ps_bagfrags.csv"
)

PS_INVENTORY = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/Swelling_Xu/ps_inventory.csv"
)

PS_EXPT_RESULT = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/Swelling_Xu/ps_exptresults.csv"
)

RF_ERROR_CSV = pkg_resources.resource_filename(
    "da_for_polymers", "ML_models/sklearn/RF/Swelling_Xu/swell_RF_error.csv"
)

KRR_ERROR_CSV = pkg_resources.resource_filename(
    "da_for_polymers", "ML_models/sklearn/KRR/Swelling_Xu/swell_KRR_error.csv"
)

PS_OHE_PATH = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/Swelling_Xu/ohe/master_ohe.csv"
)

PS_AUTO_FRAG = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/automated_fragment/master_automated_fragment.csv",
)

PS_SMILES = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/Swelling_Xu/SMILES/master_smiles.csv"
)


class Swelling:
    """
    Class that contains functions for cleaning the Polymer Swelling Dataset.
    For example: matching Label with appropriate SMILES,
    creating SELFIES from SMILES, Summation of Bag of Frags
    """

    def __init__(self, ps_bagfrag_path, ps_inventory_path, ps_expt_path):
        """
        Args:
            ps_bagfrag_path: path to data with bag of fragments
            ps_inventory_path: path to data with appropriate Species (Polymer/Solvent) and SMILES
            ps_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            None
        """
        self.ps_bagfrag = pd.read_csv(ps_bagfrag_path)
        self.ps_inventory = pd.read_csv(ps_inventory_path)
        self.ps_expt = pd.read_csv(ps_expt_path)

    def smi_match(self, ps_expt_path):
        """
        Function that will match Species Label to appropriate SMILES to the Experimental CSV
        Args:
            ps_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            ps_exptresults.csv will have filled in SMILES
        """
        # create dictionary for polymer/solvent and their indices in the inventory .csv
        inventory_dict = {}
        for index, row in self.ps_inventory.iterrows():
            species = self.ps_inventory.at[index, "Name"]
            if species not in inventory_dict:
                inventory_dict[species] = index

        # initialize polymer swelling dataframe
        self.ps_expt["Polymer_SMILES"] = ""
        self.ps_expt["Solvent_SMILES"] = ""
        for index, row in self.ps_expt.iterrows():
            polymer = self.ps_expt.at[index, "Polymer"]
            solvent = self.ps_expt.at[index, "Solvent"]
            polymer_idx = inventory_dict[polymer]
            solvent_idx = inventory_dict[solvent]
            self.ps_expt.at[index, "Polymer_SMILES"] = self.ps_inventory.at[
                polymer_idx, "SMILES"
            ]
            self.ps_expt.at[index, "Solvent_SMILES"] = self.ps_inventory.at[
                solvent_idx, "SMILES"
            ]

        self.ps_expt.to_csv(ps_expt_path, index=False)

    def sum_bag_of_frags(self, ps_expt_path):
        """
        Function that will match Species Label to appropriate SMILES to the Experimental CSV
        Args:
            ps_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            ps_exptresults.csv will have array with sum of fragments (as shown in paper)
        """
        # create dictionary for polymer/solvent and their indices in the bag_of_frags .csv
        bagfrag_dict = {}
        for index, row in self.ps_bagfrag.iterrows():
            species = self.ps_bagfrag.at[index, "Species"]
            if species not in bagfrag_dict:
                bagfrag_dict[species] = index

        # initialize polymer swelling dataframe
        self.ps_expt["Sum_of_Frags"] = ""
        unique_sum = []
        vocab_length = 0
        for index, row in self.ps_expt.iterrows():
            polymer = self.ps_expt.at[index, "Polymer"]
            solvent = self.ps_expt.at[index, "Solvent"]
            polymer_idx = bagfrag_dict[polymer]
            solvent_idx = bagfrag_dict[solvent]
            polymer_bag = list(self.ps_bagfrag.iloc[polymer_idx])[1:]
            solvent_bag = list(self.ps_bagfrag.iloc[solvent_idx])[1:]
            sum_of_bags = []
            position = 0
            while position < len(polymer_bag):
                sum_of_bags.append(int(polymer_bag[position] + solvent_bag[position]))
                position += 1
            # gets vocab length from bag of frags
            for sum in sum_of_bags:
                if sum not in unique_sum:
                    vocab_length += 1
                    unique_sum.append(sum)

            self.ps_expt.at[index, "Sum_of_Frags"] = sum_of_bags

        print("VOCAB_LENGTH: ", vocab_length)
        self.ps_expt.to_csv(ps_expt_path, index=False)

    def smi2selfies(self, ps_expt_path):
        """
        Function that will match Species Label to appropriate SMILES to the Experimental CSV
        Args:
            ps_expt_path: path to data with experimental results from Polymer Swelling paper

        Returns:
            ps_exptresults.csv will have SELFIES
        """
        # initialize polymer swelling dataframe
        self.ps_expt["Polymer_SELFIES"] = ""
        self.ps_expt["Solvent_SELFIES"] = ""
        for index, row in self.ps_expt.iterrows():
            polymer_selfies = sf.encoder(row["Polymer_SMILES"])
            solvent_selfies = sf.encoder(row["Solvent_SMILES"])
            self.ps_expt.at[index, "Polymer_SELFIES"] = polymer_selfies
            self.ps_expt.at[index, "Solvent_SELFIES"] = solvent_selfies

        self.ps_expt.to_csv(ps_expt_path, index=False)

    def clean_str(self, ps_expt_path):
        self.ps_expt = self.ps_expt[self.ps_expt.SD != "d"]
        self.ps_expt = self.ps_expt[self.ps_expt.SD != "o"]
        self.ps_expt = self.ps_expt[self.ps_expt.SD != "ld"]
        self.ps_expt.to_csv(ps_expt_path, index=False)

    def clean_outliers(self, ps_error_path, ps_expt_path):
        error_df = pd.read_csv(ps_error_path)
        print(np.std(error_df.Test_Error_0))

    def create_master_ohe(self, ps_expt_path: str, ps_ohe_path: str):
        """
        Generate a function that will one-hot encode the all of the polymer and solvent molecules. Each unique molecule has a unique number.
        Create one new column for the polymer and solvent one-hot encoded data.
        """
        master_df: pd.DataFrame = self.ps_expt
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
        master_df.to_csv(ps_ohe_path, index=False)

    def bigsmiles_from_frag(self, automated_frag: str, smiles: str):
        """
        Function that takes ordered fragments (manually by hand) and converts it into BigSMILES representation, specifically block copolymers
        Args:
            dft_automated_frag: path to data with automated fragmented polymers

        Returns:
            concatenates fragments into BigSMILES representation and returns to data
        """
        # polymer/mixture BigSMILES
        data = pd.read_csv(automated_frag)
        smi_data = pd.read_csv(smiles)
        smi_data["Polymer_BigSMILES"] = ""

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
                        == len(ast.literal_eval(data["polymer_automated_frag"][index]))
                        - 1
                    ):
                        big_smi += "[>][]}"
                    else:
                        big_smi += "[>][<]}{[>][<]"
                    position += 1

            smi_data.at[index, "Polymer_BigSMILES"] = big_smi

        smi_data.to_csv(smiles, index=False)


swelling = Swelling(PS_BAG_OF_FRAGS, PS_INVENTORY, PS_EXPT_RESULT)
# swelling.smi_match(PS_EXPT_RESULT)
# swelling.sum_bag_of_frags(PS_EXPT_RESULT)
# swelling.smi2selfies(PS_EXPT_RESULT)
# swelling.clean_str(PS_EXPT_RESULT)
# swelling.create_master_ohe(PS_EXPT_RESULT, PS_OHE_PATH)
swelling.bigsmiles_from_frag(PS_AUTO_FRAG, PS_AUTO_FRAG)
