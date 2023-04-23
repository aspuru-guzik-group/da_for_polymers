from doctest import master
import re
from rdkit import Chem
import rdkit
from rdkit.Chem import Draw, rdchem
import pkg_resources
import pandas as pd
import ast
import copy
from collections import deque
from rdkit.Chem import AllChem
import selfies as sf

CO2_INVENTORY = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/CO2_Soleimani/co2_solubility_inventory.csv"
)

CO2_EXPT_RESULT = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/CO2_Soleimani/co2_expt_data.csv"
)

master_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/CO2_Soleimani/manual_frag/master_manual_frag.csv",
)

IMG_PATH = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/CO2_Soleimani/manual_frag/"
)


class manual_frag:
    "Class that contains functions necessary to fragment molecules any way you want"

    def __init__(self, co2_inventory_path):
        """
        Instantiate class with appropriate data.

        Args:
            co2_inventory_path: path to CO2 ML data

        Returns:
            None
        """
        self.co2_inventory = pd.read_csv(co2_inventory_path)

    # pipeline
    # 1 iterate with index (master)
    # 2 show molecule with atom.index
    # 3 ask for begin/end atom index OR bond index
    # 4 fragment
    # 5 show fragmented molecule
    # 6 if correct, convert to smiles and store in new .csv
    # 7 if incorrect, go back to step 3
    # 8 NOTE: be able to manually look up any donor/acceptor and re-fragment

    def lookup(self, index: int) -> str:
        """
        Function that finds and returns SMILES from donor or acceptor .csv

        Args:
            index: index of row in dataframe

        Returns:
            smi: SMILES of looked up molecule
        """
        try:
            smi = self.co2_inventory.at[index, "SMILES"]
            name = self.co2_inventory.at[index, "Name"]
        except:
            print(
                "Max index exceeded, please try again. Max index is: ",
                len(self.co2_inventory["SMILES"]) - 1,
            )

        return smi, name

    def fragmenter(self, smi: str):
        """
        Function that asks user how to fragment molecule

        Args:
            smi: SMILES to fragment

        Returns:
            ordered_frag: molecule that was fragmented by user's input, and properly ordered
        """
        # For pervaporation data, remove all dummy atoms first to keep it consistent across datasets
        # replace dummy atoms
        mol = Chem.MolFromSmiles(smi)
        edmol_frag = Chem.EditableMol(mol)
        edmol_frag.BeginBatchEdit()
        [
            edmol_frag.RemoveAtom(atom.GetIdx())
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 0
        ]
        edmol_frag.CommitBatchEdit()
        mol = edmol_frag.GetMol()

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

        drawn = Draw.MolToFile(mol, IMG_PATH + "manual.png", size=(700, 700))

        # delete extra methyl groups
        extra_methyl = False
        while not extra_methyl:
            extra_methyl_idx: list[str] = input(
                "Methyl group atom indices to remove: "
            ).split(", ")
            if extra_methyl_idx == "na":
                extra_methyl = True
            else:
                # remove methyl group
                ed_mol = Chem.EditableMol(mol)
                ed_mol.BeginBatchEdit()
                for idx in extra_methyl_idx:
                    ed_mol.RemoveAtom(int(idx))
                ed_mol.CommitBatchEdit()

                mol = ed_mol.GetMol()

                mol.UpdatePropertyCache()

                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIdx())

                # Visualize fragmented mol
                drawn = Draw.MolToFile(mol, IMG_PATH + "manual.png", size=(700, 700))
                correct: str = input("Are the methyl groups removed correctly?: ")
                if correct == "y":
                    extra_methyl: bool = True

        fragmented = False
        # show all bond indexes with corresponding begin/atom idx
        for bond in mol.GetBonds():
            print(
                "bond: ",
                bond.GetIdx(),
                "begin, end: ",
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
            )
        while not fragmented:
            # Ex. 30, 31, 33, 34, 101, 102
            frag_idx: str = input("Bond Indices to be fragmented: ")
            if frag_idx == "None":
                mol_frag: Chem.Mol = mol
                break
            frag_tuple: tuple = tuple(map(int, frag_idx.split(", ")))
            atom_separation_idx: list[list[int]] = []
            for bond_idx in frag_tuple:
                bond: Chem.rdchem.Bond = mol.GetBondWithIdx(bond_idx)
                atom_idxs: list = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                atom_separation_idx.append(atom_idxs)
            mol_frag: Chem.rdchem.Mol = Chem.FragmentOnBonds(
                mol, frag_tuple, addDummies=False
            )
            # add arbitrary atoms for Reaction SMILES
            atomic_num: int = 64
            for atom_sep in atom_separation_idx:
                for atom in mol_frag.GetAtoms():
                    if atom.GetIdx() in atom_sep:
                        if atom.GetAtomicNum() != atomic_num - 1:
                            atom.SetAtomicNum(atomic_num)
                            atomic_num += 1

            mol_frag.UpdatePropertyCache()

            for atom in mol_frag.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
            # Visualize fragmented mol
            drawn = Draw.MolToFile(
                mol_frag, IMG_PATH + "manual_frag.png", size=(700, 700)
            )
            correct: str = input("Is the molecule fragmented correctly?: ")
            if correct == "y":
                fragmented: bool = True

        # removes atom map numbering
        [a.SetAtomMapNum(0) for a in mol_frag.GetAtoms() if a.GetAtomicNum() != 0]
        frag_list: list = Chem.MolToSmarts(mol_frag).split(".")
        # order the fragments
        frag_length: int = len(frag_list)
        ordered: bool = False
        while not ordered:
            ordered_frag: list = []
            for i in range(frag_length):
                ordered_frag.append(i)
            for frag in frag_list:
                order_idx: int = int(
                    input("Ordering of current frag (" + str(frag) + "):")
                )
                ordered_frag[order_idx] = frag
            print(ordered_frag, atom_separation_idx)
            correct: str = input("Are the fragments ordered correctly?: ")
            if correct == "y":
                ordered: bool = True

        return ordered_frag

    def shuffle_augment(self, ordered_frag: list[str]) -> list[str]:
        """
        Function that shuffles the position of the monomers, and substitutes the
        dummy atoms for a list of uncommon atoms as placeholder atoms.
        """
        # AUGMENT Donor (pre-ordered)
        augmented_donor_list = []
        donor_frag_deque = deque(copy.copy(ordered_frag))
        for j in range(len(ordered_frag)):
            frag_rotate = copy.copy(donor_frag_deque)
            frag_rotate.rotate(j)
            frag_rotate = list(frag_rotate)
            augmented_donor_list.append(frag_rotate)

        return augmented_donor_list

    def recombine(self, fragment_smarts: list[str]) -> str:
        """
        Function that recombines the rearranged molecule into the appropriate SMILES.
        """
        print("fragment_smarts", fragment_smarts)
        #  Reaction SMARTS
        curr_mol = fragment_smarts[0]
        idx = 2
        for frag in fragment_smarts[1:]:
            atomic_num: int = 64
            recombined = False
            stop_condition = 0
            while not recombined:
                if stop_condition == 50:
                    assert False
                elif idx == len(
                    fragment_smarts
                ):  # if last fragment, complete the C single bonds
                    try:
                        # Order matters
                        rxn = AllChem.ReactionFromSmarts(
                            "[#{}:1].[#{}:2]>>[C:1]-[C:2]".format(
                                atomic_num + 1, atomic_num
                            )
                        )
                        products = rxn.RunReactants(
                            [Chem.MolFromSmarts(x) for x in [curr_mol, frag]]
                        )
                        if products == ():
                            rxn = AllChem.ReactionFromSmarts(
                                "[#{}:1].[#{}:2]>>[C:1]-[C:2]".format(
                                    atomic_num, atomic_num + 1
                                )
                            )
                            products = rxn.RunReactants(
                                [Chem.MolFromSmarts(x) for x in [curr_mol, frag]]
                            )
                        curr_mol = products[0][0]  # fails if products is ()
                        curr_mol = Chem.MolToSmarts(curr_mol)
                        recombined = True
                    except:
                        try:  # Handles the end group connection
                            # Order matters
                            atomic_num_2: int = 64
                            rxn = AllChem.ReactionFromSmarts(
                                "[#{}:1].[#{}:2]>>[C:1]-[C:2]".format(
                                    atomic_num_2 + len(fragment_smarts) - 1,
                                    atomic_num_2,
                                )
                            )
                            products = rxn.RunReactants(
                                [Chem.MolFromSmarts(x) for x in [curr_mol, frag]]
                            )
                            if products == ():
                                rxn = AllChem.ReactionFromSmarts(
                                    "[#{}:1].[#{}:2]>>[C:1]-[C:2]".format(
                                        atomic_num_2,
                                        atomic_num_2 + len(fragment_smarts) - 1,
                                    )
                                )
                                products = rxn.RunReactants(
                                    [Chem.MolFromSmarts(x) for x in [curr_mol, frag]]
                                )
                            curr_mol = products[0][0]  # fails if products is ()
                            curr_mol = Chem.MolToSmarts(curr_mol)
                            recombined = True
                        except:
                            atomic_num += 1
                            stop_condition += 1
                else:
                    try:
                        # Order matters
                        rxn = AllChem.ReactionFromSmarts(
                            "[#{}:1].[#{}:2]>>[C:1]-[#{}:2]".format(
                                atomic_num, atomic_num + 1, atomic_num + 1
                            )
                        )
                        products = rxn.RunReactants(
                            [Chem.MolFromSmarts(x) for x in [curr_mol, frag]]
                        )
                        if products == ():
                            rxn = AllChem.ReactionFromSmarts(
                                "[#{}:1].[#{}:2]>>[C:1]-[#{}:2]".format(
                                    atomic_num, atomic_num + 1, atomic_num + 1
                                )
                            )
                            products = rxn.RunReactants(
                                [Chem.MolFromSmarts(x) for x in [curr_mol, frag]]
                            )
                        curr_mol = products[0][0]  # fails if products is ()
                        curr_mol = Chem.MolToSmarts(curr_mol)
                        recombined = True
                    except:
                        try:  # handles the end group connection
                            # Order matters
                            atomic_num_2: int = 64
                            rxn = AllChem.ReactionFromSmarts(
                                "[#{}:1].[#{}:2]>>[C:1]-[#{}:2]".format(
                                    atomic_num_2 + len(fragment_smarts) - 1,
                                    atomic_num_2,
                                    atomic_num_2,
                                )
                            )
                            products = rxn.RunReactants(
                                [Chem.MolFromSmarts(x) for x in [curr_mol, frag]]
                            )
                            if products == ():
                                rxn = AllChem.ReactionFromSmarts(
                                    "[#{}:1].[#{}:2]>>[C:1]-[#{}:2]".format(
                                        atomic_num_2,
                                        atomic_num_2 + len(fragment_smarts) - 1,
                                        atomic_num_2 + len(fragment_smarts) - 1,
                                    )
                                )
                                products = rxn.RunReactants(
                                    [Chem.MolFromSmarts(x) for x in [curr_mol, frag]]
                                )
                            curr_mol = products[0][0]  # fails if products is ()
                            curr_mol = Chem.MolToSmarts(curr_mol)
                            recombined = True
                        except:
                            atomic_num += 1
                            stop_condition += 1
            idx += 1

        curr_mol = Chem.MolFromSmarts(curr_mol)
        for atom in curr_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        drawn = Chem.Draw.MolToFile(
            curr_mol, IMG_PATH + "manual_rearranged.png", size=(700, 700)
        )

        Chem.SanitizeMol(curr_mol)

        for atom in curr_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        drawn = Chem.Draw.MolToFile(
            curr_mol, IMG_PATH + "manual_rearranged.png", size=(700, 700)
        )

        # removes atom map numbering
        [a.SetAtomMapNum(0) for a in curr_mol.GetAtoms() if a.GetAtomicNum() != 0]

        return Chem.MolToSmiles(curr_mol)

    def visualize_rearranged(self, recombined_smi_list: list[str]):
        recombined_mol_list: list[Chem.rdchem.Mol] = [
            Chem.MolFromSmiles(smi) for smi in recombined_smi_list
        ]
        print(recombined_mol_list)
        img = Chem.Draw.MolsToGridImage(recombined_mol_list, molsPerRow=3)
        # Visualize fragmented mol
        filename: str = IMG_PATH + "manual_rearranged.png"
        img.save(filename)

    def visualize_rearranged_with_replace(self, recombined_smi_list: list[str]):
        recombined_mol_list: list[Chem.rdchem.Mol] = [
            Chem.MolFromSmiles(smi) for smi in recombined_smi_list
        ]
        print(recombined_mol_list)
        recombined_mol_atom_map_list: list[Chem.rdchem.Mol] = []
        for mol in recombined_mol_list:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
            recombined_mol_atom_map_list.append(mol)
        img = Chem.Draw.MolsToGridImage(recombined_mol_atom_map_list, molsPerRow=3)
        # Visualize fragmented mol
        # TODO: visualize here and then replace O's.
        filename: str = IMG_PATH + "manual_rearranged.png"
        img.save(filename)

        final_mols: list = []
        for mol in recombined_mol_atom_map_list:
            replace_idx: list[str] = input(
                "Atom Indices to be Replaced with O: "
            ).split(", ")
            ed_mol = Chem.EditableMol(mol)
            if replace_idx != "na":
                ed_mol.BeginBatchEdit()
                for atom in mol.GetAtoms():
                    for idx in replace_idx:
                        if atom.GetIdx() == int(idx):
                            ed_mol.ReplaceAtom(int(idx), Chem.Atom(8))
                ed_mol.CommitBatchEdit()
            mol = ed_mol.GetMol()

            # replace end dummy group with methyl group
            ed_mol = Chem.EditableMol(mol)
            dummy_idx: list[str] = input(
                "Atom Indices for adding methyl end groups: "
            ).split(", ")
            ed_mol.BeginBatchEdit()
            for idx in dummy_idx:
                for atom in mol.GetAtoms():
                    if int(idx) == atom.GetIdx():
                        atom_idx = ed_mol.AddAtom(Chem.Atom(6))
                        ed_mol.AddBond(
                            int(idx), atom_idx, order=Chem.rdchem.BondType.SINGLE
                        )
            ed_mol.CommitBatchEdit()
            mol = ed_mol.GetMol()
            final_mols.append(ed_mol.GetMol())

        for mol in final_mols:
            # removes atom map numbering
            [a.SetAtomMapNum(0) for a in mol.GetAtoms() if a.GetAtomicNum() != 0]

        filename: str = IMG_PATH + "manual_rearranged.png"
        img = Chem.Draw.MolsToGridImage(final_mols, molsPerRow=3)
        img.save(filename)

        final_mols = [Chem.MolToSmiles(mol) for mol in final_mols]
        return final_mols

    def return_frag_dict(self):
        """
        Sifts through manual fragments and creates unique dictionary of frag2idx

        Args:
            None

        Returns:
            frag_dict: dictionary of unique fragments in the combination of donor and acceptor fragmented molecules
        """
        frag_dict = {}
        frag_dict["_PAD"] = 0
        frag_dict["."] = 1
        id = len(frag_dict)
        for i in range(len(self.co2_inventory)):
            frag_str = self.co2_inventory.at[i, "Fragments"]
            frag_list = ast.literal_eval(frag_str)
            for frag in frag_list:
                if frag not in list(frag_dict.keys()):
                    frag_dict[frag] = id
                    id += 1

        return frag_dict

    def tokenize_frag(self, list_of_frag, frag_dict, max_seq_length):
        """
        Tokenizes input list of fragment from given dictionary
        * Assumes frag_dict explains all of list_of_frig

        Args:
            list_of_frag: list of all the fragments for tokenization
            frag_dict: dictionary of unique fragments from donor and acceptor molecules
            max_seq_length: the largest number of fragments for one molecule
        """
        tokenized_list = []
        # Add pre-padding
        num_of_pad = max_seq_length - len(list_of_frag)
        for i in range(num_of_pad):
            tokenized_list.append(0)

        for frag in list_of_frag:
            tokenized_list.append(frag_dict[frag])

        return tokenized_list

    def create_manual_csv(self, frag_dict, co2_expt_path, master_manual_path):
        """
        Creates master data file for manual frags

        Args:
            frag_dict: dictionary of unique fragments from donor and acceptor molecules
            co2_expt_path: path to experimental .csv for co2 solubility data
            master_manual_path: path to master .csv file for training on manual fragments
        """
        inventory_dict = {}
        for index, row in self.co2_inventory.iterrows():
            species = self.co2_inventory.at[index, "Polymer"]
            if species not in inventory_dict:
                inventory_dict[species] = index

        manual_df = pd.read_csv(co2_expt_path)
        manual_df["Polymer_BigSMILES"] = ""
        manual_df["Polymer_manual"] = ""
        manual_df["Polymer_manual_aug"] = ""
        manual_df["Polymer_manual_str"] = ""
        manual_df["Polymer_manual_aug_str"] = ""

        aug_count = 0
        # find max_seq_length
        max_seq_length = 0
        for i in range(len(manual_df)):
            polymer_label = manual_df.at[i, "Polymer"]
            polymer_frags = list(
                ast.literal_eval(
                    self.co2_inventory.at[inventory_dict[polymer_label], "Fragments"]
                )
            )
            max_frag_list = polymer_frags
            max_frag_length = len(max_frag_list)
            if max_frag_length > max_seq_length:
                max_seq_length = max_frag_length

        print("max_frag_length: ", max_seq_length)

        for i in range(len(manual_df)):
            polymer_label = manual_df.at[i, "Polymer"]
            polymer_frags = list(
                ast.literal_eval(
                    self.co2_inventory.at[inventory_dict[polymer_label], "Fragments"]
                )
            )

            # Polymer
            # polymer_tokenized = self.tokenize_frag(
            #     polymer_frags, frag_dict, max_seq_length
            # )

            # AUGMENT Polymer (pre-ordered)
            augmented_polymer_list = []
            polymer_frag_deque = deque(copy.copy(polymer_frags))
            for j in range(len(polymer_frags)):
                frag_rotate = copy.copy(polymer_frag_deque)
                frag_rotate.rotate(j)
                frag_rotate = list(frag_rotate)
                augmented_polymer_list.append(frag_rotate)
                aug_count += 1

            # PS Pairs augmented
            # polymer_tokenized_aug = []
            # for aug_polymer in augmented_polymer_list:
            #     aug_polymer_copy = copy.copy(aug_polymer)
            #     aug_polymer_tokenized = self.tokenize_frag(
            #         aug_polymer_copy, frag_dict, max_seq_length
            #     )
            #     polymer_tokenized_aug.append(aug_polymer_tokenized)

            # ADD TO MANUAL DF from inventory (does not separate polymer and mixture)
            manual_df.at[i, "Polymer_BigSMILES"] = self.co2_inventory.at[
                inventory_dict[polymer_label], "Polymer_BigSMILES"
            ]
            manual_df.at[i, "Polymer_manual"] = polymer_frags
            manual_df.at[i, "Polymer_manual_aug"] = augmented_polymer_list

            # create string version of augmented polymers
            polymer_aug_str_list = []
            for polymer in augmented_polymer_list:
                polymer_aug_str: str = polymer[0]
                for frag in polymer[1:]:
                    polymer_aug_str += "." + frag
                print(polymer_aug_str)
                polymer_aug_str_list.append(polymer_aug_str)
            manual_df.at[i, "Polymer_manual_str"] = polymer_aug_str_list[0]
            manual_df.at[i, "Polymer_manual_aug_str"] = polymer_aug_str_list

        # number of augmented polymers
        print("AUG POLYMERS: ", aug_count)

        manual_df.to_csv(master_manual_path, index=False)

    def add_recombined_manual_and_check_smi_selfies(
        self, co2_inventory_path, master_manual_path
    ):
        """Add list of recombined SMILES to the data.
        Make sure SMILES and SELFIES are correct by replacing them with matching Names

        Args:
            co2_inventory_path (_type_): _description_
            master_manual_path (_type_): _description_
        """
        co2_inventory: pd.DataFrame = pd.read_csv(co2_inventory_path)
        master_manual: pd.DataFrame = pd.read_csv(master_manual_path)
        master_manual["Polymer_Augmented_Recombined_Fragment_SMILES"] = ""
        for idx, row in master_manual.iterrows():
            polymer: str = master_manual.at[idx, "Polymer"]
            inventory_row = co2_inventory.loc[co2_inventory["Polymer"] == polymer]
            master_manual.at[idx, "Polymer_SMILES"] = inventory_row["SMILES"].values[0]
            polymer_selfies = sf.encoder(inventory_row["SMILES"].values[0])
            master_manual.at[idx, "Polymer_SELFIES"] = polymer_selfies
            # check if smiles are unique
            augmented_recombined_fragment_smiles: list = ast.literal_eval(
                inventory_row["Augmented Recombined Fragment SMILES"].values[0]
            )
            augmented_recombined_fragment_smiles_check: list = [
                Chem.MolToSmiles(Chem.MolFromSmiles(s))
                for s in augmented_recombined_fragment_smiles
            ]
            augmented_recombined_fragment_smiles: list = list(
                dict.fromkeys(augmented_recombined_fragment_smiles_check)
            )
            master_manual.at[
                idx, "Polymer_Augmented_Recombined_Fragment_SMILES"
            ] = augmented_recombined_fragment_smiles

        master_manual.to_csv(master_manual_path, index=False)

    def fingerprint_from_frag(self, master_manual_path):
        """After adding recombined manual SMILES, fingerprint the molecules using MorganFingerprints (ECFP4)

        Args:
            master_manual_path (_type_): _description_
        """
        master_manual: pd.DataFrame = pd.read_csv(master_manual_path)
        master_manual["Polymer_manual_recombined_aug_SMILES"] = ""
        master_manual["Polymer_manual_recombined_aug_FP"] = ""
        for index, row in master_manual.iterrows():
            augmented_recombined_fragment_smiles: list = ast.literal_eval(
                master_manual.at[index, "Polymer_Augmented_Recombined_Fragment_SMILES"]
            )
            augmented_fp: list = []
            augmented_smi: list = []
            for smi in augmented_recombined_fragment_smiles:
                augmented_smi.append(smi)
                polymer_mol = Chem.MolFromSmiles(smi)
                bitvector_polymer = AllChem.GetMorganFingerprintAsBitVect(
                    polymer_mol, 3, 512
                )
                fp_list = list(bitvector_polymer.ToBitString())
                fp_map = map(int, fp_list)
                fp = list(fp_map)
                augmented_fp.append(fp)
            master_manual.at[
                index, "Polymer_manual_recombined_aug_SMILES"
            ] = augmented_smi
            master_manual.at[index, "Polymer_manual_recombined_aug_FP"] = augmented_fp

        master_manual.to_csv(master_manual_path, index=False)

    def bigsmiles_from_frag(self, co2_inventory_path):
        """
        Function that takes ordered fragments (manually by hand) and converts it into BigSMILES representation, specifically block copolymers
        Args:
            co2_inventory_path: path to data with manually fragmented polymers

        Returns:
            concatenates manual fragments into BigSMILES representation and returns to donor/acceptor data
        """
        # polymer/mixture BigSMILES
        self.co2_inventory["Polymer_BigSMILES"] = ""

        for index, row in self.co2_inventory.iterrows():
            big_smi = "{[][<]"
            position = 0
            if len(ast.literal_eval(self.co2_inventory["Fragments"][index])) == 1:
                big_smi = ast.literal_eval(self.co2_inventory["Fragments"][index])[0]
            else:
                for frag in ast.literal_eval(self.co2_inventory["Fragments"][index]):
                    big_smi += str(frag)
                    if (
                        position
                        == len(ast.literal_eval(self.co2_inventory["Fragments"][index]))
                        - 1
                    ):
                        big_smi += "[>][]}"
                    else:
                        big_smi += "[>][<]}{[>][<]"
                    position += 1

            self.co2_inventory["Polymer_BigSMILES"][index] = big_smi

        self.co2_inventory.to_csv(co2_inventory_path, index=False)


def cli_main():
    manual = manual_frag(CO2_INVENTORY) 

    # ATTENTION: Fragmenting and Order for Data Augmentation
    # iterate through inventory
    manual_df = pd.read_csv(CO2_INVENTORY)
    for idx, row in manual_df.iterrows():
        if pd.isnull(row["Fragments"]):
            print(row["Name"])
            smi = row["SMILES"]
            frag_list = manual.fragmenter(smi)
            manual_df.at[idx, "Fragments"] = str(frag_list)
            # Shuffle (Augment) and Recombine!
            aug_frag_list: list[str] = manual.shuffle_augment(frag_list)
            recombined_smi_list: list[str] = []
            for aug_frag in aug_frag_list:
                recombined_smi: str = manual.recombine(aug_frag)
                if recombined_smi not in recombined_smi_list:
                    recombined_smi_list.append(recombined_smi)
            # visualize rearranged structures
            recombined_smi_list = manual.visualize_rearranged_with_replace(
                recombined_smi_list
            )
            manual_df.at[idx, "Augmented Recombined Fragment SMILES"] = str(
                recombined_smi_list
            )
            print(recombined_smi_list)
        manual_df.to_csv(CO2_INVENTORY, index=False)

    # ATTENTION: prepare manual frag data after data augmentation
    # prepare manual frag data
    manual = manual_frag(CO2_INVENTORY)
    frag_dict = manual.return_frag_dict()
    # print(len(frag_dict))
    # manual.frag_visualization(frag_dict)
<<<<<<< HEAD
    # manual.bigsmiles_from_frag(CO2_INVENTORY)
    # manual.create_manual_csv(frag_dict, CO2_EXPT_RESULT, master_MANUAL_DATA)
=======
    manual.bigsmiles_from_frag(CO2_INVENTORY)
    manual.create_manual_csv(frag_dict, CO2_EXPT_RESULT, MASTER_MANUAL_DATA)
>>>>>>> 3518320fe8131a4d5c99874c5d2194ecbf421006
    manual.add_recombined_manual_and_check_smi_selfies(
        CO2_INVENTORY, master_MANUAL_DATA
    )
    manual.fingerprint_from_frag(master_MANUAL_DATA)


if __name__ == "__main__":
    cli_main()
    # pass
