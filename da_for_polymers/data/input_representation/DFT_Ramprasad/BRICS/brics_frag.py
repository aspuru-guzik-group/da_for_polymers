import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Draw, rdmolops
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from collections import deque
import numpy as np
import copy
import ast

MASTER_DFT_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/DFT_Ramprasad/dft_exptresults_Egc.csv"
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/DFT_Ramprasad/BRICS/master_brics_frag.csv",
)


def remove_dummy(mol):
    """
    Function that removes dummy atoms from mol and returns SMILES

    Args:
        mol: RDKiT mol object for removing dummy atoms (*)
    """
    dummy_idx = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_idx.append(atom.GetIdx())
    # remove dummy atoms altogether
    ed_mol = Chem.EditableMol(mol)
    ed_mol.BeginBatchEdit()
    for idx in dummy_idx:
        ed_mol.RemoveAtom(idx)
    ed_mol.CommitBatchEdit()
    edited_mol = ed_mol.GetMol()
    return Chem.MolToSmiles(edited_mol)


def bric_frag(dft_data: str, brics_frag_data: str):
    """
    Fragments molecules (from SMILES) using BRICS from RDKIT

    Args:
        None

    Returns:
        Creates new master_brics_frag.csv with Labels, SMILES, DA_pairs, Fragments, PCE(%)
    """
    brics_df = pd.read_csv(dft_data)

    # Iterate through row and fragment using BRICS
    # to get polymer_BRICS, solvent_BRICS, and DA_pair_BRICS
    for index, row in brics_df.iterrows():
        polymer_smi = brics_df.at[index, "smiles"]
        polymer_mol = Chem.MolFromSmiles(polymer_smi)
        polymer_brics = list(BRICS.BRICSDecompose(polymer_mol, returnMols=True))
        polymer_brics_smi = []
        for frag in polymer_brics:
            frag_smi = remove_dummy(frag)
            polymer_brics_smi.append(frag_smi)

        brics_df.at[index, "Polymer_BRICS"] = polymer_brics_smi

    brics_df.to_csv(BRICS_FRAG_DATA, index=False)


bric_frag(MASTER_DFT_DATA, BRICS_FRAG_DATA)
# print(frag_dict)
# b_frag.frag_visualization(frag_dict)
