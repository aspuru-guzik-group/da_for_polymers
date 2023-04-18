from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import pkg_resources
import pandas as pd
import numpy as np
import copy
from pathlib import Path

PS_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/preprocess/Swelling_Xu/ps_exptresults.csv",
)

PS_CFP_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/Swelling_Xu/circular_fingerprint/PS_circular_fingerprint.csv",
)


current_dir = Path(__file__).resolve().parent


# create a function that produces dimer, trimers, and polymer graph (RDKiT, add a new bond at the ends of the trimer)
def Nmer_graph(expt_results_path: str, cfp_path: str, N: int) -> pd.DataFrame:
    """Create a dataframe of polymer graphs.

    Args:
        expt_results_path (str): _description_
        cfp_path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    expt_results: pd.DataFrame = pd.read_csv(expt_results_path)
    expt_results[f"{N}mer_fp"] = ""
    for index, row in expt_results.iterrows():
        monomer_smi: str = row["Polymer_SMILES"]
        monomer_mol: Chem.rdchem.Mol = Chem.MolFromSmiles(monomer_smi)
        curr_mol: Chem.rdchem.Mol = copy.deepcopy(monomer_mol)
        for i in range(N - 1):
            curr_mol: Chem.rdchem.Mol = Chem.CombineMols(curr_mol, monomer_mol)
            # Get index of last atom of first monomer by atom index and atomic number
            dummy_atom_idx: list = []
            dummy_atom_neighbor_idx: list = []
            for atom in curr_mol.GetAtoms():
                if atom.GetAtomicNum() == 0 and len(atom.GetNeighbors()) == 1:
                    dummy_atom_idx.append(atom.GetIdx())
                    dummy_atom_neighbor_idx.append(atom.GetNeighbors()[0].GetIdx())
            # make into editable mol
            ed_mol: Chem.rdchem.EditableMol = Chem.EditableMol(curr_mol)
            # add bond between last atom of first monomer and first atom of second monomer
            ed_mol.BeginBatchEdit()
            ed_mol.AddBond(
                dummy_atom_neighbor_idx[1],
                dummy_atom_neighbor_idx[2],
                Chem.rdchem.BondType.SINGLE,
            )
            # delete dummy atom around the new bond
            ed_mol.RemoveAtom(dummy_atom_idx[1])
            ed_mol.RemoveAtom(dummy_atom_idx[2])
            ed_mol.CommitBatchEdit()
            # convert back to mol
            curr_mol: Chem.rdchem.Mol = ed_mol.GetMol()
        # convert to fingerprint
        Chem.SanitizeMol(curr_mol)
        curr_mol_fp: list = list(
            AllChem.GetMorganFingerprintAsBitVect(curr_mol, 3, nBits=512)
        )
        # add solvent SMILES fingerprint
        solvent_mol = Chem.MolFromSmiles(row["Solvent_SMILES"])
        solvent_fp: list = list(
            AllChem.GetMorganFingerprintAsBitVect(solvent_mol, 3, nBits=512)
        )
        curr_mol_fp.extend(solvent_fp)
        expt_results.at[index, f"{N}mer_fp"] = curr_mol_fp
    expt_results.to_csv(cfp_path, index=False)


def polymer_graph(expt_results_path: str, cfp_path: str, N: int) -> pd.DataFrame:
    """Create a dataframe of polymer graphs.

    Args:
        expt_results_path (str): _description_
        cfp_path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    expt_results: pd.DataFrame = pd.read_csv(expt_results_path)
    expt_results[f"{N}mer_circular_graph_fp"] = ""
    for index, row in expt_results.iterrows():
        monomer_smi: str = row["Polymer_SMILES"]
        monomer_mol: Chem.rdchem.Mol = Chem.MolFromSmiles(monomer_smi)
        curr_mol: Chem.rdchem.Mol = copy.deepcopy(monomer_mol)
        for i in range(N - 1):
            curr_mol: Chem.rdchem.Mol = Chem.CombineMols(curr_mol, monomer_mol)
            # Get index of last atom of first monomer by atom index and atomic number
            dummy_atom_idx: list = []
            dummy_atom_neighbor_idx: list = []
            for atom in curr_mol.GetAtoms():
                if atom.GetAtomicNum() == 0 and len(atom.GetNeighbors()) == 1:
                    dummy_atom_idx.append(atom.GetIdx())
                    dummy_atom_neighbor_idx.append(atom.GetNeighbors()[0].GetIdx())
            # make into editable mol
            ed_mol: Chem.rdchem.EditableMol = Chem.EditableMol(curr_mol)
            # add bond between last atom of first monomer and first atom of second monomer
            ed_mol.BeginBatchEdit()
            ed_mol.AddBond(
                dummy_atom_neighbor_idx[1],
                dummy_atom_neighbor_idx[2],
                Chem.rdchem.BondType.SINGLE,
            )
            # delete dummy atom around the new bond
            ed_mol.RemoveAtom(dummy_atom_idx[1])
            ed_mol.RemoveAtom(dummy_atom_idx[2])
            ed_mol.CommitBatchEdit()
            # convert back to mol
            curr_mol: Chem.rdchem.Mol = ed_mol.GetMol()
        # NEW: add bond from first dummy atom to last dummy atom
        dummy_atom_idx: list = []
        dummy_atom_neighbor_idx: list = []
        for atom in curr_mol.GetAtoms():
            if atom.GetAtomicNum() == 0 and len(atom.GetNeighbors()) == 1:
                dummy_atom_idx.append(atom.GetIdx())
                dummy_atom_neighbor_idx.append(atom.GetNeighbors()[0].GetIdx())
        # print(dummy_atom_idx)
        # Draw.MolToFile(
        #     curr_mol,
        #     filename=current_dir / "monomer.png",
        #     size=(500, 500),
        # )
        # make into editable mol
        ed_mol: Chem.rdchem.EditableMol = Chem.EditableMol(curr_mol)
        # add bond between last atom of first monomer and first atom of second monomer
        ed_mol.BeginBatchEdit()
        ed_mol.AddBond(
            dummy_atom_neighbor_idx[0],
            dummy_atom_neighbor_idx[-1],
            Chem.rdchem.BondType.SINGLE,
        )
        # delete dummy atom around the new bond
        ed_mol.RemoveAtom(dummy_atom_idx[0])
        ed_mol.RemoveAtom(dummy_atom_idx[-1])
        ed_mol.CommitBatchEdit()
        # convert back to mol
        curr_mol: Chem.rdchem.Mol = ed_mol.GetMol()

        # convert to fingerprint
        Chem.SanitizeMol(curr_mol)
        curr_mol_fp: list = list(
            AllChem.GetMorganFingerprintAsBitVect(curr_mol, 3, nBits=512)
        )
        # add solvent SMILES fingerprint
        solvent_mol = Chem.MolFromSmiles(row["Solvent_SMILES"])
        solvent_fp: list = list(
            AllChem.GetMorganFingerprintAsBitVect(solvent_mol, 3, nBits=512)
        )
        curr_mol_fp.extend(solvent_fp)
        expt_results.at[index, f"{N}mer_circular_graph_fp"] = curr_mol_fp
    expt_results.to_csv(cfp_path, index=False)


if __name__ == "__main__":
    Nmer_graph(PS_DATA, PS_CFP_DATA, 2)
    Nmer_graph(PS_CFP_DATA, PS_CFP_DATA, 3)
    polymer_graph(PS_CFP_DATA, PS_CFP_DATA, 3)
