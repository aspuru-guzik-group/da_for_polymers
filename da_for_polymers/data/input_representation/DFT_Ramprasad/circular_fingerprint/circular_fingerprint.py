from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import pkg_resources
import pandas as pd
import numpy as np
import copy
from pathlib import Path

DFT_RAMPRASAD = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/DFT_Ramprasad/dft_exptresults_Egc.csv"
)


CFP_DFT_RAMPRASAD = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/DFT_Ramprasad/circular_fingerprint/dft_circular_fingerprint_Egc.csv",
)


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
        monomer_smi: str = row["smiles"]
        monomer_mol: Chem.rdchem.Mol = Chem.MolFromSmiles(monomer_smi)
        curr_mol: Chem.rdchem.Mol = copy.deepcopy(monomer_mol)
        for i in range(N):
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
        curr_mol_fp: np.ndarray = np.array(
            AllChem.GetMorganFingerprintAsBitVect(curr_mol, 3, nBits=512)
        )
        expt_results.at[index, f"{N}mer_fp"] = curr_mol_fp
    expt_results.to_csv(cfp_path, index=False)


def polymer_graph(expt_results_path: str, cfp_path: str) -> pd.DataFrame:
    pass


if __name__ == "__main__":
    # Nmer_graph(CFP_DFT_RAMPRASAD, CFP_DFT_RAMPRASAD, 3)
    polymer_graph(CFP_DFT_RAMPRASAD, CFP_DFT_RAMPRASAD)
