import pathlib
from pathlib import Path
from rdkit import Chem
import pandas as pd
from rdkit.Chem import Draw

co2_data: Path = (
    pathlib.Path(__file__).parent.parent / "raw" / "CO2_Soleimani" / "co2_expt_data.csv"
)
pv_data: Path = (
    pathlib.Path(__file__).parent.parent / "raw" / "PV_Wang" / "pv_exptresults.csv"
)
swell_data: Path = (
    pathlib.Path(__file__).parent.parent / "raw" / "Swelling_Xu" / "ps_exptresults.csv"
)
dft_data: Path = (
    pathlib.Path(__file__).parent.parent
    / "raw"
    / "DFT_Ramprasad"
    / "dft_exptresults.csv"
)


def fragment_polymer(polymer: str) -> Chem.Mol:
    """Fragment a polymer molecule into its monomer units. The order of the monomer units is preserved.

    Args:
        polymer: A polymer molecule.

    Returns:
        A polymer molecule with its monomer units fragmented.
    """
    mol: Chem.Mol = Chem.MolFromSmiles(polymer)
    # visualize molecule
    # Set atom index to be displayed
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    Draw.MolToImageFile(mol, pathlib.Path(__file__).parent / "mol.png", size=(300, 300))
    # find index of astericks
    asterick_idx: list = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            asterick_idx.append(atom.GetIdx())
    backbone_idx: list = list(
        Chem.rdmolops.GetShortestPath(mol, asterick_idx[0], asterick_idx[1])
    )
    # remove astericks (first and last idx)
    asterick_idx_1: int = backbone_idx[0]
    asterick_idx_2: int = backbone_idx[-1]
    asterick_idxs: list = [asterick_idx_1, asterick_idx_2]
    backbone_idx: list = backbone_idx[1:-1]
    # fragment the smallest building blocks possible
    # If not in ring, add new fragment
    # If in ring, add new fragment if ring is not already in base_fragments
    base_fragments: list = []
    for b_idx in backbone_idx:
        atom: Chem.Atom = mol.GetAtomWithIdx(b_idx)
        if atom.IsInRing():
            # check if ring is already in base_fragments
            for i in range(0, len(base_fragments)):
                if b_idx in base_fragments[i]["backbone"]:
                    current_unit_idx: int = i
                    base_fragments[i]["backbone"].append(b_idx)
                else:
                    base_fragments.append(
                        {"backbone": [b_idx], "sidechain": [], "connections": []}
                    )
                    current_unit_idx = len(base_fragments) - 1
        else:
            base_fragments.append(
                {"backbone": [b_idx], "sidechain": [], "connections": []}
            )
            current_unit_idx = len(base_fragments) - 1
        # check connected atoms
        for neighbor in atom.GetNeighbors():
            # Ring fragment in backbone
            if neighbor.IsInRing() and neighbor.GetIdx() in backbone_idx:
                base_fragments[current_unit_idx]["backbone"].append(neighbor.GetIdx())
            # Ring fragment not in backbone
            elif neighbor.IsInRing() and neighbor.GetIdx() not in backbone_idx:
                base_fragments[current_unit_idx]["backbone"].append(neighbor.GetIdx())
            # Sidechain of a fragment
            elif (
                neighbor.GetIdx() not in backbone_idx
                and neighbor.GetIdx() not in asterick_idxs
            ):
                # fragment
                base_fragments[current_unit_idx].append(neighbor.GetIdx())
                break
    print(f"{backbone_idx=}")
    mol_idx: list = [atom.GetIdx() for atom in mol.GetAtoms()]
    mol_idx.remove(asterick_idx_1)
    mol_idx.remove(asterick_idx_2)
    print(f"{mol_idx=}")
    pass


def iterative_shuffle(fragmented: Chem.Mol) -> list:
    """Iteratively shuffle the fragments of a polymer molecule.

    Args:
        polymer: A polymer molecule.

    Returns:
        A polymer molecule with its monomer units shuffled. (list[list[str]])
    """
    pass


def augment_dataset(dataset: Path) -> pd.DataFrame:
    """Augment the dataset by iteratively shuffling the fragments of a polymer.

    Args:
        dataset: A dataset to augment.

    Returns:
        A dataset with the experimental data.
    """
    dataset_df: pd.DataFrame = pd.read_csv(dataset)
    dataset_df["augmented_polymer"] = dataset_df["smiles"].map(
        lambda x: iterative_shuffle(fragment_polymer(x))
    )

    pass


# Egc mentioned by reviewer
def filter_dataset(dataset: Path, property: str) -> pd.DataFrame:
    """Filter the dataset by removing the polymer molecules without the desired property.

    Args:
        dataset: A dataset to filter.

    Returns:
        A dataset with the experimental data.
    """
    pass


# augment_dataset(dft_data)
fragment_polymer("NC2C(O)C(OC1OC(CO)C(O(C))C(O[*])C1N)C(CO)OC2([*])")
