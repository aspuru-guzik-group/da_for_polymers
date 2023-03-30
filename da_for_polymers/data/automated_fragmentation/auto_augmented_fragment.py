import pathlib
from pathlib import Path
from rdkit import Chem
import pandas as pd

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
    # find index of astericks
    asterick_idx: list = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            asterick_idx.append(atom.GetIdx())
    backbone_idx: list = Chem.rdmolops.GetShortestPath(
        mol, asterick_idx[0], asterick_idx[1]
    )
    mol_idx: list = [atom.GetIdx() for atom in mol.GetAtoms()]
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


augment_dataset(dft_data)
