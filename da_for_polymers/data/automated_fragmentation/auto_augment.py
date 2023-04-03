import pathlib
from pathlib import Path
from rdkit import Chem
import pandas as pd
from rdkit.Chem import Draw, AllChem
import copy
import sys
from collections import deque
import numpy as np
import ast

sys.setrecursionlimit(100)

co2_data: Path = (
    pathlib.Path(__file__).parent.parent / "raw" / "CO2_Soleimani" / "co2_expt_data.csv"
)
augmented_co2_data: Path = (
    pathlib.Path(__file__).parent.parent
    / "input_representation"
    / "CO2_Soleimani"
    / "automated_fragment"
    / "master_automated_fragment.csv"
)

pv_data: Path = (
    pathlib.Path(__file__).parent.parent / "raw" / "PV_Wang" / "pv_exptresults.csv"
)
augmented_pv_data: Path = (
    pathlib.Path(__file__).parent.parent
    / "input_representation"
    / "PV_Wang"
    / "automated_fragment"
    / "master_automated_fragment.csv"
)

swell_data: Path = (
    pathlib.Path(__file__).parent.parent / "raw" / "Swelling_Xu" / "ps_exptresults.csv"
)
augmented_swell_data: Path = (
    pathlib.Path(__file__).parent.parent
    / "input_representation"
    / "Swelling_Xu"
    / "automated_fragment"
    / "master_automated_fragment.csv"
)

dft_data: Path = (
    pathlib.Path(__file__).parent.parent
    / "raw"
    / "DFT_Ramprasad"
    / "dft_exptresults.csv"
)

augmented_dft_data: Path = (
    pathlib.Path(__file__).parent.parent
    / "input_representation"
    / "DFT_Ramprasad"
    / "automated_fragment"
    / "master_automated_fragment.csv"
)


def recursively_find_sidechains(
    mol: Chem.Mol,
    sidechains_idx: list,
    current_idx: int,
    current_fragment: dict,
    backbone_idx: list,
    asterick_idx: list,
) -> list:
    """Recursively find the sidechains of a polymer molecule.

    Args:
        mol: A polymer molecule.
        sidechains_idx: The sidechain indices of the current sidechain.
        current_idx: The current index.
        current_fragment: Dictionary with backbone, sidechain, and connections indices of the fragment of interest.
        backbone_idx: List of backbone indices of the whole fragment.

    Returns:
        A polymer molecule with its monomer units fragmented.
    """
    # base case
    # print(f"{sidechains_idx=}")
    if current_idx in sidechains_idx:
        return sidechains_idx
    # recursive case
    else:
        atom: Chem.Atom = mol.GetAtomWithIdx(current_idx)
        # print(f"{atom.GetIdx()=}")
        # End of sidechain, add current idx to sidechain_idx
        if (
            len(atom.GetNeighbors()) == 1
            and current_idx not in backbone_idx
            and current_idx not in current_fragment["sidechain"]
            and current_idx not in current_fragment["backbone"]
            and current_idx not in asterick_idx
        ):
            sidechains_idx.append(current_idx)
            return recursively_find_sidechains(
                mol,
                sidechains_idx,
                current_idx,
                current_fragment,
                backbone_idx,
                asterick_idx,
            )
        # In the middle of sidechain or in a ring
        else:
            # If in ring
            # print(f"{sidechains_idx=}")
            if atom.IsInRing():
                all_atoms_in_ring_added: bool = True
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() not in sidechains_idx:
                        all_atoms_in_ring_added = False
                if all_atoms_in_ring_added:
                    sidechains_idx.append(current_idx)
                    return sidechains_idx
                else:
                    for neighbor in atom.GetNeighbors():
                        # print(f"{neighbor.GetIdx()=}")
                        if (
                            neighbor.GetIdx() not in sidechains_idx
                            and neighbor.GetIdx() not in backbone_idx
                            and neighbor.GetIdx() not in current_fragment["sidechain"]
                            and neighbor.GetIdx() not in current_fragment["backbone"]
                            and neighbor.GetIdx() not in asterick_idx
                        ):
                            sidechains_idx.append(current_idx)
                            return recursively_find_sidechains(
                                mol,
                                sidechains_idx,
                                neighbor.GetIdx(),
                                current_fragment,
                                backbone_idx,
                                asterick_idx,
                            )
            # In middle of sidechain
            else:
                for neighbor in atom.GetNeighbors():
                    # print(f"{neighbor.GetIdx()=}")
                    if (
                        neighbor.GetIdx() not in sidechains_idx
                        and neighbor.GetIdx() not in backbone_idx
                        and neighbor.GetIdx() not in current_fragment["sidechain"]
                        and neighbor.GetIdx() not in current_fragment["backbone"]
                        and neighbor.GetIdx() not in asterick_idx
                    ):
                        sidechains_idx.append(current_idx)
                        return recursively_find_sidechains(
                            mol,
                            sidechains_idx,
                            neighbor.GetIdx(),
                            current_fragment,
                            backbone_idx,
                            asterick_idx,
                        )


def get_fragment_indices(polymer: str) -> list[dict]:
    """Fragment a polymer molecule into its monomer units. The order of the monomer units is preserved.

    Args:
        polymer: A polymer molecule.

    Returns:
        A polymer molecule with its monomer units fragmented.
    """
    mol: Chem.Mol = Chem.MolFromSmiles(polymer)
    ring_info: Chem.RingInfo = mol.GetRingInfo()
    # visualize molecule
    # Set atom index to be displayed
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    # Draw.MolToImageFile(mol, pathlib.Path(__file__).parent / "mol.png", size=(300, 300))
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
    # print(f"{backbone_idx=}")
    # Get all atoms that are non-backbone
    mol_idx: list = [atom.GetIdx() for atom in mol.GetAtoms()]
    mol_idx.remove(asterick_idx_1)
    mol_idx.remove(asterick_idx_2)
    # fragment the smallest building blocks possible
    # If not in ring, add new fragment
    # If in ring, add new fragment if ring is not already in base_fragments
    base_fragments: list = []
    for b_idx in backbone_idx:
        # print(f"{base_fragments=}")
        atom: Chem.Atom = mol.GetAtomWithIdx(b_idx)
        present: bool = False
        if atom.IsInRing():
            # check if ring is already in base_fragments
            for i in range(0, len(base_fragments)):
                if b_idx in base_fragments[i]["backbone"]:
                    current_unit_idx: int = i
                    present: bool = True
            if not present:
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
            # print(f"{neighbor.GetIdx()=}")
            # Neighboring ring atoms (in the same ring)
            if atom.IsInRing():
                # Get ring info for this ring
                which_ring: tuple = ring_info.AtomMembers(atom.GetIdx())
                # find all backbone atoms that are connected to the ring
                for ring_idx in which_ring:
                    ring_idxs: tuple = ring_info.AtomRings()[ring_idx]
                    base_fragments[current_unit_idx]["backbone"].extend(ring_idxs)
                base_fragments[current_unit_idx]["backbone"]: list = list(
                    dict.fromkeys(base_fragments[current_unit_idx]["backbone"])
                )
                # find all sidechain atoms connected to backbone atoms in ring
                for ring_backbone_idx in base_fragments[current_unit_idx]["backbone"]:
                    ring_backbone_mol: Chem.Mol = mol.GetAtomWithIdx(ring_backbone_idx)
                    for r_neighbor in ring_backbone_mol.GetNeighbors():
                        if (
                            r_neighbor.GetIdx()
                            not in base_fragments[current_unit_idx]["backbone"]
                            and r_neighbor.GetIdx() not in backbone_idx
                        ):
                            # print(f"{r_neighbor.GetIdx()=}")
                            # print(f"{recursively_find_sidechains(mol, [], r_neighbor.GetIdx(), base_fragments[current_unit_idx], backbone_idx, asterick_idx)}")
                            if (
                                recursively_find_sidechains(
                                    mol,
                                    [],
                                    r_neighbor.GetIdx(),
                                    base_fragments[current_unit_idx],
                                    backbone_idx,
                                    asterick_idx,
                                )
                                is not None
                            ):
                                base_fragments[current_unit_idx]["sidechain"].extend(
                                    recursively_find_sidechains(
                                        mol,
                                        [],
                                        r_neighbor.GetIdx(),
                                        base_fragments[current_unit_idx],
                                        backbone_idx,
                                        asterick_idx,
                                    )
                                )
                # Find connections
                if (
                    neighbor.GetIdx() in backbone_idx
                    and neighbor.GetIdx()
                    not in base_fragments[current_unit_idx]["backbone"]
                ):
                    base_fragments[current_unit_idx]["connections"].append(
                        neighbor.GetIdx()
                    )
            else:
                if (
                    neighbor.GetIdx() not in backbone_idx
                    and neighbor.GetIdx() not in asterick_idxs
                ):
                    base_fragments[current_unit_idx]["sidechain"].append(
                        neighbor.GetIdx()
                    )
                # New fragment
                elif (
                    neighbor.GetIdx() in backbone_idx
                    and neighbor.GetIdx()
                    not in base_fragments[current_unit_idx]["backbone"]
                ):
                    base_fragments[current_unit_idx]["connections"].append(
                        neighbor.GetIdx()
                    )

    return mol, base_fragments


def fragment_mol_from_indices(mol: Chem.Mol, base_fragments: list[dict]) -> list[str]:
    """Fragment a polymer molecule into its monomer units. Including preprocessing for recombination.

    Args:
        mol: A polymer molecule.
        base_fragments: A list of dictionaries with backbone, sidechain, and connections indices of the fragments of interest.

    Returns:
        A list of polymer molecules with its monomer units fragmented.
    """
    if len(base_fragments) == 1:
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        mol_smi: str = Chem.MolToSmiles(mol)
        mol_smi: str = mol_smi.replace("*", "")
        return [mol_smi]
    # print(Chem.MolToSmiles(mol))
    # Fragment on connection bonds
    # print(f"{base_fragments=}" )
    bonds_to_fragment: list[int] = []
    atom_separation_idx: list[list[int]] = []
    for i in range(0, len(base_fragments) - 1):
        fragment: dict = base_fragments[i]
        next_fragment: dict = base_fragments[i + 1]
        for connection in fragment["connections"]:
            if connection in next_fragment["backbone"]:
                atom_1: int = connection
        for next_connection in next_fragment["connections"]:
            if next_connection in fragment["backbone"]:
                atom_2: int = next_connection
        # print(f"{base_fragments[i]=}")
        # print(f"{base_fragments[i+1]=}")
        # print(f"{mol.GetBondBetweenAtoms(atom_1, atom_2)=}")
        atom_separation_idx.append([atom_1, atom_2])
        bonds_to_fragment.append(mol.GetBondBetweenAtoms(atom_1, atom_2).GetIdx())
    new_mol: Chem.Mol = Chem.FragmentOnBonds(mol, bonds_to_fragment, addDummies=False)

    new_mol_ordered_fragments: list[str] = reorder_fragments(
        base_fragments, new_mol, recombine=False
    )

    return new_mol_ordered_fragments


def fragment_recombined_mol_from_indices(
    mol: Chem.Mol, base_fragments: list[dict]
) -> list[str]:
    """Fragment a polymer molecule into its monomer units. Including preprocessing for recombination.

    Args:
        mol: A polymer molecule.
        base_fragments: A list of dictionaries with backbone, sidechain, and connections indices of the fragments of interest.

    Returns:
        A list of polymer molecules with its monomer units fragmented.
    """
    if len(base_fragments) == 1:
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        mol_smi: str = Chem.MolToSmiles(mol)
        mol_smi: str = mol_smi.replace("*", "")
        return [mol_smi]
    # print(Chem.MolToSmiles(mol))
    # Fragment on connection bonds
    # print(f"{base_fragments=}" )
    bonds_to_fragment: list[int] = []
    atom_separation_idx: list[list[int]] = []
    for i in range(0, len(base_fragments) - 1):
        fragment: dict = base_fragments[i]
        next_fragment: dict = base_fragments[i + 1]
        for connection in fragment["connections"]:
            if connection in next_fragment["backbone"]:
                atom_1: int = connection
        for next_connection in next_fragment["connections"]:
            if next_connection in fragment["backbone"]:
                atom_2: int = next_connection
        # print(f"{base_fragments[i]=}")
        # print(f"{base_fragments[i+1]=}")
        # print(f"{mol.GetBondBetweenAtoms(atom_1, atom_2)=}")
        atom_separation_idx.append([atom_1, atom_2])
        bonds_to_fragment.append(mol.GetBondBetweenAtoms(atom_1, atom_2).GetIdx())
    new_mol: Chem.Mol = Chem.FragmentOnBonds(mol, bonds_to_fragment, addDummies=False)
    recombine_mol: Chem.Mol = copy.deepcopy(new_mol)
    # Add arbitrary atoms for RECOMBINATION
    atomic_num: int = 64
    for atom_sep in atom_separation_idx:
        for atom in recombine_mol.GetAtoms():
            if atom.GetIdx() in atom_sep:
                if atom.GetAtomicNum() != atomic_num - 1:
                    atom.SetAtomicNum(atomic_num)
                    atomic_num += 1

    recombine_mol.UpdatePropertyCache(strict=False)

    recombined_mol_ordered_fragments: list[str] = reorder_fragments(
        base_fragments, recombine_mol, recombine=True
    )
    return recombined_mol_ordered_fragments


def reorder_fragments(
    base_fragments: dict, mol: Chem.Mol, recombine: bool
) -> list[str]:
    """Reorder fragments based on base_fragments connections."""
    if recombine:
        mol_smi: str = Chem.MolToSmarts(mol)
    else:
        mol_smi: str = Chem.MolToSmiles(mol)
    new_frags: list[str] = mol_smi.split(".")  # split on "." to get fragments
    # Re-order fragments based on base_fragments connections
    ordered_frags: list[str] = []
    # print(f"{new_frags=}")
    for i in range(0, len(base_fragments)):
        for frag in new_frags:
            matched = True
            # all idx in backbone must be matched
            for b_idx in base_fragments[i]["backbone"]:
                str_check: str = ":" + str(b_idx) + "]"
                if str_check not in frag:
                    matched = False
            if matched:
                ordered_frags.append(frag)
    # print(f"{ordered_frags=}")
    # print(f"{ordered_frags=}")
    # Remove astericks on fragments
    ordered_fragments_without_astericks: list[str] = []
    for i in range(0, len(ordered_frags)):
        if recombine:
            mol_frag: Chem.Mol = Chem.MolFromSmarts(ordered_frags[i])
        else:
            mol_frag: Chem.Mol = Chem.MolFromSmiles(ordered_frags[i])
        # print(f"{ordered_frags[i]=}")
        # print(f"{mol_frag=}")
        # Remove atom mapping
        [a.SetAtomMapNum(0) for a in mol_frag.GetAtoms()]
        # remove dummy atom
        mol_frag_smi: str = Chem.MolToSmiles(mol_frag)
        mol_frag_smi: str = Chem.CanonSmiles(mol_frag_smi)
        mol_frag_smi: str = mol_frag_smi.replace("*", "")
        # edmol_frag: Chem.EditableMol = Chem.EditableMol(mol_frag)
        # for atom in mol_frag.GetAtoms():
        #     if atom.GetSymbol() == "*":
        #         edmol_frag.RemoveAtom(atom.GetIdx())
        # mol_frag = edmol_frag.GetMol()
        # mol_frag = Chem.rdmolops.RemoveHs(mol_frag, implicitOnly=False)
        # Draw.MolToImageFile(mol_frag, pathlib.Path(__file__).parent / "mol_frag.png", size=(300, 300))
        ordered_fragments_without_astericks.append(mol_frag_smi)

    return ordered_fragments_without_astericks


def iterative_shuffle(fragmented: list[str]) -> list[list[str]]:
    """Iteratively shuffle the fragments of a polymer molecule.

    Args:
        polymer: A polymer molecule.

    Returns:
        A polymer molecule with its monomer units shuffled. (list[list[str]])
    """
    augmented_polymer_list = []
    polymer_frag_deque = deque(copy.copy(fragmented))
    for j in range(len(fragmented)):
        frag_rotate = copy.copy(polymer_frag_deque)
        frag_rotate.rotate(j)
        frag_rotate = list(frag_rotate)
        augmented_polymer_list.append(frag_rotate)

    return augmented_polymer_list


def recombine_fragments(fragment_smarts: list[list[str]]) -> list[str]:
    """
    Function that recombines the rearranged molecule into the appropriate SMILES.
    """
    recombined_fragments: list[str] = []
    for each_shuffle in fragment_smarts:
        #  Reaction SMARTS
        curr_mol = each_shuffle[0]
        print(f"{each_shuffle=}")
        idx = 2
        for frag in each_shuffle[1:]:
            atomic_num: int = 64
            recombined = False
            stop_condition = 0
            while not recombined:
                if stop_condition == 50:
                    assert False, f"{each_shuffle=}, {curr_mol=}, {frag=}"
                elif idx == len(
                    each_shuffle
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
                                    atomic_num_2 + len(each_shuffle) - 1,
                                    atomic_num_2,
                                )
                            )
                            products = rxn.RunReactants(
                                [Chem.MolFromSmarts(x) for x in [curr_mol, frag]]
                            )
                            print(
                                f"{atomic_num_2=}, {atomic_num=}, {atomic_num_2 + len(each_shuffle) - 1=}"
                            )
                            if products == ():
                                rxn = AllChem.ReactionFromSmarts(
                                    "[#{}:1].[#{}:2]>>[C:1]-[C:2]".format(
                                        atomic_num_2,
                                        atomic_num_2 + len(each_shuffle) - 1,
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
                                    atomic_num_2 + len(each_shuffle) - 1,
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
                                        atomic_num_2 + len(each_shuffle) - 1,
                                        atomic_num_2 + len(each_shuffle) - 1,
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
        # for atom in curr_mol.GetAtoms():
        #     atom.SetAtomMapNum(atom.GetIdx())
        # drawn = Chem.Draw.MolToFile(
        #     curr_mol, IMG_PATH + "manual_rearranged.png", size=(700, 700)
        # )

        Chem.SanitizeMol(curr_mol)

        # for atom in curr_mol.GetAtoms():
        #     atom.SetAtomMapNum(atom.GetIdx())
        # drawn = Chem.Draw.MolToFile(
        #     curr_mol, IMG_PATH + "manual_rearranged.png", size=(700, 700)
        # )

        # removes atom map numbering
        [a.SetAtomMapNum(0) for a in curr_mol.GetAtoms() if a.GetAtomicNum() != 0]
        recombined_fragments.append(curr_mol)

    return recombined_fragments


def augment_dataset(dataset: Path, augmented_dataset: Path) -> pd.DataFrame:
    """Augment the dataset by iteratively shuffling the fragments of a polymer.

    Args:
        dataset: A dataset to augment.

    Returns:
        A dataset with the experimental data.
    """
    dataset_df: pd.DataFrame = pd.read_csv(dataset)
    indices: tuple[Chem.Mol, dict] = dataset_df.apply(
        lambda x: get_fragment_indices(x["smiles"]), axis=1
    )
    fragment_mols = indices.apply(lambda x: fragment_mol_from_indices(x[0], x[1]))
    dataset_df["polymer_automated_frag"] = fragment_mols.apply(lambda x: x[0])
    shuffled: list[list[str]] = fragment_mols.apply(lambda x: iterative_shuffle(x))
    dataset_df["polymer_automated_frag_aug"] = shuffled
    # recombined_fragment_mols = indices.apply(
    #     lambda x: fragment_recombined_mol_from_indices(x[0], x[1])
    # )
    # r_shuffled: list[list[str]] = recombined_fragment_mols.apply(
    #     lambda x: iterative_shuffle(x)
    # )
    # recombined: list[str] = r_shuffled.apply(lambda x: recombine_fragments(x))
    # dataset_df["polymer_automated_frag_recombined_str"] = recombined

    dataset_df.to_csv(augmented_dataset, index=False)


# Egc mentioned by reviewer
def filter_dataset(dataset: Path, property: str) -> pd.DataFrame:
    """Filter the dataset by removing the polymer molecules without the desired property.

    Args:
        dataset: A dataset to filter.

    Returns:
        A dataset with the experimental data.
    """
    pass


augment_dataset(dft_data, augmented_dft_data)
# get_fragment_indices("*c1ccc(-c2nc3cc4nc(*)oc4cc3o2)cc1")
# get_fragment_indices("NC2C(O)C(OC1OC(COCCC)C(O(C))C(O[*])C1N)C(CO)OC2([*])")
# mol, base_fragments = get_fragment_indices("NC2C(O)C(OC1OC(COC3CCCC3)C(O(C))C(O[*])C1N)C(CO)OC2([*])")
# fragmented = fragment_mol_from_indices(mol, base_fragments)
# shuffled = iterative_shuffle(fragmented)
# print(Chem.MolFromSmiles("*c1ccc(-c2nc3cc4nc(*)oc4cc3o2)cc1"))
# print(augmented_dft_data)
