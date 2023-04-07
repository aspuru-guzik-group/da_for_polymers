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
    / "preprocess"
    / "DFT_Ramprasad"
    / "dft_exptresults_Egc.csv"
)

augmented_dft_data: Path = (
    pathlib.Path(__file__).parent.parent
    / "input_representation"
    / "DFT_Ramprasad"
    / "automated_fragment"
    / "master_automated_fragment_Egc.csv"
)

current_dir: Path = pathlib.Path(__file__).parent


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
    # reorder fragments
    reordered_frags: list[str] = reorder_fragments(
        base_fragments, new_mol, recombine=True
    )
    # add end connections to atom separation idx
    if len(reordered_frags) > 1:
        frag_0: Chem.Mol = Chem.MolFromSmarts(reordered_frags[0])
        frag_1: Chem.Mol = Chem.MolFromSmarts(reordered_frags[-1])
        for atom in frag_0.GetAtoms():
            if atom.GetAtomicNum() == 0:
                neighbors: tuple[Chem.Atom] = atom.GetNeighbors()
                atom_0: int = neighbors[0].GetAtomMapNum()
        for atom in frag_1.GetAtoms():
            if atom.GetAtomicNum() == 0:
                neighbors: tuple[Chem.Atom] = atom.GetNeighbors()
                atom_1: int = neighbors[0].GetAtomMapNum()
        atom_separation_idx.append([atom_0, atom_1])

        shuffled_reordered_frags: list[list[str]] = iterative_shuffle(reordered_frags)
        shuffled_atom_separation_idx: list[list[int]] = iterative_shuffle(
            atom_separation_idx
        )
        # print(f"{reordered_frags=}")
        # Must recombine with RDKiT (cannot do str replacements)
        # atom_i = 0
        # atomic_nums: list = list(range(57, 72))
        # atomic_nums.extend(list(range(89, 104)))
        # atomic_nums.extend(list(range(104, 119)))
        mol_fragments: list[Chem.Mol] = []
        # Recombine by adding a single bond to the atoms via atom map number
        for shuffled_i in range(len(shuffled_reordered_frags)):
            # print(f"{shuffled_i=}")
            # print(f"{shuffled_reordered_frags[shuffled_i]=}")
            mol_frag_0: Chem.Mol = Chem.MolFromSmarts(
                shuffled_reordered_frags[shuffled_i][0]
            )
            i = 1
            # print(f"{shuffled_atom_separation_idx=}")
            shuffle_atom_sep: list = shuffled_atom_separation_idx[shuffled_i]
            # print(f"{shuffle_atom_sep=}")
            for atom_sep in shuffle_atom_sep[0 : len(shuffle_atom_sep) - 1]:
                mol_frag_1: Chem.Mol = Chem.MolFromSmarts(
                    shuffled_reordered_frags[shuffled_i][i]
                )
                # add bond between mol_frag_0 and mol_frag_1
                combined_mol: Chem.Mol = Chem.CombineMols(mol_frag_0, mol_frag_1)
                # print(f"{Chem.MolToSmarts(combined_mol)=}")
                atoms = []
                for atom in atom_sep:
                    # print(f"{atom=}")
                    for c_atom in combined_mol.GetAtoms():
                        if c_atom.GetAtomMapNum() == atom:
                            atoms.append(c_atom)
                            # print(f"{c_atom.GetIdx()=}")
                # print(f"{atoms=}")
                if len(atoms) == 2:
                    # print("------------------PASSED------------------")
                    editable_combined_mol: Chem.EditableMol = Chem.EditableMol(
                        combined_mol
                    )
                    editable_combined_mol.AddBond(
                        atoms[0].GetIdx(), atoms[1].GetIdx(), Chem.BondType.SINGLE
                    )
                    mol_frag_0 = editable_combined_mol.GetMol()
                    i += 1

                # print(f"{Chem.MolToSmarts(mol_frag_0)=}, FINISHED")
            # Remove all dummy atoms

            editable_combined_mol: Chem.EditableMol = Chem.EditableMol(mol_frag_0)
            dummy_atoms_to_delete: list = []
            for atom in mol_frag_0.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    dummy_atoms_to_delete.append(atom.GetIdx())
            editable_combined_mol.BeginBatchEdit()
            for atom in dummy_atoms_to_delete:
                editable_combined_mol.RemoveAtom(atom)
            editable_combined_mol.CommitBatchEdit()
            mol_frag_0 = editable_combined_mol.GetMol()
            # Where dummy atoms should be connected, add them back in
            dummy_connections: list[int] = shuffle_atom_sep[-1]
            # print(f"{dummy_connections=}")
            for atom in mol_frag_0.GetAtoms():
                if atom.GetAtomMapNum() in dummy_connections:
                    connection_idx = atom.GetIdx()
                # else:
                # print(Chem.MolToSmarts(mol_frag_0))
            dummy_idx: int = editable_combined_mol.AddAtom(Chem.rdchem.Atom(0))
            editable_combined_mol.AddBond(
                connection_idx, dummy_idx, Chem.BondType.SINGLE
            )

            mol_frag_0 = editable_combined_mol.GetMol()
            Draw.MolToFile(
                mol_frag_0,
                filename=current_dir / "mol_frag_recombined.png",
                size=(500, 500),
            )
            # assert False
            # Remove atom mapping
            Chem.SanitizeMol(mol_frag_0)
            for atom in mol_frag_0.GetAtoms():
                atom.SetAtomMapNum(0)
            mol_fragments.append(mol_frag_0)

        # Remove atom mapping and SanitizeMol
        mol_fragments_smiles: list[str] = [
            Chem.MolToSmiles(frag) for frag in mol_fragments
        ]
        # mol_fragments_smarts: list[str] = [Chem.SanitizeMol(frag) for frag in mol_fragments]
        # print(f"{mol_fragments_smiles=}")
        # Draw.MolToFile(
        #     mol_fragments[2],
        #     filename=current_dir / "mol_frag_recombined_final.png",
        #     size=(500, 500),
        # )
    else:
        # return original smiles
        mol_fragments_smiles: list[str] = [Chem.MolToSmiles(mol)]
    return mol_fragments_smiles


def reorder_fragments(
    base_fragments: dict, mol: Chem.Mol, recombine: bool
) -> list[str]:
    """Reorder fragments based on base_fragments connections."""
    if recombine:
        mol_smi: str = Chem.MolToSmarts(mol)
    else:
        mol_smi: str = Chem.MolToSmiles(mol)
    # print(f"{mol_smi=}")
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
                    for elementary_atom in range(6, 7):
                        str_check_0_idx: str = f"[#{elementary_atom}]"
                        if str_check_0_idx in frag:
                            continue
                        else:
                            matched = False
            if matched and frag not in ordered_frags:
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
        # [a.SetAtomMapNum(0) for a in mol_frag.GetAtoms()]
        # remove dummy atom
        if recombine:
            mol_frag_smi: str = Chem.MolToSmarts(mol_frag)
        else:
            mol_frag_smi: str = Chem.MolToSmiles(mol_frag)
        # mol_frag_smi: str = mol_frag_smi.replace("*", "")
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


def fingerprint_recombined_augmented_data(
    mol_fragment_smiles: list[str],
) -> list[list[int]]:
    """Fingerprint the recombined augmented data.

    Args:
        mol_fragment_smiles: A list of SMILES strings.

    Returns:
        A list of fingerprints.
    """
    fps: list[list[int]] = []
    for mol_frag in mol_fragment_smiles:
        mol_frag = Chem.MolFromSmiles(mol_frag)
        fp: list[int] = list(
            AllChem.GetMorganFingerprintAsBitVect(mol_frag, 3, nBits=512)
        )
        fps.append(fp)
    return fps


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
    dataset_df["polymer_automated_frag"] = fragment_mols.apply(lambda x: x)
    shuffled: list[list[str]] = fragment_mols.apply(lambda x: iterative_shuffle(x))
    dataset_df["polymer_automated_frag_aug"] = shuffled
    recombined_fragment_mols = indices.apply(
        lambda x: fragment_recombined_mol_from_indices(x[0], x[1])
    )
    dataset_df["polymer_automated_frag_recombined_str"] = recombined_fragment_mols
    fingerprint_recombined = recombined_fragment_mols.apply(
        lambda x: fingerprint_recombined_augmented_data(x)
    )
    dataset_df["polymer_automated_frag_recombined_fp"] = fingerprint_recombined
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
# mol, base_fragments = get_fragment_indices(
#     "C(=[O:1])([*:2])[c:3]1[cH:4][cH:5][c:6]([C:9](=[O:10])[O:11][c:12]2[cH:13][cH:14][c:15]([C:18]3([c:28]4[cH:29][cH:30][c:31]([O:34][*:35])[cH:32][cH:33]4)[c:19]4[cH:20][cH:21][cH:22][cH:23][c:24]4[C:25](=[O:26])[O:27]3)[cH:16][cH:17]2)[cH:7][cH:8]1"
# )
# print(Chem.MolToSmiles(mol), base_fragments)
# fragmented = fragment_recombined_mol_from_indices(mol, base_fragments)
# print(fragmented)
# fp = fingerprint_recombined_augmented_data(fragmented)
# print(len(fp))
# shuffled = iterative_shuffle(fragmented)
# print(Chem.MolFromSmiles("*c1ccc(-c2nc3cc4nc(*)oc4cc3o2)cc1"))
# print(augmented_dft_data)
