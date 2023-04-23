from pathlib import Path
import pandas as pd
import ast
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

current_path: Path = Path(__file__).parent

AUTO_FRAG_PATH: Path = current_path / "master_automated_fragment.csv"


def add_solvent(auto_frag_path: Path):
    data: pd.DataFrame = pd.read_csv(auto_frag_path)
    data["PS_automated_frag"] = ""
    data["PS_automated_frag_SMILES"] = ""
    data["PS_automated_frag_aug"] = ""
    data["PS_automated_frag_aug_SMILES"] = ""
    data["PS_automated_frag_aug_recombined_SMILES"] = ""
    data["PS_automated_frag_aug_recombined_fp"] = ""
    for index, row in data.iterrows():
        ps_automated_frag: list = ast.literal_eval(
            data.at[index, "polymer_automated_frag"]
        )
        ps_automated_frag.extend([".", data.at[index, "Solvent_SMILES"]])
        data.at[index, "PS_automated_frag"] = ps_automated_frag

        ps_automated_frag_SMILES = data.at[index, "polymer_automated_frag_SMILES"]
        ps_automated_frag_SMILES = (
            ps_automated_frag_SMILES + "." + data.at[index, "Solvent_SMILES"]
        )
        data.at[index, "PS_automated_frag_SMILES"] = ps_automated_frag_SMILES

        ps_automated_frag_aug: list = ast.literal_eval(
            data.at[index, "polymer_automated_frag_aug"]
        )
        for frag_aug in ps_automated_frag_aug:
            frag_aug.extend([".", data.at[index, "Solvent_SMILES"]])
        data.at[index, "PS_automated_frag_aug"] = ps_automated_frag_aug

        ps_automated_frag_aug_SMILES: list = ast.literal_eval(
            data.at[index, "polymer_automated_frag_aug_SMILES"]
        )
        for frag_aug in ps_automated_frag_aug_SMILES:
            frag_aug = frag_aug + "." + data.at[index, "Solvent_SMILES"]
        data.at[index, "PS_automated_frag_aug_SMILES"] = ps_automated_frag_aug_SMILES

        ps_automated_frag_aug_recombined_SMILES: list = ast.literal_eval(
            data.at[index, "polymer_automated_frag_aug_recombined_SMILES"]
        )
        for frag_aug in ps_automated_frag_aug_recombined_SMILES:
            frag_aug = frag_aug + "." + data.at[index, "Solvent_SMILES"]
        data.at[
            index, "PS_automated_frag_aug_recombined_SMILES"
        ] = ps_automated_frag_aug_recombined_SMILES

        ps_automated_frag_aug_recombined_fp: list = ast.literal_eval(
            data.at[index, "polymer_automated_frag_aug_recombined_fp"]
        )
        solvent_mol: Chem.Mol = Chem.MolFromSmiles(data.at[index, "Solvent_SMILES"])
        # print(solvent_mol)
        solvent_fp: list[int] = list(
            AllChem.GetMorganFingerprintAsBitVect(solvent_mol, 3, nBits=512)
        )
        for frag_aug in ps_automated_frag_aug_recombined_fp:
            frag_aug.extend(solvent_fp)
        data.at[
            index, "PS_automated_frag_aug_recombined_fp"
        ] = ps_automated_frag_aug_recombined_fp
    # data.drop(
    #     columns=[
    #         "polymer_automated_frag",
    #         "polymer_automated_frag_SMILES",
    #         "polymer_automated_frag_aug",
    #         "polymer_automated_frag_aug_SMILES",
    #         "polymer_automated_frag_aug_recombined_SMILES",
    #         "polymer_automated_frag_aug_recombined_fp",
    #     ],
    #     inplace=True,
    # )
    data.to_csv(auto_frag_path, index=False)


add_solvent(AUTO_FRAG_PATH)
