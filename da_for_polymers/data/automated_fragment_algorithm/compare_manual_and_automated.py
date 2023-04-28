from pathlib import Path
import pandas as pd
from rdkit import Chem

current_path: Path = Path(__file__).parent

co2_automated_data: Path = (
    current_path.parent
    / "input_representation/CO2_Soleimani/automated_fragment/master_automated_fragment.csv"
)

co2_manual_data: Path = (
    current_path.parent
    / "input_representation/CO2_Soleimani/manual_frag/master_manual_frag.csv"
)


def compare_manual_and_automated(auto_data: Path, manual_data: Path):
    auto_df: pd.DataFrame = pd.read_csv(auto_data)
    manual_df: pd.DataFrame = pd.read_csv(manual_data)
    for index, row in auto_df.iterrows():
        auto_smi: str = auto_df.at[index, "polymer_automated_frag_SMILES"]
        manual_smi: str = manual_df.at[index, "Polymer_manual_SMILES"]
        print(auto_smi, "///", manual_smi)


compare_manual_and_automated(co2_automated_data, co2_manual_data)
