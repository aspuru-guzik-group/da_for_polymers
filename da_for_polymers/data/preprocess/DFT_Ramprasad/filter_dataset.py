"""Create a function that filters the master_automated_fragment.csv dataset for "Egc" values only."""
import pandas as pd
from pathlib import Path

current_dir: Path = Path(__file__).parent


def filter_dataset(current_directory: Path):
    """Filter the master_automated_fragment.csv dataset for "Egc" values only."""
    df = pd.read_csv(
        current_directory.parent.parent
        / "raw"
        / "DFT_Ramprasad"
        / "dft_exptresults.csv"
    )
    df = df[df["property"] == "Egc"]
    df.to_csv(current_directory / "dft_exptresults_Egc.csv", index=False)


filter_dataset(current_dir)
