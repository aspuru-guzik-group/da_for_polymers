from argparse import ArgumentParser
from pathlib import Path

from numpy import mean, std
import pandas as pd

# Database
def summarize(config: dict):
    folder_path = config["folder_path"]
    folder_path: Path = Path(folder_path)
    for file in folder_path.iterdir():
        if file.name == "progress_report.csv":
            data: pd.DataFrame = pd.read_csv(file)
            r_mean = mean(data["r"])
            r_std = std(data["r"])
            r2_mean = mean(data["r2"])
            r2_std = std(data["r2"])
            rmse_mean = mean(data["rmse"])
            rmse_std = std(data["rmse"])
            mae_mean = mean(data["mae"])
            mae_std = std(data["mae"])
            summary_path: Path = file.parent / "summary.csv"
            summary_data: pd.DataFrame = pd.DataFrame(
                columns=[
                    "Dataset",
                    "num_of_folds",
                    "Features",
                    "Targets",
                    "Model",
                    "r_mean",
                    "r_std",
                    "r2_mean",
                    "r2_std",
                    "rmse_mean",
                    "rmse_std",
                    "mae_mean",
                    "mae_std",
                    "num_of_data",
                ]
            )
            summary_data.at[0, "Dataset"] = "Swelling_Xu"
            summary_data.at[0, "num_of_folds"] = 7
            summary_data.at[0, "Features"] = folder_path.parent.name
            summary_data.at[0, "Targets"] = folder_path.name
            summary_data.at[0, "Model"] = folder_path.parent.parent.name
            summary_data.at[0, "r_mean"] = r_mean
            summary_data.at[0, "r_std"] = r_std
            summary_data.at[0, "r2_mean"] = r2_mean
            summary_data.at[0, "r2_std"] = r2_std
            summary_data.at[0, "rmse_mean"] = rmse_mean
            summary_data.at[0, "rmse_std"] = rmse_std
            summary_data.at[0, "mae_mean"] = mae_mean
            summary_data.at[0, "mae_std"] = mae_std
            summary_data.at[0, "num_of_data"] = 77
            summary_data.to_csv(summary_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder_path", type=str)
    args = parser.parse_args()
    config = vars(args)
    summarize(config)
