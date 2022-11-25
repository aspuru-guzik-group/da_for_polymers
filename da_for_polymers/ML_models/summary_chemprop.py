from argparse import ArgumentParser
from pathlib import Path
from requests import delete
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from numpy import mean, std
import pandas as pd
import numpy as np


def main(config):
    """_summary_

    Args:
        config (_type_): _description_
    """
    gt_paths: list[str] = config["ground_truth"]
    pred_paths: list[str] = config["predictions"]
    data_dir: Path = Path(gt_paths[0]).parent
    r_scores: list[float] = []
    r2_scores: list[float] = []
    mae_scores: list[float] = []
    rmse_scores: list[float] = []
    folds: list[int] = []
    fold_idx = 0
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        gt_data: pd.DataFrame = pd.read_csv(gt_path)
        pred_data: pd.DataFrame = pd.read_csv(pred_path)
        target: str = config["target"]
        gt: pd.Series = gt_data[target]
        pred: pd.Series = pred_data[target]
        idx = 0
        delete_idxs: list = []
        for gt_val, pred_val in zip(gt, pred):
            try:
                gt_val = float(gt_val)
                pred_val = float(pred_val)
            except:
                delete_idxs.append(idx)
            idx += 1
        print(delete_idxs)
        gt: pd.Series = gt.drop(delete_idxs)
        pred: pd.Series = pred.drop(delete_idxs)
        gt: pd.Series = gt.astype(float)
        pred: pd.Series = pred.astype(float)
        gt: np.ndarray = gt.to_numpy()
        pred: np.ndarray = pred.to_numpy()
        print(gt, pred)
        r = np.corrcoef(gt, pred)[0, 1]
        r2 = r2_score(gt, pred)
        mae = mean_absolute_error(gt, pred)
        rmse = np.sqrt(mean_squared_error(gt, pred))
        r_scores.append(r)
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        folds.append(fold_idx)
        fold_idx += 1

    progress: dict = {
        "folds": folds,
        "r": r_scores,
        "r2": r2_scores,
        "mae": mae_scores,
        "rmse": rmse_scores,
    }
    progress_df: pd.DataFrame = pd.DataFrame.from_dict(progress, orient="columns")
    progress_path: Path = data_dir / "progress_report.csv"
    progress_df.to_csv(progress_path, index=False)

    summary: dict = {
        "Dataset": [data_dir.parent.parent.stem],
        "num_of_folds": [fold_idx],
        "Features": [config["features"]],
        "Targets": [config["target"]],
        "Model": ["chemprop"],
        "r_mean": [mean(r_scores)],
        "r_std": [std(r_scores)],
        "r2_mean": [mean(r2_scores)],
        "r2_std": [std(r2_scores)],
        "mae_mean": [mean(mae_scores)],
        "mae_std": [std(mae_scores)],
        "rmse_mean": [mean(rmse_scores)],
        "rmse_std": [std(rmse_scores)],
    }
    print(summary)
    summary_df: pd.DataFrame = pd.DataFrame.from_dict(summary)
    summary_path: Path = data_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ground_truth", type=str, nargs="+", help="Filepaths to ground truth data."
    )
    parser.add_argument(
        "--predictions", type=str, nargs="+", help="Filepaths to predictions."
    )
    parser.add_argument("--target", type=str, help="1 target to compute metrics.")
    parser.add_argument(
        "--features",
        type=str,
        help="Name of features used. Found in chemprop_train.csv",
    )
    args = parser.parse_args()
    config = vars(args)
    main(config)

# python summary_chemprop.py --ground_truth ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/manual_frag_str/KFold/chemprop_test_*.csv --predictions ~/Research/Repos/polymer_chemprop_data/results/OPV_Min/manual_frag_str/KFold/predictions_*.csv --target calc_PCE_percent --features DA_manual_str
