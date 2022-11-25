import pandas as pd
import argparse
from pathlib import Path

from da_for_polymers.visualization.path_utils import (
    gather_results,
    path_to_result,
)


def aggregate_results(config):
    """Aggregates all the summary results into a .csv.

    Args:
        config (_type_): _description_
    """
    path_config: dict = {
        "datasets": [],
        "models": [],
        "input_representations": [],
        "feature_names": [],
        "target_names": [],
        "x": "Model",
        "hue": "Features",
        "metrics": "r2",
    }
    # Combine 2 dictionaries together
    config.update(path_config)
    print(config)

    progress_paths: list[Path] = path_to_result(config, "summary")
    aggregate_path: Path = Path(config["results_path"])
    aggregate: pd.DataFrame = gather_results(progress_paths)
    aggregate: pd.DataFrame = aggregate[aggregate.Features != "SELFIES"]
    aggregate: pd.DataFrame = aggregate[aggregate.Features != "BRICS"]
    # get unique values from datasets column
    datasets: list = list(aggregate.Dataset.unique())
    for dataset in datasets:
        subset: pd.DataFrame = aggregate[aggregate["Dataset"] == dataset]
        # sort by metrics (highest to lowest)
        subset.sort_values(by=["r2_mean"], axis=0, ascending=False, inplace=True)
        # create new csv with sorted metrics and different files
        subset_path: Path = aggregate_path / dataset / "{}_results.csv".format(dataset)
        subset.to_csv(subset_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_training",
        type=str,
        help="Filepath to directory called 'training' which contains all outputs from train.py",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Directory path to location of all_results.csv",
    )
    args = parser.parse_args()
    config = vars(args)
    aggregate_results(config)
