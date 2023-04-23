import argparse
from ast import Str
from copy import deepcopy
from email import generator
from pathlib import Path
from typing import Iterable
from matplotlib.container import BarContainer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import textwrap
import json


from da_for_polymers.visualization.path_utils import (
    gather_results,
    path_to_result,
)


def heatmap(config: dict):
    """
    Args:
        config: outlines the parameters to select for the appropriate configurations for comparison
    """

    with open(config["config_path"]) as f:
        plot_config: dict = json.load(f)
    plot_config: dict = plot_config[config["config_name"]]

    # Combine 2 dictionaries together
    config.update(plot_config)
    print(config)

    summary_paths: list[Path] = path_to_result(config, "summary")

    summary: pd.DataFrame = gather_results(summary_paths)
    summary: pd.DataFrame = summary.replace(
        {
            "manual_frag": "Fragments",
            "manual_frag_aug": "Augmented Fragments",
            "manual_frag_str": "Fragment (SMILES)",
            "manual_frag_aug_str": "Augmented Fragment (SMILES)",
            "manual_recombined_aug_SMILES": "Recombined \n Augmented (SMILES)",
            "manual_recombined_aug_fingerprint": "Recombined \n Augmented \n (Fingerprints)",
            "fingerprint": "Fingerprints",
            "Augmented_SMILES": "Augmented SMILES",
            "manual_frag_SMILES": "Fragments (SMILES)",
            "manual_frag_aug_SMILES": "Augmented Fragments \n (SMILES)",
            "BRT": "XGBoost",
        }
    )

    # Plot Axis
    fig, ax = plt.subplots(figsize=(18, 8))
    # Title
    ax.set_title(
        "Heatmap of CO2 Solubility in Polymers".format(config["datasets"][0]),
        fontsize=18,
    )
    # ax.set_title(
    #     "Heatmap of Polymer Pervaporation".format(config["datasets"][0]), fontsize=18
    # )
    # ax.set_title(
    #     "Heatmap of Polymer Swelling".format(config["datasets"][0]), fontsize=18
    # )

    # Color Brewer color palette
    custom_palette = sns.color_palette("Greens", as_cmap=True)
    # custom_palette = sns.color_palette("Oranges", as_cmap=True)
    # custom_palette = sns.color_palette("Blues", as_cmap=True)
    # Heatmap
    mean_metric: str = config["metrics"] + "_mean"
    std_metric: str = config["metrics"] + "_std"

    mean_summary: pd.DataFrame = summary.pivot("Model", "Features", mean_metric)
    x = ["SVM", "RF", "XGBoost", "NN", "LSTM"]
    y = [
        "SMILES",
        "SELFIES",
        "BigSMILES",
        "BRICS",
        "Fragments",
        "Fingerprints",
        "Augmented SMILES",
        "Augmented Fragments",
        "Augmented Fragments \n (SMILES)",
        "Recombined \n Augmented (SMILES)",
        "Recombined \n Augmented \n (Fingerprints)",
    ]
    mean_summary: pd.DataFrame = mean_summary.reindex(index=x, columns=y)

    summary_annotated: pd.DataFrame = deepcopy(summary)
    for index, row in summary.iterrows():
        m: float = round(summary.at[index, mean_metric], 2)
        s: float = round(summary.at[index, std_metric], 2)
        annotate_label: str = str(m) + "\n" + "(" + str(s) + ")"
        summary_annotated.at[index, "annotate_label"] = annotate_label

    summary_annotated: pd.DataFrame = summary_annotated.pivot(
        "Model", "Features", "annotate_label"
    )
    summary_annotated: pd.DataFrame = summary_annotated.reindex(index=x, columns=y)
    summary_annotated: np.ndarray = summary_annotated.to_numpy()
    sns.set(font_scale=1.4)
    res = sns.heatmap(
        mean_summary,
        annot=summary_annotated,
        annot_kws={"size": 18},
        cmap=custom_palette,
        fmt="",
        cbar_kws={
            "label": "$Avg.\;R^2$ \n ($±\;StDev.\;R^2$)".format(mean_metric, std_metric)
        },
    )

    res.set_xticklabels(res.get_xmajorticklabels(), fontsize=14)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize=14, rotation=0)
    res.set_ylabel("Models", fontsize=18)
    res.set_xlabel("Input Representation", fontsize=18)
    # for plotting/saving
    # fig.subplots_adjust(left=0.4)
    plot_path: Path = Path(config["plot_path"])
    plot_path: Path = plot_path / "{}_{}_heatmap.png".format(
        config["config_name"], config["metrics"]
    )
    plt.savefig(plot_path, dpi=500, bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_training",
        type=str,
        help="Filepath to directory called 'training' which contains all outputs from train.py",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Filepath to config.json which contains most of the necessary parameters to create informative barplots",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        help="Input key of config you'd like to visualize. It is specified in barplot_config.json",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        help="Directory path to location of plotting.",
    )
    args = parser.parse_args()
    config = vars(args)
    heatmap(config)
