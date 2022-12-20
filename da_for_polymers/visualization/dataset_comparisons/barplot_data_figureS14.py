import argparse
from ast import Str
from copy import deepcopy
from email import generator
import json
from pathlib import Path
from typing import Iterable
from matplotlib.container import BarContainer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import textwrap

from da_for_polymers.visualization.path_utils import (
    gather_results,
    path_to_result,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

### MAIN FUNCTION
def wrap_labels(ax, width: int = 10):
    print(ax)
    labels: list = []
    for label in ax.get_xticklabels():
        text: str = label.get_text()
        text_split: list = text.split(",")
        wrapped_text_split: list = []
        idx = 0
        for text in text_split:
            if len(text) > width:
                text: str = textwrap.fill(text, width)
            wrapped_text_split.append(text)
            if idx < len(text_split):
                wrapped_text_split.append("\n")
            idx += 1
        final_text: str = "".join(wrapped_text_split)
        labels.append(final_text)
    ax.set_xticklabels(labels, rotation=30)


# def rename_features(filtered_summary: pd.DataFrame, input_representation: str):
#     """Rename features to eliminate differences between datasets.

#     Args:
#         filtered_summary (pd.DataFrame): _description_
#         input_representation (str):
#     """
#     for index, row in filtered_summary.iterrows():
#         if (
#             "manual_frag_aug" in filtered_summary.at[index, "Features"]
#             or "manual_frag_aug_str" in filtered_summary.at[index, "Features"]
#         ):
#             filtered_summary.at[index, "Features"] = "Augmented Manual Fragments"
#         else:
#             filtered_summary.at[index, "Features"] = "Manual Fragments"

#     return filtered_summary


def barplot(config: dict):
    """Creates a bar plot of the model performance from several configurations.
    Args:
        config: outlines the parameters to select for the appropriate       configurations for comparison
    Returns:
        bar_plot: saves a bar plot comparison of all the configurations in the current working directory.
    """
    with open(config["config_path"]) as f:
        plot_config: dict = json.load(f)
    plot_config: dict = plot_config[config["config_name"]]

    # Combine 2 dictionaries together
    config.update(plot_config)
    print(config)

    progress_paths: list[Path] = path_to_result(config, "summary")

    summary: pd.DataFrame = gather_results(progress_paths)
    # TODO: Divide augmented num_of_data by original num_of_data
    # summary["factor"] = 0
    # for idx, row in summary.iterrows():
    #     if summary.at[idx, "Features"] == "manual_frag_aug":
    #         for idx2, row in summary.iterrows():
    #             if (
    #                 summary.at[idx2, "Features"] == "manual_frag"
    #                 and summary.at[idx2, "Dataset"] == summary.at[idx, "Dataset"]
    #             ):
    #                 num_of_original = summary.at[idx2, "num_of_data"]
    #         summary.at[idx, "factor"] = summary.at[idx, "num_of_data"] / num_of_original
    #     if summary.at[idx, "Features"] == "manual_frag_aug_str":
    #         for idx2, row in summary.iterrows():
    #             if (
    #                 summary.at[idx2, "Features"] == "manual_frag_str"
    #                 and summary.at[idx2, "Dataset"] == summary.at[idx, "Dataset"]
    #             ):
    #                 num_of_original = summary.at[idx2, "num_of_data"]
    #         summary.at[idx, "factor"] = summary.at[idx, "num_of_data"] / num_of_original
    #     if summary.at[idx, "Features"] == "manual_recombined_aug_fingerprint":
    #         for idx2, row in summary.iterrows():
    #             if (
    #                 summary.at[idx2, "Features"] == "fingerprint"
    #                 and summary.at[idx2, "Dataset"] == summary.at[idx, "Dataset"]
    #             ):
    #                 num_of_original = summary.at[idx2, "num_of_data"]
    #         summary.at[idx, "factor"] = summary.at[idx, "num_of_data"] / num_of_original
    summary: pd.DataFrame = summary.sort_values(config["hue"])
    summary: pd.DataFrame = summary.replace(
        {
            "manual_frag": "Fragments",
            "manual_frag_aug": "Augmented Fragments",
            "manual_frag_str": "Fragment (SMILES)",
            "manual_frag_aug_str": "Augmented Fragment (SMILES)",
            "manual_recombined_aug_SMILES": "Augmented Recombined (SMILES)",
            "manual_recombined_aug_fingerprint": "Recombined \n Augmented Fingerprints",
            "fingerprint": "Fingerprints",
            "Augmented_SMILES": "Augmented SMILES",
            "SMILES": "Non-Augmented Data",
        }
    )

    # Plot Axis
    fig, ax = plt.subplots(figsize=(9, 6))
    # Title
    # ax.set_title(
    #     "Barplot of {} for {}".format(
    #         config["config_name"], config["models"][0]
    #     )  # config["models"][0])
    # )
    ax.set_xlabel(config["x"][0], fontsize=14)
    plt.tick_params(labelsize=10)
    # Customization
    # Barplot
    sns.set_style("whitegrid")

    # Font
    sns.set(font_scale=1)

    # Color
    colors = ["#41ab5d", "#f16913", "#4292c6"]
    sns.set_palette(sns.color_palette(colors))

    # Font

    if "data" in config["config_name"]:
        if "smiles" in config["config_name"]:
            ax.set_title(
                "Amount of Data: Augmentation of SMILES",
                fontsize=16,
            )
            sns.barplot(
                x=summary[config["x"]],
                y=summary[config["metrics"]],
                order=["SMILES", "Augmented_SMILES"],
                ax=ax,
                hue=summary[config["hue"]],
            )
            for container in ax.containers:
                ax.bar_label(container)
        elif "recombined" in config["config_name"]:
            ax.set_title(
                "Amount of Data: Augmented and Recombined Fragments",
                fontsize=16,
            )
            sns.barplot(
                x=summary[config["x"]],
                y=summary[config["metrics"]],
                ax=ax,
                hue=summary[config["hue"]],
            )
            for container in ax.containers:
                ax.bar_label(container)

        else:
            ax.set_title(
                "Amount of Data: Before and After Data Augmentation",
                fontsize=16,
            )
            sns.barplot(
                x=summary[config["x"]],
                y=summary[config["metrics"]],
                order=[
                    "Non-Augmented Data",
                    "Augmented SMILES",
                    "Augmented Fragments",
                    "Recombined \n Augmented Fingerprints",
                ],
                ax=ax,
                hue=summary[config["hue"]],
            )
            for container in ax.containers:
                ax.bar_label(container)

    # Y-axis Limits
    max_yval: float = max(summary[config["metrics"]]) * 1.1
    # min_idx_yval: int = np.argmin(summary[config["metrics"]])
    # min_yval: float = min_yval - list(summary["r_std"])[min_idx_yval]
    # min_yval: float = min_yval * 0.9
    ax.set_ylim(0, max_yval)
    ax.set_ylabel("Number of Datapoints", fontsize=14)

    # wrap_labels(ax)
    # for plotting/saving
    plt.tight_layout()
    plot_path: Path = Path(config["plot_path"])
    plot_path: Path = plot_path / "{}_{}_{}_barplot.png".format(
        config["config_name"], config["metrics"], config["models"]
    )
    plt.savefig(plot_path, dpi=500)


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
    barplot(config)

# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_comparison --plot_path ./dataset_comparisons/
