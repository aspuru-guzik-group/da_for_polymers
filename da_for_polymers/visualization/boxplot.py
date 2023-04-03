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

### master FUNCTION
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


def rename_features(filtered_summary: pd.DataFrame, input_representation: str):
    """Rename features to eliminate differences between datasets.

    Args:
        filtered_summary (pd.DataFrame): _description_
        input_representation (str):
    """
    for index, row in filtered_summary.iterrows():
        if "aug" in filtered_summary.at[index, "Features"]:
            filtered_summary.at[index, "Features"] = "Augmented Manual Fragments"
        else:
            filtered_summary.at[index, "Features"] = "Manual Fragments"

    return filtered_summary


def boxplot(config: dict):
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

    # Get paths of progress.csv
    progress_paths: list[Path] = path_to_result(config, "progress_report")

    # Get summary dataframes
    summary: pd.DataFrame = gather_results(progress_paths)
    print(summary)
    summary: pd.DataFrame = summary.sort_values(config["hue"])
    summary: pd.DataFrame = summary.replace(
        {
            "manual_frag": "Fragments",
            "manual_frag_aug": "Augmented Fragments",
            "manual_frag_str": "Fragment (SMILES)",
            "manual_frag_aug_str": "Augmented Fragment (SMILES)",
            "manual_recombined_aug_SMILES": "Recombined Augmented (SMILES)",
            "manual_recombined_aug_fingerprint": "Recombined Augmented Fingerprints",
            "fingerprint": "Fingerprints",
            "Augmented_SMILES": "Augmented SMILES",
            "CO2_Soleimani": "CO2 Solubility",
            "PV_Wang": "Pervaporation",
            "Swelling_Xu": "Swelling",
        }
    )

    # Plot Axis
    fig, ax = plt.subplots(figsize=(8, 6.5))
    # Title
    # ax.set_title(
    #     "boxplot of {} for {}".format(
    #         config["config_name"], config["models"][0]
    #     )  # config["models"][0])
    # )
    ax.set_xlabel(config["x"][0], fontsize=14)
    ax.set_ylabel(config["metrics"][0], fontsize=14)
    plt.tick_params(labelsize=12)
    # Customization
    # boxplot
    sns.set_style("whitegrid")

    # Font
    sns.set(font_scale=1.1)

    # Color
    colors = ["#41ab5d", "#f16913", "#4292c6"]
    sns.set_palette(sns.color_palette(colors))
    # sns.set_palette(sns.color_palette("husl", 8))

    # Font

    if "data" in config["config_name"]:
        if "smiles" in config["config_name"]:
            ax.set_title(
                "Amount of Data: Before and After Augmentation of SMILES",
                fontsize=16,
            )
        else:
            ax.set_title(
                "Amount of Data: Before and After Augmentation of Fragments",
                fontsize=16,
            )
        sns.boxplot(
            x=summary[config["x"]],
            y=summary[config["metrics"]],
            ax=ax,
            hue=summary[config["hue"]],
            order=["Fragments", "Augmented Fragments", "Augmented SMILES"],
        )
        for container in ax.containers:
            ax.bar_label(container)
    elif (
        "comparison" in config["config_name"] and "recombined" in config["config_name"]
    ):
        ax.set_title(
            "Property Prediction Performance using {} and Recombined Augmented Fingerprints".format(
                config["models"][0]
            ),
            fontsize=12,
        )  # config["models"][0])
        if "fingerprint" in config["config_name"]:
            sns.boxplot(
                x=summary[config["x"]],
                y=summary[config["metrics"]],
                ax=ax,
                hue=summary[config["hue"]],
                order=[
                    "Fingerprints",
                    "Recombined Augmented Fingerprints",
                    # "Augmented Fragments",
                ],
                linewidth=1,
            )
        else:
            sns.boxplot(
                x=summary[config["x"]],
                y=summary[config["metrics"]],
                ax=ax,
                hue=summary[config["hue"]],
                order=[
                    "SMILES",
                    "Augmented_SMILES",
                ],
            )

    elif "comparison" in config["config_name"]:
        ax.set_title(
            "Property Prediction Performance using {} and Augmented Fragments".format(
                config["models"][0]
            ),
            fontsize=15,
        )  # config["models"][0])
        if "frag" in config["config_name"]:
            sns.boxplot(
                x=summary[config["x"]],
                y=summary[config["metrics"]],
                ax=ax,
                hue=summary[config["hue"]],
                order=[
                    "Fragments",
                    "Augmented Fragments",
                    # "Fragment (SMILES)",
                    # "Augmented Fragment (SMILES)",
                ],
                linewidth=1,
            )
        else:
            sns.boxplot(
                x=summary[config["x"]],
                y=summary[config["metrics"]],
                ax=ax,
                hue=summary[config["hue"]],
                order=[
                    "SMILES",
                    "Augmented_SMILES",
                ],
            )
    else:
        ax.set_title(
            "{} : Comparison of Augmentation of Fragments".format(
                config["datasets"][0]
            ),
            fontsize=14,
        )  # config["models"][0])
        sns.boxplot(
            x=summary[config["x"]],
            y=summary[config["metrics"]],
            ax=ax,
            order=["RF", "BRT", "NN", "LSTM"],
            hue=summary[config["hue"]],
            hue_order=[
                "Fragments",
                "Augmented Fragments",
                "Fragment SMILES",
                "Augmented Fragment SMILES",
            ],
        )
        # Y-axis Limits
        min_yval: float = min(summary[config["metrics"]])
        # min_idx_yval: int = np.argmin(summary[config["metrics"]])
        # min_yval: float = min_yval - list(summary["r_std"])[min_idx_yval]
        # min_yval: float = min_yval * 0.9
        ax.set_ylim(min_yval, 1)
    ax.set_ylabel("$R^2$", fontsize=14)

    # wrap_labels(ax)
    # for plotting/saving
    plt.tight_layout()
    plot_path: Path = Path(config["plot_path"])
    plot_path: Path = plot_path / "{}_{}_{}_boxplot.png".format(
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
        help="Filepath to config.json which contains most of the necessary parameters to create informative boxplots",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        help="Input key of config you'd like to visualize. It is specified in boxplot_config.json",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        help="Directory path to location of plotting.",
    )
    args = parser.parse_args()
    config = vars(args)
    boxplot(config)

# python boxplot.py --path ../training/ --config_path ./boxplot_config.json --config_name augment_frag_comparison --plot_path ./dataset_comparisons/
