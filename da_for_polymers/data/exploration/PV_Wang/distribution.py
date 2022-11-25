import ast
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

PV_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/PV_Wang/pv_exptresults.csv"
)

DISTRIBUTION_PLOT = pkg_resources.resource_filename(
    "da_for_polymers", "data/exploration/PV_Wang/pv_distribution_plot.png"
)

PV_AUG_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/PV_Wang/manual_frag/master_manual_frag.csv"
)


AUGMENTED_DISTRIBUTION_PLOT = pkg_resources.resource_filename(
    "da_for_polymers", "data/exploration/PV_Wang/pv_augmented_polymer_distribution_plot.png"
)

AUGMENTED_OUTPUT_DISTRIBUTION_PLOT = pkg_resources.resource_filename(
    "da_for_polymers", "data/exploration/PV_Wang/pv_augmented_polymer_output_distribution_plot.png"
) 

class Distribution:
    """
    Class that contains functions to determine the distribution of each variable in the dataset.
    Each dataset will have slightly different variable names.
    Must be able to handle numerical and categorical variables.
    """

    def __init__(self, data):
        self.data = pd.read_csv(data)

    def histogram(self, columns_list_idx):
        """
        Function that plots the histogram of all variables in the dataset
        NOTE: you must know the variable names beforehand

        Args:
            columns_list_idx: select which columns you want to plot in the histogram

        Returns:
            Histogram plots of all the variables.
        """
        columns = self.data.columns
        columns_dict = {}
        index = 0
        while index < len(columns):
            columns_dict[columns[index]] = index
            index += 1

        print(columns_dict)

        # prepares the correct number of (x,y) subplots
        num_columns = len(columns_list_idx)
        x_columns = round(np.sqrt(num_columns))
        if x_columns == np.sqrt(num_columns):
            y_rows = x_columns
        elif x_columns == np.floor(np.sqrt(num_columns)):
            y_rows = x_columns + 1
        elif x_columns == np.ceil(np.sqrt(num_columns)):
            y_rows = x_columns
        print(x_columns, y_rows)

        if x_columns == 1:
            fig, ax = plt.subplots(x_columns, figsize=(y_rows * 4, x_columns * 3))
            fig.tight_layout()
            current_column = columns[columns_list_idx[0]]
            current_test_list = self.data[current_column].tolist()
            current_test_list = [
                item for item in current_test_list if not (pd.isnull(item)) == True
            ]
            ax.set_title(current_column)
            if isinstance(current_test_list[0], str):
                n, bins, patches = ax.hist(current_test_list, bins="auto")
            elif isinstance(current_test_list[0], float):
                n, bins, patches = ax.hist(current_test_list, bins=30)
            start = 0
            end = n.max()
            stepsize = end / 5
            y_ticks = list(np.arange(start, end, stepsize))
            y_ticks.append(end)
            ax.yaxis.set_ticks(y_ticks)
            total = "Total: " + str(len(current_test_list))
            anchored_text = AnchoredText(total, loc="upper right")
            ax.add_artist(anchored_text)
            ax.set_xlabel(current_column)
        else:
            fig, axs = plt.subplots(
                y_rows, x_columns, figsize=(y_rows * 3, x_columns * 4)
            )

            x_idx = 0
            y_idx = 0
            for i in columns_list_idx:
                current_column = columns[i]
                current_test_list = self.data[current_column].tolist()
                current_test_list = [
                    item for item in current_test_list if not (pd.isnull(item)) == True
                ]
                axs[y_idx, x_idx].set_title(current_column)
                if isinstance(current_test_list[0], str):
                    n, bins, patches = axs[y_idx, x_idx].hist(
                        current_test_list, bins="auto"
                    )
                elif isinstance(current_test_list[0], float):
                    n, bins, patches = axs[y_idx, x_idx].hist(
                        current_test_list, bins=30
                    )
                else:
                    n, bins, patches = axs[y_idx, x_idx].hist(
                        current_test_list, bins=30
                    )
                start = 0
                end = n.max()
                stepsize = end / 5
                y_ticks = list(np.arange(start, end, stepsize))
                y_ticks.append(end)
                axs[y_idx, x_idx].yaxis.set_ticks(y_ticks)
                total = "Total: " + str(len(current_test_list))
                anchored_text = AnchoredText(total, loc="upper right")
                axs[y_idx, x_idx].add_artist(anchored_text)
                if isinstance(current_test_list[0], str):
                    axs[y_idx, x_idx].tick_params(axis="x", labelrotation=90)
                    axs[y_idx, x_idx].tick_params(axis="x", labelsize=6)
                y_idx += 1
                if y_idx == y_rows:
                    y_idx = 0
                    x_idx += 1

        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.3  # the amount of width reserved for blank space between subplots
        hspace = 0.6  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        plt.savefig(DISTRIBUTION_PLOT)

    def gather_augmented_data(self, manual_frag_path: str):
        """
        Function that creates a new dataframe with the count of augmented polymers in the dataset.
        Args:
            manual_frag_path: path to data with augmented data (augmented fragments, and recombined augmented fragments)

        Returns:
            augmented_distribution_df: dataframe with distribution of augmented polymers
        """
        manual: pd.DataFrame = pd.read_csv(manual_frag_path)
        augment: dict = {'Polymer': [], 'num_of_original': [], 'num_of_augmented': [], 'num_of_recombined_augmented': []}

        # iterate through each polymer
        for polymer in manual["Polymer"].unique():
        # iterate through dataframe each time
            original: int = 0
            augmented: int = 0
            recombined: int = 0
            for index, row in manual.iterrows():
                if manual.at[index, "Polymer"] == polymer:
                    original += 1
                    augmented_polymers: list = ast.literal_eval(manual.at[index, "PS_manual_aug"])
                    augmented += len(augmented_polymers)
                    recombined_polymers: list = ast.literal_eval(manual.at[index, "PS_manual_recombined_aug_SMILES"])
                    recombined += len(recombined_polymers)
            augment["Polymer"].append(polymer)
            augment["num_of_original"].append(original)
            augment["num_of_augmented"].append(augmented)
            augment["num_of_recombined_augmented"].append(recombined)

        return augment

    def plot_distribution_of_augmented(self, augment: dict):
        """
        Function that plots the distribution of augmented data by polymer.
        Args:
            augment: dict with keys of [Polymer, num_of_original, num_of_augmented, num_of_recombined_augmented]

        Returns:
            Distribution plot!
        """
        fig, ax = plt.subplots(figsize=(12,5))
        _X = np.arange(len(augment["Polymer"]))
        plt.bar(_X - 0.2, height=augment["num_of_original"], label = "Original", width=0.2)
        plt.bar(_X, height=augment["num_of_augmented"], label= "Augmented", width=0.2)
        plt.bar(_X + 0.2, height=augment["num_of_recombined_augmented"], label= "Recombined Augmented", width=0.2)
        plt.xticks(_X, augment["Polymer"])
        plt.legend(loc="upper right")
        plt.xlabel("Polymer")
        plt.ylabel("Number of Datapoints")
        plt.title("Distribution of Datapoints by Polymers after Data Augmentation")
        plt.tight_layout()
        plt.savefig(AUGMENTED_DISTRIBUTION_PLOT)
    
    def gather_augment_output(self, manual_frag_path: str):
        """Function that creates a new dataframe with the count of augmented polymers in the dataset.
        Args:
            manual_frag_path: path to data with augmented data (augmented fragments, and recombined augmented fragments)

        Returns:
            augmented_distribution_df: dataframe with distribution of augmented polymers
        """
        manual: pd.DataFrame = pd.read_csv(manual_frag_path)
        original: dict = {'Polymer': [], 'J_Total_flux_kg_m_2h_1': []}
        augment: dict = {'Polymer': [], 'J_Total_flux_kg_m_2h_1': []}
        recombined: dict = {'Polymer': [], 'J_Total_flux_kg_m_2h_1': []}
        for index, row in manual.iterrows():
            original["Polymer"].append(manual.at[index, "Polymer"])
            original["J_Total_flux_kg_m_2h_1"].append(manual.at[index, "J_Total_flux_kg_m_2h_1"])
            augment_data: list = ast.literal_eval(manual.at[index, "PS_manual_aug"])
            for d in augment_data:
                augment["Polymer"].append(manual.at[index, "Polymer"])
                augment["J_Total_flux_kg_m_2h_1"].append(manual.at[index, "J_Total_flux_kg_m_2h_1"])
            recombined_data: list = ast.literal_eval(manual.at[index, "PS_manual_recombined_aug_SMILES"])
            for r in recombined_data:
                recombined["Polymer"].append(manual.at[index, "Polymer"])
                recombined["J_Total_flux_kg_m_2h_1"].append(manual.at[index, "J_Total_flux_kg_m_2h_1"])
        
        return original, augment, recombined

    def plot_distribution_of_augmented_outputs(self, original: dict, augment: dict, recombined: dict):
        """
        Function that plots the distribution of augmented data by polymer via shaded histogram.
        Args:
            

        Returns:
            Shaded histogram distribution plot!
        """
        fig, ax = plt.subplots(figsize=(10,5))
        # output_dict: dict = {"original": original["J_Total_flux_kg_m_2h_1"], "augmented": augment["J_Total_flux_kg_m_2h_1"], "recombined_augmented": recombined["J_Total_flux_kg_m_2h_1"]}
        # output: pd.DataFrame = pd.DataFrame.from_dict(output_dict)
        # print(output)
        plt.hist(augment["J_Total_flux_kg_m_2h_1"], bins=60, label="Augmented", alpha=0.4, color="tab:orange")
        plt.hist(recombined["J_Total_flux_kg_m_2h_1"], bins=60, label="Recombined Augmented", alpha=0.4, color="tab:green")
        plt.hist(original["J_Total_flux_kg_m_2h_1"], bins=60, label="Original", alpha=0.6, color="tab:blue")
        handles, labels =  plt.gca().get_legend_handles_labels()
        order_handles = [2, 0, 1]
        order_labels = [2, 0, 1]
        plt.legend([handles[idx] for idx in order_handles],[labels[idx] for idx in order_labels], loc="upper right")
        plt.xlabel("J - Total Flux (kg m^(-2) h^(-1))")
        plt.ylabel("Number of Datapoints")
        plt.title("Distribution of Property of Interest (Total Flux) after Data Augmentation")
        plt.tight_layout()
        plt.savefig(AUGMENTED_OUTPUT_DISTRIBUTION_PLOT)
    
    def count_polymer(self, polymers_to_count: list) -> float:
        """
        Function that counts number of polymer in dataset out of all of them
        """
        polymer_data: pd.Series = self.data["Polymer"]
        poly_count: int = 0
        total_count: int = 0
        for polymer in polymer_data:
            if polymer in polymers_to_count:
                poly_count += 1
            total_count += 1
        
        print(poly_count / total_count)

dist = Distribution(PV_DATA)

# J, alpha are dependent variable
# dist.histogram([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# augment: dict = dist.gather_augmented_data(PV_AUG_DATA)
# dist.plot_distribution_of_augmented(augment)


# original, augment, recombined = dist.gather_augment_output(PV_AUG_DATA)
# dist.plot_distribution_of_augmented_outputs(original, augment, recombined)

dist.count_polymer(["pva", "alginate", "chitosan", "pdms"])