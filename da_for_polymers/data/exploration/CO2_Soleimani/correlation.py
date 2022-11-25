from cmath import nan
import math
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
import itertools
from sklearn.metrics import mean_squared_error

CO2_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/CO2_Soleimani/co2_expt_data.csv"
)

CORRELATION_PLOT = pkg_resources.resource_filename(
    "da_for_polymers", "data/exploration/CO2_Soleimani/co2_correlation_plot.png"
)

CORRELATION_HEATMAP_PLOT = pkg_resources.resource_filename(
    "da_for_polymers", "data/exploration/CO2_Soleimani/co2_correlation_heatmap.png"
)

RMSE_HEATMAP_PLOT = pkg_resources.resource_filename(
    "da_for_polymers", "data/exploration/CO2_Soleimani/co2_rmse_heatmap.png"
)


class Correlation:
    """
    Class that contains all functions for creating correlations between each variable.
    """

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def parity_plot(self, columns_list_idx):
        """
        Function that plots the parity plots between each variable.
        NOTE: you must know the variable names beforehand

        Args:
            columns_list_idx: select which columns you want to plot in the parity plot

        Returns:
            Parity plots of each relationship.
            Layout will be:
            Var 1|X     PLOT  PLOT
            Var 2|PLOT  X     PLOT
            Var 3|PLOT  PLOT  X
                 |Var 1|Var 2|Var 3|
        """
        columns = self.data.columns
        columns_dict = {}
        index = 0
        while index < len(columns):
            columns_dict[columns[index]] = index
            index += 1

        print(columns_dict)
        # select which columns you want to plot
        x_columns = len(columns_list_idx)
        y_rows = len(columns_list_idx)

        # create column index dictionary (for subplot referencing)
        column_idx_dict = {}
        index = 0
        for i in columns_list_idx:
            column_idx_dict[i] = index
            index += 1

        permutations = list(itertools.permutations(columns_list_idx, 2))

        fig, axs = plt.subplots(
            y_rows,
            x_columns,
            figsize=(len(columns_list_idx) * 8, len(columns_list_idx) * 8),
        )

        for pair in permutations:
            column_idx_0 = pair[0]
            column_idx_1 = pair[1]
            column_name_0 = columns[column_idx_0]
            column_name_1 = columns[column_idx_1]
            column_0 = self.data[column_name_0]
            column_1 = self.data[column_name_1]
            # handle unequal number of data points
            # mask values with True or False if nan
            isna_column_0 = column_0.isna().tolist()
            isna_column_1 = column_1.isna().tolist()
            filtered_column_0 = []
            filtered_column_1 = []
            index = 0
            while index < len(column_0):
                if not isna_column_0[index] and not isna_column_1[index]:
                    filtered_column_0.append(column_0[index])
                    filtered_column_1.append(column_1[index])
                index += 1

            # subplot
            x_axis_idx = column_idx_dict[column_idx_0]
            y_axis_idx = column_idx_dict[column_idx_1]
            axs[x_axis_idx, y_axis_idx].scatter(
                filtered_column_1, filtered_column_0, s=(20 / len(columns_list_idx))
            )

            # set xlabel and ylabel
            axs[x_axis_idx, y_axis_idx].set_xlabel(column_name_1)
            axs[x_axis_idx, y_axis_idx].set_ylabel(column_name_0)

            # handle different data types (str, float, int)
            if not isinstance(filtered_column_0[0], str) and not isinstance(
                filtered_column_1[0], str
            ):
                # find slope and y-int of linear line of best fit
                m, b = np.polyfit(
                    filtered_column_1,
                    filtered_column_0,
                    1,
                )
                # find correlation coefficient
                corr_coef = np.corrcoef(
                    filtered_column_1,
                    filtered_column_0,
                )[0, 1]
                # find rmse
                rmse = np.sqrt(
                    mean_squared_error(
                        filtered_column_1,
                        filtered_column_0,
                    )
                )
                axs[x_axis_idx, y_axis_idx].plot(
                    np.array(filtered_column_1),
                    m * np.array(filtered_column_1) + b,
                    color="black",
                )
                textstr = (
                    "R: "
                    + str(round(corr_coef, 3))
                    + "  "
                    + "RMSE: "
                    + str(round(rmse, 3))
                )
                anchored_text = AnchoredText(textstr, loc="lower right")
                axs[x_axis_idx, y_axis_idx].add_artist(anchored_text)

        plt.savefig(CORRELATION_PLOT)

    def heatmap_plot(self, columns_list_idx, option):
        """
        Function that plots the parity plots between each variable.
        NOTE: you must know the variable names beforehand

        Args:
            columns_list_idx: select which columns you want to plot in the parity plot

        Returns:
            Parity plots of each relationship.
            Layout will be:
            Var 1|X     PLOT  PLOT
            Var 2|PLOT  X     PLOT
            Var 3|PLOT  PLOT  X
                 |Var 1|Var 2|Var 3|
        """
        columns = self.data.columns
        columns_dict = {}
        index = 0
        while index < len(columns):
            columns_dict[columns[index]] = index
            index += 1

        print(columns_dict)
        # select which columns you want to plot
        x_columns = len(columns_list_idx)
        y_rows = len(columns_list_idx)

        # create column index dictionary (for subplot referencing)
        column_idx_dict = {}
        index = 0
        for i in columns_list_idx:
            column_idx_dict[i] = index
            index += 1

        column_name_list: list = list(columns[columns_list_idx])

        permutations = list(itertools.permutations(columns_list_idx, 2))

        fig, ax = plt.subplots(figsize=(8, 8))

        if option == "r":
            heatmap_array = np.ones((y_rows, x_columns))
        elif option == "rmse":
            heatmap_array = np.zeros((y_rows, x_columns))

        print(heatmap_array.shape)

        for pair in permutations:
            column_idx_0 = pair[0]
            column_idx_1 = pair[1]
            print(f"{column_idx_0=}", f"{column_idx_1=}")
            column_name_0 = columns[column_idx_0]
            column_name_1 = columns[column_idx_1]
            column_0 = self.data[column_name_0]
            column_1 = self.data[column_name_1]
            # handle unequal number of data points
            # mask values with True or False if nan
            isna_column_0 = column_0.isna().tolist()
            isna_column_1 = column_1.isna().tolist()
            filtered_column_0 = []
            filtered_column_1 = []
            index = 0
            while index < len(column_0):
                if not isna_column_0[index] and not isna_column_1[index]:
                    filtered_column_0.append(column_0[index])
                    filtered_column_1.append(column_1[index])
                index += 1
            print(filtered_column_0, filtered_column_1)

            # handle different data types (str, float)
            if isinstance(filtered_column_0[0], float) and isinstance(
                filtered_column_1[0], float
            ):
                if option == "r":
                    # find correlation coefficient
                    result = np.corrcoef(
                        filtered_column_1,
                        filtered_column_0,
                    )[0, 1]
                elif option == "rmse":
                    # find rmse
                    result = np.sqrt(
                        mean_squared_error(
                            filtered_column_1,
                            filtered_column_0,
                        )
                    )
                heatmap_array[column_idx_0 - 1][column_idx_1 - 1] = round(result, 4)
            else:
                heatmap_array[column_idx_0 - 1][column_idx_1 - 1] = nan

        print(heatmap_array)

        custom_colormap = sns.color_palette("icefire", as_cmap=True)
        custom_colormap.set_bad("grey")

        sns.heatmap(
            heatmap_array,
            annot=True,
            cmap=custom_colormap,
            fmt=".4f",
            cbar_kws={"label": "Pearson Correlation Coefficient (r)"},
            vmin=-1.0,
            vmax=1.0,
            mask=heatmap_array == nan,
        )

        # Show all ticks and label them with the respective list entries
        print(column_name_list)
        ax.set_xticklabels(column_name_list, rotation=90)
        ax.set_yticklabels(column_name_list, rotation=0)

        if option == "r":
            ax.set_title("Heatmap of Correlation Coefficient")
            fig.tight_layout()
            plt.savefig(CORRELATION_HEATMAP_PLOT)

        elif option == "rmse":
            ax.set_title("Heatmap of RMSE")
            fig.tight_layout()
            plt.savefig(RMSE_HEATMAP_PLOT)


corr_plot = Correlation(CO2_DATA)
# corr_plot.parity_plot([1, 2, 3, 4])
corr_plot.heatmap_plot([1, 2, 3], "r")
