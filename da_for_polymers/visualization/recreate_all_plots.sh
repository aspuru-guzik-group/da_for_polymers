# BARPLOTS - Representation Comparisons across Datasets
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_smiles_comparison --plot_path ./dataset_comparisons/
python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_smiles_lstm_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_lstm_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_rf_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_brt_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_rf_fingerprint_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_brt_fingerprint_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_svm_fingerprint_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_nn_fingerprint_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_lstm_fingerprint_comparison --plot_path ./dataset_comparisons/

# Data Comparisons
# python barplot_data.py --path ../training/ --config_path ./barplot_config.json --config_name augment_smiles_data_comparison --plot_path ./dataset_comparisons/
# python barplot_data.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_data_comparison --plot_path ./dataset_comparisons/
# python barplot_data.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_fingerprint_data_comparison --plot_path ./dataset_comparisons/

# HEATMAP
# python heatmap.py --path_to_training ../training/ --config_path ./heatmap_config.json --config_name CO2_Soleimani --plot_path ./CO2_Soleimani
# python heatmap.py --path_to_training ../training/ --config_path ./heatmap_config.json --config_name PV_Wang --plot_path ./PV_Wang
# python heatmap.py --path_to_training ../training/ --config_path ./heatmap_config.json --config_name DFT_Ramprasad --plot_path ./DFT_Ramprasad

# BOX PLOT
# python boxplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_comparison --plot_path ./dataset_comparisons/
# python boxplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_nn_fingerprint_comparison --plot_path ./dataset_comparisons/


# Figures from Paper
# Figure 3
# python ./CO2_Soleimani/heatmap_figure3.py --path_to_training ../training/ --config_path ./heatmap_config.json --config_name CO2_Soleimani --plot_path ./CO2_Soleimani
# Figure 4
# python ./PV_Wang/heatmap_figure4.py --path_to_training ../training/ --config_path ./heatmap_config.json --config_name PV_Wang --plot_path ./PV_Wang
# Figure 5

# Figure 6
# python ./dataset_comparisons/barplot_figure6.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_comparison --plot_path ./dataset_comparisons/
# Figure 7
# python ./dataset_comparisons/barplot_figure7.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_nn_fingerprint_comparison --plot_path ./dataset_comparisons/
# Figure S14
# python ./dataset_comparisons/barplot_data_figureS14.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_data_comparison --plot_path ./dataset_comparisons/