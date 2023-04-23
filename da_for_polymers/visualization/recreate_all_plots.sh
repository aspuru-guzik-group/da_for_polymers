# BARPLOTS
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name CO2_Soleimani --plot_path ./CO2_Soleimani
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name CO2_Soleimani_augment --plot_path ./CO2_Soleimani
# # python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name OPV_Min --plot_path ./OPV_Min
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name OPV_Min_augment --plot_path ./OPV_Min
# # python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name PV_Wang --plot_path ./PV_Wang
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name PV_Wang_augment --plot_path ./PV_Wang
# # python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name Swelling_Xu --plot_path ./Swelling_Xu
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name Swelling_Xu_augment --plot_path ./Swelling_Xu


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
python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_nn_fingerprint_comparison --plot_path ./dataset_comparisons/
# python barplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_lstm_fingerprint_comparison --plot_path ./dataset_comparisons/

# Data Comparisons
# python barplot_data.py --path ../training/ --config_path ./barplot_config.json --config_name augment_smiles_data_comparison --plot_path ./dataset_comparisons/
# python barplot_data.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_data_comparison --plot_path ./dataset_comparisons/
# python barplot_data.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_fingerprint_data_comparison --plot_path ./dataset_comparisons/

# HEATMAP
# python heatmap.py --path_to_training ../training/ --config_path ./heatmap_config.json --config_name CO2_Soleimani --plot_path ./CO2_Soleimani
# python heatmap.py --path_to_training ../training/ --config_path ./heatmap_config.json --config_name PV_Wang --plot_path ./PV_Wang
# python heatmap.py --path_to_training ../training/ --config_path ./heatmap_config.json --config_name Swelling_Xu --plot_path ./Swelling_Xu
# python heatmap.py --path_to_training ../training/ --config_path ./heatmap_config.json --config_name DFT_Ramprasad --plot_path ./DFT_Ramprasad

# BOX PLOT
# python boxplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_frag_comparison --plot_path ./dataset_comparisons/
# python boxplot.py --path ../training/ --config_path ./barplot_config.json --config_name augment_recombined_nn_fingerprint_comparison --plot_path ./dataset_comparisons/
