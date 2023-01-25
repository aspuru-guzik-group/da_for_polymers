# ML Models

To run the models you will need to choose `pytorch or sklearn` and one of the `Datasets`. Please use the `*_train_run_auto.sh` to run locally and comment or uncomment specific lines to run the desired model-representation pair.

In the first line of the file, you will find the available models. <br />
For sklearn: `model_types=('RF' 'BRT' 'SVM')`  <br />
For pytorch: `model_types=('NN' 'LSTM')`