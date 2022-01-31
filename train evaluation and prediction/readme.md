# Train, Evaluation and Prediction

The training .ipynb and the final prediction .ipynb are divided into two files, and the trained model is used for prediction and inference with a one-to-one version relationship.

The training run time environment for the model is in kaggle, and offline training is required for submitting the results, so offline configuration of Centermask2 and other environments is required. Both online and offline training versions of the code are provided here.

detectron2-download-code-for-offline-install-ii.ipynb provides the code to generate the files for offline installation of Detectron2.

For details(relevant data and code) on training, evaluation and prediction see the .ipynb files.

## Offline training and prediction

### Version 4

* **Pretrained model and config**

  base on Anchor free Architecture using SH-SY5Y dataset.

  Download:  [config][https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/shsy5y_config.yaml], [model][https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/SHSY5Y/LIVECell_anchor_free_shsy5y_model.pth].

* **Training model and configs**

  centermask2-lite-V-39-eSE-FPN-ms-4x

* **Iteration**: 15000

* **Training run time**: 4 hours and 24 mins

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

### Version 3

* **Pretrained model and config**

  base on Anchor free Architecture using SH-SY5Y dataset.

  Download:  [config][https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/shsy5y_config.yaml], [model][https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/SHSY5Y/LIVECell_anchor_free_shsy5y_model.pth].

* **Training model and configs**

  centermask2-lite-V-39-eSE-FPN-ms-4x

* **Iteration**: 7000

* **Training run time**:  3 hours and 53 mins

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

### Version 2

* **Pretrained model and config**

  none

* **Training model and configs**

  centermask2-lite-V-39-eSE-FPN-ms-4x

* **Iteration**: 10000. From checkpoint (Iterations==14000) of version 1 to iteration==24000

* **Training run time**:  5 hours and 35 mins

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

### Version 1

* **Pretrained model and config**

  none

* **Training model and configs**

  centermask2-lite-V-39-eSE-FPN-ms-4x

* **Iteration**: 14000. 

* **Training run time**:  8 hours

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

### Version 0

* **Pretrained model and config**

  none

* **Training model and configs**

  centermask_lite_V_39_eSE_FPN_ms_4x

* **Iteration**: 12000. 

* **Training run time**:  6 hours 24 mins

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

## Online training

* **Pretrained model and config**

  none

* **Training model and configs**

  centermask-lite-V-19-eSE-slim-FPN-ms-4x

* **Iteration: **14000. 

* **Training run time:** 6 hours 38 mins

* **GPU: **Tesla P100-PCIE-16GB (arch=6.0)
