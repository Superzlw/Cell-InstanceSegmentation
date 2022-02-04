# Introduction

[CenterMask2](https://github.com/youngwanLEE/centermask2) using [Detectron2](https://github.com/facebookresearch/detectron2) library perform well in object detection and instance segmentation, and the paper also demonstrates good results on the Livecell dataset, so CenterMask2 and Detectron2 are used as training frameworks to select appropriate variants and adjust hyperparameters for cell instance segmentation and prediction. 

The code in this branch is based on the kaggle notebook development Environment.

The basic pipeline is to analysis the cell shape, creating coco annotated dataset and splitting it, training and evaluating the model with centermask 1/2, optimizing the results using Weighted Segments Fusion (WSF) and TTA (Test time augmentation), and testing the model.

# Sartorius: cell shape analysis

Post-processing of detections will be key in this competition. To do proper filtering if detections we need a clear picture of the statistics of each of the three cell types. The aim of this notebook is to give some guidelines for post processing features regardless of which model is used. 

# Sartorius: Create COCO annotations

There are many semantic segmentation tools available, and they all require image annotations in one of several specific formats. In this notebook we will create COCO annotations for the Sartoruis dataset.

# Train and Evaluation

The training with evaluation noteboook and the test notebook are divided into two parts, the trained model is used for testing, prediction and inference with a one-to-one version relationship.

The training run time environment for the model is in kaggle, and offline training is required for submitting the results, so offline configuration of Centermask2 and other environments is required. Both online and offline training versions of the code are provided here.

List of data resources needed to read and write at runtime:

* [Sartorius - Cell Instance Segmentation](https://www.kaggle.com/c/sartorius-cell-instance-segmentation) dataset
* Detectron2 offline installed files
* Configs and models from [CenterMask2](https://github.com/youngwanLEE/centermask2) 
* Dataset with coco annotations after splitting

`detectron2_download_code_for_offline_install.ipynb` provides the code to generate the files for offline installation of Detectron2.

`sartorius_create_coco_annotations.ipynb` provides the code to split dataset with coco annotations.

## Training and model-related information
#### offline_train_evaluation_pretrained_cm2liteV39_iter7000to15000.ipynb

* **Pretrained model and config**

  base on Anchor free Architecture using SH-SY5Y dataset.

  Download:  [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/shsy5y_config.yaml), [model](https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/SHSY5Y/LIVECell_anchor_free_shsy5y_model.pth).

* **Training model and configs**

  centermask2-lite-V-39-eSE-FPN-ms-4x

* **Iteration**: 8000. From checkpoint (Iterations==7000) of version 3 to iteration==15000

* **Training run time**: 4 hours and 24 mins

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

#### offline_train_evaluation_pretrained_cm2liteV39_iter7000.ipynb

* **Pretrained model and config**

  base on Anchor free Architecture using SH-SY5Y dataset.

  Download:  [config](https://github.com/sartorius-research/LIVECell/blob/main/model/anchor_free/shsy5y_config.yaml), [model](https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/models/Anchor_free/SHSY5Y/LIVECell_anchor_free_shsy5y_model.pth).

* **Training model and configs**

  centermask2-lite-V-39-eSE-FPN-ms-4x

* **Iteration**: 7000

* **Training run time**:  3 hours and 53 mins

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

#### offline_train_evaluation_cm2liteV39_iter14000to24000.ipynb

* **Pretrained model and config**

  none

* **Training model and configs**

  centermask2-lite-V-39-eSE-FPN-ms-4x

* **Iteration**: 10000. From checkpoint (Iterations==14000) of version 1 to iteration==24000

* **Training run time**:  5 hours and 35 mins

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

#### offline_train_evaluation_cm2liteV39_iter14000.ipynb

* **Pretrained model and config**

  none

* **Training model and configs**

  centermask2-lite-V-39-eSE-FPN-ms-4x

* **Iteration**: 14000. 

* **Training run time**:  8 hours

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

#### offline_train_evaluation_cm1liteV39_iter12000.ipynb

* **Pretrained model and config**

  none

* **Training model and configs**

  centermask_lite_V_39_eSE_FPN_ms_4x

* **Iteration**: 12000. 

* **Training run time**:  6 hours 24 mins

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

#### online_train_evaluation_cm1liteV19_iter14000.ipynb

* **Pretrained model and config**

  none

* **Training model and configs**

  centermask-lite-V-19-eSE-slim-FPN-ms-4x

* **Iteration**: 14000. 

* **Training run time**:  6 hours 38 mins

* **GPU**: Tesla P100-PCIE-16GB (arch=6.0)

# Sartorius: TTA with weighted segments fusion

Most ML learner are familiar with Weighted Boxes Fusion in object detection. What about Weighted Segments Fusion (WSF) for use in instance segmentation? There are papers on the topic, but not much code, here gives a try. There are many possible approaches to this problem; take on it is a two-stage process: Start with WBF on the bounding boxes, and then fuse the segments based on the WBF output. WSF can of course used for ensembling models too, not only TTA(Test time augmentation).

References:

- [Create COCO annotations for Sartorius dataset](https://www.kaggle.com/mistag/sartorius-create-coco-annotations)
- [Cell shape analysis](https://www.kaggle.com/mistag/sartorius-cell-shape-analysis)
- [Trained CenterMask2 model](https://www.kaggle.com/mistag/train-sartorius-detectron2-centermask2)
- [Competition Metric : mAP IoU](https://www.kaggle.com/theoviel/competition-metric-map-iou)
- [Weighted boxes fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) by [ZFTurbo](https://kaggle.com/zfturbo)

List of data resources needed to read and write at runtime:

* [Sartorius - Cell Instance Segmentation](https://www.kaggle.com/c/sartorius-cell-instance-segmentation) dataset
* Detectron2 offline installed files
* Configs and models(check point) from trained model with the same suffix in train_evaluation.
* Dataset with coco annotations after splitting
* Sartorius: Cell shape analysis
* Weighted boxes fusion offline installation

`sartorius_cell_shape_analysis.ipynb` provides the code to create shape_data.pkl file.

`weighted_boxes_fusion_offline_installation.ipynb` provides the code for weighted boxes fusion offline installation.

# Test

Each test here corresponds to a trained model with the same suffix in train_evaluation.

List of data resources needed to read and write at runtime:

* [Sartorius - Cell Instance Segmentation](https://www.kaggle.com/c/sartorius-cell-instance-segmentation) dataset
* Detectron2 offline installed files
* Configs and models(check point) from trained model with the same suffix in train_evaluation.
* Dataset with coco annotations after splitting
* Sartorius: Cell shape analysis
* TTA with Weighted Segments Fusion

`sartorius_tta_with_weighted_segments_fusion.ipynb` provides the code for TTA with Weighted Segments Fusion.
