# Infrastructure for training
GPU: Tesla P100-PCIE-16GB (arch=6.0)

# Configuration of the data folder structure
## Kaggle
Before training the model on the kaggle platform, we need to prepare the relevant dataset folder according to the code.

First, download the compressed package required for training at the following link: https://drive.google.com/file/d/1EOCecaCxrXNNt1HYhNvNw1vvJu7jSWq2/view?usp=sharing

Create the following folder in your kaggle account and import it to the notebook before training.

* `apexforcellis`: Here is the `apex.zip` in the compressed package.

* `cellis`: Here is the `cellis.zip` in the compressed package.

* `cellisckp`: Here is the checkpoint file `epoch_10.pth` in the compressed package, the `epoch_10.pth` here is the checkpoint for the training of mask RCNN based on competition dataset.

* `competconfig`: Here is the config file named `mask_rcnn_r50_fpn_compet.py` in the compressed package, which is for the training of mask RCNN based on competition dataset.

* `competdata`: Here is the `competdata.zip` in the compressed package.

* `competdataplus`: Here is the `competdataplus.zip` in the compressed package. 