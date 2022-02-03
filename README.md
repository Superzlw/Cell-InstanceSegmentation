# Introduction
This repo contains everything from the Data Science2 project, i.e. Cell Instance Segmentation, including data processing, model building, model training, result evaluation, and different methods for deploying models to web pages (Streamlit, Flask & Heroku), This topic comes from the Kaggle competition, further information can be found here: [kaggle](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)

## Folders
In addition to the relevant folders mentioned above, referenced papers and Canvas files are also included here.

* `analytics`: It contains files related to data analysis (on Live-Cell dataset and Competition dataset) in the format of `.ipynb`

* `app_Flask`: Here are the relevant files for deploying the model to Heroku based on Flask.

	* 'app'

		* 'static': This folder is used to save temporary images, including uploaded images, processed images, and images that need to be downloaded.

		* `templates`: Here are the codes of different web pages saved.

* `app_Streamlit`: Here are the relevant files for deploying the model to Streamlit-cloud based on streamlit.

	* `static`: This folder is used to save temporary images, including uploaded images, processed images, and images that need to be downloaded.

	* `utils`: Some basic constant definitions, and helper functions inside the main function

* `checkpoints`: Used to store downloaded model files.

* `config`: Inherited from mmdetection's configs folder, according to mmdetection's model construction principle, rely on this folder to build a complete model.The Mask RCNN model used in this project is saved in the mask_rcnn folder, the file name is: mask_rcnn_r50_fpn_compet.py.

* `dataset`: Download the dataset from the specified [url](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/data) and save it in this folder.

* `experiment`: The relevant code for converting the dataset to COCO form is saved here. The file form is `.ipynb`, and the platform is: Kaggle.

* `install`: A `.md` file is included here, which details how to install mmdetection.

* `Models`: Model initialization and inference codes for different models are saved. Currently, there is only Mask RCNN.

* `pytorch_pipeline`: The relevant code of the pytorch-based baseline model is saved here, which is a complete Mask RCNN project.

* `results`: This is used to save all intermediate files, such as processed datasets, model evaluation files, etc.\\

## Python environment configuration and installation of related dependencies
The introduction here is mainly based on Anaconda.

For the part based on pycharm, please refer to: https://github.com/Superzlw/DS2/blob/main/install/INSTALLATION.md

### base dependency
update conda:

> conda update conda

Create a virtual environment with conda and specify the python version:

> conda create -n venv python=3.8

Activate the virtual environment:

> conda activate venv

Install pytorch(1.6.0), cuda(10.1) and torchvision(0.7.0):

> pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html 

For other versions, please refer to the pytorch official website: https://pytorch.org/

### Install MMDetection
For the detailed installation process of MMDetection, please refer to: https://github.com/Superzlw/DS2/blob/main/install/INSTALLATION.md
### Install and use Streamlit
For the detailed of this part, please refer to:
https://github.com/Superzlw/DS2/blob/main/app_Streamlit/README.md

## Git Workflow and Review

The collaboration requires some rules, to keep the code clean:

* We are using GitHub's pull requests model to integrate new code into our repository.
* Every pull request is reviewed by one of our maintainers.
* We foster a git history that is clean, easy to read and to review.

To guarantee this we follow the best practices described [here](https://www.git-tower.com/learn/git/ebook/en/command-line/appendix/best-practices#start).

We use a rebase workflow with short living branches for features and bug fixes (see [here](https://www.git-tower.com/learn/git/ebook/en/command-line/advanced-topics/rebase#start)).

One useful link for forked git repo:
https://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/