# Introduction
This repo serves for the project **Cell Instance Segmentation** within the course Data Science II. The whole pipeline 
consists of data pre-processing, model building, model training, model evaluation, and deployment of candidate models to 
web pages (Tools: 1. Streamlit; 2. Flask & Heroku). In addition, the data analytics is performed as a preliminary step before
the whole pipeline. 

The topic comes from the Kaggle competition, further information can be found here: [kaggle](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)

## Folders
The relevant folders are as follows:

* `analytics`: It contains files related to data analysis (on Live-Cell dataset and Competition dataset) in the format of `.ipynb`

* `app_Flask`: Here are the relevant files for deploying the model to Heroku based on Flask.

    * `app`

        * `static`: This folder is used to save temporary images, including uploaded images, processed images, and images that need to be downloaded.

        * `templates`: Here are the codes of different web pages saved.

* `app_Streamlit`: Here are the relevant files for deploying the model to Streamlit-cloud based on streamlit.

    * `static`: This folder is used to save temporary images, including uploaded images, processed images, and images that need to be downloaded.

    * `utils`: Some basic constant definitions, and helper functions inside the main function

* `checkpoints`: Used to store downloaded model checkpoints (.pth).

* `configs`: Inherited from mmdetection's configs folder, according to mmdetection's model construction principle. The config
file in this folder is used to build a complete machine learning pipeline spanning from data pre-processing to model evaluation. 
The Mask RCNN model used in this project is saved in the `mask_rcnn` folder, the file name is: `mask_rcnn_r50_fpn_compet.py`.

* `dataset`: Download the dataset from the specified [url](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/data) and save it in this folder.

* `experiments`: It contains python scripts for converting the dataset to COCO form as well as building the training, 
validation and testing pipeline. The file form is `.ipynb`, and the platform for using these scripts can be: **Kaggle**
or **colab**.

* `install`: A markdown file is included here, which mainly instructs how to configure the virtual env and install mmdetection.

* `Models`: Model initialization and inference codes for different models are saved. Currently, there is only Mask RCNN.

* `pytorch_pipeline`: The implementation of the pytorch-based machine learning pipeline including baseline model (Mask R-CNN + R50)
is saved here.

* `results`: This is used to save all intermediate files, such as processed datasets, model evaluation files, diagrams etc.

## Python environment configuration and installation of related dependencies
The introduction here is mainly based on Anaconda.

For the part based on pycharm, please refer to: https://github.com/Superzlw/DS2/blob/main/install/INSTALLATION.md

### base dependency
update conda:

> conda update conda

Create a virtual environment with conda and specify the python version (py3.8 is tested):

> conda create -n <virtual env name> python=3.8

Activate the virtual environment:

> conda activate <virtual env name>

Install pytorch(1.6.0), cuda(10.1) and torchvision(0.7.0):

> pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html 

For other versions, please refer to the pytorch official website: https://pytorch.org/

### Install MMDetection
For the detailed installation instruction of MMDetection, please refer to the chapter **Install mmdetection in your virtual 
environment** from the [markdown file](https://github.com/Superzlw/DS2/blob/main/install/INSTALLATION.md). 
### Install Streamlit
1. Install the relevant packages in your virtual environment in the terminal, as follows:

> Command `pip install streamlit`

2. Validate your installation of **streamlit** in your virtual environment in the terminal:

> Command `python`
> 
> Command `import streamlit as st` in your python interpreter

(Some words about Streamlit: it provides many functions for layout, such as `st.sidebar`, `st.columns`, `st.title`, etc. .)

### Install other essential python packages 
Command `pip install -r requirements.txt` in your virtual environment in the terminal

## Usage
After finishing the installation, 
1. Navigate to **app_Streamlit** dir of your project directory **DS2** in the terminal and then:

> Command `streamlit run run.py`

to view the results on web page in your localhost.

2. Then you can navigate to [this link](https://share.streamlit.io/superzlw/ds2/main/app_Streamlit/run.py) to check the
corresponding result on the remote web page apart from the page shown in localhost.


(Note:
If you are still interested in building the web service on your own, you can then upload the current repo of this 
project to your own github account, and then register an account in Streamlit and connect the APP with github.)

## Training, Validation and Testing
If you would like to reproduce the machine learning pipeline to obtain the model by yourself, please refer to the directory
**experiments**, where there are only notebooks. The reason why no python script is written for the pipeline is that
we have no access to local GPUs or GPUs provided by any institute on campus, instead we can only use the open-source
**colab** and **Kaggle** for our training on GPU.
### Usage
1. First use the notebook named as "data_preprocessing_convert_to_coco_colab.ipynb" and run it on colab. The desired result
after running should be the dataset in COCO format, which applies to both **LIVECell** and **Conmpetition** dataset.
2. Then run any notebook with the name beginning with prefix "train" can start the training. To understand what kind of model,
what kind of dataset, and what kind of platform, and others, you are using, please refer to the following naming convention
for the notebooks:
   1. "*train_<model_name>_<dataset(s)>_<training_platform>.ipynb*"
   2. One example, when you would like to train the model "cascade_mask_rcnn_cbv2_swin_tiny" only based on the dataset "LIVECell"
   on the platform "colab", then you can use the notebook "train_cascade_mask_rcnn_cbv2_swin_tiny_only_livecell_colab.ipynb"
   3. Another example, when you would like to train the model "cascade_mask_rcnn_cbv2_swin_small" both based on the dataset "LIVECell"
   and "Competition" on the platform "colab", then you can use the notebook "train_cascade_mask_rcnn_cbv2_swin_small_livecell_compet_colab.ipynb"

## Git Workflow and Review

The collaboration requires some rules, to keep the code clean:

* We are using GitHub's pull requests model to integrate new code into our repository.
* Every pull request is reviewed by one of our maintainers.
* We foster a git history that is clean, easy to read and to review.

To guarantee this we follow the best practices described [here](https://www.git-tower.com/learn/git/ebook/en/command-line/appendix/best-practices#start).

We use a rebase workflow with short living branches for features and bug fixes (see [here](https://www.git-tower.com/learn/git/ebook/en/command-line/advanced-topics/rebase#start)).

One useful link for forked git repo:
https://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/