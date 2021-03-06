# Configure project dependencies
## Configure virtual environment: using *venv*
When using venv (standard python lib) for setting up virtual environment (**preferred** in terms of Python 3.3 or newer),
you can perform the following to setup the virtual env:
> `py -3 -m venv venv`

Reference website: 
1. https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
2. https://flask.palletsprojects.com/en/1.1.x/installation/#installation
### Special case: Using PyCharm as IDE
When using PyCharm, there is another convenient way, that is, to navigate to the **Settings > Project:[project name] > Python Interpreter** 
to build the virtual environment.

### FAQs:
> Under the situation that the error in terms of loading the file with the .\env\Scripts\Activate.ps1 rises when 
> switching to the virtual environment --> Reference website: https://blog.csdn.net/qq_41574947/article/details/106939020

## Configure virtual environment: *using virtualenv*
When using virtualenv (older variant) --> 
IF python is not installed on current OS, issue the following commands in terminal
1. `sudo apt-get install python3-pip python3.8` (assume that you want to install python3.8)
2. `ls /usr/bin/python*` (check if python3.8 is successfully installed)

OTHERWISE (by checking it with `python -V`), ignore the commands above and issue the following commands

1. `pip3 install virtualenv`
2. `python -m virtualenv -p python3.8 .env` (if you've installed python3.8; if operated in Linux, python3 may be necessary.)
3. `source .env/bin/activate` (on Linux or Mac) or `.env\Scripts\activate` (on Win)
4. `deactivate` (when quitting)

### Install packages in virtual environment
Reference website: https://pip.pypa.io/en/stable/user_guide/#requirements-files

## Install mmdetection in your virtual environment
Official instruction: https://mmdetection.readthedocs.io/en/latest/get_started.html#install-mmdetection.

The general procedure (not all-inclusive) summarized from the two instructions above is:
> Install CUDA and cuDNN (**not necessary if you plan to deploy the candidate model in web service**)

> Install pytorch corresponding to CUDA's version (**not necessary if you plan to deploy the candidate model in web service**)

> Check the right version of wheels of mmcv-full given your cuda version and pytporch version using the following 
> link for example: https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html, when your venv is equipped with
> cuda 10.1 and torch 1.6.0 (**Note: the cuda mentioned here does not necessarily mean that you need to install cuda, it
> means only the cuda bundled with pytorch instead.**). And then download the right version to some dir as you wish. 
>> Example: when your venv is equipped with cuda 10.1 and torch 1.6.0, you navigate to the [link](https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html) 
> mentioned above, and then click the downloading link named as "./mmcv_full-1.4.4-cp38-cp38-win_amd64.whl" for downloading
> when your OS is Windows, or "./mmcv_full-1.4.4-cp38-cp38-manylinux1_x86_64.whl" when your OS is Linux-based. This assures
> that you would download the latest version.

> Navigate to another directory other than the current project directory in the terminal, because we need to git clone a repo named as **mmcv**, 
> which is a essential repo for **mmdetection**, onto this directory.
>> command `git clone https://github.com/open-mmlab/mmcv.git` under the directory you chose
>>
>> command `cd mmcv`
>>
>> command `pip install -r requirements.txt`
>>
> In this way, you have installed mmcv dependencies and keep the project directory clean as well.

> Navigate to the dir where you have downloaded the wheel for mmcv (say you downloaded "./mmcv_full-1.4.4-cp38-cp38-win_amd64.whl")
> in the terminal, and then command `pip install mmcv_full-1.4.4-cp38-cp38-win_amd64.whl`.

> Install mmdetection of a specific version from source

>> Command `git clone -b dev-v2.19.0 https://github.com/open-mmlab/mmdetection.git` under the same directory in the terminal 
> where the downloaded wheel for mmcv is located

>> Command `cd mmdetection`

>> Command `pip install -r requirements`

>> Command `pip install -v -e .`  

> Validation the installation using the corresponding [repo](https://github.com/open-mmlab/mmdetection) of 
> corresponding branch version and the code snippet given in the 
> [official instruction](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html)

> Sometimes more adaptations of the [repo](https://github.com/open-mmlab/mmdetection) need to be implemented, which can
> be referenced to the [useful instruction](https://blog.csdn.net/qq_26755205/article/details/115999553) mentioned above

### FAQs:
> **torchvision==0.8.0+cu101** for **torch==1.7.0+cu101** cannot be found in the official resources. As an alternative,
> you can refer to **torch==1.6.0+cu101**, which is also feasible in out project.
## Others
### Check CUDA version (cross-platform applicable)
- `nvidia-smi` - This driver API
- `nvcc --version` - This runtime API reports the CUDA runtime version, which actually reflects your expectation.

The difference between them see 
[this link](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi).

### Shortcut keys for convenience in coding
Helpful [links](https://www.pythonforbeginners.com/comments/shortcut-to-comment-out-multiple-lines-in-python) to 
shortcuts for commenting out in different IDEs.

- In Pycharm, press `ctrl+shift+/` to comment out multiple lines.
- In Jupyter Notebook, press `ctrl+/` to comment out multiple lines.
