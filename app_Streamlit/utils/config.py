# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:18:52 2022

@author: SuperWZL
"""

from easydict import EasyDict as edict
import os


_C = edict()
cfg = _C

# Set the config path relative to the main.py
#_C.CONFIG = '../configs/mask_rcnn/mask_rcnn_r50_fpn_compet.py'
_C.CONFIG = './configs/mask_rcnn/mask_rcnn_r50_fpn_compet.py'
# Set the path to the candidate checkpoint relative to the main.py
#_C.CHECKPOINT = '../checkpoints/mask_rcnn_r50_20e_compet.pth'
_C.CHECKPOINT = './checkpoints/mask_rcnn_r50_20e_compet.pth'

# URL of Google Drive
_C.IDS = '1sDUUANLST4wCZUVwnTRKwsTkJgcXJRJY'
_C.URL = f'https://drive.google.com/uc?id={_C.IDS}'

# Path for uploaded imgs and processed imgs
_C.TEMP = 'app_Streamlit/static/temp_imgs'# here add 'app_Streamlit/' before deployment
_C.TEMP_PROCESSED = os.path.join(_C.TEMP, 'processed_imgs')
_C.TEMP_ORIGNAL = os.path.join(_C.TEMP, 'orignal_imgs')

# post processing parameters
_C.RANDOM_ORDER = False  # when true, the value of c_order does not matter
_C.C_ORDER = True  # when true, read row first, otherwise, read column first
_C.THRESHOLDS = [.15, .35, .55]
_C.MIN_PIXELS = [75, 150, 75]

#allowed image formats
_C.ALLOWED_EXTENSIONS = {'png', 'tif'}

#show the resulting image with masks printed after prediction
_C.SHOW_RESULT = True