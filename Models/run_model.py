# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 21:05:54 2022

@author: SuperWZL
"""

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

# import other useful python packages
import os
import PIL
import numpy as np
import pandas as pd
import mmcv
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from app.config import cfg  # since app is the base package
import streamlit as st

@st.cache(allow_output_mutation=True)
def init_model(config, checkpoint):
    model = init_detector(config, checkpoint, device='cpu')
    return model

def inference(model, test_imgs, processed_filename, processed_img):
    result = inference_detector(model, test_imgs)
    model.show_result(test_imgs, result, score_thr=0.5, show=False,
                win_name=processed_filename,
                out_file=processed_img)