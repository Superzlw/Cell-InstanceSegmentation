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
import matplotlib.pyplot as plt
import mmcv
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from app_Streamlit.utils.config import cfg
import streamlit as st
import time

@st.cache(allow_output_mutation=True)
def init_model(config, checkpoint):
    model = init_detector(config, checkpoint, device='cpu')
    return model

def tmp_inference(model, test_imgs, processed_filename, processed_img):
    result = inference_detector(model, test_imgs)
    model.show_result(test_imgs, result, score_thr=0.5, show=False,
                win_name=processed_filename,
                out_file=processed_img)
    
def create_pred_result(model, test_imgs):
    for test_img in test_imgs:
        result = inference_detector(model, test_img)
        name_id = os.path.basename(test_img).split('.')[0]
        yield name_id, result

@st.cache
def inference(model, test_img):
    values = inference_detector(model, test_img)
    return values

@st.cache
def show_image(model, test_img, values, processed_filename, processed_img, thr):
    #for test_img in test_imgs:
    #values = _inference(model, test_img)
    model.show_result(test_img, values, score_thr=thr, show=False,
            win_name=processed_filename,
            out_file=processed_img)
    bbox_lst, mask_lst = values
    if len(mask_lst[0]) >= len(mask_lst[1]) and len(mask_lst[0]) >= len(mask_lst[2]):
        img_type = 1
    elif len(mask_lst[1]) >= len(mask_lst[2]):
        img_type = 2
    else:
        img_type = 3
    img_mask_lst = [mask+0 for i, mask in enumerate(mask_lst[img_type-1]) if bbox_lst[img_type-1][i][4]>=thr]
    left_img = plt.imread(processed_img)#(520,704,3)
    #left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
    right_img = left_img.copy()#(520,704,3)
    sum_mask = sum(img_mask_lst)
    sum_mask[sum_mask>1]=1
    zero_dim = np.zeros(sum_mask.shape)
    sum_mask=np.expand_dims(sum_mask,axis=2).repeat(3,axis=2)
    sum_mask[:,:,1] = zero_dim
    sum_mask[:,:,2] = zero_dim
    sum_mask *= 255
    cv2.addWeighted(sum_mask, 0.6, left_img, 1 - 0.6, 0, right_img, dtype=cv2.CV_32F)
    final_img = np.concatenate((left_img, right_img), axis=1)
    save_path = os.path.join(cfg.TEMP_PROCESSED, f'mask_{processed_filename}')
    cv2.imwrite(save_path, sum_mask)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        