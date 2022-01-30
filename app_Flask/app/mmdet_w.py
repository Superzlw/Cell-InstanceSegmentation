# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:18:08 2022

@author: SuperWZL
"""

import requests
import os
from os import chdir, system, path
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, show_result_pyplot


def run_model(config, checkpoint, test_imgs,
                   processed_filename="demo.png", processed_img="demo_detected.png"):
    print("Start inf")
    model = init_detector(config, checkpoint, device='cpu')
    result = inference_detector(model, test_imgs)

    model.show_result(test_imgs, result, score_thr=0.5, show=False,
                win_name=processed_filename,
                out_file=processed_img) 
    print("end_inf")
    
def create_pred_result(model, test_imgs):
    for test_img in test_imgs:
        result = inference_detector(model, test_img)
        name_id = os.path.basename(test_img).split('.')[0]
        yield name_id, result