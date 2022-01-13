# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:15:49 2022

@author: SuperWZL
"""

import requests
from os import chdir, system, path
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, show_result_pyplot
import argparse
from PIL import Image
import streamlit as st
from app.config import cfg
import gdown
from run_model import get_prediction
from utils.utils_func import download_from_google

ids = '1I2WrZhlF5ABDBuPXj0Yoj9KGZ_dhzNw0'
url = f'https://drive.google.com/uc?id={ids}'
checkpoint_path = './checkpoints/mask_rcnn_r50_20e_compet.pth'
config = './configs/mask_rcnn/mask_rcnn_r50_fpn_compet.py'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    
    args = parser.parse_args()
    return args
  

def main():
    st.title('DS2-Project: Cell Instance Segmentation')
    source = ("Mask RCNN", "Demo-Faster RCNN")
    source_index = st.sidebar.selectbox("Model", range(
        len(source)), format_func = lambda x: source[x])
    
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader("upload Image", type=['png', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text = 'Image Loading...'):
                st.sidebar.image(uploaded_file)
                img = Image.open(uploaded_file)
                img = img.save(f'app/static/temp_imgs/{uploaded_file.name}')
        else:
            is_valid = False
    else:
        pass
            
    if is_valid:
        print('valid')
        if st.button("Start testing"):
            if source_index == 0:
                download_from_google()
                with st.spinner(text='Preparing Image...'):
                    processed_img = './app/static/temp_processed_imgs/processed_img.png'
                    processed_filename = 'processed_img.png'
                    res_csv = get_prediction(cfg.CONFIG, cfg.CHECKPOINT, img,
                                             processed_filename, processed_img)
                    '''
                    model = init_detector(cfg.CONFIG, cfg.CHECKPOINT, device='cpu')
                    result = inference_detector(model, img)
                    model.show_result(img, result, score_thr=0.5, show=False,
                                win_name=processed_filename,
                                out_file=processed_img)
                    st.write("Text end")
                    st.balloons()
                    '''
                    st.image(processed_img)
                    st.download_button(label='download .csv file', data=res_csv,
                       file_name=f'{uploaded_file.name}.csv')
            else:
                pass
if __name__ == '__main__':
    main()