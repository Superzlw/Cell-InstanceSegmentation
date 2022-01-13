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
import gdown

ids = '1I2WrZhlF5ABDBuPXj0Yoj9KGZ_dhzNw0'
url = f'https://drive.google.com/uc?id={ids}'
checkpoint_path = './checkpoints/mask_rcnn_r50_20e_compet.pth'
config = './configs/mask_rcnn/mask_rcnn_r50_fpn_compet.py'

def down_from_google():
    with st.spinner(text='downing model...'):
        gdown.download(url, checkpoint_path, quiet=True)
    print("finish")
    

def main():
    st.title('Project Test')
    source = ("Mask RCNN", "Faster RCNN")
    
    source_index = st.sidebar.selectbox("Input", range(
        len(source)), format_func = lambda x: source[x])
    
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader("upload Image", type=['png', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text = 'Image Loading...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'app/static/temp_imgs/{uploaded_file.name}')
        else:
            is_valid = False
    else:
        pass
            
    if is_valid:
        print('valid')
        if st.button("Start testing"):
            if source_index == 0:
                down_from_google()
                with st.spinner(text='Preparing Image'):
                    model = init_detector(config, checkpoint_path, device='cuda:0')
                    result = inference_detector(model, picture)
                    processed_img = './app/static/temp_processed_imgs/processed_img.png'
                    processed_filename = 'processed_img.png'
                    model.show_result(picture, result, score_thr=0.5, show=False,
                                win_name=processed_filename,
                                out_file=processed_img)
                    st.write("Text end")
                    st.balloons()
            else:
                pass
if __name__ == '__main__':
    main()