# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:15:49 2022

@author: SuperWZL
"""

import requests
from os import chdir, system, path
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, show_result_pyplot
import argparse
from PIL import Image
import streamlit as st
from utils.config import cfg
import gdown
from run_model import get_prediction
from utils.utils_func import download_from_google, to_excel, remove_files
import pandas as pd

ids = '1I2WrZhlF5ABDBuPXj0Yoj9KGZ_dhzNw0'
url = f'https://drive.google.com/uc?id={ids}'
checkpoint_path = './checkpoints/mask_rcnn_r50_20e_compet.pth'
config = './configs/mask_rcnn/mask_rcnn_r50_fpn_compet.py'

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
                up_img = Image.open(uploaded_file)
                up_img.save(f'app/static/temp_imgs/{uploaded_file.name}')
                img = f'app/static/temp_imgs/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        pass
            
    if is_valid:
        st.write(os.listdir('app/static/temp_imgs'))
        print('valid')
        if st.button("Start testing"):
            if source_index == 0:
                if not 'mask_rcnn_r50_20e_compet.pth' in os.listdir('./checkpoints'):
                    download_from_google()
                with st.spinner(text='Preparing Image...'):
                    processed_img = './app/static/temp_processed_imgs/processed_img.png'
                    processed_filename = 'processed_img.png'
                    res_df = get_prediction(cfg.CONFIG, cfg.CHECKPOINT, img,
                                             processed_filename, processed_img)
                    res_excel = to_excel(res_df)
                    st.image(processed_img)
                    st.download_button(label='Download the Result(.xlxs)', data=res_excel,
                       file_name=f'{uploaded_file.name}.xlsx')
                st.write('tmp')
            else:
                pass
if __name__ == '__main__':
    main()