# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:15:49 2022

@author: SuperWZL
"""

import requests
import os
import argparse
from PIL import Image
import pandas as pd
import streamlit as st
from utils.config import cfg
from run_model import get_prediction
from utils.utils_func import download_from_google, to_excel, init_folder

def main():
    init_folder(cfg.TEMP_ORIGNAL)
    init_folder(cfg.TEMP_PROCESSED)
    
    st.title('Cell Instance Segmentation')
    source = ("Mask RCNN", "Demo-Faster RCNN")
    source_index = st.sidebar.selectbox("Model", range(
        len(source)), format_func = lambda x: source[x])
    
    if source_index == 0:
        uploaded_files = st.sidebar.file_uploader("upload Image", type=['png', 'jpg'], accept_multiple_files = True)
        if not uploaded_files == []:
            is_valid = True
            with st.spinner(text = 'Image Loading...'):
                for uploaded_file in uploaded_files:
                    st.sidebar.image(uploaded_file)
                    up_img = Image.open(uploaded_file)
                    up_img.save(os.path.join(cfg.TEMP_ORIGNAL, uploaded_file.name))
                    #img = f'app/static/temp_imgs/{uploaded_file.name}'
        else:
            is_valid = False
    else:
        pass
            
    if is_valid:
        st.write(os.listdir(cfg.TEMP_ORIGNAL))
        print('valid')
        if st.button("Start testing"):
            if source_index == 0:
                if not 'mask_rcnn_r50_20e_compet.pth' in os.listdir('./checkpoints'):
                    download_from_google()
                with st.spinner(text='Preparing Image...'):
                    processed_imgs = []
                    processed_filenames = []
                    processed_filename2res = {}
                    imgs = os.listdir(cfg.TEMP_ORIGNAL)
                    for img in imgs:
                        processed_filename = f'processed_{img}.png'
                        img = os.path.join(cfg.TEMP_ORIGNAL, img)
                        processed_img = os.path.join(cfg.TEMP_PROCESSED, processed_filename)                    
                        res_df = get_prediction(cfg.CONFIG, cfg.CHECKPOINT, img,
                                                 processed_filename, processed_img)
                        processed_imgs.append(processed_img)
                        processed_filenames.append(processed_filename)
                        processed_filename2res[processed_filename] = res_df
                    st.image(processed_imgs)
                    processed_filenames.append('ALL')
                    selected_option = []
                    selected_option = st.multiselect("Select one or more options:",processed_filenames)
                    if 'ALL' in selected_option:
                        processed_filenames.remove('ALL')
                        selected_option = processed_filenames
                    if not selected_option == []:
                        res_lst = [processed_filename2res[selected_filename] for selected_filename in selected_option]
                        res_out_df = pd.concat(res_lst)
                        res_excel = to_excel(res_out_df)
                        st.download_button(label='Download the Result(.xlxs)', data=res_excel,
                           file_name='result.xlsx')
            else:
                pass
if __name__ == '__main__':
    main()