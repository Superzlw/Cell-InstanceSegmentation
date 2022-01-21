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
#from run_model import get_prediction
from utils.utils_func import download_from_google, to_excel, init_folder, downlaod_result, zipFiles, save4download
from Models.run_model import init_model, inference

def main():   
    st.title('Cell Instance Segmentation')
    source = ("Mask RCNN", "Demo-Faster RCNN")
    source_index = st.sidebar.selectbox("Model", range(
        len(source)), format_func = lambda x: source[x])
    #st.write(os.listdir(cfg.TEMP_ORIGNAL))
    if source_index == 0:
        init_folder(cfg.TEMP_ORIGNAL)
        init_folder(cfg.TEMP_PROCESSED)
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
        #st.write(os.listdir(cfg.TEMP_ORIGNAL))
        print('valid')
        if st.button("Start testing"):
            if source_index == 0:
                if not 'mask_rcnn_r50_20e_compet.pth' in os.listdir('./checkpoints'):
                    download_from_google()
                with st.spinner(text='Preparing Image...'):
                    processed_imgs = []
                    model = init_model(cfg.CONFIG, cfg.CHECKPOINT)
                    processed_filenames = ['ALL']
                    processed_filename2res = {}
                    imgs = os.listdir(cfg.TEMP_ORIGNAL)
                    #st.write(uploaded_files)
                    for img in uploaded_files:
                        processed_filename = f'processed_{img.name}'
                        img = os.path.join(cfg.TEMP_ORIGNAL, img.name)
                        processed_img = os.path.join(cfg.TEMP_PROCESSED, processed_filename)
                        inference(model, img, processed_filename, processed_img)
                        #res_df = get_prediction(cfg.CONFIG, cfg.CHECKPOINT, img,
                        #                         processed_filename, processed_img)
                        processed_imgs.append(processed_img)
                        processed_filenames.append(processed_filename)
                        #processed_filename2res[processed_filename] = res_df
                st.image(processed_imgs)
                #with st.form(key='Select Image(s)'):
                    #st.write(os.listdir(cfg.TEMP_PROCESSED))
                selected_option = st.multiselect("Select one or more options:",processed_filenames)
                if 'ALL' in selected_option:
                    processed_filenames.remove('ALL')
                    selected_option = processed_filenames
                save4download(selected_option)
                zipFiles()
                    #submit_button = st.form_submit_button(label='Submit')
                #if submit_button:
                    #downlaod_result(processed_filename2res, processed_filenames, selected_option)
                path = os.path.join(cfg.TEMP, 'download/result.zip')
                with open(path, "rb") as fp:
                    btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="result.zip",
                        mime="application/zip"
                    )

if __name__ == '__main__':
    main()