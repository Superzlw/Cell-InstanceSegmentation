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
import sys 
#sys.path.append("..")
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from Models.m_rcnn import inference, show_image
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def main():   
    st.title('cell instance segmentor')
    source = ("Mask RCNN", "Demo-Faster RCNN")
    source_index = st.sidebar.selectbox("Model", range(
        len(source)), format_func = lambda x: source[x])
    thr = st.sidebar.slider('Threshold', 0., 1., 0.5, 0.05)
    if source_index == 0:
        #init_folder(cfg.TEMP_ORIGNAL)
        #init_folder(cfg.TEMP_PROCESSED)
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
        cols = st.columns(3)
        cols[0].header('Orignal input-images')
        cols[1].header('Processed with label')
        cols[2].header('Processed only mask')
        processed_filenames = ['ALL']
        if st.button("Start testing"):
            if source_index == 0:
                if not 'mask_rcnn_r50_20e_compet.pth' in os.listdir('./checkpoints'):
                    download_from_google()
                #model = init_model(cfg.CONFIG, cfg.CHECKPOINT)
                with st.spinner(text='Preparing Image...'):
                    processed_imgs = []
                                        
                    processed_filename2res = {}
                    imgs = os.listdir(cfg.TEMP_ORIGNAL)
                    #st.write(uploaded_files)
                    for img in uploaded_files:
                        processed_filename = f'processed_{img.name}'
                        img = os.path.join(cfg.TEMP_ORIGNAL, img.name)
                        processed_img = os.path.join(cfg.TEMP_PROCESSED, processed_filename)
                        #values = test_inf(model, img)
                        values = inference(cfg.CONFIG, cfg.CHECKPOINT, img)
                        show_image(cfg.CONFIG, cfg.CHECKPOINT, img,values, processed_filename, processed_img, thr)
                        processed_imgs.append(processed_img)
                        processed_filenames.append(processed_filename)
                        #processed_filename2res[processed_filename] = res_df
                        cols[0].image(img)
                        cols[1].image(processed_img)
                        cols[2].image(os.path.join(cfg.TEMP_PROCESSED, f'mask_{processed_filename}'))
                #st.image(processed_imgs)
                #with st.form(key='Select Image(s)'):
                    #st.write(os.listdir(cfg.TEMP_PROCESSED))
                selected_option = st.multiselect("Select one or more options:",processed_filenames)
                if 'ALL' in selected_option:
                    processed_filenames.remove('ALL')
                    selected_option = processed_filenames
                print(selected_option)
                save4download(selected_option)
                zipFiles()
                    #submit_button = st.form_submit_button(label='Submit')
                #if submit_button:
                    #downlaod_result(processed_filename2res, processed_filenames, selected_option)
                path = os.path.join(cfg.TEMP, 'result.zip')
                with open(path, "rb") as fp:
                    btn = st.download_button(
                        label="Download ZIP",
                        data=fp,
                        file_name="result.zip",
                        mime="application/zip"
                            )

if __name__ == '__main__':
    main()