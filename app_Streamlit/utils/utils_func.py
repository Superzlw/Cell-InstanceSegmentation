# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:17:19 2022

@author: SuperWZL
"""

import gdown
from utils.config import cfg
import streamlit as st
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import pandas as pd
import os
import shutil

def download_from_google():
    """
    download the checkpoint-file from google drive
    """
    try:
        with st.spinner(text='downing model...'):
            gdown.download(cfg.URL, cfg.CHECKPOINT, quiet=True)
    except Exception:
        st.error('Something wrong while downloading the checkpoint file')
    else:
        print("downloading the checkpoint file: finish")
    
def download_from_app(img_name, csv_file):
    st.download_button(label='download .csv file', data=csv_file,
                       file_name=f'{img_name}.csv')
    
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def _remove_files(folder):
    files_lst = os.listdir(folder)
    for file in files_lst:
        file_path = os.path.join(folder, file)
        os.remove(file_path)
        
def _mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)
    
def init_folder(path):
    _mkdir(path)
    _remove_files(path)
    
def downlaod_result(processed_filename2res, processed_filenames, selected_option):
    if 'ALL' in selected_option:
        processed_filenames.remove('ALL')
        selected_option = processed_filenames
    res_lst = [processed_filename2res[selected_filename] for selected_filename in selected_option]
    res_out_df = pd.concat(res_lst)
    res_excel = to_excel(res_out_df)
    st.error("可以")
    st.download_button(label='Download the Result(.xlxs)', data=res_excel,
       file_name='result.xlsx')
    
def zipFiles():
    """
    Copy the images(processed images and mask images) to be downloaded to a specific folder
    """
    target = os.path.join(cfg.TEMP, 'result')
    init_folder(target)
    orignal = os.path.join(cfg.TEMP, 'download')
    shutil.make_archive(target, 'zip', orignal)
    
def save4download(imgs_lst):
    """
    Copy the images(processed images and mask images) to be downloaded to a specific folder
    :param imgs_lst: list list of images, it contains processed images with labels
    """
    download_path = os.path.join(cfg.TEMP, 'download')
    init_folder(download_path)
    print(imgs_lst)
    if not imgs_lst == []:
        for img in imgs_lst:
            old_label = os.path.join(cfg.TEMP_PROCESSED, img)
            old_mask = os.path.join(cfg.TEMP_PROCESSED, f'mask_{img}')
            new_label = os.path.join(download_path, img)
            new_mask = os.path.join(download_path, f'mask_{img}')
            shutil.copyfile(old_label, new_label)
            shutil.copyfile(old_mask, new_mask)
        
        
    
    





















