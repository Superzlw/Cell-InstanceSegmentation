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

def download_from_google():
    try:
        with st.spinner(text='downing model...'):
            gdown.download(cfg.URL, cfg.CHECKPOINT, quiet=True)
    except Exception:
        st.error('Something wrong while downloading the checkpoint-file')
    else:
        print("finish")
    
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

def remove_files(folder):
    files_lst = os.listdir(folder)
    for file in files_lst:
        file_path = os.path.join(folder, file)
        os.remove(file_path)
        
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)
    else:
        pass





















