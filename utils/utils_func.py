# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:17:19 2022

@author: SuperWZL
"""

import gdown
from utils.config import cfg
import streamlit as st

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