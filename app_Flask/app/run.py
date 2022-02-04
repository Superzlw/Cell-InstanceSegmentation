# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:40:27 2022

@author: SuperWZL
"""

from flask import Flask, render_template, redirect, url_for, send_from_directory, send_file, request, jsonify
from flask_uploads import IMAGES, UploadSet, configure_uploads
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField, FileRequired, FileAllowed

import numpy as np
from redis import Redis
import os
import time
import gdown

from rq import Queue
from rq.job import Job
from worker import conn
import pathlib2
from config import cfg  # since app is the base package
from mmdet_model import run_model

def download_from_google():
    """
    download the checkpoint-file from google drive
    """
    try:
        gdown.download(cfg.URL, cfg.CHECKPOINT, quiet=True)
    except Exception:
        raise Exception('Something wrong while downloading the checkpoint file')
    else:
        print("downloading the checkpoint file: finish")


current_dir = pathlib2.Path.cwd()
project_dir = current_dir.parent
ds_path = project_dir / 'test_imgs'
print(f"the path to the folder holding the testing imgs: {ds_path}")

app = Flask(__name__)

#app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
app.config['UPLOADED_IMAGES_DEST'] = 'static/temp_imgs'
app.config['DOWNLOAD_IMAGES_DEST'] = 'static/temp_processed_imgs'
app.config['PREFERRED_URL_SCHEME'] = 'https'

task_queue = Queue('detector',connection=conn)

images = UploadSet('images', IMAGES)
configure_uploads(app, images)

class Upload(FlaskForm):
    image = FileField('image', validators=[
        FileRequired(),
        FileAllowed(images, 'Images only!')])
    submit = SubmitField('Submit')

class Download(FlaskForm):
    submit = SubmitField('Download')
    
    
@app.route('/', methods=['GET', 'POST'])
def index():
    upload_form = Upload()
    if request.method == 'POST':
        if not 'mask_rcnn_r50_20e_compet.pth' in os.listdir('../../checkpoints'):
            download_from_google()
        if upload_form.validate_on_submit():
            filename = images.save(upload_form.image.data)
            processed_filename = f"detected_{filename}"
            img = os.path.join(app.config['UPLOADED_IMAGES_DEST'], filename)
            print(img)
            print(os.listdir(app.config['UPLOADED_IMAGES_DEST']))
            processed_img = os.path.join(app.config['DOWNLOAD_IMAGES_DEST'], processed_filename)
            print("processed-", processed_img)
            
            test_imgs = "/".join(img.split('/')[1:])
            processed_filename = processed_filename
            run_model(cfg.CONFIG, cfg.CHECKPOINT, img, processed_filename, processed_img)
            return redirect(url_for('show', picture=filename, processed_filename=processed_filename))
    else:
        return render_template('index.html', form=upload_form)

    
@app.route('/show/<picture>/<processed_filename>', methods=['GET', 'POST'])    
def show(picture, processed_filename):
    download_form = Download()
    print(os.listdir(app.config['DOWNLOAD_IMAGES_DEST']))
    print(processed_filename)
    job_complete = processed_filename in os.listdir(app.config['DOWNLOAD_IMAGES_DEST'])
    filename = f"{picture}"
    processed_filename = f"detected_{picture}"
    img = f"{app.config['UPLOADED_IMAGES_DEST']}{filename}"
    processed_img = f"{app.config['UPLOADED_IMAGES_DEST']}{processed_filename}"
    if download_form.validate_on_submit() and job_complete:
        print(processed_img)
        return send_from_directory(app.config['DOWNLOAD_IMAGES_DEST'], processed_filename, as_attachment=True)
    return render_template('show.html', pic=filename, job_complete=job_complete, form=download_form)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=os.getenv("PORT"),debug=True,use_reloader=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


