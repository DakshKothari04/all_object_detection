# Helmet Detection Application

## Description

The Helmet Detection Application is designed to detect whether a person is wearing a helmet or not in images using a pre-trained model. It utilizes a Haar cascade classifier to identify helmet features and provides visual feedback by highlighting the detected helmets. The application also includes a helmet detection model trained using the provided dataset.

## Features

- Helmet detection in images
- Visual feedback with highlighted helmet regions
- Pre-trained helmet detection model (`helmetmodel.h5`)
- Haar cascade files for helmet detection (`haar_cascade_files`)
- Additional helmet detection model (`helmodel.h5`)

## Installation

Ensure you have Python 3.8.10 installed. You can install the required packages using pip:

Install all library to run :

``pip install -r requirements.txt``


```bash
pip install opencv-python==4.7.0.72
pip install numpy==1.23.0
pip install keras==2.12.0
pip install pygame==2.4.0
pip install pickle==0.7.5
pip install random


## Run code
python3 helmet_detect.py

##For Training the Helmet Detection Model
pip install random
pip install pickle
pip install opencv-python
pip install keras
pip install numpy

