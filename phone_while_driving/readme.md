
# Phone while Driving Detection Model

## Description

It is an algorithm ,where it detects if a person is using a phone or not while driving. It makes a bounding box around the person,phone and wheel.

### Dataset
Dataset wals collected from roboflow -https://universe.roboflow.com/pruebadeteccioncelulareschoferes/cellphone-detector-drivers-ma3vx

Extraxt train,test,val and data.yaml and put them in your local folder


## Deployment

To deploy this project you can create an virtual environment(venv)

I have used anaconda prompt
```bash
  conda create -n spit python=3.10 anaconda 
```
```bash
  conda activate spit
```




## Installation

Insure you have python 3.10 or higher

```bash
  pip install ultralytics
```
```bash
  python
```
```bash
  import torch
  torch.__version__
  exit()
```
if have  nvidia gpu then

 ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
else
```bash
  pip3 install torch torchvision torchaudio
```

## Run

```bash
  yolo task=detect mode=train epochs=100 data=phonedata.yaml model=yolov8m.pt imgsz=640 batch=8
```
```bash
  yolo task =detect mode=predict model=best.pt show=True conf=0.5 source="C:\others\phone_while_driving\test\images\1.jpg" save_txt=True line_thickness=1
```
Reduce the number of epochs if you are running on CPU as it will take much time to train