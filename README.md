# MobileNetV2-PoseEstimation

Apr 29, 2019 Under construction  
**[Caution] The behavior of RraspberryPi+NCS2 is very unstable. It looks like a bug in the API but is not detected properly.**

## Introduction
This repository has its own implementation, impressed by ildoonet's achievements.  
Thank you, **[ildoonet](https://github.com/ildoonet)**.  
**https://github.com/ildoonet/tf-pose-estimation.git**  
  
I will make his implementation even faster with CPU only.  

## Environment
- Ubuntu 16.04 x86_64
- OpenVINO 2019 R1.0.1
- Tensorflow v1.12.0 + Tensorflow Lite
- USB Camera
- Python 3.5

## Environment construction and training procedure
**[Learn "Openpose" from scratch with MobileNetv2 + MS-COCO and deploy it to OpenVINO/TensorflowLite Part.1](https://qiita.com/PINTO/items/2316882e18715c6f138c)**  

## Core i7 only + OpenVINO + Openpose + Sync mode (disabled GPU)
![01](media/01.gif)  

## Usage
```console
$ git clone https://github.com/PINTO0309/MobileNetV2-PoseEstimation.git
$ cd MobileNetV2-PoseEstimation
```
**CPU - Sync Mode**  
```console
$ python3 openvino-usbcamera-cpu-ncs2-sync.py -d CPU
```
**NCS2 - Sync Mode**  
```console
$ python3 openvino-usbcamera-cpu-ncs2-sync.py -d MYRIAD
```
  
**CPU - Async Mode**  
```console
$ python3 openvino-usbcamera-cpu-ncs2-async.py -d CPU
```
**NCS2 - Async - Single Stick Mode**  
```console
$ python3 openvino-usbcamera-cpu-ncs2-async.py -d MYRIAD
```
**NCS2 - Async - Multi Stick Mode**  
```console
$ python3 openvino-usbcamera-cpu-ncs2-async.py -d MYRIAD -numncs 2
```
## Reference articles, Very Thanks!!
**https://github.com/ildoonet/tf-pose-estimation.git**  
**https://www.tensorflow.org/api_docs/python/tf/image/resize_area**  
**[Python OpenCVの基礎 resieで画像サイズを変えてみる - Pythonの学習の過程とか - ピーハイ](http://peaceandhilightandpython.hatenablog.com/entry/2016/01/09/214333)**  
**[Blurring and Smoothing - OpenCV with Python for Image and Video Analysis 8](https://youtu.be/sARklx6sgDk?t=228)**  
**https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/**  
**https://teratail.com/questions/169393**  
