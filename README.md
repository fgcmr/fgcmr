# Multi-Model Network for Fine-Grained Cross-Media Retrieval
Introduction
------------
This is the source code for our paper **Multi-Model Network for Fine-Grained Cross-Media Retrieval**.

Network Architecture
--------------------
The architecture of our proposed model is as follows
![network](network.png)

Installation
-------------
* **Requirement**

    - pytorch, tested on [v1.2]
    - CUDA, tested on v10.0
    - Language: Python 3.6

How to use
-------------
The code is currently tested only on GPU.

* **Download dataset**

Please download from this page. http://59.108.48.34/tiki/FGCrossNet/

* **Demo**

    If you want to quickly test the performance, please follow subsequent steps:
    
    - Download trained model from 
    ```
    wget https://fgcmr.oss-cn-hongkong.aliyuncs.com/audio.pkl
    wget https://fgcmr.oss-cn-hongkong.aliyuncs.com/common_network.pkl
    wget https://fgcmr.oss-cn-hongkong.aliyuncs.com/image.pkl
    wget https://fgcmr.oss-cn-hongkong.aliyuncs.com/text.pkl
    wget https://fgcmr.oss-cn-hongkong.aliyuncs.com/video.pkl
    ```
    - Modify CUDA_VISIBLE_DEVICES to proper cuda device id in test.sh

    - Activate virtual environment(e.g. conda) and then run the script `bash test.sh`

* **Source Code**

    If you want to train the whole network from begining using source code, please follow subsequent steps:
    - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `train_image.sh`
    - Activate virtual environment(e.g. conda) and then run the script `bash train_image.sh`
    ------------
    - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `train_video.sh`
    - Activate virtual environment(e.g. conda) and then run the script `bash train_video.sh`
    ------------
    - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `train_audio.sh`
    - Activate virtual environment(e.g. conda) and then run the script `bash train_audio.sh`
    ------------
    - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `train_text.sh`
    - Activate virtual environment(e.g. conda) and then run the script `bash train_text.sh`
    ------------
    - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `train_common.sh`
    - Activate virtual environment(e.g. conda) and then run the script `bash train_common.sh`
    ------------
    - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `test.sh`
    - Activate virtual environment(e.g. conda) and then run the script `bash test.sh`

