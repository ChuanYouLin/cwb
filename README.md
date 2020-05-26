# 實戰教學
## 安裝環境
Be sure you have installed miniconda or anaconda.<br>
直接安裝我們提供的環境:
```
# Create environment
conda env create -f environment.yaml
```
開一個環境裝Pytorch:
```
# Create environment
conda create --name cwb python=3.6
# Pytorch no_gpu for windows/linux
conda install pytorch torchvision cpuonly -c pytorch
```
## 預測pm2.5
Check usage:
```
python main.py -h
```
Evaluation only:
```
python main.py --do_eval
```
Training and evaluation:
```
python main.py --do_train --do_eval
```
## 數字辨識