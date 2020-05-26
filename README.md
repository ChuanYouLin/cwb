# 實戰教學
## 安裝環境
Be sure you have miniconda(anaconda).
Create environment:
```
conda env create -f environment.yaml
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