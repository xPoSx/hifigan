# HiFiGAN
## How to train the model


### Preparations
```shell
!git clone https://github.com/xPoSx/hifigan.git
!pip install -r hifigan/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

!wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
!tar -xjf LJSpeech-1.1.tar.bz2
!mv LJSpeech-1.1 hifigan/
```
### Train
```shell
!cd hifigan && python3 train.py
```
### Weights
Веса обученного генератора хранятся в файле hifigan165957
### Test
Запуск генератора с подгрузкой весов на тестовых и тренировочных данных (для вторых нужно скачать датасет) в ноутбуке Hifigan_test.ipynb (в гите не отображаются аудио, поэтому лучше запустить).
