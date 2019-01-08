# Source code

This [GitHub](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DeconvNet)

# Data and weights

Create a folder **data/** at the root of the repo and put inside the folder **weights/** with the weights associated with VGG (trained on ImageNet: save the weights obtained with the Keras model VGG, see *models/models.py*) and the two small CNNs "Vonc" and "Conv" (trained with CIFAR-10 training set).

+ For "Conv" model, at the end of training:

```bash
python2.7 process_model.py --tmodel conv --tdata CIFAR-10 --trun training --trained 0 --epoch 10 --lr 0.0001 --optimizer Adam --batch 128
```

FINAL | TRAINING | VALIDATION |
---------|--------|-------------------------
ACCURACY | 0.566 | 0.515
LOSS | 1.213 | 1.344

+ For "Vonc" model, at the end of training:

```bash
python2.7 process_model.py --tmodel vonc --tdata CIFAR-10 --trun training --trained 0 --epoch 250 --lr 0.0001 --optimizer Adam --batch 128
```

FINAL | TRAINING | VALIDATION |
---------|--------|-------------------------
ACCURACY | 0.818 | 0.758
LOSS | 0.517 | 0.707

# Python and modules

see requirements.txt. run

```bash
(sudo apt-get install python-pip)
pip install -r requirements.txt
```

Python version is 2.7.9 

For cyvlfeat:

```bash
wget -N http://www.di.ens.fr/willow/teaching/recvis/assignment1/install_cyvlfeat.py
python2.7 install_cyvlfeat.py
```
