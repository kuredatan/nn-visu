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

# Final pipeline (tests on deconvoluted feature maps)

Call:

```bash
python2.7 process_model.py --tmodel conv --trained 1 --trun final --batch 32 --tdata siamese --lr 0.001 --optimizer Adam --loss categorical_crossentropy --epoch 10
```

For analysis using BoW, SIFT descriptors and Harris detectors on feature map in *conv/convfeature_map_layer_conv1.png* against images from dataset *CATS*:

```bash
python3.6 analysis_fmaps.py --tname conv/convfeature_map_layer_conv1.png --tdata CATS --tmethod {bow|harris|sift}
```

# Python and modules

see requirements.txt. run

```bash
(sudo apt-get install python-pip)
pip install -r requirements.txt
```

Python version is 2.7.9 for the deconvolution. Python version is 3.6.5 for the analysis.

For cyvlfeat:

```bash
wget -N http://www.di.ens.fr/willow/teaching/recvis/assignment1/install_cyvlfeat.py
python2.7 install_cyvlfeat.py
```
