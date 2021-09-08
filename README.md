<img src="info/logo.png" width="512"/>
A framework to learn Shallow Water Equations with Deep Neural Networks.
<br/><br/>

<img src="https://github.com/halixness/deepSWE/blob/main/info/prediction_2.gif" width="250" height="250"/> <img src="https://github.com/halixness/deepSWE/blob/main/info/prediction_baganza.gif" width="250" height="250"/> 

## Introduction

This framework aims to provide tools for experiments on physics-informed deep neural networks. It stems from the research conducted at the University of Parma from the group of prof. Alessandro Dal Palú, in collaboration with the [HyLab research team](http://www.hylab.unipr.it/it/). 

- [Wiki (in Italian)](https://github.com/halixness/deepSWE/wiki)
- [Experiments](https://github.com/halixness/deepSWE/tree/main/runs)
- [Models](https://github.com/halixness/deepSWE/tree/main/models)

## Running experiments

Two scripts are available to either train or test the model:

- `train.py`: loads the dataset, initiates a model with random weights and performs training. The outputs are plotted for each epoch and saved in the session folder under `runs/` (train_XXX)
- `test.py`: loads the dataset, loads the weights from a trained network, generates new samples from a test dataset. The results are plotted and saved ina the session folder under `runs/` (eval_XXX)

### Training

Loads a .npy dataset and performs training:
```
python train.py -dset arda_dataset.npy -epochs 100 -lr 0.001
```

### Testing

Loads a .npy dataset and performs training:
```
python test.py -dset arda_dataset.npy -weights model.weights
```

**Please note:** the latent space size, defined with `-ls`, must be compatible with the latent space size defined during the training phase (thus the network weights).


## Credits

- [julianstastny/VAE-ResNet18-PyTorch](https://github.com/julianstastny/VAE-ResNet18-PyTorch)
- [thunil/Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction)
- [Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)

