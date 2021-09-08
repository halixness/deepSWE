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

You can use `python train.py -h` to check out the parameters.
Load a dataset from disk (PARFLOOD format), filter dynamic sequences and train deepSWE@32 filters:
```
python train.py -root /path -epochs 200 -future_frames 4 -filters 32 -dynamicity 0.5
```

### Testing

You can use `python test.py -h` to check out the parameters.
Load a dataset from disk (PARFLOOD format), filter dynamic sequences, train deepSWE@32 filters and apply 5cm of approximation for the accuracy:
```
python test.py -root /path -weights model.weights -filters 32 -dynamicity 0.5 -accuracy_threshold 0.05
```

**Please note:** the network hyperparameters (filters, lr) in the test script, must be compatible with the ones used during training.


## Credits

- [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch)
- [Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
- [Andreas Holm Nielsen](https://holmdk.github.io/)


