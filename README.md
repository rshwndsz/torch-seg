# Torch-Seg

<img src="https://img.shields.io/github/v/tag/rshwndsz/torch-seg?include_prereleases&label=version"></img> 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10QGnghYxi0h4oMQ0JcSt_FAX8Ez0VTTe)

A simple, flexible & bare-bones PyTorch pipeline for segmentation

## Installation

Start by cloning the repository

```bash
git clone https://github.com/rshwndsz/torch-seg.git
```

If you don't already have an environment with PyTorch 1.x, it's better to create a new conda environment with Python 3.6+.

Install `conda-env` if you haven't already.

```bash
conda install -c conda conda-env
```

To create a new environment and install required packages available on conda, run

```console
$ conda env create
Fetching package metadata: ...
Solving package specifications: .Linking packages ...
[      COMPLETE      ] |#################################################| 100%
#
# To activate this environment, use:
# $ source activate torchseg-env
#
# To deactivate this environment, use:
# $ source deactivate
#
```

Move into your new environment

```console
$ source activate torchseg-env
(torchseg-env) $
```

Install PyTorch and the CUDA Toolkit based on your local configuration. Get the command for the installation from [the official website](https://pytorch.org/).

The command for a Linux system with a GPU and CUDA 10.1 installed is given below.

```console
(torchseg-env) $ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Install packages not available through conda using pip

```console
(torchseg-env) $ pip install -r requirements.txt
```

## Setup

Get the [Multi-organ Kumar](https://monuseg.grand-challenge.org/Data/) dataset by running the download script

```console
(torchseg-env) $ ./install.sh
```

or download your own dataset into `dataset/raw/`.

Edit the dataset APIs in `torchseg/data.py` and `test.py` as required.

## Training

To train the model defined in `torchseg/model.py` using default parameters run

```console
(torchseg-env) $ python train.py
```

This trains the model for 10 epochs and saves the best model (based on validation loss) in `checkpoints/model-saved.pth`

You can specify a lot of parameters as command line arguments.
To find out which parameters can be provided run

```console
(torchseg-env) $ python train.py --help
```

## Testing

To test the model run

```console
(torchseg-env) $ python test.py
```
