### Generative Adversarial Network (GAN) Model

This script implements a Generative Adversarial Network (GAN) using PyTorch. It consists of two neural networks, the Generator and the Discriminator, which are trained simultaneously in a competitive setting. The Generator creates fake images from random noise, while the Discriminator tries to distinguish between real and fake images. The goal is to train the Generator to produce images that are indistinguishable from real images, according to the Discriminator.

Usage:
    `python gan.py --dataset <dataset_name> --dataroot <path_to_dataset> [options]`

Arguments:
```
    --dataset (str): The dataset to use. Options include 'cifar10', 'lsun', 'mnist', 'imagenet', 'folder', 'lfw', 'fake'.
    --dataroot (str): Path to the dataset (required for most datasets).
    --workers (int): Number of data loading workers (default: 2).
    --batchSize (int): Input batch size (default: 64).
    --imageSize (int): Height/width of the input image to the network (default: 64).
    --nz (int): Size of the latent z vector (default: 100).
    --ngf (int): Number of generator filters (default: 64).
    --ndf (int): Number of discriminator filters (default: 64).
    --niter (int): Number of epochs to train for (default: 25).
    --lr (float): Learning rate (default: 0.0002).
    --beta1 (float): Beta1 for Adam optimizer (default: 0.5).
    --cuda (bool): Enable CUDA (default: False).
    --dry-run (bool): Check a single training cycle works (default: False).
    --ngpu (int): Number of GPUs to use (default: 1).
    --netG (str): Path to the generator model to continue training (default: '').
    --netD (str): Path to the discriminator model to continue training (default: '').
    --outf (str): Folder to output images and model checkpoints (default: '.').
    --manualSeed (int): Manual seed for random number generation.
    --classes (str): Comma-separated list of classes for the LSUN dataset (default: 'bedroom').
    --mps (bool): Enable macOS GPU training (default: False).
```

Example:
    `python gan.py --dataset cifar10 --dataroot ./data --batchSize 128 --niter 50`

The script initializes the datasets, creates DataLoader objects, sets up the generator and discriminator networks, and runs the training loop, logging the loss for both networks and saving generated images at specified intervals.

## Installation
Ensure you have Python 3.7 or higher installed on your machine.

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

```
docker build -t dcgan:1.0.<minor-version> .
docker run -it dcgan:1.0.<minor-version>
```
