# Car-Designs-Generator-using-GAN
Car Design generator using GAN and Kaggle Principles 
This project implements a Generative Adversarial Network (GAN) to generate realistic car images using the Kaggle Cars dataset. The GAN consists of a generator and a discriminator, which are trained adversarially to improve each other's performance.

Table of Contents
Installation
Dataset
Model Architecture
Training
Results

Required Libraries
The project requires the following libraries:
numpy
pandas
pillow
matplotlib
seaborn
tensorflow
opencv-python
opencv-contrib-python

Dataset
The dataset used in this project is the Kaggle Cars dataset, attached with in the main file

Model Architecture
The GAN consists of two main components:

Generator: Takes random noise as input and generates images.
Discriminator: Takes an image as input and outputs whether it is real or fake.
Generator
The generator model transforms random noise into 32x32 car images using a series of dense, convolutional, and upsampling layers.

Discriminator
The discriminator model classifies 32x32 images as real or fake using a series of convolutional layers followed by dense layers.

Results
Generated images will be saved and can be visualized during and after training. The loss values of the generator and discriminator will also be printed to the console.
