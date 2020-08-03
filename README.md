# CycleGAN-Unpaired-Image-translation 
With tensorflow 1.0.0
Image to Image translation involves generating a new synthetic version of a given image with some modifications. Obtaining and constructing paired image datasets is  expensive and sometimes impossible. This project implements Cycle GAN to achieve image translation from Apple to Orange and Horse to Zebra and vice versa.

## Introduction

Training a model for image-to-image translation typically requires a large dataset of paired examples.These datasets can be difficult and expensive to prepare, and in some cases impossible, such as photographs of paintings by long dead artists. The CycleGAN is a technique that involves the automatic training of image-to-image translation models without paired examples. The models are trained in an unsupervised manner using a collection of images from the source and target domain that do not need to be related in any way.

