# Bootstrapping-Object-Pose-Detection
Repository for class project in CMU 10-807 (Deep Learning)

Members of Project:
* Tushar Agrawal
* Cole Gulino
* Erik Sjoberg

Dataset:
* LineMOD Dataset: http://campar.in.tum.de/Main/StefanHinterstoisser
* Berkeley Instance Recognition Dataset: http://rll.berkeley.edu/bigbird/
* ShapeNet: https://shapenet.org/
-- Link to find all of synsetId: http://image-net.org/archive/words.txt or http://wordnetweb.princeton.edu/perl/webwn3.0
-- Ex: airplane, aeroplane, plane -> 02691156
* Voxel representation using binvox: http://www.patrickmin.com/binvox/
-- Binvox command example: ./binvox ModelNet10/bathtub/train/bathtub_0101.off -d 50 -cb

Folders:
* /autoencoder
-- Simple autoencoder implementation on mnist dataset
* /vae
-- Simple variational autoencoder implementation on mnist dataset
-- Paper: https://arxiv.org/pdf/1312.6114v10.pdf