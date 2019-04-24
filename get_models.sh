#!/usr/bin/env bash

mkdir ./ehpi_action_recognition/data
mkdir ./ehpi_action_recognition/data/models
cd ./ehpi_action_recognition/data/models

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/ehpi_v1.pth
wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/yolov3.weights
wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/pose_resnet_50_256x192.pth.tar