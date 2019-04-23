I'm currently cleaning up the code, in the next few days the repository and all dependencies (especially nobos_commons and nobos_torch_lib) should be fully available, I'll then update the README with instructions etc.

# Installation
TODO: nobos_commons, nobos_torch_lib, get_models.sh

# Reconstruct paper results

## ITSC 2019

```bash
mkdir ./ehpi_action_recognition/data
mkdir ./ehpi_action_recognition/data/datasets
mkdir ./ehpi_action_recognition/data/models

cd ./ehpi_action_recognition/data/datasets

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_datasets.tar.gz

cd ../models

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_models.tar.gz

```

## ITS ITSC 2018 Special Issue

```bash
mkdir ./ehpi_action_recognition/data
mkdir ./ehpi_action_recognition/data/datasets
mkdir ./ehpi_action_recognition/data/models

cd ./ehpi_action_recognition/data/datasets

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_datasets.tar.gz

cd ../models

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_lstm_models.tar.gz

```