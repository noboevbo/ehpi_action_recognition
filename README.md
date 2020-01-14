This repository contains the code for our real-time pose based action recognition pipeline, which was introduced in our paper [Simple yet efficient real-time pose-based action recognition](https://arxiv.org/abs/1904.09140) (submitted to ITSC 2019, as of 2019-04-23). There is also paper result reconstruction code for another publication (submitted to Transactions on ITS SE ITSC 2018, as of 2019-04-23), XYZ, based on an alternative approach using LSTM.

The approach is based on encoding human poses over a period of time in an image-like data format (Encoded Human Pose Images) that can then be classified using standard CNNs. The entire pipeline is lightweight and runs 20-60 FPS, depending on settings. It was tested on an XMG Ultra 17 Laptop with an i7-8700 Desktop CPU, NVIDIA GTX1080 GPU, 32GB RAM using KDE Neon (Ubuntu 18.04) and CUDA 10.0 with CUDNN 7.5.

We currently provide example code and a pre-trained model for our use-case action recognition which required us to differentiate between the actions idle, walk and wave. We provide training code etc. which can be used to train your own models with different / more actions. To generate a EHPI see the feature_vec_producer for an EHPI vector / the action recognition network. I plan to provide source code for our database powered dataset management tool which contains importers and exporters for different datasets / formats, but I need a little cleanup time for this.
- Visualization of our use case evaluation sequence: https://youtu.be/xSIaUd_nJMM
- Subsequence with EHPI overlay and training data examples: https://youtu.be/UqzLqjS0OR4

![EHPI Example](/images/ehpi_example.jpg)

# Installation
## Prerequisites
- Python 3.6+
- CUDA (tested with 9.0 and 10.0)
- CUDNN (tested with 7.5)
- PyTorch (tests with 1.0)
- OpenCV with Python bindings

A basic setup guide for Ubuntu 18.04 is available at: https://dennisnotes.com/note/20180528-ubuntu-18.04-machine-learning-setup/.
I set up my system like this, with the difference that I now use CUDA 10.0 and CUDNN 7.5, the blogpost will be updated sometime.

Note: The code runs on Windows but with decreased performance, see Known Bugs.

## Setup
I use two of my libraries in this code, nobos_commons and nobos_torch_lib. These and their dependencies have to be installed first. In the following code example I assume a Python installation with virtualenvwrapper, if this is not used the code must be adapted accordingly.
A new virtual environment is created in the code, then PyTorch (with CUDA 10.0) is installed and then the required repositories cloned, dependencies installed and finally the required model weights loaded from our web server. The weights for YoloV3 and 2D Human Pose Recognition are originally from https://github.com/ayooshkathuria/pytorch-yolo-v3 and https://github.com/Microsoft/human-pose-estimation.pytorch. We have the weights on our server to ensure availability and version.

```bash
git clone https://github.com/noboevbo/nobos_commons.git
git clone https://github.com/noboevbo/nobos_torch_lib.git
git clone https://github.com/noboevbo/ehpi_action_recognition.git
mkvirtualenv ehpi_action_recognition -p python3 --system-site-packages
workon ehpi_action_recognition
pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip install torchvision
pip install -r nobos_commons/requirements.txt 
pip install -r nobos_torch_lib/requirements.txt
pip install -r ehpi_action_recognition/requirements.txt  
pip install --upgrade nobos_commons
pip install --upgrade nobos_torch_lib
cd ehpi_action_recognition
sh get_models.sh

```
An example showing the whole pipeline on the webcam can be executed as follows:
```bash
export PYTHONPATH="~/path/to/ehpi_action_recognition:$PYTHONPATH"
python ehpi_action_recognition/run_ehpi.py
```
I haven't adapted the whole thing to the command line yet, changes can be made in the code. Examples for training and evaluation can be found in the files "train_ehpi.py" and "evaluate_ehpi.py".

## Configuration Options

There are some configuration options available in run_ehpi.py:

- image_size = ImageSize(width=640, height=360): The image size to be used. Higher resolutions usually help Yolo to detect objects.
- camera_number = 0: The webcam id
- fps = 30: FPS which should be used for the input source (webcam or image folder)
- buffer_size = 20: The size of the action buffer, in this project not really used, just the detected humans from frame n-1.
- action_names = [Action.IDLE.name, Action.WALK.name, Action.WAVE.name]: The corresponding names to the action class vector outputed by the action recognition network. Need to be updated when you train your own models with different action classes.
- use_action_recognition = True: Turns the action recognition on / off
- use_quick_n_dirty = False: If set to true it deactivates the object recognition completly after a human skeleton has been found. Continues to track this skeleton but won't recognize new humans. Improves the performance by a huge margin.

## Known Bugs
- The code runs on Windows, but there is somewhere a bug, so the whole thing runs on our system with only 10-30% of the FPS on Linux (Ubuntu 18.04).
- When use_quick_n_dirty is set to zero there's sometimes a merge bug where a person gets two skeletons assigned.

# Reconstruct paper results
This repository contains code for our (submitted, as of 23.04.2019) publication on ITSC 2019 and ITS Journal Special Issue ITSC 2018. As the EHPI publication is not yet published and citable, we have used an LSTM approach for action recognition for the ITS Journal publication, which is based on the normalized EHPI inputs. We want to ensure that the results can be reproduced from our papers. Therefore, we provide our training and evaluation code in this repository. The results in our papers are reported as mean values from five training sessions with different seeds. As seeds we use 0, 104, 123, 142 and 200. We use fixed values so that the results are 100% reproducible, seeds 142 and 200 are randomly selected, 0 and 123 are seeds often used in other work and 104 is our office room number. 

## IEEE Intelligent Transportation Systems Conference (ITSC 2019)
- Our datasets are available here: https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_datasets.tar.gz
- Our trained models are available here: https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_models.tar.gz

Here is an example of the standard setup that should allow our training and evaluation code to be used directly:
```bash
mkdir ./ehpi_action_recognition/data
mkdir ./ehpi_action_recognition/data/datasets
mkdir ./ehpi_action_recognition/data/models

cd ./ehpi_action_recognition/data/datasets

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_datasets.tar.gz
tar -xvf itsc_2019_datasets.tar.gz

cd ../models

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/itsc_2019_models.tar.gz
tar -xvf itsc_2019_models.tar.gz

```
Here is the direct link to the training code for the JHMDB dataset: [JHMDB Training Code](ehpi_action_recognition/paper_reproduction_code/trainings/ehpi/train_ehpi_itsc_2019_jhmdb.py)<br/>
And here to the evaluation code: [JHMDB Evaluation Code](ehpi_action_recognition/paper_reproduction_code/evaluations/ehpi/test_ehpi_itsc_2019_jhmdb.py)

Here is the direct link to the training code for the Use Case dataset: [Use Case Training Code](ehpi_action_recognition/paper_reproduction_code/trainings/ehpi/train_ehpi_itsc_2019_ofp.py)</br>
And here to the evaluation code: [Use Case Evaluation Code](ehpi_action_recognition/paper_reproduction_code/evaluations/ehpi/test_ehpi_itsc_2019_ofp.py)

## IEEE Transactions on Intelligent Transportation Systems - Special Issue 21st IEEE Intelligent Transportation Systems Conference (ITSC 2018)
- Our datasets are available here: https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_datasets.tar.gz
- Our trained models are available here: https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_lstm_models.tar.gz

Here is an example of the standard setup that should allow our training and evaluation code to be used directly:
```bash
mkdir ./ehpi_action_recognition/data
mkdir ./ehpi_action_recognition/data/datasets
mkdir ./ehpi_action_recognition/data/models

cd ./ehpi_action_recognition/data/datasets

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_datasets.tar.gz
tar -xvf its_2019_datasets.tar.gz

cd ../models

wget https://cogsys.reutlingen-university.de/pub/files/2019_04_ehpi/its_2019_lstm_models.tar.gz
tar -xvf its_2019_lstm_models.tar.gz

```
Here is the direct link to the training code for the both datasets (ActionSim and Office): [ITS Training Code](ehpi_action_recognition/paper_reproduction_code/trainings/lstm/train_its_journal_2019.py)<br/>
And here to the evaluation code: [ITS Evaluation Code](ehpi_action_recognition/paper_reproduction_code/evaluations/lstm/test_its_journal_2019.py)

# Citation
Please cite the following papers if this code is helpful in your research.
Currently the publications to this repository are submitted, but not yet accepted or published. I will update the entries as soon as I have feedback about the submissions. A preprint for the ITSC 2019 publication is available [here](https://arxiv.org/abs/1904.09140) on arxiv.org.

Edit 2019-06-25: The EHPI ITSC 2019 publication is accepted and will be presented at ITSC 2019 (Oct 27-30). 
```bash
D. Ludl, T. Gulde, and C. Curio, “Simple yet efficient real-time pose-based action recognition,” in 22nd Int. Conf. on Intelligent Transportation Systems (ITSC), 2019, pp. 581–588, doi: 10.1109/ITSC.2019.8917128.
```


# Open Source Acknowledgments
I used parts of the following open source projects in my code:

- YoloV3: https://github.com/ayooshkathuria/pytorch-yolo-v3
- 2D Human Pose Estimation: https://github.com/Microsoft/human-pose-estimation.pytorch
- Imbalanced dataset sampler: https://github.com/ufoym/imbalanced-dataset-sampler/

Thank you for making this code available!
